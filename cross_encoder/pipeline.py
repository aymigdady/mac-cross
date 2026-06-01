"""
Pipeline glue between the cross-search shortlist and the cross-encoder reranker.

This is the *only* file the cross-search code needs to import from
:mod:`cross_encoder`. Keeping the integration point narrow makes it trivial to
swap implementations later (Cohere Rerank API, fine-tuned models, etc.) without
touching ``app.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .reranker import (
    CrossEncoderReranker,
    get_reranker,
    is_cross_encoder_enabled,
)

logger = logging.getLogger(__name__)


def cross_encode_and_rank(
    query: str,
    candidates: Sequence[Tuple[Any, str]],
    *,
    redis_client=None,
    reranker: Optional[CrossEncoderReranker] = None,
    penalty_by_label: Optional[Dict[Any, float]] = None,
    top_n: Optional[int] = None,
) -> List[Tuple[Any, str, float]]:
    """Score ``(query, candidate_text)`` pairs with the cross-encoder.

    ``candidates`` is ``[(label, text), ...]``. Returns
    ``[(label, text, score), ...]`` sorted by score descending. ``label`` is
    opaque to this function — typically the work-frame index label.

    ``penalty_by_label`` (Phase 3a) is *subtracted* from the cross-encoder score
    before sorting. Hard-filtered candidates should not appear in
    ``candidates`` in the first place (they were dropped earlier in the
    pipeline); this only carries the soft penalties forward.

    Returns an empty list when:
    - ``CCAPR_USE_CROSS_ENCODER=0`` (kill switch),
    - the model fails to load,
    - or any inference exception (which is logged and returned as an empty
      list so the caller falls back to the Phase 2 hybrid rerank).
    """
    if not query or not candidates or not is_cross_encoder_enabled():
        return []

    rr = reranker or get_reranker()
    texts = [t or "" for _, t in candidates]
    try:
        scores = rr.score_pairs(query, texts, redis_client=redis_client)
    except Exception:
        logger.exception("Cross-encoder scoring failed; caller should fall back to Phase 2 rerank")
        return []

    if scores.size == 0:
        return []

    penalty_by_label = penalty_by_label or {}
    out: List[Tuple[Any, str, float]] = []
    for (label, text), s in zip(candidates, scores):
        adjusted = float(s) - float(penalty_by_label.get(label, 0.0))
        out.append((label, text, adjusted))
    out.sort(key=lambda x: x[2], reverse=True)
    if top_n is not None:
        out = out[: int(top_n)]
    return out
