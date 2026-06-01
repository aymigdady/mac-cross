"""Rerank shortlists: lexical, hybrid embedding, cross-encoder."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from comparison_engine import COL_DESC, nullable_str

from .constants import (
    _CROSS_MATCH_RERANK_TOP_N,
    _HYBRID_RERANK_LEX_WEIGHT,
    _abstain_threshold,
    _attribute_filter_enabled,
)
from .text import _cross_desc_similarity_score, _normalize_cross_desc_for_match

logger = logging.getLogger(__name__)

def _cross_match_rerank_shortlist_by_similarity(
    work: pd.DataFrame,
    query_texts: List[str],
    *,
    top_n: int = _CROSS_MATCH_RERANK_TOP_N,
) -> pd.DataFrame:
    """
    Second stage after BM25: score each row with ``_cross_desc_similarity_score`` against every
    non-empty query (best score wins), sort descending, keep the top ``top_n`` rows.
    """
    if work.empty or COL_DESC not in work.columns:
        return work
    nonempty = [nullable_str(q) for q in query_texts if nullable_str(q)]
    q_norms = [_normalize_cross_desc_for_match(q) for q in nonempty]
    q_norms = [qn for qn in q_norms if len(qn) >= 2]
    if not q_norms:
        return work.head(top_n).copy() if len(work) > top_n else work.copy()
    desc_series = work[COL_DESC].fillna("").astype(str)
    scored: List[Tuple[float, int]] = []
    for i in range(len(work)):
        rn = _normalize_cross_desc_for_match(desc_series.iloc[i])
        best = 0.0
        for qn in q_norms:
            best = max(best, _cross_desc_similarity_score(qn, rn))
        scored.append((best, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    take = min(top_n, len(scored))
    order = [scored[j][1] for j in range(take)]
    return work.iloc[order].copy()

def _apply_attribute_filter_to_shortlist(
    work: pd.DataFrame,
    shortlist_labels: List[Any],
    query_desc: str,
) -> Tuple[List[Any], Dict[Any, float], Dict[Any, str]]:
    """Phase 3a: drop hard-mismatched candidates and compute soft penalties.

    Returns ``(kept_labels_in_input_order, penalty_by_label, reason_by_label)``.
    Empty / missing inputs ⇒ no-op (returns the input unchanged).

    The function tolerates *any* failure of the extractor / filter (missing
    Redis, malformed cache, import error) by returning the input unchanged so
    the cross-search pipeline stays operational even if Phase 3 is broken.
    """
    if not shortlist_labels or not query_desc or not _attribute_filter_enabled():
        return list(shortlist_labels), {}, {}
    try:
        from attributes.extractor import get_extractor
        from attributes.filters import attribute_filter_score
        from session_store import get_redis_client_for_cache_use
    except Exception:
        return list(shortlist_labels), {}, {}

    if work is None or work.empty or COL_DESC not in work.columns:
        return list(shortlist_labels), {}, {}

    redis_client = get_redis_client_for_cache_use()
    extractor = get_extractor()

    try:
        query_attrs = extractor.extract(query_desc, redis_client=redis_client)
    except Exception:
        return list(shortlist_labels), {}, {}

    try:
        candidate_descs = [
            str(work.loc[lb][COL_DESC]) if lb in work.index else "" for lb in shortlist_labels
        ]
        candidate_attrs_list = extractor.extract_many(
            candidate_descs, redis_client=redis_client
        )
    except Exception:
        return list(shortlist_labels), {}, {}

    kept: List[Any] = []
    penalty_by_label: Dict[Any, float] = {}
    reason_by_label: Dict[Any, str] = {}
    for lb, cattrs in zip(shortlist_labels, candidate_attrs_list):
        try:
            decision = attribute_filter_score(query_attrs, cattrs)
        except Exception:
            kept.append(lb)
            continue
        if not decision.keep:
            reason_by_label[lb] = decision.reason
            continue
        kept.append(lb)
        if decision.penalty > 0.0:
            penalty_by_label[lb] = decision.penalty
            reason_by_label[lb] = decision.reason
    return kept, penalty_by_label, reason_by_label

def _cross_match_cross_encoder_rerank(
    work: pd.DataFrame,
    query_texts: List[str],
    *,
    top_n: int,
    penalty_by_label: Optional[Dict[Any, float]] = None,
    abstain_callback: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """Phase 3b: cross-encoder rerank of the work-frame candidates.

    Returns ``None`` (so callers fall back to the Phase-2 hybrid lex+emb
    reranker) when:
    - the cross-encoder kill switch is off (``CCAPR_USE_CROSS_ENCODER=0``),
    - the work-frame is empty / lacks the description column,
    - or the cross-encoder import / inference fails for any reason.

    Phase 3c: when ``abstain_callback`` is provided and the top-1 score is
    below the calibrated threshold, the callback is invoked with
    ``(top_score, top_label, top_text)`` so callers can surface the
    no-confident-match state to the UI.

    The cross-encoder is the precision lever in Phase 3 — it scores the
    (query, candidate) pair jointly so paraphrase / synonym wins from
    embeddings finally survive into the candidate window.
    """
    if work.empty or COL_DESC not in work.columns:
        return None
    nonempty = [nullable_str(q) for q in query_texts if nullable_str(q)]
    if not nonempty:
        return None
    try:
        from cross_encoder.pipeline import cross_encode_and_rank
        from cross_encoder.reranker import is_cross_encoder_enabled
        from session_store import get_redis_client_for_cache_use
    except Exception:
        return None
    if not is_cross_encoder_enabled():
        return None

    desc_series = work[COL_DESC].fillna("").astype(str)
    candidates = list(zip(work.index.tolist(), desc_series.tolist()))
    try:
        ranked = cross_encode_and_rank(
            nonempty[0],
            candidates,
            redis_client=get_redis_client_for_cache_use(),
            penalty_by_label=penalty_by_label,
            top_n=int(top_n),
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Cross-encoder rerank failed; falling back to Phase 2 hybrid rerank"
        )
        return None
    if not ranked:
        return None

    if abstain_callback is not None:
        top_label, top_text, top_score = ranked[0]
        try:
            abstain_callback(
                float(top_score),
                top_label,
                top_text,
                bool(float(top_score) < _abstain_threshold()),
            )
        except Exception:
            pass  # never let an audit hook break the pipeline

    ordered_labels = [lb for lb, _, _ in ranked]
    return work.loc[ordered_labels].copy()

def _cross_match_hybrid_rerank_shortlist(
    work: pd.DataFrame,
    query_texts: List[str],
    company: str,
    *,
    top_n: int = _CROSS_MATCH_RERANK_TOP_N,
    penalty_by_label: Optional[Dict[Any, float]] = None,
    out_top_score: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Phase 2 reranker: combines lexical similarity (``_cross_desc_similarity_score``)
    with embedding cosine similarity (per row's canonical description vs the query
    embedding) using a fixed convex weight.

    Phase 3 addition: ``penalty_by_label`` (label → soft penalty in [0, 1]) is
    *subtracted* from the rerank score for that row. This is how the structured
    attribute filter discounts mismatched-but-not-clearly-wrong candidates.

    Falls back to the lexical-only reranker when:
    - embeddings are disabled (or unavailable) for ``company``,
    - the query embedding cannot be produced,
    - or the work-frame is empty / lacks the description column.

    This is the "missing piece" that lets the embedding signal survive into the
    candidate window the LLM actually sees — without it, lexical rerank discards
    paraphrase / synonym wins from the BM25-∪-embedding shortlist.
    """
    if work.empty or COL_DESC not in work.columns:
        return work
    penalty_by_label = penalty_by_label or {}

    nonempty = [nullable_str(q) for q in query_texts if nullable_str(q)]
    q_norms = [_normalize_cross_desc_for_match(q) for q in nonempty]
    q_norms = [qn for qn in q_norms if len(qn) >= 2]
    if not q_norms:
        return work.head(top_n).copy() if len(work) > top_n else work.copy()

    # Try to enable embedding rerank. Any failure ⇒ silent fallback to lexical.
    emb_query_vec = None
    company_store = None
    try:
        from embeddings.builder import embed_query
        from embeddings.hybrid import (
            get_company_store,
            is_embeddings_enabled_for_company,
        )
        from ingest.canonical_desc import canonicalize_description
        from session_store import get_redis_client_for_cache_use

        if is_embeddings_enabled_for_company(company):
            company_store = get_company_store(company)
            if company_store is not None and company_store.size > 0:
                # Phase 4A — embed *every* query (orig + paraphrases) so the
                # embedding rerank scores each candidate against the closest of
                # the queries we have. Without this, the lexical max-over-queries
                # below works (line 3386) but the embedding score only reflects
                # the original query — which is exactly the failure mode that
                # nullifies query-expansion's shortlist gains in the candidate
                # window. Embedding lookup is Redis-cached (~0.5 ms each), so
                # the per-call overhead is < 2 ms even cold.
                rc = get_redis_client_for_cache_use()
                emb_query_vecs: List[np.ndarray] = []
                for q in nonempty:
                    try:
                        v = embed_query(q, redis_client=rc)
                    except Exception:
                        v = None
                    if v is not None:
                        emb_query_vecs.append(np.asarray(v, dtype=np.float32))
                emb_query_vec = emb_query_vecs[0] if emb_query_vecs else None
    except Exception:
        emb_query_vec = None
        company_store = None
        emb_query_vecs = []

    if emb_query_vec is None or company_store is None or company_store.size == 0:
        return _cross_match_rerank_shortlist_by_similarity(
            work, query_texts, top_n=top_n
        )

    # Score the candidates (we look up each row's canonical-desc vector lazily).
    from ingest.canonical_desc import canonicalize_description

    desc_series = work[COL_DESC].fillna("").astype(str)
    w = _HYBRID_RERANK_LEX_WEIGHT
    one_minus_w = 1.0 - w
    # Pre-normalise *every* query embedding so the per-row scoring is just N
    # cheap dot products (N ≤ 4 with default QE budget).
    qvec_units: List[np.ndarray] = []
    for v in emb_query_vecs:
        n = float(np.linalg.norm(v)) or 1.0
        qvec_units.append(v / n)

    # Build a tiny look-up of canonical-desc -> stored vector index.
    # Cheap because shortlist size is ~50 rows.
    canonical_descs = [canonicalize_description(s) for s in desc_series.tolist()]
    label_by_desc = company_store._id_by_desc  # type: ignore[attr-defined]

    work_index_list = list(work.index)
    scored: List[Tuple[float, int]] = []
    for i, (raw_desc, canon) in enumerate(zip(desc_series.tolist(), canonical_descs)):
        rn = _normalize_cross_desc_for_match(raw_desc)
        lex = 0.0
        for qn in q_norms:
            lex = max(lex, _cross_desc_similarity_score(qn, rn))
        emb_sim = 0.0
        label = label_by_desc.get(canon) if canon else None
        if label is not None:
            try:
                row_vec = company_store._index.get(int(label))  # type: ignore[attr-defined]
                if row_vec is not None:
                    rv = np.asarray(row_vec, dtype=np.float32).reshape(-1)
                    rv_norm = float(np.linalg.norm(rv)) or 1.0
                    rv_unit = rv / rv_norm
                    # Phase 4A: max cosine similarity over orig + paraphrases.
                    # This is the rerank-side counterpart of the shortlist-side
                    # query expansion union — without it, candidates pulled in
                    # by a paraphrase would lose to original-query matches in
                    # the rerank stage.
                    for qu in qvec_units:
                        s = float(np.dot(rv_unit, qu))
                        if s > emb_sim:
                            emb_sim = s
                    if emb_sim < 0.0:
                        emb_sim = 0.0  # avoid penalising unrelated rows below lex-only
            except Exception:
                emb_sim = 0.0
        score = w * lex + one_minus_w * emb_sim
        # Phase 3a: subtract the soft attribute-filter penalty (e.g., -0.4 for
        # low-conf size mismatch). Hard-filter drops happened earlier in the
        # pipeline so any label still here only carries a soft penalty.
        row_label = work_index_list[i]
        score -= float(penalty_by_label.get(row_label, 0.0))
        scored.append((score, i))

    scored.sort(key=lambda x: (-x[0], x[1]))
    take = min(top_n, len(scored))
    order = [scored[j][1] for j in range(take)]
    # Phase 3 inv. — surface the top-1 score so callers can decide whether the
    # hybrid result is confident enough to skip the more expensive CE rerank.
    if out_top_score is not None and scored:
        out_top_score["top1"] = float(scored[0][0])
        if len(scored) >= 2:
            out_top_score["margin"] = float(scored[0][0] - scored[1][0])
    return work.iloc[order].copy()

