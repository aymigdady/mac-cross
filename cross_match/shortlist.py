"""BM25 + embedding hybrid shortlist for cross-match."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from comparison_engine import nullable_str
from bm25_erp_index_cache import get_or_build_erp_bm25_index

from .text import tokenize_cross_desc_for_bm25

logger = logging.getLogger(__name__)

def _cross_match_expand_query(query: str) -> List[str]:
    """Phase 4A — domain-aware query expansion.

    Returns ``[query, paraphrase1, paraphrase2, ...]`` (always non-empty when
    the input is non-empty, with the original query as the first element).

    Each call is a Redis cache lookup; on a miss, one Haiku call. Results are
    cached forever per canonical-description hash so the steady-state cost is
    a single Redis ``GET`` (~0.3 ms).

    Hard fail-safe: any failure (Redis down, no API key, LLM error, parse
    error) returns ``[query]`` so the rest of the pipeline runs unchanged.
    The caller passing ``[query]`` to the shortlist builder is exactly the
    pre-Phase-4 baseline behaviour, so this helper can never *reduce* recall.
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        from query_expansion import expand_query, is_query_expansion_enabled
    except Exception:
        return [q]
    if not is_query_expansion_enabled():
        return [q]
    try:
        from session_store import get_redis_client_for_cache_use

        redis_client = get_redis_client_for_cache_use()
    except Exception:
        redis_client = None
    try:
        out = expand_query(q, redis_client=redis_client)
    except Exception:
        return [q]
    return out or [q]

def _cross_match_bm25_shortlist_index_labels(
    store: Dict[str, Any],
    tab: str,
    hist: pd.DataFrame,
    work: pd.DataFrame,
    query_texts: List[str],
    *,
    per_query_top_k: int = 50,
    max_total_rows: int = 500,
) -> Optional[List[Any]]:
    """
    BM25 shortlist over the full ERP tab: top ``per_query_top_k`` rows per non-empty query,
    merged by max score, capped at ``max_total_rows``. Returns index labels valid for ``work.loc``.
    """
    fp = nullable_str(store.get("erp_file_sha256"))
    if not fp:
        return None
    nonempty_queries = [nullable_str(q) for q in query_texts if nullable_str(q)]
    if not nonempty_queries:
        return None
    try:
        bm25_ix = get_or_build_erp_bm25_index(fp, tab, hist)
    except Exception:
        return None
    if bm25_ix.bm25 is None:
        return None
    work_index = work.index
    aggregate_max: Dict[Any, float] = {}
    for q in nonempty_queries:
        toks = tokenize_cross_desc_for_bm25(q)
        if not toks:
            continue
        scores = bm25_ix.scores_for_query_tokens(toks)
        if scores.size == 0:
            continue
        top_pos = np.argsort(-scores)[:per_query_top_k]
        for pos in top_pos:
            pos_i = int(pos)
            if pos_i < 0 or pos_i >= len(bm25_ix.row_index_labels):
                continue
            label = bm25_ix.row_index_labels[pos_i]
            if label not in work_index:
                continue
            s = float(scores[pos_i])
            prev = aggregate_max.get(label)
            if prev is None or s > prev:
                aggregate_max[label] = s
    if not aggregate_max:
        return None
    ordered = sorted(aggregate_max.keys(), key=lambda lb: aggregate_max[lb], reverse=True)
    ordered = [lb for lb in ordered if lb in work_index]
    return ordered[:max_total_rows]

def _cross_match_hybrid_shortlist_index_labels(
    store: Dict[str, Any],
    tab: str,
    hist: pd.DataFrame,
    work: pd.DataFrame,
    query_texts: List[str],
    *,
    bm25_per_query_top_k: int = 30,
    embedding_per_query_top_k: int = 30,
    max_total_rows: int = 50,
) -> Optional[List[Any]]:
    """
    Phase-2 hybrid shortlist: BM25 top-N ∪ embedding top-N, fused via reciprocal-rank.

    Falls back transparently to the BM25-only path when the company has no
    embedding index, or when the global / per-company embedding flag is off,
    or when any embedding step fails. This preserves the **rollback** acceptance
    criterion: ``CCAPR_ENABLE_EMBEDDINGS=0`` returns identical results to Phase 1.
    """
    bm25_labels = _cross_match_bm25_shortlist_index_labels(
        store,
        tab,
        hist,
        work,
        query_texts,
        per_query_top_k=bm25_per_query_top_k,
        max_total_rows=max_total_rows * 5,  # leave headroom for the union/dedupe
    ) or []

    if not _embeddings_hybrid_available_for_tab(tab):
        return bm25_labels[:max_total_rows] if bm25_labels else None

    nonempty_queries = [nullable_str(q) for q in query_texts if nullable_str(q)]
    if not nonempty_queries:
        return bm25_labels[:max_total_rows] if bm25_labels else None

    try:
        from embeddings.hybrid import (
            embedding_top_labels_for_query,
            hybrid_shortlist_labels,
        )

        from session_store import get_redis_client_for_cache_use

        redis_client = get_redis_client_for_cache_use()
        emb_label_scores: List[Tuple[Any, float]] = []
        for q in nonempty_queries:
            emb_label_scores.extend(
                embedding_top_labels_for_query(
                    tab,
                    work,
                    q,
                    top_k=embedding_per_query_top_k,
                    redis_client=redis_client,
                )
            )
        fused = hybrid_shortlist_labels(
            tab,
            work,
            bm25_labels,
            emb_label_scores,
            union_max=max_total_rows,
        )
        if fused:
            return fused
    except Exception:
        logging.getLogger(__name__).exception(
            "Hybrid embedding shortlist failed for tab %s; falling back to BM25 only", tab
        )
    return bm25_labels[:max_total_rows] if bm25_labels else None

def _embeddings_hybrid_available_for_tab(tab: str) -> bool:
    """Cheap pre-check — never imports the embedder unless the flags say yes."""
    try:
        from embeddings.hybrid import (
            get_company_store,
            is_embeddings_enabled_for_company,
        )
    except Exception:
        return False
    if not is_embeddings_enabled_for_company(tab):
        return False
    store = get_company_store(tab)
    return store is not None and store.size > 0

