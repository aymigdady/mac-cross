"""
Hybrid recall: union of BM25 top-N and embedding top-N candidates.

The cross-search pipeline historically used BM25 alone for the top-50 shortlist.
Hard-case golden-set runs showed BM25 misses 75% of paraphrase / synonym /
brand-swap pairs because they share no rare lexical tokens. Dense embeddings
(bge-m3) are the recall fix; combining both signals is empirically better than
either alone — BM25 catches exact codes/sizes, embeddings catch semantics.

This module is the **single integration point** for the cross-search pipeline:
``hybrid_shortlist_for_query`` returns a ranked, deduped list of work-frame
labels that callers can plug straight into the existing rerank step.

Decisions:

- **Per-canonical-description vector index** (not per-row). Saves 5-10× compute
  on real ERPs. To return *row labels* we expand each top-K canonical-desc hit
  into the work-frame rows that share it, using a ``canonical_desc → labels``
  reverse map built lazily and cached on the work-frame identity.
- **Score-merge by max**, not weighted sum. Since BM25 and cosine are not on the
  same scale, the safest fusion before rerank is reciprocal-rank fusion (RRF):
  each candidate gets ``1/(k + bm25_rank) + 1/(k + emb_rank)`` and we take the
  top-N. RRF is the standard in production search systems for exactly this case.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from comparison_engine import COL_DESC
from ingest.canonical_desc import canonicalize_description

from .builder import embed_query, ensure_company_embedding_index
from .embedder import EMBEDDING_DIM
from .hnsw_store import HnswStore, hnsw_cache_root

logger = logging.getLogger(__name__)


# Knobs (env-overridable so we can A/B without redeploying).
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


HYBRID_BM25_TOP_K = _int_env("CCAPR_HYBRID_BM25_TOP_K", 30)
HYBRID_EMB_TOP_K = _int_env("CCAPR_HYBRID_EMB_TOP_K", 30)
HYBRID_UNION_MAX = _int_env("CCAPR_HYBRID_UNION_MAX", 50)
RRF_K = _int_env("CCAPR_HYBRID_RRF_K", 60)


# --- Per-company HNSW store cache (one per process) -------------------------
_STORE_CACHE: Dict[str, HnswStore] = {}
_STORE_CACHE_LOCK = threading.Lock()


def get_company_store(company: str) -> Optional[HnswStore]:
    """Return the cached :class:`HnswStore` for ``company`` (loading from disk on miss)."""
    if not company:
        return None
    with _STORE_CACHE_LOCK:
        st = _STORE_CACHE.get(company)
        if st is None:
            try:
                cache_root = hnsw_cache_root()
                if cache_root:
                    try:
                        from .gcs_hnsw_backend import download_if_missing

                        download_if_missing(company, cache_root)
                    except Exception as exc:
                        logger.warning("GCS download before open failed for %s: %s", company, exc)
                st = HnswStore.open_or_create(company, dim=EMBEDDING_DIM)
            except Exception as exc:
                logger.warning("Could not open HNSW for %s: %s", company, exc)
                return None
            _STORE_CACHE[company] = st
        return st


def reset_store_cache() -> None:
    """Clear the in-process HNSW cache (used by tests)."""
    with _STORE_CACHE_LOCK:
        _STORE_CACHE.clear()


def is_embeddings_enabled_for_company(company: str) -> bool:
    """Honor both the global kill-switch and the per-company allow-list.

    Default rollout per the Phase 2 plan: embeddings on for **MBL only**. Adjust
    via ``CCAPR_ENABLE_EMBEDDINGS_PER_COMPANY=MBL,IFAS`` (comma-separated).
    """
    if (os.environ.get("CCAPR_ENABLE_EMBEDDINGS") or "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    allow_raw = (os.environ.get("CCAPR_ENABLE_EMBEDDINGS_PER_COMPANY") or "MBL").strip()
    allow = {c.strip().upper() for c in allow_raw.split(",") if c.strip()}
    return company.strip().upper() in allow


# --- Reverse-map cache (canonical_desc -> work-frame labels) ----------------
_REVERSE_MAP_CACHE: Dict[int, Dict[str, List[Any]]] = {}
_REVERSE_MAP_CACHE_LOCK = threading.Lock()


def _reverse_map_for(work: pd.DataFrame) -> Dict[str, List[Any]]:
    """Lazy ``canonical_desc → list[work_label]`` map, cached on work-frame id."""
    cache_key = id(work)
    with _REVERSE_MAP_CACHE_LOCK:
        cached = _REVERSE_MAP_CACHE.get(cache_key)
        if cached is not None:
            return cached
    out: Dict[str, List[Any]] = {}
    if work is None or work.empty or COL_DESC not in work.columns:
        with _REVERSE_MAP_CACHE_LOCK:
            _REVERSE_MAP_CACHE[cache_key] = out
        return out
    descs = work[COL_DESC].astype(str).tolist()
    labels = list(work.index)
    for label, desc in zip(labels, descs):
        cd = canonicalize_description(desc)
        if not cd:
            continue
        bucket = out.get(cd)
        if bucket is None:
            out[cd] = [label]
        else:
            bucket.append(label)
    with _REVERSE_MAP_CACHE_LOCK:
        # Defensive bound on cache size — only retain recent maps.
        if len(_REVERSE_MAP_CACHE) > 8:
            _REVERSE_MAP_CACHE.clear()
        _REVERSE_MAP_CACHE[cache_key] = out
    return out


# --- Public API --------------------------------------------------------------


def embedding_top_labels_for_query(
    company: str,
    work: pd.DataFrame,
    query_text: str,
    *,
    top_k: int = HYBRID_EMB_TOP_K,
    redis_client=None,
) -> List[Tuple[Any, float]]:
    """Run the embedding lookup only. Returns ``[(work_label, similarity), ...]``.

    Empty list if embeddings are disabled, the index is missing, or the query
    canonicalizes to empty.
    """
    if not query_text:
        return []
    store = get_company_store(company)
    if store is None or store.size == 0:
        return []
    qvec = embed_query(query_text, redis_client=redis_client)
    if qvec is None:
        return []
    hits = store.search(qvec, top_k=int(top_k))
    if not hits:
        return []
    rev = _reverse_map_for(work)
    out: List[Tuple[Any, float]] = []
    for h in hits:
        labels = rev.get(h.canonical_desc) or []
        for lb in labels:
            out.append((lb, h.similarity))
    return out


def hybrid_shortlist_labels(
    company: str,
    work: pd.DataFrame,
    bm25_labels_in_rank_order: List[Any],
    embedding_label_scores: List[Tuple[Any, float]],
    *,
    union_max: int = HYBRID_UNION_MAX,
    rrf_k: int = RRF_K,
) -> List[Any]:
    """Reciprocal-Rank-Fusion of BM25 and embedding shortlists.

    ``bm25_labels_in_rank_order`` is what the current pipeline already produces.
    ``embedding_label_scores`` is the output of :func:`embedding_top_labels_for_query`.

    Both inputs are already restricted to labels in ``work.index``; the function
    returns up to ``union_max`` labels in the fused order.
    """
    if not bm25_labels_in_rank_order and not embedding_label_scores:
        return []

    work_index_set = set(work.index)

    bm25_rank_by_label: Dict[Any, int] = {}
    for rank, lb in enumerate(bm25_labels_in_rank_order):
        if lb in work_index_set and lb not in bm25_rank_by_label:
            bm25_rank_by_label[lb] = rank

    # Embedding hits may include duplicate (label, sim) when several canonical
    # descs map to the same label — keep the best similarity.
    emb_best_sim: Dict[Any, float] = {}
    for lb, sim in embedding_label_scores:
        if lb not in work_index_set:
            continue
        prev = emb_best_sim.get(lb)
        if prev is None or sim > prev:
            emb_best_sim[lb] = sim

    # Embedding rank: order by similarity desc.
    emb_rank_by_label: Dict[Any, int] = {}
    for rank, (lb, _sim) in enumerate(
        sorted(emb_best_sim.items(), key=lambda kv: kv[1], reverse=True)
    ):
        emb_rank_by_label[lb] = rank

    all_labels = set(bm25_rank_by_label) | set(emb_rank_by_label)
    if not all_labels:
        return []

    fused: List[Tuple[Any, float]] = []
    for lb in all_labels:
        score = 0.0
        if lb in bm25_rank_by_label:
            score += 1.0 / (rrf_k + bm25_rank_by_label[lb])
        if lb in emb_rank_by_label:
            score += 1.0 / (rrf_k + emb_rank_by_label[lb])
        fused.append((lb, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return [lb for lb, _ in fused[: int(union_max)]]


def explain_hybrid_membership(
    bm25_labels_in_rank_order: List[Any],
    embedding_label_scores: List[Tuple[Any, float]],
    fused_labels: List[Any],
) -> Dict[Any, str]:
    """For each fused label, return a one-token provenance: ``bm25``, ``emb``, or ``hybrid``.

    Useful for the per-row audit trail (Phase 1's ``_set_cross_search_audit``).
    """
    bm25_set = set(bm25_labels_in_rank_order)
    emb_set = {lb for lb, _ in embedding_label_scores}
    out: Dict[Any, str] = {}
    for lb in fused_labels:
        in_b = lb in bm25_set
        in_e = lb in emb_set
        if in_b and in_e:
            out[lb] = "hybrid"
        elif in_b:
            out[lb] = "bm25"
        elif in_e:
            out[lb] = "emb"
        else:
            out[lb] = "?"
    return out
