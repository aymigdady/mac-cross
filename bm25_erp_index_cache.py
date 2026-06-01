"""
Phase 5 — BM25 ERP index cache (Redis, same lifecycle as ERP parse cache).

- **One index per (file fingerprint, company tab)**. Fingerprint is SHA-256 of the uploaded
  workbook bytes — identical to ``get_erp_parse_cache`` / ``set_erp_parse_cache``.
- **TTL** uses ``session_store.ERP_CACHE_TTL`` (``CCAPR_ERP_CACHE_TTL``, default **24 hours**).
  A new daily export changes the hash → new keys → new indexes; old keys expire without
  manual invalidation.
- **Cold parse**: BM25 is built inside ``_store_historical_df`` / ``_store_historical_multi`` as
  soon as DataFrames are written to the session (single paid build per upload; compare loads
  from Redis).
- **ERP parse cache hit**: warm with ``ensure_erp_bm25_cache_for_store`` so after a restart the
  session DataFrame may be rehydrated from Redis while in-process BM25 is gone; Redis then
  serves or rebuilds the index.
- **Serialization**: values are **pickle** blobs — fine for internal Redis where writer and
  reader are the same app. Pickle is **not** safe across arbitrary Python upgrades: after
  bumping the interpreter in the Docker image, flush stale BM25 keys before traffic resumes, e.g.::

    redis-cli --scan --pattern 'ccapr:erp:bm25:*' | xargs redis-cli del

Session key ``erp_file_sha256`` is set on every successful ERP upload (including ERP parse
cache hits) so callers can resolve the fingerprint without re-reading the file.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, Optional

import pandas as pd

from bm25_erp_index import (
    COMPANY_TABS,
    ErpBm25Index,
    build_erp_bm25_index_for_dataframe,
    normalize_company_tab,
)
from session_store import get_erp_bm25_index_blob, set_erp_bm25_index_blob

logger = logging.getLogger(__name__)

_BM25_INDEX_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

# Bump this whenever the tokenizer, corpus document shape, or index parameters
# change in a way that makes pre-existing cached indexes unsafe to reuse.
# The version is baked into the Redis cache key so old keys silently expire and
# new builds replace them — no manual ``redis-cli del 'ccapr:erp:bm25:*'`` needed.
#
# Version history:
#   1 — initial layout (description + item-no, k1=1.5, b=0.75)
#   2 — phase-1 cross-search: SKU-preserving + light-stem tokenizer, deduped corpus
_BM25_INDEX_VERSION = 2


def erp_bm25_cache_key_description(fingerprint: str, company_tab: str) -> str:
    """Human-readable key pattern (actual Redis key is built in ``session_store``)."""
    tab = normalize_company_tab(company_tab)
    return f"ccapr:erp:bm25:v{_BM25_INDEX_VERSION}:{fingerprint}:{tab}"


def _loads_index(blob: bytes) -> Optional[ErpBm25Index]:
    try:
        obj = pickle.loads(blob)
        return obj if isinstance(obj, ErpBm25Index) else None
    except Exception as exc:
        logger.warning("BM25 index pickle load failed: %s", exc)
        return None


def load_erp_bm25_index_cached(fingerprint: str, company_tab: str) -> Optional[ErpBm25Index]:
    blob = get_erp_bm25_index_blob(fingerprint, company_tab, _BM25_INDEX_VERSION)
    if not blob:
        return None
    return _loads_index(blob)


def save_erp_bm25_index_cached(fingerprint: str, company_tab: str, index: ErpBm25Index) -> None:
    blob = pickle.dumps(index, protocol=_BM25_INDEX_PICKLE_PROTOCOL)
    set_erp_bm25_index_blob(fingerprint, company_tab, blob, _BM25_INDEX_VERSION)


def _index_matches_tab_frame(hit: ErpBm25Index, tab: str, df: pd.DataFrame) -> bool:
    if normalize_company_tab(hit.company_tab) != tab:
        return False
    return hit.size == len(df)


def ensure_erp_bm25_cache_for_store(fingerprint: str, store: Dict[str, Any]) -> None:
    """
    Ensure BM25 indexes exist in Redis for this upload fingerprint.

    Uses session ``historical_by_company`` (multi-tab) or ``historical`` (single-tab). Skips
    tabs with no DataFrame rows. On Redis hit, skips rebuild when corpus size matches the
    current frame (guards against rare desync).
    """
    # Use per-company slices whenever present. Do not require ``_multi_company_erp`` (older Redis
    # sessions may omit that flag after rehydration); cross/BM25 must still index each tab.
    per = store.get("historical_by_company")
    if isinstance(per, dict) and per:
        for tab, sl in per.items():
            if tab not in COMPANY_TABS:
                continue
            if not isinstance(sl, dict):
                continue
            df = sl.get("historical")
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            tnorm = normalize_company_tab(tab)
            hit = load_erp_bm25_index_cached(fingerprint, tnorm)
            if hit is not None and _index_matches_tab_frame(hit, tnorm, df):
                continue
            idx = build_erp_bm25_index_for_dataframe(df, tnorm)
            save_erp_bm25_index_cached(fingerprint, tnorm, idx)
        return

    df = store.get("historical")
    if isinstance(df, pd.DataFrame) and not df.empty:
        tnorm = normalize_company_tab(None)
        hit = load_erp_bm25_index_cached(fingerprint, tnorm)
        if hit is not None and _index_matches_tab_frame(hit, tnorm, df):
            return
        idx = build_erp_bm25_index_for_dataframe(df, tnorm)
        save_erp_bm25_index_cached(fingerprint, tnorm, idx)


def get_or_build_erp_bm25_index(
    fingerprint: str,
    company_tab: str,
    df: pd.DataFrame,
) -> ErpBm25Index:
    """
    Resolve BM25 for one tab: Redis hit if fingerprint + frame size match, else build and cache.
    """
    tab = normalize_company_tab(company_tab)
    if df.empty:
        return build_erp_bm25_index_for_dataframe(df, tab)
    hit = load_erp_bm25_index_cached(fingerprint, tab)
    if hit is not None and _index_matches_tab_frame(hit, tab, df):
        return hit
    idx = build_erp_bm25_index_for_dataframe(df, tab)
    save_erp_bm25_index_cached(fingerprint, tab, idx)
    return idx
