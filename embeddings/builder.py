"""
Idempotent build/load of per-company embedding indexes.

This is the integration glue between Phase 1 (canonical-description registry)
and Phase 2 (embedder + HNSW). One call to :func:`ensure_company_embedding_index`
will:

1. Load any existing on-disk HNSW index for the company.
2. Diff the canonical descriptions in the session's registry against the index.
3. Embed only the *new* descriptions (with Redis-cached lookups so previous
   container generations don't re-pay the cost).
4. Append them to the index and persist.

A typical weekly upload adds dozens of new descriptions, which embeds in seconds.
Cold first build of MBL (~30k unique descriptions) takes ~5 min on CPU.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

from .embedder import (
    EMBEDDING_DIM,
    Embedder,
    canonicalize_for_embedding,
    get_embedder,
)
from .gcs_hnsw_backend import (
    download_if_missing,
    gcs_meta_fingerprint,
    upload_async,
)
from .hnsw_store import HnswStore, hnsw_cache_root

logger = logging.getLogger(__name__)


# Module-level lock — at most one cold build runs concurrently per process.
_BUILD_LOCK = threading.Lock()
_BUILDING_COMPANIES: Set[str] = set()


@dataclass
class IndexBuildResult:
    company: str
    initial_size: int
    added_count: int
    final_size: int
    elapsed_s: float
    persisted: bool
    cache_hit: bool  # False when we had to embed at least one new description


def ensure_company_embedding_index(
    company: str,
    canonical_descriptions: Set[str],
    *,
    redis_client=None,
    embedder: Optional[Embedder] = None,
    persist: bool = True,
    fingerprint: Optional[str] = None,
) -> Optional[HnswStore]:
    """Build or extend the HNSW index for ``company`` so it covers every
    description in ``canonical_descriptions``.

    Returns the live :class:`HnswStore` (or ``None`` on hard failure such as
    ``usearch`` import error). Callers can use the returned store directly to
    issue queries.

    Concurrency: only one cold build of a given company runs at a time per
    process — concurrent callers wait on the same lock so we never spend
    embedding compute twice.
    """
    started = time.time()
    if not canonical_descriptions:
        return None

    with _BUILD_LOCK:
        if company in _BUILDING_COMPANIES:
            # Another caller is mid-build; we trust their result.
            logger.info(
                "Embedding index for %s already being built by another caller", company
            )
            return None
        _BUILDING_COMPANIES.add(company)

    cache_root: Optional[str] = None
    try:
        cache_root = hnsw_cache_root()
        if cache_root:
            try:
                download_if_missing(company, cache_root)
            except Exception as exc:
                logger.warning("GCS download before open failed for %s: %s", company, exc)

        store = HnswStore.open_or_create(company, dim=EMBEDDING_DIM)

        if fingerprint and cache_root:
            try:
                if gcs_meta_fingerprint(company, cache_root) == str(fingerprint).strip():
                    logger.info(
                        "fingerprint matches GCS index for %s, skipping rebuild", company
                    )
                    return store
            except Exception as exc:
                logger.warning("GCS fingerprint check failed for %s: %s", company, exc)

        initial_size = store.size

        # Filter to descriptions not yet in the index. The registry already keys
        # by canonical form, but we re-canonicalize defensively in case a future
        # caller passes raw text.
        wanted = [
            canonicalize_for_embedding(d) for d in canonical_descriptions if d
        ]
        wanted = [d for d in wanted if d]
        missing = store.descriptions_missing_from(wanted)

        if not missing:
            logger.info(
                "Embedding index for %s already covers all %d descriptions",
                company,
                len(wanted),
            )
            return store

        emb = embedder or get_embedder()
        logger.info(
            "Building embedding index for %s: %d existing + %d new descriptions",
            company,
            initial_size,
            len(missing),
        )

        # Embed in chunks so we get progress logs on a 30k cold build.
        chunk = 1024
        added = 0
        # Persist every Nth chunk so that operators (and the /api/embeddings/status
        # endpoint) see the index file growing, and so a container crash mid-build
        # only loses the work since the last flush. Flushing every chunk would be
        # safer but ~10% slower on a 30k build; 4 chunks is a pragmatic middle.
        persist_every_chunks = 4
        for i in range(0, len(missing), chunk):
            sub = missing[i : i + chunk]
            vecs = emb.encode_canonical_descriptions_with_cache(
                sub, redis_client=redis_client
            )
            added += store.add_descriptions(sub, vecs)
            chunk_idx = i // chunk
            if persist and (chunk_idx + 1) % persist_every_chunks == 0:
                store.persist(fingerprint=fingerprint)
            if len(missing) > chunk:
                logger.info(
                    "  %s embedding progress: %d / %d (%.1fs)",
                    company,
                    min(i + chunk, len(missing)),
                    len(missing),
                    time.time() - started,
                )

        if persist:
            if store.persist(fingerprint=fingerprint) and cache_root:
                try:
                    upload_async(company, cache_root)
                except Exception as exc:
                    logger.warning("GCS upload_async failed for %s: %s", company, exc)

        return store
    finally:
        with _BUILD_LOCK:
            _BUILDING_COMPANIES.discard(company)


def embed_query(
    raw_description: str,
    *,
    embedder: Optional[Embedder] = None,
    redis_client=None,
) -> Optional["np.ndarray"]:  # noqa: F821 — np imported lazily by callers
    """One-shot query embedding (uses the description-level cache so common queries are free)."""
    canonical = canonicalize_for_embedding(raw_description)
    if not canonical:
        return None
    emb = embedder or get_embedder()
    vecs = emb.encode_canonical_descriptions_with_cache(
        [canonical], redis_client=redis_client
    )
    if vecs.shape[0] == 0:
        return None
    return vecs[0]
