"""
bge-m3 embedder with description-level Redis cache.

Why description-level (not row-level)
-------------------------------------
Real ERP exports re-issue the same physical product under multiple item codes.
Phase 1's canonical-description registry showed one MBL description living under
**111 distinct codes**. If we keyed embeddings on rows we would compute the same
1024-dim vector 111 times. Keying on the *canonical description* gives a free
5-10× compression with no quality loss.

Why a singleton
---------------
``sentence_transformers.SentenceTransformer('BAAI/bge-m3')`` is ~5s to construct
on a cold CPU and consumes ~1 GB of memory. We want exactly one instance per
process, lazily loaded so that:

- Imports / unit tests that don't actually need embeddings pay zero cost.
- Each Gunicorn worker pays the load cost once on first cross-search.
- The harness can override the embedder with a fake (no PyTorch download in CI).

Cache contract
--------------
Redis key:   ``ccapr:emb:m3:v=<EMBEDDING_INDEX_VERSION>:<sha256(canonical_desc)>``
Value:       raw bytes ``np.float32`` little-endian, length == EMBEDDING_DIM.
TTL:         none (embeddings of a given description never change for a fixed
             model/version, and Phase 5 will add explicit eviction tooling).

Bumping :data:`EMBEDDING_INDEX_VERSION` mechanically invalidates every cached
embedding without manual ``redis-cli del`` — the same pattern Phase 1 uses for
the BM25 index cache.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ingest.canonical_desc import canonicalize_description

logger = logging.getLogger(__name__)


# Bump if the model OR the canonicalization changes incompatibly.
EMBEDDING_INDEX_VERSION = 1
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_DTYPE = np.float32

EMBEDDING_REDIS_PREFIX = "ccapr:emb:m3:"
DEFAULT_BATCH_SIZE = 32


def _redis_key_for(canonical_desc: str) -> str:
    """Stable Redis key for one canonical description."""
    h = hashlib.sha256((canonical_desc or "").encode("utf-8")).hexdigest()
    return f"{EMBEDDING_REDIS_PREFIX}v={EMBEDDING_INDEX_VERSION}:{h}"


def _vector_to_bytes(vec: np.ndarray) -> bytes:
    if vec.dtype != EMBEDDING_DTYPE:
        vec = vec.astype(EMBEDDING_DTYPE, copy=False)
    if vec.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Embedding has shape {vec.shape}, expected ({EMBEDDING_DIM},)"
        )
    return vec.tobytes(order="C")


def _vector_from_bytes(blob: bytes) -> Optional[np.ndarray]:
    if blob is None:
        return None
    expected = EMBEDDING_DIM * np.dtype(EMBEDDING_DTYPE).itemsize
    if len(blob) != expected:
        # Length mismatch ⇒ stale / wrong-version blob; treat as cache miss.
        return None
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).copy()


class Embedder:
    """Thread-safe wrapper around the bge-m3 model with Redis cache integration.

    Concurrency: ``encode`` is safe to call from multiple threads. Internal
    forward pass is serialized with a lock because PyTorch is not reliably
    re-entrant on CPU and our Gunicorn deployment is gthread (single process,
    multiple threads) so contention is bounded.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
        cache_folder: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = int(batch_size)
        self._device = device
        self._cache_folder = cache_folder or os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME"
        ) or "/opt/hf-cache"
        self._model = None  # lazy
        self._model_load_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._model_load_lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer  # heavy import

            logger.info(
                "Loading sentence-transformer %s (device=%s, cache=%s) …",
                self._model_name,
                self._device,
                self._cache_folder,
            )
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                cache_folder=self._cache_folder,
            )
            return self._model

    # --- Public API -----------------------------------------------------------

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single string. Convenience wrapper around :meth:`encode_many`."""
        out = self.encode_many([text])
        return out[0]

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch and return an ``(N, EMBEDDING_DIM)`` float32 array.

        Empty / whitespace-only inputs return zero vectors (rather than raising)
        so callers can keep the input/output alignment without filtering.
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=EMBEDDING_DTYPE)
        cleaned = [t if isinstance(t, str) and t.strip() else "" for t in texts]
        non_empty_idx = [i for i, t in enumerate(cleaned) if t]
        out = np.zeros((len(cleaned), EMBEDDING_DIM), dtype=EMBEDDING_DTYPE)
        if not non_empty_idx:
            return out
        non_empty_texts = [cleaned[i] for i in non_empty_idx]
        model = self._ensure_model()
        with self._encode_lock:
            vecs = model.encode(
                non_empty_texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        vecs = np.asarray(vecs, dtype=EMBEDDING_DTYPE)
        for slot, src in enumerate(non_empty_idx):
            out[src] = vecs[slot]
        return out

    def encode_canonical_descriptions_with_cache(
        self,
        canonical_descriptions: Sequence[str],
        *,
        redis_client=None,
    ) -> np.ndarray:
        """Cache-aware bulk embed. ``canonical_descriptions`` should already be
        canonicalised (see :func:`ingest.canonical_desc.canonicalize_description`)
        because the cache key is derived from the canonical form.

        - Hits are returned from Redis; misses are encoded then back-filled.
        - Empty / whitespace descriptions short-circuit to zero vectors with no
          cache I/O and no model invocation.
        - When ``redis_client is None`` the function still works (cache disabled),
          which is the path tests use.
        """
        n = len(canonical_descriptions)
        out = np.zeros((n, EMBEDDING_DIM), dtype=EMBEDDING_DTYPE)
        if n == 0:
            return out

        # Slot positions that hold a real (non-empty) description; only these
        # ever touch Redis or the model.
        non_empty_positions = [
            i for i, d in enumerate(canonical_descriptions) if isinstance(d, str) and d.strip()
        ]
        if not non_empty_positions:
            return out

        miss_positions: List[int] = []
        cached_keys: List[str] = []

        if redis_client is not None:
            keys = [_redis_key_for(canonical_descriptions[i]) for i in non_empty_positions]
            try:
                blobs = redis_client.mget(keys)
            except Exception as exc:
                logger.warning("Redis mget failed for embedding cache: %s", exc)
                blobs = [None] * len(keys)
            for offset, (blob, key) in enumerate(zip(blobs, keys)):
                src = non_empty_positions[offset]
                vec = _vector_from_bytes(blob) if blob else None
                if vec is None:
                    miss_positions.append(src)
                    cached_keys.append(key)
                else:
                    out[src] = vec
        else:
            miss_positions = list(non_empty_positions)
            cached_keys = [_redis_key_for(canonical_descriptions[i]) for i in non_empty_positions]

        if miss_positions:
            miss_texts = [canonical_descriptions[i] for i in miss_positions]
            logger.info(
                "Embedding cache: %d hits, %d misses (computing fresh)",
                len(non_empty_positions) - len(miss_positions),
                len(miss_positions),
            )
            fresh = self.encode_many(miss_texts)
            for slot, src in enumerate(miss_positions):
                out[src] = fresh[slot]
            if redis_client is not None:
                pipe = redis_client.pipeline(transaction=False)
                for slot, src in enumerate(miss_positions):
                    try:
                        pipe.set(cached_keys[slot], _vector_to_bytes(fresh[slot]))
                    except Exception as exc:
                        logger.debug("Skipping bad embedding write: %s", exc)
                try:
                    pipe.execute()
                except Exception as exc:
                    logger.warning("Redis pipeline failed for embedding cache: %s", exc)

        return out


# --- Process-level singleton -------------------------------------------------

_GLOBAL_EMBEDDER: Optional[Embedder] = None
_GLOBAL_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Embedder:
    """Return the per-process embedder singleton (lazy)."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is not None:
        return _GLOBAL_EMBEDDER
    with _GLOBAL_EMBEDDER_LOCK:
        if _GLOBAL_EMBEDDER is None:
            _GLOBAL_EMBEDDER = Embedder()
        return _GLOBAL_EMBEDDER


def set_embedder(emb: Optional[Embedder]) -> None:
    """Replace (or clear) the singleton — used by tests to inject a fake embedder."""
    global _GLOBAL_EMBEDDER
    with _GLOBAL_EMBEDDER_LOCK:
        _GLOBAL_EMBEDDER = emb


def canonicalize_for_embedding(raw: str) -> str:
    """Single source of truth: descriptions are canonicalized exactly once before
    they are embedded or looked up in the cache. Re-uses Phase 1's normaliser."""
    return canonicalize_description(raw)


def canonicalize_many_for_embedding(rows: Iterable[str]) -> List[str]:
    return [canonicalize_for_embedding(r) for r in rows]
