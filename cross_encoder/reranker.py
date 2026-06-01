"""
bge-reranker-v2-m3 cross-encoder with Redis pair-level score cache.

Why a cross-encoder
-------------------
Phase 2 lifted the recall *ceiling* (BM25 ∪ bge-m3 finds the right candidate
in the top-50 for 33 % of Tab-2 cases vs 27 % BM25-only). But the lexical
reranker that follows discards those wins because token overlap can't separate
"wireless headset" from "wireless mouse". A cross-encoder scores the
``(query, candidate)`` pair *jointly* with full bidirectional attention, which
is exactly the signal the lexical step is missing.

Why bge-reranker-v2-m3 specifically
-----------------------------------
- Multilingual (handles the occasional Arabic transliteration in MBL/iFAS).
- ~600 MB on disk, fits cleanly into the existing ~1 GB bge-m3 layer.
- CPU inference at batch_size=32 ≈ 600 ms for 50 pairs, comfortably inside
  Phase 3's p95 < 2 s budget.
- Same project (``BAAI``) and tokenizer family as the bge-m3 embedder, so
  there is one model-vendor surface to track for security updates.

Cache contract
--------------
Cache key:    ``ccapr:rerank:v=<RERANK_INDEX_VERSION>:<sha256(canonical_query)>:<sha256(canonical_candidate)>``
Value:        4-byte little-endian float32 score.
TTL:          none — bumping :data:`RERANK_INDEX_VERSION` invalidates everything
              cleanly (same versioning pattern as Phase 1 BM25 + Phase 2 embeddings).

Hit-rate matters: Phase 2's HNSW index converges quickly, which means the same
~50 candidates surface for many queries. After ~1 month of real usage the hit
rate is essentially 100 %.
"""
from __future__ import annotations

import hashlib
import logging
import os
import struct
import threading
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ingest.canonical_desc import canonicalize_description

logger = logging.getLogger(__name__)


# Bump if the model OR the canonicalization changes incompatibly.
RERANK_INDEX_VERSION = int(os.environ.get("CCAPR_RERANK_INDEX_VERSION", "1") or 1)
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_REDIS_PREFIX = "ccapr:rerank:"
DEFAULT_RERANK_BATCH_SIZE = 32


def _canonical_pair_key(canonical_query: str, canonical_candidate: str) -> str:
    qh = hashlib.sha256((canonical_query or "").encode("utf-8")).hexdigest()
    ch = hashlib.sha256((canonical_candidate or "").encode("utf-8")).hexdigest()
    return f"{RERANK_REDIS_PREFIX}v={RERANK_INDEX_VERSION}:{qh}:{ch}"


def _score_to_bytes(score: float) -> bytes:
    return struct.pack("<f", float(score))


def _score_from_bytes(blob: Optional[bytes]) -> Optional[float]:
    if not blob or len(blob) != 4:
        return None
    try:
        return float(struct.unpack("<f", blob)[0])
    except struct.error:
        return None


class CrossEncoderReranker:
    """Thread-safe wrapper around bge-reranker-v2-m3 with Redis cache integration.

    Concurrency: ``score_pairs`` is safe to call from multiple threads. Internal
    forward pass is serialized with a lock because PyTorch is not reliably
    re-entrant on CPU and our Gunicorn deployment is gthread.
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL_NAME,
        *,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        device: str = "cpu",
        cache_folder: Optional[str] = None,
        max_length: int = 256,
    ) -> None:
        self._model_name = model_name
        self._batch_size = int(batch_size)
        self._device = device
        self._cache_folder = cache_folder or os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME"
        ) or "/opt/hf-cache"
        self._max_length = int(max_length)
        self._model = None  # lazy
        self._model_load_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._model_load_lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import CrossEncoder  # heavy import

            logger.info(
                "Loading cross-encoder %s (device=%s, cache=%s) …",
                self._model_name,
                self._device,
                self._cache_folder,
            )
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
                cache_folder=self._cache_folder,
            )
            return self._model

    # --- Public API -----------------------------------------------------------

    def score_pairs(
        self,
        query: str,
        candidates: Sequence[str],
        *,
        redis_client=None,
    ) -> np.ndarray:
        """Return float32 scores aligned with ``candidates`` for one query.

        - Uses canonical descriptions for the cache key (so the same product
          under reformatted descriptions hits the cache).
        - Empty / whitespace candidates yield score 0.0 with no model call.
        - Cache misses are computed in one batched forward pass.
        """
        n = len(candidates)
        out = np.zeros(n, dtype=np.float32)
        if n == 0 or not query:
            return out

        canonical_query = canonicalize_description(query)
        if not canonical_query:
            return out

        canonical_candidates = [
            canonicalize_description(c) if c else "" for c in candidates
        ]

        miss_positions: List[int] = []
        miss_keys: List[str] = []

        if redis_client is not None:
            keys: List[Optional[str]] = [
                _canonical_pair_key(canonical_query, cc) if cc else None
                for cc in canonical_candidates
            ]
            try:
                non_empty = [k for k in keys if k]
                blobs = redis_client.mget(non_empty) if non_empty else []
            except Exception as exc:
                logger.warning("Redis mget failed for rerank cache: %s", exc)
                blobs = [None] * len(non_empty)
            blob_iter = iter(blobs)
            for i, k in enumerate(keys):
                if k is None:
                    continue  # empty candidate ⇒ score stays 0.0
                blob = next(blob_iter)
                cached_score = _score_from_bytes(blob)
                if cached_score is None:
                    miss_positions.append(i)
                    miss_keys.append(k)
                else:
                    out[i] = cached_score
        else:
            miss_positions = [i for i, cc in enumerate(canonical_candidates) if cc]
            miss_keys = [
                _canonical_pair_key(canonical_query, canonical_candidates[i])
                for i in miss_positions
            ]

        if miss_positions:
            # Pass the *raw* texts to the cross-encoder (more signal than the
            # canonical form for paraphrase / case / punctuation cues).
            pairs: List[Tuple[str, str]] = [
                (query, candidates[i]) for i in miss_positions
            ]
            model = self._ensure_model()
            with self._encode_lock:
                fresh = model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            fresh = np.asarray(fresh, dtype=np.float32).reshape(-1)
            for slot, src in enumerate(miss_positions):
                out[src] = fresh[slot]
            if redis_client is not None and miss_keys:
                pipe = redis_client.pipeline(transaction=False)
                for slot, src in enumerate(miss_positions):
                    try:
                        pipe.set(miss_keys[slot], _score_to_bytes(fresh[slot]))
                    except Exception as exc:
                        logger.debug("Skipping bad rerank cache write: %s", exc)
                try:
                    pipe.execute()
                except Exception as exc:
                    logger.warning("Redis pipeline failed for rerank cache: %s", exc)

        return out


# --- Process-level singleton -------------------------------------------------

_GLOBAL_RERANKER: Optional[CrossEncoderReranker] = None
_GLOBAL_RERANKER_LOCK = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is not None:
        return _GLOBAL_RERANKER
    with _GLOBAL_RERANKER_LOCK:
        if _GLOBAL_RERANKER is None:
            _GLOBAL_RERANKER = CrossEncoderReranker()
        return _GLOBAL_RERANKER


def set_reranker(rr: Optional[CrossEncoderReranker]) -> None:
    """Test seam — replace or clear the singleton."""
    global _GLOBAL_RERANKER
    with _GLOBAL_RERANKER_LOCK:
        _GLOBAL_RERANKER = rr


def is_cross_encoder_enabled() -> bool:
    """Honor the global kill-switch. Default ON; set to '0' to fall back to Phase-2 hybrid rerank."""
    return (os.environ.get("CCAPR_USE_CROSS_ENCODER") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
