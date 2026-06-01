"""Tests for the cross-encoder reranker + Redis pair-cache.

We don't load the real bge-reranker-v2-m3 model in CI (heavy, network-dependent).
Instead we inject a fake reranker that returns deterministic scores so we can
exercise the cache contract without any ML stack.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from cross_encoder.pipeline import cross_encode_and_rank
from cross_encoder.reranker import (
    RERANK_INDEX_VERSION,
    RERANK_REDIS_PREFIX,
    CrossEncoderReranker,
    _canonical_pair_key,
    _score_from_bytes,
    _score_to_bytes,
)


# ---- Fake redis + fake reranker ------------------------------------------


class FakeRedis:
    def __init__(self) -> None:
        self.store: Dict[str, bytes] = {}
        self.mget_calls = 0
        self.set_calls = 0

    def mget(self, keys):
        self.mget_calls += 1
        return [self.store.get(k) for k in keys]

    def pipeline(self, transaction: bool = False):
        outer = self

        class _Pipe:
            def __init__(self) -> None:
                self.queue: List = []

            def set(self, k: str, v: bytes) -> None:
                self.queue.append((k, v))

            def execute(self) -> None:
                for k, v in self.queue:
                    outer.store[k] = v
                    outer.set_calls += 1

        return _Pipe()


def _seeded_score(query: str, candidate: str) -> float:
    """Stable deterministic 'score' so tests can assert on relative ordering."""
    h = hashlib.sha256(f"{query}|||{candidate}".encode("utf-8")).digest()[:4]
    seed = int.from_bytes(h, "big")
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-2.0, 5.0))


class FakeReranker(CrossEncoderReranker):
    """Returns deterministic scores; never touches sentence-transformers."""

    def __init__(self) -> None:
        super().__init__()
        self.encode_calls = 0
        self.last_pairs: List[Tuple[str, str]] = []

    def _ensure_model(self):
        return None

    def score_pairs(  # type: ignore[override]
        self,
        query: str,
        candidates: Sequence[str],
        *,
        redis_client=None,
    ):
        # Re-implement the cache+score loop using the seeded fake instead of a
        # real model so this fixture works without sentence-transformers
        # installed (CI / local-dev offline environments).
        n = len(candidates)
        out = np.zeros(n, dtype=np.float32)
        if n == 0 or not query:
            return out

        from ingest.canonical_desc import canonicalize_description

        canonical_query = canonicalize_description(query)
        if not canonical_query:
            return out
        canonical_candidates = [
            canonicalize_description(c) if c else "" for c in candidates
        ]
        miss_positions: List[int] = []
        miss_keys: List[str] = []
        if redis_client is not None:
            keys = [
                _canonical_pair_key(canonical_query, cc) if cc else None
                for cc in canonical_candidates
            ]
            non_empty = [k for k in keys if k]
            blobs = redis_client.mget(non_empty) if non_empty else []
            blob_iter = iter(blobs)
            for i, k in enumerate(keys):
                if k is None:
                    continue
                blob = next(blob_iter)
                cached = _score_from_bytes(blob)
                if cached is None:
                    miss_positions.append(i)
                    miss_keys.append(k)
                else:
                    out[i] = cached
        else:
            miss_positions = [i for i, cc in enumerate(canonical_candidates) if cc]
            miss_keys = [
                _canonical_pair_key(canonical_query, canonical_candidates[i])
                for i in miss_positions
            ]

        if miss_positions:
            self.encode_calls += 1
            self.last_pairs = [(query, candidates[i]) for i in miss_positions]
            for slot, src in enumerate(miss_positions):
                out[src] = _seeded_score(query, candidates[src])
            if redis_client is not None and miss_keys:
                pipe = redis_client.pipeline(transaction=False)
                for slot, src in enumerate(miss_positions):
                    pipe.set(miss_keys[slot], _score_to_bytes(out[src]))
                pipe.execute()
        return out


# ---- Round-trip helpers --------------------------------------------------


def test_score_to_bytes_roundtrip():
    for s in (-2.5, 0.0, 1.234, 5.0):
        assert _score_from_bytes(_score_to_bytes(s)) == pytest.approx(s, abs=1e-6)


def test_score_from_bytes_rejects_wrong_length():
    assert _score_from_bytes(b"\x00\x00") is None
    assert _score_from_bytes(None) is None


def test_redis_key_includes_version_and_two_hashes():
    key = _canonical_pair_key("ball valve 1 inch", "stop valve 1 inch")
    assert key.startswith(f"{RERANK_REDIS_PREFIX}v={RERANK_INDEX_VERSION}:")
    parts = key.split(":")
    assert len(parts) == 5  # ccapr, rerank, v=N, qhash, chash
    assert len(parts[3]) == 64
    assert len(parts[4]) == 64


# ---- Cache behaviour ----------------------------------------------------


def test_first_call_misses_then_caches():
    rr = FakeReranker()
    r = FakeRedis()
    out = rr.score_pairs("ball valve 1 inch", ["stop valve 1 inch", "broom stick"], redis_client=r)
    assert out.shape == (2,)
    assert rr.encode_calls == 1
    assert r.set_calls == 2
    assert len(r.store) == 2


def test_second_call_is_pure_cache_hit():
    rr = FakeReranker()
    r = FakeRedis()
    rr.score_pairs("ball valve", ["candidate a", "candidate b"], redis_client=r)
    out2 = rr.score_pairs("ball valve", ["candidate a", "candidate b"], redis_client=r)
    assert rr.encode_calls == 1, "Second call must not re-encode"
    out1 = rr.score_pairs("ball valve", ["candidate a", "candidate b"], redis_client=None)
    np.testing.assert_array_almost_equal(out2, out1, decimal=5)


def test_partial_cache_hit_only_encodes_misses():
    rr = FakeReranker()
    r = FakeRedis()
    rr.score_pairs("q1", ["a"], redis_client=r)
    rr.encode_calls = 0
    rr.score_pairs("q1", ["a", "b", "c"], redis_client=r)
    assert rr.encode_calls == 1
    assert {p[1] for p in rr.last_pairs} == {"b", "c"}


def test_empty_candidates_skip_model_and_cache():
    rr = FakeReranker()
    r = FakeRedis()
    out = rr.score_pairs("q", ["", "  ", None], redis_client=r)  # type: ignore[list-item]
    assert out.shape == (3,)
    assert np.all(out == 0)
    assert rr.encode_calls == 0
    assert r.set_calls == 0


def test_canonical_keying_collapses_punctuation_variants():
    """The cache key uses the canonical form, so e.g. 'BALL-VALVE 1\"' and
    'ball valve 1' map to the same cached score."""
    rr = FakeReranker()
    r = FakeRedis()
    rr.score_pairs("ball valve 1\"", ["candidate"], redis_client=r)
    rr.encode_calls = 0
    out2 = rr.score_pairs("BALL-VALVE 1!", ["candidate"], redis_client=r)
    assert rr.encode_calls == 0  # canonical key collision ⇒ cache hit
    assert out2.shape == (1,)


# ---- Pipeline ranking ---------------------------------------------------


def test_pipeline_returns_sorted_descending():
    rr = FakeReranker()
    out = cross_encode_and_rank(
        "query xyz",
        [("L1", "candidate a"), ("L2", "candidate b"), ("L3", "candidate c")],
        reranker=rr,
    )
    assert len(out) == 3
    scores = [s for _, _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_applies_penalty_per_label():
    rr = FakeReranker()
    raw = cross_encode_and_rank(
        "q",
        [("L1", "a"), ("L2", "b")],
        reranker=rr,
    )
    raw_scores = {lb: s for lb, _, s in raw}
    # Apply a penalty large enough to flip the ranking.
    biased = cross_encode_and_rank(
        "q",
        [("L1", "a"), ("L2", "b")],
        reranker=rr,
        penalty_by_label={"L1": raw_scores["L1"] - raw_scores["L2"] + 1.0},
    )
    # The penalised label must NOT be the top.
    assert biased[0][0] != "L1"


def test_pipeline_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("CCAPR_USE_CROSS_ENCODER", "0")
    rr = FakeReranker()
    out = cross_encode_and_rank("q", [("L1", "a")], reranker=rr)
    assert out == []


def test_pipeline_top_n_truncates():
    rr = FakeReranker()
    out = cross_encode_and_rank(
        "q",
        [(f"L{i}", f"c{i}") for i in range(20)],
        reranker=rr,
        top_n=5,
    )
    assert len(out) == 5
