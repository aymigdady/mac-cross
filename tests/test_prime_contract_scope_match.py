"""Prime contract scope matching endpoint logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def test_match_lines_to_scope_in_scope(monkeypatch):
    from prime_contract.scope_match import match_lines_to_scope

    class _FakeEmbedder:
        def encode_canonical_descriptions_with_cache(self, texts, redis_client=None):
            n = len(texts)
            out = np.zeros((n, 4), dtype=np.float32)
            for i in range(n):
                out[i, i % 4] = 1.0
            return out

    monkeypatch.setenv("CCAPR_PRIME_SCOPE_MIN_SCORE", "0.5")
    monkeypatch.setattr("embeddings.embedder.get_embedder", lambda: _FakeEmbedder())

    def _fake_embed_query(raw_description, redis_client=None, embedder=None):
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return vec

    monkeypatch.setattr("embeddings.builder.embed_query", _fake_embed_query)

    out = match_lines_to_scope(
        scope_items=["Supply and install pumps", "Painting works"],
        lines=[{"itemKey": "1", "description": "Supply pumps for HVAC"}],
        scope_fingerprint="fp1",
        redis_client=MagicMock(),
    )
    assert len(out) == 1
    assert out[0]["itemKey"] == "1"
    assert out[0]["inScope"] is True
    assert out[0]["confidence"] >= 0.5


def test_match_lines_empty_scope():
    from prime_contract.scope_match import match_lines_to_scope

    out = match_lines_to_scope(
        scope_items=[],
        lines=[{"itemKey": "1", "description": "Anything"}],
    )
    assert out[0]["inScope"] is False
    assert out[0]["method"] == "empty_scope"
