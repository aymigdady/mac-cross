"""Tests for hybrid (BM25 ∪ embedding) shortlist fusion."""
from __future__ import annotations

from typing import List

import pandas as pd

from comparison_engine import COL_DESC, COL_ITEM_NO
from embeddings.hybrid import (
    explain_hybrid_membership,
    hybrid_shortlist_labels,
)


def _work_df(rows: List[tuple]) -> pd.DataFrame:
    """Tiny work-df builder. ``rows`` = ``[(label, item_no, desc), ...]``."""
    df = pd.DataFrame(
        [{COL_ITEM_NO: code, COL_DESC: desc} for _, code, desc in rows],
        index=[lb for lb, _, _ in rows],
    )
    return df


def test_hybrid_returns_empty_when_both_inputs_empty():
    work = _work_df([(0, "A", "alpha")])
    assert hybrid_shortlist_labels("MBL", work, [], []) == []


def test_hybrid_keeps_bm25_only_when_embedding_empty():
    work = _work_df([(0, "A", "a"), (1, "B", "b"), (2, "C", "c")])
    out = hybrid_shortlist_labels("MBL", work, [0, 2, 1], [])
    assert out == [0, 2, 1]


def test_hybrid_keeps_embedding_only_when_bm25_empty():
    work = _work_df([(0, "A", "a"), (1, "B", "b")])
    out = hybrid_shortlist_labels("MBL", work, [], [(1, 0.9), (0, 0.7)])
    assert out == [1, 0]


def test_hybrid_promotes_intersection_with_rrf():
    """A label that appears in BOTH lists should outrank labels that appear in only one."""
    work = _work_df([(0, "A", "a"), (1, "B", "b"), (2, "C", "c")])
    bm25 = [0, 1, 2]            # A first by BM25
    emb = [(2, 0.99), (0, 0.95)]  # C first, then A by embedding
    fused = hybrid_shortlist_labels("MBL", work, bm25, emb)
    # A appears in both at decent ranks; C appears in only one but at top emb rank.
    # With RRF (k=60), score(A) = 1/(60+0) + 1/(60+1) ≈ 0.0334
    #                    score(C) = 1/(60+2) + 1/(60+0) ≈ 0.0328
    # so A should be first, C second, B last.
    assert fused[0] == 0
    assert set(fused[:3]) == {0, 1, 2}


def test_hybrid_dedupes_when_emb_repeats_a_label():
    work = _work_df([(0, "A", "a"), (1, "B", "b")])
    # The embedding stage can produce the same label multiple times when many
    # canonical descriptions resolve to the same row — make sure the fusion
    # keeps the best similarity and only counts the row once.
    fused = hybrid_shortlist_labels("MBL", work, [0], [(1, 0.5), (1, 0.9)])
    assert sorted(fused) == [0, 1]


def test_hybrid_drops_labels_not_in_work_index():
    work = _work_df([(0, "A", "a"), (1, "B", "b")])
    # Stale label 99 should be silently filtered out — important when work_df
    # is dedup'd vs the historical/BM25 corpus.
    fused = hybrid_shortlist_labels("MBL", work, [99, 0], [(1, 0.9), (99, 0.99)])
    assert sorted(fused) == [0, 1]


def test_explain_hybrid_membership_tags_correctly():
    work = _work_df([(0, "A", "a"), (1, "B", "b"), (2, "C", "c")])
    bm25 = [0, 1]
    emb = [(1, 0.9), (2, 0.7)]
    fused = hybrid_shortlist_labels("MBL", work, bm25, emb)
    prov = explain_hybrid_membership(bm25, emb, fused)
    assert prov[0] == "bm25"
    assert prov[1] == "hybrid"
    assert prov[2] == "emb"


def test_hybrid_respects_union_max():
    work = _work_df([(i, f"X{i}", f"x{i}") for i in range(20)])
    bm25 = list(range(20))
    emb = [(i, 1.0 - i * 0.01) for i in range(20)]
    fused = hybrid_shortlist_labels("MBL", work, bm25, emb, union_max=5)
    assert len(fused) == 5
