"""
Phase 10 — Golden-set checks for the cross-match pipeline.

For each case we assert:
1) ``expected_item_no`` appears in the BM25 shortlist (≤50),
2) it survives rerank into the top-10 list,
3) full ``_cross_match_reference_work_df`` output matches rerank (same curated window).

With ``ANTHROPIC_API_KEY`` set, ``test_golden_llm_picks_expected_row`` also checks Haiku chooses
the expected ``reference_item_no`` for one representative case (live API; skipped in CI by default).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from cross_pipeline import (
    _cross_match_reference_work_df,
    cross_match_pipeline_diagnose,
)
from bm25_erp_index_cache import ensure_erp_bm25_cache_for_store
from comparison_engine import (
    COL_DESC,
    COL_ITEM_NO,
    COL_PO,
    COL_PO_DATE,
    COL_UNIT,
    COL_UNIT_COST,
    COL_VENDOR,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cross_match_golden_set.json"


def _row(
    item_no: str,
    desc: str,
    *,
    po_date: str = "2024-06-01",
) -> Dict[str, Any]:
    return {
        COL_ITEM_NO: item_no,
        COL_DESC: desc,
        COL_UNIT_COST: 1.0,
        COL_UNIT: "ea",
        COL_VENDOR: "VendorCo",
        COL_PO: f"PO-{item_no}",
        COL_PO_DATE: po_date,
    }


def _build_deep_corpus_titanium():
    import pandas as pd

    rows = [_row(str(i), "standard steel bracket assembly") for i in range(599)]
    rows.append(_row("599", "titanium cryogenic valve assembly low temperature"))
    return pd.DataFrame(rows)


def _build_near_exact_three_way():
    import pandas as pd

    return pd.DataFrame(
        [
            _row("a", "unrelated bolt m12 zinc"),
            _row("b", "widget type a 12mm stainless extra words"),
            _row("c", "widget type a 12mm"),
        ]
    )


def _build_hastelloy_among_steel_noise():
    import pandas as pd

    rows = [_row(str(i), "carbon steel pipe sch40 line stock") for i in range(40)]
    rows.append(_row("TARGET", "hastelloy reactor flange specialty alloy"))
    return pd.DataFrame(rows)


def _build_copper_elbow_distinct():
    import pandas as pd

    return pd.DataFrame(
        [
            _row("P-1", "pvc conduit elbow 20mm"),
            _row("P-2", "galvanized tee fitting"),
            _row("CU-22", "copper press elbow 22mm type b"),
            _row("P-4", "stainless nipple hex"),
        ]
    )


_BUILDERS: Dict[str, Callable[[], Any]] = {
    "deep_corpus_titanium": _build_deep_corpus_titanium,
    "near_exact_three_way": _build_near_exact_three_way,
    "hastelloy_among_steel_noise": _build_hastelloy_among_steel_noise,
    "copper_elbow_distinct": _build_copper_elbow_distinct,
}


def _load_manifest() -> List[Dict[str, Any]]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return list(data.get("cases") or [])


def _store_for_df(df, fp: str) -> Dict[str, Any]:
    return {"historical": df, "erp_file_sha256": fp, "_multi_company_erp": False}


@pytest.mark.parametrize(
    "case",
    _load_manifest(),
    ids=lambda c: str(c.get("id", "unknown")),
)
def test_golden_bm25_rerank_and_pipeline(case: Dict[str, Any]) -> None:
    cid = case["id"]
    if cid not in _BUILDERS:
        pytest.skip(f"No builder registered for golden case id={cid!r}")
    df = _BUILDERS[cid]()
    fp = uuid.uuid4().hex
    store = _store_for_df(df, fp)
    ensure_erp_bm25_cache_for_store(fp, store)

    query = str(case["query"])
    expected = str(case["expected_item_no"]).strip()

    diag = cross_match_pipeline_diagnose(store, "MBL", [query])
    assert diag is not None
    assert not diag["legacy_head_500_fallback"], f"{cid}: expected BM25 path, got legacy fallback"

    bm25 = diag["bm25_shortlist_item_nos"]
    rerank = diag["rerank_top_item_nos"]
    pipe = diag["pipeline_output_item_nos"]

    assert expected in bm25, (
        f"{cid}: expected item {expected!r} missing from BM25 top-50 — tune k1/b or tokenizer. bm25={bm25!r}"
    )
    assert expected in rerank, (
        f"{cid}: expected item {expected!r} missing from reranked top-10 — check lexical scorer. rerank={rerank!r}"
    )
    assert rerank == pipe, f"{cid}: pipeline output should match rerank top-10; rerank={rerank!r} pipe={pipe!r}"

    final_df = _cross_match_reference_work_df(store, "MBL", bm25_query_texts=[query])
    assert final_df is not None
    assert final_df[COL_ITEM_NO].astype(str).str.strip().tolist() == pipe


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") or "").strip(),
    reason="Set ANTHROPIC_API_KEY to run live Haiku golden check",
)
def test_golden_llm_picks_expected_row() -> None:
    from ai_service import AIService, external_ai_enabled

    if not external_ai_enabled():
        pytest.skip("external AI disabled")

    df = _build_copper_elbow_distinct()
    fp = uuid.uuid4().hex
    store = _store_for_df(df, fp)
    ensure_erp_bm25_cache_for_store(fp, store)
    query = "copper press elbow 22mm"
    expected = "CU-22"

    cross_df = _cross_match_reference_work_df(store, "MBL", bm25_query_texts=[query])
    assert cross_df is not None and not cross_df.empty
    tsv = cross_df.head(10).to_csv(sep="\t", index=False)

    ai = AIService()
    if not ai.available():
        pytest.skip("AI service not available")

    out = ai.match_cross_company_descriptions(
        reference_tsv=tsv,
        items=[{"item_no": "LINE1", "item_description": query}],
    )
    assert not out.get("error"), out.get("error")
    parsed = out.get("result") or {}
    from cross_pipeline import _extract_cross_matches_from_parsed

    matches = _extract_cross_matches_from_parsed(parsed)
    assert matches, f"No matches in model output: {parsed!r}"
    ref_no = str(
        matches[0].get("reference_item_no")
        or matches[0].get("referenceItemNo")
        or ""
    ).strip()
    assert ref_no == expected, (
        f"LLM picked {ref_no!r}, expected {expected!r} — adjust Haiku prompt or TSV layout. matches[0]={matches[0]!r}"
    )
