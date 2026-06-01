"""
Phase-1 cross-search hardening: smarter tokenizer, dedupe, exact-code shortcut,
3-score confidence + provenance, audit trace.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

import pandas as pd

from cross_pipeline import (
    _apply_cross_company_match_pipeline,
    _blend_cross_search_confidence,
    _cross_match_dedupe_receipt_rows,
    _cross_match_prepare_work_df,
    _normalize_cross_desc_for_match,
    tokenize_cross_desc_for_bm25,
)
from comparison_engine import (
    COL_DESC,
    COL_ITEM_NO,
    COL_PO,
    COL_PO_DATE,
    COL_UNIT,
    COL_UNIT_COST,
    COL_VENDOR,
)


def _row(item_no: str, desc: str, *, cost: float = 1.0, vendor: str = "VendorCo", po_date: str = "2024-06-01") -> Dict[str, Any]:
    return {
        COL_ITEM_NO: item_no,
        COL_DESC: desc,
        COL_UNIT_COST: cost,
        COL_UNIT: "ea",
        COL_VENDOR: vendor,
        COL_PO: f"PO-{item_no}",
        COL_PO_DATE: po_date,
    }


def test_tokenizer_fuses_digit_unit_pairs() -> None:
    """`12 mm` should tokenize the same as `12mm` so spec variants match."""
    a = tokenize_cross_desc_for_bm25("copper press elbow 22 mm type b")
    b = tokenize_cross_desc_for_bm25("copper press elbow 22mm type b")
    assert "22mm" in a
    assert "22mm" in b
    assert "22" not in a, "digit + unit should not leak as bare digit"


def test_tokenizer_preserves_sku_hyphens() -> None:
    """`PD-100039641` should appear as one fused token, not split at the hyphen."""
    toks = tokenize_cross_desc_for_bm25("PD-100039641 RENTAL CHARGES IFAS")
    assert any(t.startswith("pd100039641") for t in toks), toks
    # Nothing should be the bare "pd"
    assert "pd" not in toks


def test_tokenizer_light_stem_for_long_tokens() -> None:
    """Plurals/gerunds map to a shared stem so corpus and query agree."""
    a = set(tokenize_cross_desc_for_bm25("RENTAL CHARGES BUILDING"))
    b = set(tokenize_cross_desc_for_bm25("RENT CHARGE BUILDINGS"))
    # At least one shared longer stem (e.g. "rental"/"rent" share "rent" only after stem;
    # we mainly assert "buildings"/"building" collapse).
    assert "build" in {t for t in a | b if t.startswith("build")} or any(
        t.startswith("buil") for t in a & b
    ), f"expected building/buildings to share a stem; a={a} b={b}"


def test_normalizer_does_not_break_short_tokens() -> None:
    """Tokens shorter than 6 chars should not be stemmed (avoids `id` -> `''`)."""
    norm = _normalize_cross_desc_for_match("ID NO 12345")
    assert "id" in norm and "no" in norm


def test_dedupe_collapses_identical_receipt_rows() -> None:
    """Same item/desc/cost/vendor across multiple PO dates should reduce to one row (most recent)."""
    df = pd.DataFrame(
        [
            _row("X-1", "rental services", cost=100.0, po_date="2022-01-01"),
            _row("X-1", "rental services", cost=100.0, po_date="2024-08-01"),
            _row("X-1", "rental services", cost=100.0, po_date="2023-05-01"),
            _row("X-2", "rental services", cost=100.0, po_date="2024-06-01"),
        ]
    )
    sorted_df = df.sort_values(COL_PO_DATE, ascending=False)
    out = _cross_match_dedupe_receipt_rows(sorted_df)
    assert len(out) == 2
    # X-1 keeps the 2024-08-01 row (most recent in the sort order).
    x1 = out[out[COL_ITEM_NO].astype(str) == "X-1"]
    assert len(x1) == 1
    assert str(x1.iloc[0][COL_PO_DATE]).startswith("2024-08")


def test_exact_code_shortcut_skips_bm25_and_sets_provenance() -> None:
    """When the CCAPR item code AND description match the target tab, the shortcut copies the row directly."""
    erp = pd.DataFrame(
        [
            _row("ABC-123", "specialty alloy reactor flange", cost=4500.0),
            _row("OTHER-1", "carbon steel pipe sch40", cost=10.0),
        ]
    )
    fp = uuid.uuid4().hex
    store = {"historical": erp, "erp_file_sha256": fp, "_multi_company_erp": False}
    rows: List[Dict[str, Any]] = [
        {"item_no": "ABC-123", "description": "specialty alloy reactor flange", "has_history": False}
    ]
    items = [{"itemNo": "ABC-123", "itemDescription": "specialty alloy reactor flange"}]
    _apply_cross_company_match_pipeline(store, rows, items, "MBL", "VendorCo")
    assert rows[0]["match_provenance"] == "exact_code"
    assert rows[0]["cross_search_confidence_pct"] == 100.0
    assert rows[0]["lowest_hist_unit_cost"] == 4500.0
    audit = rows[0]["cross_search_audit"]
    assert audit["provenance"] == "exact_code"
    # Audit records the real lexical similarity that let the shortcut fire (here ~100 %).
    assert audit["lexical_pct"] >= 60.0


def test_exact_code_shortcut_blocked_when_descriptions_diverge(monkeypatch) -> None:
    """Same Item No. across companies but unrelated descriptions must NOT short-circuit.

    Pre-Phase-1.1 the shortcut silently stamped 100 % confidence on cross-company SKU
    collisions, so Haiku never ran and the row inherited an unrelated product's
    benchmark.  The lex gate (``CCAPR_CROSS_EXACT_CODE_MIN_LEX``, default 0.6) now
    forces such lines into the BM25 + Haiku path.  With external AI disabled in
    tests, that means the row receives no exact-code stamp and is left for the
    local fallback to reconcile.
    """
    monkeypatch.setenv("CCAPR_CROSS_EXACT_CODE_MIN_LEX", "0.6")
    erp = pd.DataFrame(
        [
            _row("ABC-123", "carbon steel pipe sch40", cost=4500.0),
            _row("OTHER-1", "alloy reactor flange", cost=10.0),
        ]
    )
    fp = uuid.uuid4().hex
    store = {"historical": erp, "erp_file_sha256": fp, "_multi_company_erp": False}
    rows: List[Dict[str, Any]] = [
        {"item_no": "ABC-123", "description": "office stationery copier toner", "has_history": False}
    ]
    items = [{"itemNo": "ABC-123", "itemDescription": "office stationery copier toner"}]
    _apply_cross_company_match_pipeline(store, rows, items, "MBL", "VendorCo")
    # The shortcut must not have fired: provenance is anything except "exact_code"
    # (most likely missing, or "local_lex" if the BM25 fallback resolved it).
    assert rows[0].get("match_provenance") != "exact_code"
    # And the silent 100 % is gone.
    assert rows[0].get("cross_search_confidence_pct") != 100.0


def test_blended_confidence_formula() -> None:
    """Blend = 0.55 * lexical + 0.45 * AI when both exist."""
    assert _blend_cross_search_confidence(80.0, 60.0) == 71.0
    assert _blend_cross_search_confidence(50.0, None) == 50.0
    assert _blend_cross_search_confidence(None, 90.0) == 90.0
    assert _blend_cross_search_confidence(None, None) is None
    # Cap at 100.
    assert _blend_cross_search_confidence(120.0, 110.0) == 100.0
