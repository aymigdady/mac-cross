"""Build and normalize cross-match working DataFrames."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from comparison_engine import (
    COL_DESC,
    COL_ITEM_NO,
    COL_PO,
    COL_PO_DATE,
    COL_PROJECT,
    COL_QTY,
    COL_UNIT,
    COL_UNIT_COST,
    COL_VENDOR,
    _clean_numeric_series,
    clean_ccapr_item_no_input,
    normalized_item_key_from_input,
)

from .columns import _cross_match_ensure_unit_cost_column
from .parsing import (
    _normalize_new_po_source_tab,
    _normalize_vendor_key_local,
    _parse_reference_unit_cost_scalar,
    _po_date_iso_from_cell,
)
from .text import _normalize_cross_desc_for_match

logger = logging.getLogger(__name__)

def _effective_historical_store(
    store: Dict[str, Any],
    new_po_tab_raw: Optional[str],
    *,
    strict_tab: bool = True,
) -> Dict[str, Any]:
    """Merge session store with the PO-line slice for the selected NEW PO company tab (multi-tab ERP)."""
    tab = _normalize_new_po_source_tab(new_po_tab_raw)
    per = store.get("historical_by_company")
    if not isinstance(per, dict) or not per:
        if strict_tab:
            raise ValueError(
                f"{tab} ERP tab is not loaded for cross search. "
                "Upload the multi-company ERP workbook or reload default ERP."
            )
        return store
    sl = per.get(tab)
    if sl is None:
        if strict_tab:
            raise ValueError(f"{tab} ERP tab is missing from historical_by_company.")
        sl = per.get("MBL")
    if sl is None:
        if strict_tab:
            raise ValueError(f"{tab} ERP tab is missing from historical_by_company.")
        sl = next(iter(per.values()))
    merged = dict(store)
    merged["historical"] = sl["historical"]
    merged["latest_by_item"] = sl["latest_by_item"]
    merged["lowest_by_item"] = sl["lowest_by_item"]
    merged["_multi_company_erp"] = True
    return merged

def _cross_match_apply_mbl_positional_fallback(work: pd.DataFrame) -> pd.DataFrame:
    """
    MBL exports can arrive with unstable headers. For cross-search only, use fixed MBL column
    positions as fallback:

    - B  -> Ref. PO No.
    - H  -> QTY  (header is sometimes "Qty"/"Quantity" but column letter is stable)
    - I  -> UOM
    - J  -> Unit price (header reads "Price" — this is the per-unit rate, not a line total)
    - P  -> Vendor name
    - W  -> Ref. PO Date

    The fallback only fills a canonical column when it is missing OR has no usable values,
    so well-formed exports with proper headers are left untouched.
    """
    if work is None or work.empty:
        return work
    # Need at least 23 columns (indices 0..22) so column W is reachable.
    if work.shape[1] <= 22:
        return work

    out = work.copy()
    col_po = out.iloc[:, 1]    # B  -> Ref. PO No.
    col_qty = out.iloc[:, 7]   # H  -> QTY
    col_unit = out.iloc[:, 8]  # I  -> UOM
    col_price = out.iloc[:, 9]  # J  -> Unit price ("Price")
    col_vendor = out.iloc[:, 15]  # P  -> Vendor name
    col_date = out.iloc[:, 22]  # W  -> Ref. PO Date

    def _nonempty_count(s: Any) -> int:
        if not isinstance(s, pd.Series):
            return 0
        return int(s.fillna("").astype(str).str.strip().ne("").sum())

    def _price_count(s: Any) -> int:
        if not isinstance(s, pd.Series):
            return 0
        return int(s.map(_parse_reference_unit_cost_scalar).map(lambda v: v is not None and np.isfinite(float(v))).sum())

    def _numeric_count(s: Any) -> int:
        if not isinstance(s, pd.Series):
            return 0
        return int(_clean_numeric_series(s).notna().sum())

    if COL_PO not in out.columns or _nonempty_count(out[COL_PO]) == 0:
        out[COL_PO] = col_po
    # QTY drives line-amount math (qty * unit_cost) and the cross-search summary;
    # fall back to column H whenever the canonical "QTY" column is missing or non-numeric.
    if COL_QTY not in out.columns or _numeric_count(out[COL_QTY]) == 0:
        out[COL_QTY] = col_qty
    if COL_UNIT not in out.columns or _nonempty_count(out[COL_UNIT]) == 0:
        out[COL_UNIT] = col_unit
    # Price is critical for NEW/benchmark decisions; prefer column J when it has numeric values.
    if COL_UNIT_COST not in out.columns or _price_count(out[COL_UNIT_COST]) == 0:
        out[COL_UNIT_COST] = col_price
    if COL_VENDOR not in out.columns or _nonempty_count(out[COL_VENDOR]) == 0:
        out[COL_VENDOR] = col_vendor
    if COL_PO_DATE not in out.columns or _nonempty_count(out[COL_PO_DATE]) == 0:
        out[COL_PO_DATE] = col_date

    return out

def _cross_match_prepare_work_df(
    store: Dict[str, Any], tab_raw: Optional[str]
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, str]]:
    """
    Build the cross-match working frame (sorted by PO date when available) and return
    ``(work, hist, tab)``. ``hist`` is the effective historical DataFrame for the tab.
    """
    eff = _effective_historical_store(
        store,
        tab_raw,
        strict_tab=isinstance(store.get("historical_by_company"), dict) and bool(store.get("historical_by_company")),
    )
    hist = eff.get("historical")
    if not isinstance(hist, pd.DataFrame) or hist.empty:
        return None
    tab = _normalize_new_po_source_tab(tab_raw)
    desc_series = None
    if COL_DESC in hist.columns:
        desc_series = hist[COL_DESC]
        if isinstance(desc_series, pd.DataFrame):
            desc_series = desc_series.iloc[:, 0]
    if desc_series is None:
        desc_col_index_by_tab = {"MBL": 6, "IFAS": 10, "MSB": 8}
        desc_idx = desc_col_index_by_tab.get(tab, -1)
        if desc_idx >= 0 and hist.shape[1] > desc_idx:
            try:
                desc_series = hist.iloc[:, desc_idx]
            except Exception:
                desc_series = None
    if desc_series is None:
        desc_series = pd.Series([""] * len(hist), index=hist.index)
    work = hist.copy()
    if tab == "MBL":
        work = _cross_match_apply_mbl_positional_fallback(work)
    work = _cross_match_ensure_unit_cost_column(work)
    work["__cross_desc__"] = desc_series.fillna("").astype(str)
    cols = [c for c in [COL_ITEM_NO, "__cross_desc__", COL_UNIT_COST, COL_UNIT, COL_VENDOR, COL_PO, COL_PO_DATE, COL_PROJECT] if c in work.columns]
    if not cols:
        return None
    work = work[cols].dropna(how="all").rename(columns={"__cross_desc__": COL_DESC})
    if work.empty:
        return None
    if COL_PO_DATE in work.columns:
        iso_dates = work[COL_PO_DATE].map(_po_date_iso_from_cell)
        sort_key = pd.to_datetime(iso_dates, errors="coerce")
        work = work.assign(__cross_sort_po_date=sort_key)
        work = work.sort_values("__cross_sort_po_date", ascending=False, na_position="last")
        work = work.drop(columns=["__cross_sort_po_date"])
    # De-duplicate identical receipt rows: ERP exports often emit the same PO line once
    # per receipt event. Group by (item_no, normalized description, rounded unit cost,
    # vendor) and keep the **first** occurrence (which is the most recent because rows
    # are already date-sorted descending). Drops dozens of dupes per cold load and prevents
    # the BM25 + Haiku pool from being dominated by repetitions.
    work = _cross_match_dedupe_receipt_rows(work)
    work = _cross_match_drop_unpriceable_rows(work)
    return work, hist, tab

def _cross_match_drop_unpriceable_rows(work: pd.DataFrame) -> pd.DataFrame:
    """Drop ERP rows missing item no or unit cost — they cannot back a cross-match benchmark."""
    if work is None or work.empty:
        return work
    item_ok = (
        work[COL_ITEM_NO].apply(lambda v: bool(clean_ccapr_item_no_input(v)))
        if COL_ITEM_NO in work.columns
        else pd.Series([False] * len(work), index=work.index)
    )
    cost_ok = (
        work[COL_UNIT_COST].apply(lambda v: _parse_reference_unit_cost_scalar(v) is not None)
        if COL_UNIT_COST in work.columns
        else pd.Series([False] * len(work), index=work.index)
    )
    kept = item_ok & cost_ok
    dropped = int((~kept).sum())
    if dropped:
        logger.debug(
            "Cross-match work df: dropped %s row(s) without usable item no + unit cost",
            dropped,
        )
    return work.loc[kept].copy()

def _cross_match_dedupe_receipt_rows(work: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate receipt rows (see ``_cross_match_prepare_work_df`` note)."""
    if work is None or work.empty:
        return work
    if COL_ITEM_NO not in work.columns:
        return work
    item_key = work[COL_ITEM_NO].map(
        lambda v: normalized_item_key_from_input(clean_ccapr_item_no_input(v))
    )
    desc_key = (
        work[COL_DESC].fillna("").astype(str).map(_normalize_cross_desc_for_match)
        if COL_DESC in work.columns
        else pd.Series([""] * len(work), index=work.index)
    )
    if COL_UNIT_COST in work.columns:
        cost_key = work[COL_UNIT_COST].map(_parse_reference_unit_cost_scalar)
        cost_key = cost_key.map(lambda v: round(float(v), 4) if v is not None and np.isfinite(v) else None)
    else:
        cost_key = pd.Series([None] * len(work), index=work.index)
    if COL_VENDOR in work.columns:
        vendor_key = work[COL_VENDOR].fillna("").astype(str).map(_normalize_vendor_key_local)
    else:
        vendor_key = pd.Series([""] * len(work), index=work.index)
    composite = list(zip(item_key.tolist(), desc_key.tolist(), cost_key.tolist(), vendor_key.tolist()))
    work = work.assign(__cross_dedupe_key=composite)
    out = work.drop_duplicates(subset="__cross_dedupe_key", keep="first").drop(columns=["__cross_dedupe_key"])
    return out

