"""
Pure comparison / historical lookup logic for CCAPR.

Imported by app.py only (no Flask, no import of app).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Expected ERP column names (header row is row index 2 in Excel)
COL_ITEM_NO = "Item No."
COL_DESC = "Item Description"
COL_UNIT_COST = "Unit Cost"
COL_UNIT = "Unit"
COL_PO = "PO Number"
COL_PO_DATE = "PO Order Date"
COL_VENDOR = "PO Co Name"
COL_QTY = "QTY"
# Optional in ERP export — mapped when a matching column title exists.
COL_PROJECT = "Project Name"

# Compare benchmarks: lowest unit cost using PO Order Date within this many years of today;
# items with no PO in the window fall back to the most recent PO strictly before that window.
BENCHMARK_LOOKBACK_YEARS = 2


def clean_ccapr_item_no_input(raw: Any) -> str:
    """
    Strip BOM / ZWSP / NBSP and normalize Unicode before hashing item keys.
    Matches pasted values from Excel or ERP grids that would otherwise not match history.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFKC", str(raw))
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        s = s.replace(ch, "")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_item_no_one(val: Any) -> str:
    """
    Stable key for Item No. matching (historical ERP vs CCAPR input).

    Excel often stores numeric item numbers as floats; pandas yields 12345.0 which becomes
    the string \"12345.0\" and would never match user input \"12345\". Strip integer floats
    and trailing .0 / .00 string forms so keys align with the UI normalizeItemKey logic.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, (float, np.floating)):
        if not np.isfinite(float(val)):
            return ""
        xf = float(val)
        if xf == int(xf):
            return str(int(xf)).upper()
        return str(xf).strip().upper()
    if isinstance(val, (int, np.integer)):
        return str(int(val)).upper()
    t = str(val).strip()
    t = re.sub(r"\s+", " ", t)
    # String forms of Excel floats: "12345.0", "12345.00"
    m = re.fullmatch(r"(\d+)\.0+", t)
    if m:
        t = m.group(1)
    return t.upper()


def _normalize_item_no_series(s: pd.Series) -> pd.Series:
    """
    Per-cell item key (see `_normalize_item_no_one`). This replaces a naive
    `s.astype(str).str.strip().str.upper()` series: Excel often emits floats and
    \"12345.0\" strings that would not match CCAPR / `normalizeItemKey` without it.
    """
    return s.apply(_normalize_item_no_one)


def normalized_item_key_from_input(raw: Any) -> str:
    s = clean_ccapr_item_no_input(raw)
    if not s:
        return ""
    return _normalize_item_no_series(pd.Series([s])).iloc[0]


def _item_no_series_from_df(df: pd.DataFrame) -> pd.Series:
    """Single Series for Item No.; duplicate column names from imports become a DataFrame."""
    s = df[COL_ITEM_NO]
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s


_UOM_CANONICAL = {
    "PH": "PH",
    "P/H": "PH",
}


def _canonical_uom_token(normalized: str) -> str:
    """Map known aliases to one token (P/H and PH are the same)."""
    return _UOM_CANONICAL.get(normalized, normalized)


def _normalize_uom(s: str) -> str:
    """Normalize unit of measure for comparison (EA vs ea vs Ea; P/H vs PH)."""
    if not s or not str(s).strip():
        return ""
    out = (
        str(s)
        .strip()
        .upper()
        .replace(".", "")
        .replace(" ", "")
    )
    return _canonical_uom_token(out)


def _normalize_vendor_key(s: Any) -> str:
    return (
        str(s or "")
        .strip()
        .upper()
        .replace("\t", " ")
        .replace("\n", " ")
    )


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)

    raw = s.fillna("").astype(str)
    cleaned = raw.str.replace(r"[^0-9,.\-]", "", regex=True)

    has_comma = cleaned.str.contains(",", regex=False)
    has_dot = cleaned.str.contains(".", regex=False)
    cleaned = cleaned.mask(has_comma & has_dot, cleaned.str.replace(",", "", regex=False))
    cleaned = cleaned.mask(has_comma & ~has_dot, cleaned.str.replace(",", ".", regex=False))

    return pd.to_numeric(cleaned, errors="coerce")


def _parse_po_dates(s: pd.Series) -> pd.Series:
    """
    Parse PO dates from mixed ERP exports.

    Some sheets store Excel serial day numbers (e.g. 45265). Plain ``pd.to_datetime``
    interprets those as epoch-based integers, collapsing many rows to 1970. We
    explicitly map plausible Excel serial ranges to the Excel origin.
    """
    out = pd.to_datetime(s, errors="coerce", utc=False)

    nums = pd.to_numeric(s, errors="coerce")
    # Plausible modern Excel serial dates (roughly 1990..2040).
    excel_mask = nums.notna() & nums.between(29500, 65000)
    if excel_mask.any():
        out = out.copy()
        out.loc[excel_mask] = pd.Timestamp("1899-12-30") + pd.to_timedelta(
            nums.loc[excel_mask], unit="D"
        )
    return out


def _to_nullable_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int, np.floating, np.integer)):
        if pd.isna(x):
            return None
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        cleaned = re.sub(r"[^0-9,.\-]", "", s)
        cleaned = cleaned.replace(",", "")
        try:
            v = float(cleaned)
        except Exception:
            return None
        if pd.isna(v):
            return None
        return v
    return None


def _nullable_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    # Duplicate canonical column names can make r[COL_*] a Series; prefer first non-empty cell.
    if isinstance(x, pd.Series):
        for v in x:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            t = str(v).strip()
            if t:
                return t
        return ""
    return str(x).strip()


# Shown when Item Description cannot be resolved from the ERP export ("Item Description" column only).
ITEM_DESC_MISSING_PLACEHOLDER = "non"


def _item_description_from_historical(hist_df: Optional[pd.DataFrame], item_key: str) -> str:
    """First non-empty value from the ERP column `Item Description` for this item key (any matching row)."""
    if hist_df is None or hist_df.empty or not item_key:
        return ""
    if COL_ITEM_NO not in hist_df.columns or COL_DESC not in hist_df.columns:
        return ""
    work = hist_df.copy()
    work["__item_key"] = _normalize_item_no_series(_item_no_series_from_df(work))
    sub = work[work["__item_key"] == item_key]
    if sub.empty:
        return ""
    for _, r in sub.iterrows():
        d = _nullable_str(r.get(COL_DESC))
        if d:
            return d
    return ""


def _finalize_item_description(
    primary_desc: str,
    hist_full: Optional[pd.DataFrame],
    item_key: str,
) -> str:
    """Use ERP Item Description only; if missing anywhere, return the placeholder."""
    d = _nullable_str(primary_desc)
    if d:
        return d
    d = _item_description_from_historical(hist_full, item_key)
    if d:
        return d
    return ITEM_DESC_MISSING_PLACEHOLDER


def _site_from_series(r: Any) -> str:
    """Project / PO site name from a historical row (COL_PROJECT)."""
    if r is None:
        return ""
    if isinstance(r, pd.DataFrame):
        if r.empty:
            return ""
        r = r.iloc[0]
    try:
        if COL_PROJECT not in r.index:
            return ""
        v = r[COL_PROJECT]
        if pd.isna(v):
            return ""
        return _nullable_str(v)
    except Exception:
        return ""


def _optional_new_unit_cost_from_item(it: Dict[str, Any]) -> Optional[float]:
    """Quoted new unit price from the CCAPR input row, if provided."""
    v = it.get("newUnitCost")
    if v is None:
        return None
    if isinstance(v, str) and not str(v).strip():
        return None
    return _to_nullable_float(v)


def _input_item_description(it: Dict[str, Any]) -> str:
    """Manual Item Description from CCAPR JSON (`itemDescription` or legacy `description`)."""
    return _nullable_str(it.get("itemDescription") or it.get("description"))


def build_latest_by_item(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per Item No. (normalized): most recent by PO Order Date.
    """
    work = df.copy()
    work["__item_key"] = _normalize_item_no_series(_item_no_series_from_df(work))
    work = work[work["__item_key"] != ""].copy()

    work["__po_date"] = _parse_po_dates(work[COL_PO_DATE])
    work["__unit_cost"] = _clean_numeric_series(work[COL_UNIT_COST])

    work = work.sort_values(["__item_key", "__po_date"], ascending=[True, False], na_position="last")
    latest = work.drop_duplicates(subset=["__item_key"], keep="first")

    return latest


def build_lowest_by_item(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per Item No.: row with the **lowest** Unit Cost in history.
    Ties: use the most recent PO date among rows tied on price.
    """
    work = df.copy()
    work["__item_key"] = _normalize_item_no_series(_item_no_series_from_df(work))
    work = work[work["__item_key"] != ""].copy()

    work["__po_date"] = _parse_po_dates(work[COL_PO_DATE])
    work["__unit_cost"] = _clean_numeric_series(work[COL_UNIT_COST])
    valid = work[work["__unit_cost"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=work.columns)

    valid = valid.sort_values(
        ["__item_key", "__unit_cost", "__po_date"],
        ascending=[True, True, False],
        na_position="last",
    )
    cheapest = valid.groupby("__item_key", as_index=False).head(1)
    return cheapest


def _benchmark_window_cutoff() -> pd.Timestamp:
    """Start of PO-date window (inclusive): today at midnight minus ``BENCHMARK_LOOKBACK_YEARS``."""
    return pd.Timestamp.now().normalize() - pd.DateOffset(years=BENCHMARK_LOOKBACK_YEARS)


def _filter_rows_po_date_in_benchmark_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose PO Order Date parses and falls on/after the benchmark window cutoff."""
    if df is None or df.empty or COL_PO_DATE not in df.columns:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()

    work = df.copy()
    po = _parse_po_dates(work[COL_PO_DATE])
    cutoff = _benchmark_window_cutoff()
    m = po.notna() & (po >= cutoff)
    return work.loc[m].copy()


def build_most_recent_by_item_before_benchmark_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per item: the line with the **most recent** PO Order Date among rows **strictly before**
    the benchmark window cutoff (older than ``BENCHMARK_LOOKBACK_YEARS``). Requires parsable date,
    valid unit cost, and non-empty item key. Used only as fallback when the in-window benchmark
    is missing for that item.
    """
    if df is None or df.empty or COL_PO_DATE not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["__item_key"] = _normalize_item_no_series(_item_no_series_from_df(work))
    work = work[work["__item_key"] != ""].copy()
    work["__po_date"] = _parse_po_dates(work[COL_PO_DATE])
    work["__unit_cost"] = _clean_numeric_series(work[COL_UNIT_COST])
    cutoff = _benchmark_window_cutoff()
    sub = work[
        work["__po_date"].notna()
        & (work["__po_date"] < cutoff)
        & work["__unit_cost"].notna()
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(["__item_key", "__po_date"], ascending=[True, False], na_position="last")
    return sub.drop_duplicates(subset=["__item_key"], keep="first")


def build_lowest_by_item_with_window_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per item: **lowest** unit cost among POs in the last ``BENCHMARK_LOOKBACK_YEARS``.

    If an item has no qualifying line in that window, fall back to the **most recent** PO **before**
    the window (by PO Order Date), not the all-time lowest price.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    in_window = _filter_rows_po_date_in_benchmark_window(df)
    low_win = build_lowest_by_item(in_window) if not in_window.empty else pd.DataFrame()
    low_fb = build_most_recent_by_item_before_benchmark_window(df)
    if low_fb.empty:
        return low_win
    if low_win.empty:
        return low_fb
    keys = set(low_win["__item_key"].astype(str))
    tail = low_fb[~low_fb["__item_key"].astype(str).isin(keys)]
    return pd.concat([low_win, tail], ignore_index=True)


def run_compare(
    store: Dict[str, Any],
    items: List[Dict[str, Any]],
    new_po_user: str,
    ccapr_vendor: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Core /api/compare logic. Returns (rows, None) on success, (None, error_message) on client error.
    """
    if "historical" not in store or store.get("historical") is None:
        return None, "Please upload the historical Excel file first."

    if not items:
        return None, "Add at least one line with an Item No."

    exclude_po_norm = (new_po_user or "").strip()
    hist_for_bench = store["historical"].copy()
    if exclude_po_norm:
        hist_for_bench["__po_norm"] = hist_for_bench[COL_PO].fillna("").astype(str).str.strip()
        hist_for_bench = hist_for_bench[hist_for_bench["__po_norm"] != exclude_po_norm].copy()

    latest = build_lowest_by_item_with_window_fallback(hist_for_bench)
    if latest.empty:
        idx_last = pd.DataFrame().set_index(pd.Index([], name="__item_key"))
    else:
        idx_last = latest.set_index("__item_key", drop=False)
    idx_same_vendor_latest = pd.DataFrame().set_index(pd.Index([], name="__item_key"))
    idx_all_vendor_latest = pd.DataFrame().set_index(pd.Index([], name="__item_key"))
    vendor_key = _normalize_vendor_key(ccapr_vendor)
    if hist_for_bench is not None:
        hv_all = hist_for_bench.copy()
        hv_all["__vendor_key"] = hv_all[COL_VENDOR].fillna("").astype(str).map(_normalize_vendor_key)
        if vendor_key:
            hv_all = hv_all[hv_all["__vendor_key"] != vendor_key].copy()
        hv_all["__item_key"] = _normalize_item_no_series(_item_no_series_from_df(hv_all))
        hv_all["__po_date"] = _parse_po_dates(hv_all[COL_PO_DATE])
        hv_all["__unit_cost"] = _clean_numeric_series(hv_all[COL_UNIT_COST])
        valid_all = hv_all[
            (hv_all["__item_key"] != "")
            & hv_all["__unit_cost"].notna()
        ].copy()
        if not valid_all.empty:
            lowest_all = build_lowest_by_item_with_window_fallback(valid_all)
            if not lowest_all.empty:
                idx_all_vendor_latest = lowest_all.set_index("__item_key", drop=False)

    if vendor_key and "historical" in store:
        hv = hist_for_bench.copy() if hist_for_bench is not None else store["historical"].copy()
        hv["__item_key"] = _normalize_item_no_series(_item_no_series_from_df(hv))
        hv["__vendor_key"] = hv[COL_VENDOR].fillna("").astype(str).map(_normalize_vendor_key)
        hv["__po_date"] = _parse_po_dates(hv[COL_PO_DATE])
        hv["__unit_cost"] = _clean_numeric_series(hv[COL_UNIT_COST])
        valid_sv = hv[
            (hv["__item_key"] != "")
            & (hv["__vendor_key"] == vendor_key)
            & hv["__unit_cost"].notna()
        ].copy()
        if not valid_sv.empty:
            lowest_sv = build_lowest_by_item_with_window_fallback(valid_sv)
            if not lowest_sv.empty:
                idx_same_vendor_latest = lowest_sv.set_index("__item_key", drop=False)

    hist_full: Optional[pd.DataFrame] = store.get("historical")
    rows: List[Dict[str, Any]] = []
    rid = 0
    for it in items:
        raw_item = clean_ccapr_item_no_input(it.get("itemNo"))
        if not raw_item:
            continue

        item_key = normalized_item_key_from_input(raw_item)
        qty = _to_nullable_float(it.get("qty"))
        if qty is None:
            qty = 1.0
        input_unit = _nullable_str(it.get("unit"))
        input_desc = _input_item_description(it)

        if item_key not in idx_last.index:
            quoted = _optional_new_unit_cost_from_item(it)
            new_uc = quoted
            new_line = (new_uc * qty) if new_uc is not None else None
            desc_new = _finalize_item_description(input_desc, hist_full, item_key)
            rows.append(
                {
                    "__id": rid,
                    "item_no": raw_item,
                    "item_key": item_key,
                    "previous_po_number": "",
                    "new_po_number": new_po_user,
                    "ccapr_vendor": ccapr_vendor,
                    "description": desc_new,
                    "unit": input_unit,
                    "hist_unit_last_po": "",
                    "unit_mismatch": False,
                    "lowest_hist_unit_cost": None,
                    "same_vendor_latest_hist_unit_cost": None,
                    "same_vendor_benchmark_vendor": "",
                    "same_vendor_benchmark_po_date": None,
                    "same_vendor_benchmark_po_number": "",
                    "lowest_benchmark_vendor": "",
                    "lowest_benchmark_po_date": None,
                    "lowest_benchmark_po_number": "",
                    "lowest_benchmark_hist_unit": "",
                    "lowest_benchmark_site_name": "",
                    "same_vendor_benchmark_site_name": "",
                    "same_vendor_hist_unit": "",
                    "new_unit_cost": new_uc,
                    "qty": qty,
                    "benchmark_line_cost": None,
                    "new_line_cost": new_line,
                    "difference_sar": None,
                    "pct_change": None,
                    "vendor_last": "",
                    "last_po_date": None,
                    "has_history": False,
                }
            )
            rid += 1
            continue

        r_last = idx_last.loc[item_key]
        if isinstance(r_last, pd.DataFrame):
            r_last = r_last.iloc[0]

        desc = _finalize_item_description(
            input_desc or _nullable_str(r_last[COL_DESC]),
            hist_full,
            item_key,
        )
        hist_unit_last = _nullable_str(r_last[COL_UNIT])
        prev_po_num = _nullable_str(r_last[COL_PO])
        vendor = _nullable_str(r_last[COL_VENDOR])
        po_date = r_last["__po_date"]
        last_po_date = None
        if pd.notna(po_date):
            try:
                last_po_date = pd.Timestamp(po_date).date().isoformat()
            except Exception:
                last_po_date = None

        lowest_uc = None
        lowest_benchmark_vendor = ""
        lowest_benchmark_po_number = ""
        lowest_benchmark_po_date = None
        lowest_site = ""
        lowest_benchmark_hist_unit = ""
        if item_key in idx_all_vendor_latest.index:
            r_low = idx_all_vendor_latest.loc[item_key]
            if isinstance(r_low, pd.DataFrame):
                r_low = r_low.iloc[0]
            lowest_uc = _to_nullable_float(r_low["__unit_cost"])
            lowest_benchmark_vendor = _nullable_str(r_low[COL_VENDOR])
            lowest_benchmark_po_number = _nullable_str(r_low[COL_PO])
            lowest_benchmark_hist_unit = _nullable_str(r_low[COL_UNIT])
            lowest_site = _site_from_series(r_low)
            po_date_low = r_low["__po_date"]
            if pd.notna(po_date_low):
                try:
                    lowest_benchmark_po_date = pd.Timestamp(po_date_low).date().isoformat()
                except Exception:
                    lowest_benchmark_po_date = None

        same_vendor_latest_uc = None
        same_vendor_benchmark_vendor = ""
        same_vendor_benchmark_po_date = None
        same_vendor_benchmark_po_number = ""
        same_sv_site = ""
        same_sv_unit = ""
        if item_key in idx_same_vendor_latest.index:
            r_sv = idx_same_vendor_latest.loc[item_key]
            if isinstance(r_sv, pd.DataFrame):
                r_sv = r_sv.iloc[0]
            same_vendor_latest_uc = _to_nullable_float(r_sv["__unit_cost"])
            same_vendor_benchmark_vendor = _nullable_str(r_sv[COL_VENDOR])
            same_vendor_benchmark_po_number = _nullable_str(r_sv[COL_PO])
            same_sv_site = _site_from_series(r_sv)
            same_sv_unit = _nullable_str(r_sv[COL_UNIT])
            po_date_sv = r_sv["__po_date"]
            if pd.notna(po_date_sv):
                try:
                    same_vendor_benchmark_po_date = pd.Timestamp(po_date_sv).date().isoformat()
                except Exception:
                    same_vendor_benchmark_po_date = None

        unit_display = input_unit or hist_unit_last
        unit_mismatch = False
        if input_unit and hist_unit_last:
            unit_mismatch = _normalize_uom(input_unit) != _normalize_uom(hist_unit_last)

        quoted = _optional_new_unit_cost_from_item(it)
        new_uc = quoted if quoted is not None else lowest_uc

        bench_line = (lowest_uc * qty) if lowest_uc is not None else None
        new_line = (new_uc * qty) if new_uc is not None else None
        diff = (
            ((new_uc - lowest_uc) * qty)
            if (lowest_uc is not None and new_uc is not None)
            else None
        )
        pct = (
            ((new_uc - lowest_uc) / lowest_uc * 100.0)
            if (lowest_uc is not None and new_uc is not None and lowest_uc != 0)
            else None
        )

        rows.append(
            {
                "__id": rid,
                "item_no": raw_item,
                "item_key": item_key,
                "previous_po_number": prev_po_num,
                "new_po_number": new_po_user,
                "ccapr_vendor": ccapr_vendor,
                "description": desc,
                "unit": unit_display,
                "hist_unit_last_po": hist_unit_last,
                "unit_mismatch": unit_mismatch,
                "lowest_hist_unit_cost": lowest_uc,
                "same_vendor_latest_hist_unit_cost": same_vendor_latest_uc,
                "same_vendor_benchmark_vendor": same_vendor_benchmark_vendor,
                "same_vendor_benchmark_po_date": same_vendor_benchmark_po_date,
                "same_vendor_benchmark_po_number": same_vendor_benchmark_po_number,
                "same_vendor_benchmark_site_name": same_sv_site,
                "same_vendor_hist_unit": same_sv_unit,
                "lowest_benchmark_vendor": lowest_benchmark_vendor,
                "lowest_benchmark_po_date": lowest_benchmark_po_date,
                "lowest_benchmark_po_number": lowest_benchmark_po_number,
                "lowest_benchmark_hist_unit": lowest_benchmark_hist_unit,
                "lowest_benchmark_site_name": lowest_site,
                "new_unit_cost": new_uc,
                "qty": qty,
                "benchmark_line_cost": bench_line,
                "new_line_cost": new_line,
                "difference_sar": diff,
                "pct_change": pct,
                "vendor_last": vendor,
                "last_po_date": last_po_date,
                "has_history": True,
            }
        )
        rid += 1

    if not rows:
        return None, "No valid rows (enter at least one Item No.)."

    return rows, None


def run_item_reference(
    store: Dict[str, Any],
    items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Returns (rows, optional_message_when_no_upload)."""
    hist_ir = store.get("historical")
    if hist_ir is None:
        return [], "Upload historical data first."
    if isinstance(hist_ir, pd.DataFrame) and hist_ir.empty and not store.get("_multi_company_erp"):
        return [], "Upload historical data first."

    latest = build_lowest_by_item_with_window_fallback(hist_ir)
    if latest.empty:
        idx_last = pd.DataFrame().set_index(pd.Index([], name="__item_key"))
    else:
        idx_last = latest.set_index("__item_key", drop=False)
    seen = set()
    rows_out: List[Dict[str, Any]] = []
    hist_full_ir: Optional[pd.DataFrame] = store.get("historical")

    for it in items:
        raw_item = clean_ccapr_item_no_input(it.get("itemNo"))
        if not raw_item:
            continue

        item_key = normalized_item_key_from_input(raw_item)
        if item_key in seen:
            continue
        seen.add(item_key)
        input_desc = _input_item_description(it)

        if item_key not in idx_last.index:
            rows_out.append(
                {
                    "item_no": raw_item,
                    "item_key": item_key,
                    "description": _finalize_item_description(input_desc, hist_full_ir, item_key),
                    "last_po_unit_cost": None,
                    "lowest_hist_unit_cost": None,
                    "hist_unit_last_po": "",
                    "previous_po_number": "",
                    "vendor_last": "",
                    "last_po_date": None,
                    "in_history": False,
                }
            )
            continue

        r_last = idx_last.loc[item_key]
        if isinstance(r_last, pd.DataFrame):
            r_last = r_last.iloc[0]

        last_uc = _to_nullable_float(r_last["__unit_cost"])
        lowest_uc = last_uc
        desc = _finalize_item_description(
            input_desc or _nullable_str(r_last[COL_DESC]),
            hist_full_ir,
            item_key,
        )
        hist_unit_last = _nullable_str(r_last[COL_UNIT])
        prev_po_num = _nullable_str(r_last[COL_PO])
        vendor = _nullable_str(r_last[COL_VENDOR])
        po_date = r_last["__po_date"]
        last_po_date = None
        if pd.notna(po_date):
            try:
                last_po_date = pd.Timestamp(po_date).date().isoformat()
            except Exception:
                last_po_date = None

        rows_out.append(
            {
                "item_no": raw_item,
                "item_key": item_key,
                "description": desc,
                "last_po_unit_cost": last_uc,
                "lowest_hist_unit_cost": lowest_uc,
                "hist_unit_last_po": hist_unit_last,
                "previous_po_number": prev_po_num,
                "vendor_last": vendor,
                "last_po_date": last_po_date,
                "in_history": True,
            }
        )

    return rows_out, None


def run_item_po_history(
    historical_df: pd.DataFrame,
    item_no_raw: str,
) -> List[Dict[str, Any]]:
    item_no_raw = clean_ccapr_item_no_input(item_no_raw)
    if not item_no_raw:
        return []

    df = historical_df
    if COL_PROJECT not in df.columns:
        df = df.copy()
        df[COL_PROJECT] = ""

    work = df.copy()
    work["__item_key"] = _normalize_item_no_series(_item_no_series_from_df(work))
    item_key = normalized_item_key_from_input(item_no_raw)
    sub = work[work["__item_key"] == item_key].copy()
    if sub.empty:
        return []

    sub["__po_date"] = _parse_po_dates(sub[COL_PO_DATE])
    sub["__unit_cost"] = _clean_numeric_series(sub[COL_UNIT_COST])
    sub["__qty"] = _clean_numeric_series(sub[COL_QTY])

    sub = sub.sort_values(["__po_date", COL_PO], ascending=[False, False], na_position="last")

    rows_out: List[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        po_date = r["__po_date"]
        po_date_s = None
        if pd.notna(po_date):
            try:
                po_date_s = pd.Timestamp(po_date).date().isoformat()
            except Exception:
                po_date_s = None

        proj = _nullable_str(r[COL_PROJECT]) if COL_PROJECT in sub.columns else ""

        rows_out.append(
            {
                "po_number": _nullable_str(r[COL_PO]),
                "project_name": proj,
                "po_date": po_date_s,
                "unit": _nullable_str(r[COL_UNIT]),
                "description": _nullable_str(r[COL_DESC]),
                "vendor": _nullable_str(r[COL_VENDOR]),
                "unit_cost": _to_nullable_float(r["__unit_cost"]),
                "qty": _to_nullable_float(r["__qty"]),
            }
        )

    return rows_out


def normalize_po_lookup_key(val: Any) -> str:
    """
    Canonical key for matching user-typed PO numbers to ERP ``COL_PO`` cells.

    Collapses whitespace, uppercases, and normalizes Excel numeric PO cells (e.g. 4500123.0 → 4500123)
    so typed values align with float/string exports.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, (float, np.floating)):
        xf = float(val)
        if not np.isfinite(xf):
            return ""
        if abs(xf - round(xf)) < 1e-9:
            t = str(int(round(xf)))
        else:
            t = str(xf).strip()
            m0 = re.fullmatch(r"(\d+)\.0+", t)
            if m0:
                t = m0.group(1)
    elif isinstance(val, (int, np.integer)):
        t = str(int(val))
    else:
        t = str(val).strip()
        m1 = re.fullmatch(r"(\d+)\.0+", t)
        if m1:
            t = m1.group(1)
    t = re.sub(r"\s+", " ", t.strip())
    return t.upper()


def run_po_history_lines(
    df: pd.DataFrame,
    po_number_raw: str,
    source_company: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return one row per ERP line for the given PO.

    ``source_company`` is the issuing company's tab key (``"IFAS"`` / ``"MBL"``
    / ``"MSB"``). It enables a narrow, opt-in positional fallback for the
    project-code source: when the ERP doesn't expose one of the known
    project-code header aliases, iFAS exports specifically use **column D**
    (positional index 3) for the project code. We only apply this fallback
    for iFAS to avoid silently mismapping in MBL/MSB ERPs whose column
    layouts differ.
    """
    work = df.copy()
    work["__po_key"] = work[COL_PO].map(normalize_po_lookup_key)
    target_po = normalize_po_lookup_key(po_number_raw)
    if not target_po:
        return []
    sub = work[work["__po_key"] == target_po].copy()

    if sub.empty:
        return []

    sub["__po_date"] = _parse_po_dates(sub[COL_PO_DATE])
    sub["__unit_cost"] = _clean_numeric_series(sub[COL_UNIT_COST])
    sub["__qty"] = _clean_numeric_series(sub[COL_QTY])
    sub["__line_idx"] = range(len(sub))

    # Project code source:
    # Prefer explicit project-code-ish headers only.
    # Do NOT use positional fallback (column index) by default because import
    # normalization / merges can shift dataframe column order and accidentally
    # map to unrelated columns (e.g. PO Status). The iFAS-specific fallback
    # below is gated on ``source_company == "IFAS"`` for that exact reason.
    def _norm_header(s: Any) -> str:
        t = unicodedata.normalize("NFKC", str(s or ""))
        t = t.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
        t = re.sub(r"\s+", " ", t).strip().lower()
        return t

    _pc_aliases = {
        "po project no",
        "po project no.",
        "po project number",
        "po project code",
        "project code",
        "projectcode",
        "project no",
        "project no.",
        "project number",
        "project #",
        "job code",
        "job no",
        "job no.",
        "site code",
    }
    project_code_col: Optional[str] = None
    for c in sub.columns:
        k = _norm_header(c)
        if k in _pc_aliases:
            project_code_col = str(c)
            break

    # iFAS-only positional fallback: column D (zero-indexed 3) carries the
    # project code in iFAS ERP exports, and the column header in those files
    # is not one of the standard aliases above. We only consult column D when:
    #   1. ``source_company`` is explicitly iFAS (no silent fallback for other tabs),
    #   2. the named-alias lookup found nothing,
    #   3. the dataframe actually has at least 4 columns.
    # An additional guard rejects column D candidates whose header looks like a
    # known *non*-project field (PO number/date/vendor/qty/cost) so we never
    # mismap if the column order has been re-arranged in the import pipeline.
    if (
        project_code_col is None
        and isinstance(source_company, str)
        and source_company.strip().upper() == "IFAS"
        and len(sub.columns) >= 4
    ):
        cand = str(sub.columns[3])
        cand_key = _norm_header(cand)
        _blocked_cand_keys = {
            "po number",
            "po no",
            "po no.",
            "po order date",
            "po date",
            "po status",
            "vendor",
            "vendor name",
            "supplier",
            "supplier name",
            "qty",
            "quantity",
            "unit cost",
            "unit price",
            "uom",
            "unit",
            "item",
            "item no",
            "item no.",
            "item number",
            "item description",
            "description",
            "desc",
        }
        if cand_key not in _blocked_cand_keys:
            project_code_col = cand

    sub = sub.sort_values(["__po_date", COL_ITEM_NO], ascending=[False, True], na_position="last")

    rows_out: List[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        po_date = r["__po_date"]
        po_date_s = None
        if pd.notna(po_date):
            try:
                po_date_s = pd.Timestamp(po_date).date().isoformat()
            except Exception:
                po_date_s = None

        proj = _nullable_str(r[COL_PROJECT]) if COL_PROJECT in sub.columns else ""
        proj_code = _nullable_str(r[project_code_col]) if project_code_col and project_code_col in sub.columns else ""

        rows_out.append(
            {
                "po_number": _nullable_str(r[COL_PO]),
                "po_date": po_date_s,
                "vendor": _nullable_str(r[COL_VENDOR]),
                "project_name": proj,
                "project_code": proj_code,
                "item_no": _nullable_str(r[COL_ITEM_NO]),
                "description": _nullable_str(r[COL_DESC]),
                "unit": _nullable_str(r[COL_UNIT]),
                "unit_cost": _to_nullable_float(r["__unit_cost"]),
                "qty": _to_nullable_float(r["__qty"]),
            }
        )

    return rows_out


def build_section2_prefill_from_po(
    df: pd.DataFrame,
    po_number_raw: str,
    source_company: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build Section 2 autofill payload: dominant vendor / site from ERP lines for this PO,
    plus one grid row per historical line (item, description, qty, unit, unit cost).

    ``source_company`` is forwarded to :func:`run_po_history_lines` so the
    iFAS-only column-D project-code fallback engages when appropriate.
    """
    raw = str(po_number_raw or "").strip()
    if not raw:
        return {
            "ok": False,
            "matched": False,
            "error": "Missing PO number.",
            "vendorName": "",
            "projectName": "",
            "projectCode": "",
            "lines": [],
            "lineCount": 0,
            "poNumberDisplay": "",
        }

    rows = run_po_history_lines(df, raw, source_company=source_company)
    if not rows:
        return {
            "ok": False,
            "matched": False,
            "error": "No ERP lines found for that PO number.",
            "vendorName": "",
            "projectName": "",
            "projectCode": "",
            "lines": [],
            "lineCount": 0,
            "poNumberDisplay": raw,
        }

    def _fmt_cell_num(x: Any) -> str:
        if x is None:
            return ""
        try:
            f = float(x)
        except (TypeError, ValueError):
            return str(x).strip()
        if not np.isfinite(f):
            return ""
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        s = f"{f:.6f}".rstrip("0").rstrip(".")
        return s or "0"

    vend_c = Counter((str(r.get("vendor") or "").strip()) for r in rows if str(r.get("vendor") or "").strip())
    proj_c = Counter(
        (str(r.get("project_name") or "").strip()) for r in rows if str(r.get("project_name") or "").strip()
    )
    vendor_name = vend_c.most_common(1)[0][0] if vend_c else ""
    project_name = proj_c.most_common(1)[0][0] if proj_c else ""

    code_c = Counter((str(r.get("project_code") or "").strip()) for r in rows if str(r.get("project_code") or "").strip())
    project_code = code_c.most_common(1)[0][0] if code_c else ""

    lines_out: List[Dict[str, str]] = []
    for r in rows:
        lines_out.append(
            {
                "itemNo": str(r.get("item_no") or "").strip(),
                "itemDescription": str(r.get("description") or "").strip(),
                "qty": _fmt_cell_num(r.get("qty")),
                "unit": str(r.get("unit") or "").strip(),
                "unitCost": _fmt_cell_num(r.get("unit_cost")),
            }
        )

    display_po = str(rows[0].get("po_number") or "").strip() or raw

    return {
        "ok": True,
        "matched": True,
        "error": "",
        "poNumberDisplay": display_po,
        "vendorName": vendor_name,
        "projectName": project_name,
        "projectCode": project_code,
        "lines": lines_out,
        "lineCount": len(lines_out),
    }


nullable_str = _nullable_str
