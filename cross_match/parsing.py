"""Parsing and normalization helpers for cross-match."""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from comparison_engine import _clean_numeric_series

def _normalize_new_po_source_tab(raw: Optional[str]) -> str:
    s = (raw or "").strip().upper()
    if s in ("IFAS", "MBL", "MSB"):
        return s
    return "MBL"

def _normalize_vendor_key_local(raw: Any) -> str:
    """Normalize vendor name to a comparable key: uppercase, strip legal suffixes, alphanumeric only."""
    s = str(raw or "").strip().replace("\t", " ").replace("\n", " ").upper()
    for suffix in (
        " CO.",
        " CO",
        " COMPANY",
        " LTD",
        " LLC",
        " EST.",
        " EST",
        " ESTABLISHMENT",
        " CORP",
        " CORPORATION",
        " INC",
        " W.L.L",
        " WLL",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def _vendor_keys_match(key_a: str, key_b: str) -> bool:
    """True if normalized vendor keys are equal or share ≥ 85% sequence similarity."""
    if not key_a or not key_b:
        return False
    if key_a == key_b:
        return True
    ratio = difflib.SequenceMatcher(None, key_a, key_b).ratio()
    return ratio >= 0.85

def _dict_first(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            return d[k]
    lk = {str(a).lower(): a for a in d}
    for want in keys:
        ak = lk.get(want.lower())
        if ak is not None and d[ak] is not None and d[ak] != "":
            return d[ak]
    return None

def _parse_reference_unit_cost_scalar(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    if isinstance(val, (int, float, np.floating)):
        if isinstance(val, float) and pd.isna(val):
            return None
        f = float(val)
        return f if np.isfinite(f) else None
    s = str(val).strip()
    if not s:
        return None
    s_cur = re.sub(r"(?i)\s*(SAR|ر\.س|USD|EUR)\s*", "", s)
    s_flat = s_cur.replace(",", "").replace(" ", "")
    try:
        f = float(s_flat)
        return f if np.isfinite(f) else None
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val).replace(",", ""))
        if m:
            try:
                f = float(m.group(0))
                return f if np.isfinite(f) else None
            except ValueError:
                pass
    try:
        n = _clean_numeric_series(pd.Series([s]))
        if len(n.index) and pd.notna(n.iloc[0]):
            f = float(n.iloc[0])
            return f if np.isfinite(f) else None
    except Exception:
        pass
    return None

def _cross_match_series_has_numeric_prices(s: pd.Series) -> bool:
    """True if any cell parses as a finite unit-cost-like number (same cleaner as compare)."""
    if s is None or len(s) == 0:
        return False
    n = _clean_numeric_series(s)
    if not n.notna().any():
        return False
    v = n[n.notna()].astype(float)
    return bool(np.isfinite(v).any())

def _po_date_iso_from_cell(val: Any) -> Optional[str]:
    """Parse ERP PO date cells: datetimes, date strings, or Excel serial day numbers."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (bytes, bytearray)):
        try:
            val = val.decode("utf-8", errors="replace").strip()
        except Exception:
            return None
    # Excel stores dates as numeric day counts; pd.Timestamp(45321) would misinterpret.
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
        f = float(val)
        if np.isfinite(f) and 29500 <= f <= 65000:
            try:
                dt = pd.Timestamp("1899-12-30") + pd.to_timedelta(f, unit="D")
                if pd.notna(dt) and 1990 <= dt.year <= 2040:
                    return dt.date().isoformat()
            except Exception:
                pass
    try:
        ts = pd.Timestamp(val)
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    except Exception:
        return None

def _strip_col(c: Any) -> str:
    return str(c).strip()

