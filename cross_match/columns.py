"""ERP column header normalization for cross-match work frames."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

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
)

from .parsing import _cross_match_series_has_numeric_prices, _parse_reference_unit_cost_scalar, _strip_col

def _build_header_alias_map() -> Dict[str, str]:
    """
    Map common ERP / export header text (lowercase) -> canonical column names
    used in comparison_engine (COL_*). Ported from cost-control-app reference.
    """
    m: Dict[str, str] = {}

    def add(keys: List[str], canonical: str) -> None:
        for k in keys:
            m[k.strip().lower()] = canonical

    add(
        [
            "item no.",
            "item no",
            "item",
            "item number",
            "item #",
            "item#",
            "material",
            "material number",
            "material no",
            "material no.",
            "part no",
            "part no.",
            "part number",
            "sku",
            "stock code",
            "stockcode",
        ],
        COL_ITEM_NO,
    )
    add(
        [
            "item description",
            "item desc",
            "material description",
            "description",
            "long description",
            "po description",
            "line description",
            "line item description",
        ],
        COL_DESC,
    )
    add(
        [
            "unit cost",
            "unit price",
            "net price",
            "net unit cost",
            "rate",
            "cost",
            "unit rate",
            "price",
            "unit cost (sar)",
            "unit price (sar)",
            "net unit cost (sar)",
            "unit cost sar",
            "price (sar)",
            "price sar",
            "unit price (sar)",
            "po price",
            "line price",
            "mbl price",
            "unit amount",
            "net amount",
            "valuation",
            "local value",
            "unit value",
            "unite cost",
        ],
        COL_UNIT_COST,
    )
    add(["unit", "uom", "uom.", "unit of measure", "um", "u/m"], COL_UNIT)
    add(
        [
            "po number",
            "po",
            "po #",
            "p.o.",
            "p.o",
            "p.o. number",
            "purchase order",
            "purchase order no",
            "purchase order no.",
            "po no",
            "po no.",
            "client po",
            "document no.",
            "document no",
        ],
        COL_PO,
    )
    add(
        [
            "po order date",
            "ordered date",
            "order date",
            "po date",
            "date ordered",
            "order placed date",
            "po.order date",
        ],
        COL_PO_DATE,
    )
    add(["status date", "changed date", "last status date"], COL_PO_DATE)
    add(
        [
            "po co name",
            "po company",
            "po company name",
            "po.company name",
            "vendor name",
            "vendor",
            "supplier",
            "supplier name",
            "company",
            "po vendor",
        ],
        COL_VENDOR,
    )
    add(
        [
            "qty",
            "quantity",
            "q.ty",
            "order qty",
            "order quantity",
            "qty ordered",
            "ordered qty",
            "line qty",
            "line quantity",
            "ord qty",
            "po line qty",
            "po qty",
        ],
        COL_QTY,
    )
    add(
        [
            "project name",
            "project",
            "job name",
            "site",
            "po site",
            "po site name",
            "po.site",
            "project number",
            "project no",
            "project no.",
        ],
        COL_PROJECT,
    )
    return m

_HEADER_ALIASES = _build_header_alias_map()

def _normalize_header_cell_text(x: Any) -> str:
    """Normalize Excel header text for row scanning and rename keys (NBSP, unicode, collapse spaces)."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def _cross_match_guess_price_column(work: pd.DataFrame) -> Optional[str]:
    """
    Last resort: pick the column whose header looks price-like and has the most numeric cells.
    Skips obvious totals/qty columns.
    """
    best_col: Optional[str] = None
    best_n = 0
    for c in work.columns:
        cs = str(c)
        if cs.startswith("__"):
            continue
        key = _normalize_header_cell_text(_strip_col(cs))
        if not key or key.startswith("unnamed"):
            continue
        if any(
            bad in key
            for bad in (
                "total",
                "subtotal",
                "extended",
                "line total",
                "order qty",
                "qty ordered",
                "quantity",
                "line qty",
            )
        ):
            continue
        if "qty" in key and "price" not in key and "cost" not in key:
            continue
        if not any(
            good in key
            for good in ("price", "cost", "rate", "value", "amount", "sar", "tariff", "valuation")
        ):
            continue
        n = _clean_numeric_series(work[c])
        nn = int(n.notna().sum())
        if nn > best_n:
            best_n = nn
            best_col = cs
    return best_col if best_n > 0 else None

def _cross_match_ensure_unit_cost_column(work: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-match TSV must expose a unit cost column the model can read.
    MBL exports often use ``Price`` or non-canonical headers. If ``Unit Cost`` exists but is empty
    while another column holds prices, remap — otherwise AI enrichment and local fallback see no costs.
    """
    work = work.copy()
    if COL_UNIT_COST in work.columns and _cross_match_series_has_numeric_prices(work[COL_UNIT_COST]):
        return work

    for c in list(work.columns):
        if str(c).startswith("__"):
            continue
        key = _normalize_header_cell_text(_strip_col(str(c)))
        if not key or key.startswith("unnamed"):
            continue
        if _HEADER_ALIASES.get(key) == COL_UNIT_COST:
            work[COL_UNIT_COST] = work[c]
            if _cross_match_series_has_numeric_prices(work[COL_UNIT_COST]):
                return work
    for c in list(work.columns):
        if str(c).startswith("__"):
            continue
        key = _normalize_header_cell_text(_strip_col(str(c)))
        if not key or key.startswith("unnamed"):
            continue
        if "total" in key or "subtotal" in key or "extended" in key:
            continue
        if "price" in key or "unit cost" in key or key in ("rate", "tariff"):
            work[COL_UNIT_COST] = work[c]
            if _cross_match_series_has_numeric_prices(work[COL_UNIT_COST]):
                return work

    guess = _cross_match_guess_price_column(work)
    if guess is not None:
        work[COL_UNIT_COST] = work[guess]
    return work

