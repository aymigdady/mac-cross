"""Claude cross-match response parsing and field extraction."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from comparison_engine import (
    COL_ITEM_NO,
    COL_PO,
    COL_PO_DATE,
    COL_PROJECT,
    COL_UNIT,
    COL_UNIT_COST,
    COL_VENDOR,
    clean_ccapr_item_no_input,
    normalized_item_key_from_input,
    nullable_str,
)

from .parsing import _dict_first, _parse_reference_unit_cost_scalar

def _cross_match_list_looks_like_matches(lst: List[Any]) -> bool:
    if not lst:
        return False
    for x in lst[:8]:
        if not isinstance(x, dict):
            continue
        ks = {str(k).lower() for k in x}
        if ks & {
            "item_no",
            "itemno",
            "item_number",
            "reference_unit_cost",
            "referenceunitcost",
            "matched",
        }:
            return True
    return False

def _extract_cross_matches_from_parsed(parsed: Any) -> List[Dict[str, Any]]:
    """Claude may nest `matches`, use camelCase keys, or wrap JSON in extra objects."""
    if parsed is None:
        return []
    if isinstance(parsed, list):
        if _cross_match_list_looks_like_matches(parsed):
            return [x for x in parsed if isinstance(x, dict)]
        return []
    if not isinstance(parsed, dict):
        return []
    if "raw_text" in parsed and len(parsed) <= 2:
        return []
    for key in parsed:
        if str(key).lower() == "matches":
            v = parsed[key]
            if isinstance(v, list):
                dicts = [x for x in v if isinstance(x, dict)]
                if dicts and (
                    _cross_match_list_looks_like_matches(dicts)
                    or len(dicts) == len(v)
                ):
                    return dicts
    for v in parsed.values():
        if isinstance(v, (dict, list)):
            sub = _extract_cross_matches_from_parsed(v)
            if sub:
                return sub
    return []

def _cross_match_confidence_pct_from_dict(m: Dict[str, Any]) -> Optional[float]:
    """
    Semantic-match confidence from the model (0..1 or 0..100). Returns percentage 0..100 for UI.
    """
    if not isinstance(m, dict):
        return None
    v = _dict_first(m, "confidence", "Confidence", "semantic_confidence", "match_confidence")
    if v is None:
        for nk in (
            "reference_row",
            "referenceRow",
            "best_match",
            "bestMatch",
            "matched_row",
            "matchedRow",
        ):
            inner = m.get(nk)
            if isinstance(inner, dict):
                v = _dict_first(inner, "confidence", "Confidence")
                if v is not None:
                    break
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    if 0.0 <= x <= 1.0:
        return round(float(x) * 100.0, 1)
    if 1.0 < x <= 100.0:
        return round(float(x), 1)
    return None

def _cross_ai_matched_flag(m: Dict[str, Any]) -> Optional[bool]:
    v = _dict_first(m, "matched", "Matched", "is_matched")
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None

def _cross_match_effective_unit_cost(m: Dict[str, Any]) -> Optional[float]:
    """Top-level or nested (Claude sometimes puts the historical line in a sub-object)."""
    for key in (
        "reference_unit_cost",
        "referenceUnitCost",
        "unit_cost",
        "unitCost",
        "historical_unit_cost",
        "historicalUnitCost",
        "benchmark_unit_cost",
        "unit_price",
        "Unit Cost",
    ):
        if key in m:
            uc = _parse_reference_unit_cost_scalar(m.get(key))
            if uc is not None:
                return uc
    lk = {str(a).lower(): a for a in m}
    for want in ("reference_unit_cost", "referenceunitcost", "unit_cost", "unitcost"):
        ak = lk.get(want)
        if ak is not None:
            uc = _parse_reference_unit_cost_scalar(m.get(ak))
            if uc is not None:
                return uc
    for nested_key in (
        "reference_row",
        "referenceRow",
        "historical_row",
        "historicalRow",
        "best_match",
        "bestMatch",
        "matched_row",
        "matchedRow",
        "reference",
    ):
        inner = m.get(nested_key)
        if not isinstance(inner, dict):
            continue
        for key in (
            "reference_unit_cost",
            "referenceUnitCost",
            "unit_cost",
            "unitCost",
            COL_UNIT_COST,
            "Unit Cost",
            "price",
        ):
            if key in inner:
                uc = _parse_reference_unit_cost_scalar(inner.get(key))
                if uc is not None:
                    return uc
    return None

def _reference_fields_from_cross_match(m: Dict[str, Any]) -> Tuple[Optional[float], str, str, str, str, str]:
    """Unit cost plus vendor / PO / site from top-level or nested reference row."""
    uc = _cross_match_effective_unit_cost(m)
    ref_vendor = ""
    ref_unit = ""
    ref_po = ""
    ref_date = ""
    ref_site = ""
    nested: Optional[Dict[str, Any]] = None
    for nested_key in (
        "reference_row",
        "referenceRow",
        "historical_row",
        "historicalRow",
        "best_match",
        "bestMatch",
        "matched_row",
        "matchedRow",
        "reference",
    ):
        inner = m.get(nested_key)
        if isinstance(inner, dict):
            nested = inner
            break
    src_v = m
    if nested is not None and not _dict_first(m, "reference_vendor", "referenceVendor", "vendor"):
        src_v = nested
    ref_vendor = nullable_str(_dict_first(src_v, "reference_vendor", "referenceVendor", "vendor", COL_VENDOR, "PO Co Name"))
    ref_unit = nullable_str(_dict_first(src_v, "reference_unit", "referenceUnit", "unit", COL_UNIT))
    ref_po = nullable_str(_dict_first(src_v, "reference_po_number", "referencePoNumber", "po_number", "poNumber", COL_PO))
    ref_date = nullable_str(_dict_first(src_v, "reference_po_date", "referencePoDate", "po_date", COL_PO_DATE))
    ref_site = nullable_str(
        _dict_first(m, "reference_site", "referenceSite", "site", COL_PROJECT)
        or _dict_first(m, "matched_description_snippet", "matchedDescriptionSnippet")
    )
    if uc is None and nested is not None:
        uc = _cross_match_effective_unit_cost(nested)
    return uc, ref_vendor, ref_unit, ref_po, ref_date, ref_site

def _reference_item_no_from_cross_match_dict(m: Dict[str, Any]) -> str:
    """Item No. from the model's match object or nested reference row."""
    direct = nullable_str(_dict_first(m, "reference_item_no", "referenceItemNo", "matched_reference_item_no"))
    if direct:
        return clean_ccapr_item_no_input(direct)
    for nk in (
        "reference_row",
        "referenceRow",
        "matched_row",
        "matchedRow",
        "historical_row",
        "historicalRow",
        "best_match",
        "bestMatch",
    ):
        inner = m.get(nk)
        if isinstance(inner, dict):
            v = _dict_first(inner, COL_ITEM_NO, "item_no", "ItemNo", "item_number", "Item Number")
            if v is not None and str(v).strip():
                return clean_ccapr_item_no_input(v)
    return ""

def _cross_match_description_for_similarity(desc: str) -> str:
    """Strip placeholder CCAPR descriptions so we do not pick a nonsense ``best'' row."""
    s = nullable_str(desc).strip()
    if len(s) < 3:
        return ""
    if s.lower() in ("non", "n/a", "na", "none", "tbd", "-", "--", ".", "…"):
        return ""
    return s

def _cross_match_lookup_query_for_df(m: Dict[str, Any], ccapr_description: str) -> str:
    """
    Text used to resolve a row in ``cross_df``: real CCAPR description, else the model's snippet of the
    matched ERP line (when CCAPR text is missing or placeholder ``non``).
    """
    q = _cross_match_description_for_similarity(ccapr_description)
    if q:
        return q
    sn = nullable_str(
        _dict_first(
            m,
            "matched_description_snippet",
            "matchedDescriptionSnippet",
            "reference_description",
            "referenceDescription",
        )
    )
    return _cross_match_description_for_similarity(sn)

def _cross_match_inferred_reference_item_no(m: Dict[str, Any], ccapr_item_no: str) -> str:
    """
    Prefer explicit ``reference_item_no`` from the model. If the model put the *ERP* code in ``item_no``
    (different from the CCAPR line item), use that for dataframe lookup — common when keys misalign.
    """
    direct = _reference_item_no_from_cross_match_dict(m)
    if direct:
        return direct
    ccapr_key = normalized_item_key_from_input(ccapr_item_no or "")
    ai_raw = _dict_first(m, "item_no", "itemNo", "item_number", "ItemNo")
    if ai_raw is None or str(ai_raw).strip() == "":
        return ""
    ai_clean = clean_ccapr_item_no_input(str(ai_raw))
    if not ai_clean:
        return ""
    if ccapr_key and normalized_item_key_from_input(ai_clean) == ccapr_key:
        return ""
    return ai_clean

def _cross_match_apply_failure_hint(matches: List[Dict[str, Any]]) -> str:
    if not matches or not isinstance(matches[0], dict):
        return "matches empty or not objects"
    m0 = matches[0]
    uc = _cross_match_effective_unit_cost(m0)
    mf = _cross_ai_matched_flag(m0)
    parts: List[str] = []
    if uc is None:
        parts.append(
            "no_unit_cost_in_response_model_may_have_found_no_plausible_TSV_row"
            if mf is False
            else "no_numeric_unit_cost_in_response"
        )
    if not parts:
        parts.append("item_no_keys_still_mismatch_after_index_pairing")
    return "; ".join(parts)

def _cross_reference_item_no_is_ccapr_echo(ref: str, ccapr_item_no: str) -> bool:
    """True when ``reference_item_no`` is just the CCAPR / Section 2 code (not a distinct ERP material no.)."""
    if not ref or not ccapr_item_no:
        return False
    return normalized_item_key_from_input(ref) == normalized_item_key_from_input(ccapr_item_no)

