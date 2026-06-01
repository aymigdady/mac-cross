"""
Cross-company match pipeline (extracted from cost-control-app app.py).
"""
from __future__ import annotations

import difflib
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from cross_ai import AIService, external_ai_enabled
from bm25_erp_index_cache import get_or_build_erp_bm25_index
from comparison_engine import (
    _clean_numeric_series,
    COL_DESC,
    COL_ITEM_NO,
    COL_PO,
    COL_PO_DATE,
    COL_PROJECT,
    COL_QTY,
    COL_UNIT,
    COL_UNIT_COST,
    COL_VENDOR,
    clean_ccapr_item_no_input,
    normalized_item_key_from_input,
    nullable_str,
)
from session_store import DATA_STORE, get_redis_client_for_cache_use

logger = logging.getLogger(__name__)
_AI_SERVICE = AIService()

_CROSS_MATCH_BM25_SHORTLIST_MAX = 50
_CROSS_MATCH_RERANK_TOP_N = 10
_CROSS_MATCH_CANDIDATES_PER_LINE = int(os.environ.get("CCAPR_CROSS_CANDIDATES_PER_LINE", "6"))
_CROSS_MATCH_REMATCH_CANDIDATES_PER_LINE = int(os.environ.get("CCAPR_CROSS_REMATCH_CANDIDATES_PER_LINE", "8"))
_HYBRID_RERANK_LEX_WEIGHT = float(os.environ.get("CCAPR_HYBRID_RERANK_LEX_WEIGHT", "0.5"))
CROSS_CCAPR_ITEM_COL = "CCAPR Item No."
CROSS_ABSTAIN_COL = "_CCAPR_Abstain"
CROSS_CE_TOP_SCORE_COL = "_CCAPR_CE_TopScore"

def _normalize_new_po_source_tab(raw: Optional[str]) -> str:
    s = (raw or "").strip().upper()
    if s in ("IFAS", "MBL", "MSB"):
        return s
    return "MBL"

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


def _cross_match_reference_tsv(store: Dict[str, Any], tab_raw: Optional[str]) -> str:
    df = _cross_match_reference_work_df(store, tab_raw)
    if df is None or df.empty:
        return ""
    tsv = df.to_csv(sep="\t", index=False)
    max_chars = 120_000
    return tsv[:max_chars] if len(tsv) > max_chars else tsv

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

def _backfill_cross_search_confidence_from_df(
    rows: List[Dict[str, Any]],
    cross_df: Optional[pd.DataFrame],
) -> None:
    """When the model omits ``confidence``, derive an approximate % from description similarity to the picked ERP row."""
    if cross_df is None or cross_df.empty or COL_DESC not in cross_df.columns:
        return
    for r in rows:
        if r.get("cross_search_confidence_pct") is not None:
            continue
        if not r.get("has_history"):
            continue
        q = nullable_str(r.get("description") or "")
        if not q.strip():
            continue
        idx = _cross_match_reference_row_index_for_row(r, cross_df)
        if idx is None:
            continue
        qn = _normalize_cross_desc_for_match(q)
        rn = _normalize_cross_desc_for_match(str(cross_df.iloc[idx].get(COL_DESC) or ""))
        sim = _cross_desc_similarity_score(qn, rn)
        r["cross_search_confidence_pct"] = round(max(0.0, min(1.0, sim)) * 100.0, 1)

def _ensure_reference_cross_descriptions(
    rows: List[Dict[str, Any]],
    cross_df: Optional[pd.DataFrame],
) -> None:
    """
    Cross search UI: show the reference (cross-company) ERP line description, not Section 2 manual text.
    Only rows with ``has_history`` get a value; NEW lines keep manual ``description`` only.
    """
    if cross_df is None or cross_df.empty or COL_DESC not in cross_df.columns:
        return
    for r in rows:
        if not r.get("has_history"):
            continue
        idx = _cross_match_reference_row_index_for_row(r, cross_df)
        if idx is None:
            continue
        raw = cross_df.iloc[idx].get(COL_DESC)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        txt = nullable_str(str(raw))
        if txt:
            r["reference_cross_description"] = txt

def _cross_df_per_line_candidate_max(cross_df: Optional[pd.DataFrame]) -> int:
    """Largest number of candidate rows for any single CCAPR line (for Haiku prompt sizing)."""
    if cross_df is None or cross_df.empty or CROSS_CCAPR_ITEM_COL not in cross_df.columns:
        return 2
    try:
        g = cross_df.groupby(cross_df[CROSS_CCAPR_ITEM_COL].astype(str).str.strip()).size()
        return max(2, min(int(g.max()), 12))
    except Exception:
        return 2

def _rematch_lexical_best_among_tagged_candidates(
    rows: List[Dict[str, Any]],
    slim_items: List[Dict[str, Any]],
    cross_df: pd.DataFrame,
    ccapr_vendor: str,
) -> None:
    """
    After Haiku / fallback, force the benchmark to the **lexically** strongest ERP row among
    candidates tagged for that CCAPR line (max description similarity). Improves re-match when
    the model picks a suboptimal row from a widened shortlist.
    """
    if not rows or not slim_items or cross_df is None or cross_df.empty:
        return
    if CROSS_CCAPR_ITEM_COL not in cross_df.columns or COL_DESC not in cross_df.columns:
        return
    n = min(len(rows), len(slim_items))
    for j in range(n):
        r = rows[j]
        ccapr_no = clean_ccapr_item_no_input(slim_items[j].get("item_no") or "")
        if not ccapr_no:
            continue
        want = normalized_item_key_from_input(ccapr_no)
        q = nullable_str(
            slim_items[j].get("item_description")
            or slim_items[j].get("description")
            or r.get("description")
            or ""
        )
        qn = _normalize_cross_desc_for_match(q)
        if len(qn) < 2:
            continue
        best_i: Optional[int] = None
        best_s = -1.0
        for i in range(len(cross_df)):
            tag_raw = cross_df.iloc[i].get(CROSS_CCAPR_ITEM_COL)
            tag_key = normalized_item_key_from_input(clean_ccapr_item_no_input(tag_raw))
            if tag_key != want:
                continue
            uc = _parse_reference_unit_cost_scalar(cross_df.iloc[i].get(COL_UNIT_COST))
            if uc is None or not np.isfinite(float(uc)):
                continue
            rn = _normalize_cross_desc_for_match(str(cross_df.iloc[i].get(COL_DESC) or ""))
            s = _cross_desc_similarity_score(qn, rn)
            if s > best_s:
                best_s = s
                best_i = i
        if best_i is None:
            continue
        _apply_cross_benchmark_from_reference_row(
            r, cross_df, best_i, ccapr_vendor, q, po_default="Ref. re-match (lexical)"
        )
        r["cross_erp_description_match"] = True
        _set_cross_search_audit(
            r,
            provenance="rematch_lexical",
            lexical_pct=r.get("cross_search_confidence_pct"),
        )

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

def _reference_item_no_from_cross_df(cross_df: pd.DataFrame, description: str) -> str:
    """Resolve reference sheet Item No. using the same description row-picker as the local fallback."""
    if cross_df is None or cross_df.empty or COL_ITEM_NO not in cross_df.columns:
        return ""
    idx = _best_cross_reference_row_index(cross_df, description or "")
    if idx is None:
        return ""
    raw = cross_df.iloc[idx].get(COL_ITEM_NO)
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    return clean_ccapr_item_no_input(raw) or str(raw).strip()

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

def _first_cross_df_row_with_cost_for_item(cross_df: pd.DataFrame, ref_item: str) -> Optional[int]:
    """First row index matching ``ref_item`` whose ``COL_UNIT_COST`` parses to a finite number."""
    if not ref_item or COL_ITEM_NO not in cross_df.columns or COL_UNIT_COST not in cross_df.columns:
        return None
    want = normalized_item_key_from_input(ref_item)
    if not want:
        return None
    for i in range(len(cross_df)):
        raw = cross_df.iloc[i].get(COL_ITEM_NO)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        if normalized_item_key_from_input(raw) != want:
            continue
        uc = _parse_reference_unit_cost_scalar(cross_df.iloc[i].get(COL_UNIT_COST))
        if uc is not None and np.isfinite(float(uc)):
            return i
    return None

def _cross_df_resolve_row_index_for_match(
    cross_df: pd.DataFrame,
    m: Dict[str, Any],
    ccapr_description: str,
    *,
    ccapr_item_no: str = "",
) -> Optional[int]:
    """
    Map a model match object + CCAPR description to a row index in ``cross_df`` (same rules as a compare row:
    prefer ``reference_item_no`` from the match, disambiguate by description, else best description row).
    """
    lookup_q = _cross_match_lookup_query_for_df(m, ccapr_description)
    tmp: Dict[str, Any] = {"description": lookup_q}
    ref_item = _cross_match_inferred_reference_item_no(m, ccapr_item_no)
    if ref_item:
        tmp["reference_item_no"] = ref_item
    return _cross_match_reference_row_index_for_row(tmp, cross_df)

def _cross_match_enrich_pick_row_index(
    cross_df: pd.DataFrame,
    m: Dict[str, Any],
    ccapr_description: str,
    *,
    ccapr_item_no: str = "",
) -> Optional[int]:
    """
    Choose a reference row that has a usable unit cost: resolve match, then same-item fallback,
    then description similarity (standard and low-threshold rescue).
    """
    if COL_UNIT_COST not in cross_df.columns:
        return None
    lookup_q = _cross_match_lookup_query_for_df(m, ccapr_description)
    ref_item = _cross_match_inferred_reference_item_no(m, ccapr_item_no)

    def _uc_at(i: int) -> Optional[float]:
        v = _parse_reference_unit_cost_scalar(cross_df.iloc[i].get(COL_UNIT_COST))
        if v is None or not np.isfinite(float(v)):
            return None
        return float(v)

    idx = _cross_df_resolve_row_index_for_match(
        cross_df, m, ccapr_description, ccapr_item_no=ccapr_item_no
    )
    if idx is not None and _uc_at(idx) is not None:
        return idx
    if ref_item:
        alt = _first_cross_df_row_with_cost_for_item(cross_df, ref_item)
        if alt is not None:
            return alt
    if lookup_q:
        j = _best_cross_reference_row_index(cross_df, lookup_q, min_score=0.22)
        if j is not None and _uc_at(j) is not None:
            return j
        j2 = _best_cross_reference_row_index(cross_df, lookup_q, min_score=0.06)
        if j2 is not None and _uc_at(j2) is not None:
            return j2
    return None

def _cross_match_enrich_from_reference_df(
    m: Dict[str, Any],
    cross_df: Optional[pd.DataFrame],
    ccapr_description: str,
    *,
    ccapr_item_no: str = "",
) -> Dict[str, Any]:
    """
    When the model omits ``reference_unit_cost`` but identifies a line (or description can resolve one),
    copy unit cost from ``cross_df`` so apply / index-pairing can succeed. Fills ``reference_item_no`` when
    missing so downstream PO sync matches the cost row.
    """
    if not isinstance(m, dict) or cross_df is None or cross_df.empty:
        return m
    if COL_UNIT_COST not in cross_df.columns:
        return m
    if _cross_match_effective_unit_cost(m) is not None:
        return m
    ref_item = _cross_match_inferred_reference_item_no(m, ccapr_item_no)
    lookup_q = _cross_match_lookup_query_for_df(m, ccapr_description)
    if not ref_item and not lookup_q:
        return m
    idx = _cross_match_enrich_pick_row_index(
        cross_df, m, ccapr_description, ccapr_item_no=ccapr_item_no
    )
    if idx is None:
        return m
    uc = _parse_reference_unit_cost_scalar(cross_df.iloc[idx].get(COL_UNIT_COST))
    if uc is None or not np.isfinite(float(uc)):
        return m
    out = dict(m)
    out["reference_unit_cost"] = float(uc)
    h = cross_df.iloc[idx]
    if COL_ITEM_NO in cross_df.columns and not _reference_item_no_from_cross_match_dict(out):
        ref_no = clean_ccapr_item_no_input(h.get(COL_ITEM_NO))
        if ref_no:
            out["reference_item_no"] = ref_no
    if not nullable_str(_dict_first(out, "reference_vendor", "referenceVendor", "vendor")) and COL_VENDOR in cross_df.columns:
        v = nullable_str(h.get(COL_VENDOR))
        if v:
            out["reference_vendor"] = v
    if not nullable_str(_dict_first(out, "reference_unit", "referenceUnit", "unit")) and COL_UNIT in cross_df.columns:
        u = nullable_str(h.get(COL_UNIT))
        if u:
            out["reference_unit"] = u
    if not nullable_str(_dict_first(out, "reference_po_number", "referencePoNumber", "po_number", "poNumber")) and COL_PO in cross_df.columns:
        po = nullable_str(h.get(COL_PO))
        if po:
            out["reference_po_number"] = po
    if COL_PO_DATE in cross_df.columns:
        iso = _po_date_iso_from_cell(h.get(COL_PO_DATE))
        if iso and not nullable_str(_dict_first(out, "reference_po_date", "referencePoDate", "po_date")):
            out["reference_po_date"] = iso
    if not nullable_str(_dict_first(out, "reference_site", "referenceSite", "site", COL_PROJECT)) and COL_PROJECT in cross_df.columns:
        site = nullable_str(h.get(COL_PROJECT))
        if site:
            out["reference_site"] = site
    return out

def _set_cross_reference_item_no_on_row(
    r: Dict[str, Any],
    m: Dict[str, Any],
    cross_df: Optional[pd.DataFrame],
    *,
    section2_description: Optional[str] = None,
) -> None:
    """
    Persist the **target ERP** Item No. on the compare row for cross-search UI.

    Models sometimes echo the CCAPR / Section 2 code in ``reference_item_no`` or leave it blank
    while still returning a unit cost; resolve from ``cross_df`` using Section 2 manual text when needed.
    """
    ref = _reference_item_no_from_cross_match_dict(m)
    ccapr_no = nullable_str(r.get("item_no") or "")
    if ref and _cross_reference_item_no_is_ccapr_echo(ref, ccapr_no):
        ref = ""
    lookup_desc = (section2_description or "").strip() or nullable_str(r.get("description") or "")
    if not ref and cross_df is not None:
        q = lookup_desc
        if q.strip():
            ref = _reference_item_no_from_cross_df(cross_df, q)
    if ref and _cross_reference_item_no_is_ccapr_echo(ref, ccapr_no):
        ref = ""
    if not ref and cross_df is not None and not cross_df.empty:
        idx = _cross_match_enrich_pick_row_index(
            cross_df,
            m,
            lookup_desc,
            ccapr_item_no=ccapr_no,
        )
        if idx is not None and COL_ITEM_NO in cross_df.columns:
            raw = cross_df.iloc[idx].get(COL_ITEM_NO)
            if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                cand = clean_ccapr_item_no_input(raw) or str(raw).strip()
                if cand:
                    ref = cand
    if ref:
        r["reference_item_no"] = ref

def _backfill_cross_reference_item_nos(
    rows: List[Dict[str, Any]],
    slim_items: List[Dict[str, Any]],
    cross_df: Optional[pd.DataFrame],
) -> None:
    """
    After Haiku / reconcile, ensure ``reference_item_no`` is the **viewed company's** material code when
    the row has a cross benchmark but the reference slot is missing or still echoes the CCAPR Item No.
    """
    if not rows or cross_df is None or cross_df.empty or COL_ITEM_NO not in cross_df.columns:
        return
    tagged = CROSS_CCAPR_ITEM_COL in cross_df.columns
    n_slim = len(slim_items) if slim_items else 0
    for j, r in enumerate(rows):
        if not r.get("has_history"):
            continue
        ccapr_no = nullable_str(r.get("item_no") or "")
        cur = nullable_str(r.get("reference_item_no") or "")
        if cur and not _cross_reference_item_no_is_ccapr_echo(cur, ccapr_no):
            continue
        section2 = ""
        if j < n_slim and isinstance(slim_items[j], dict):
            section2 = nullable_str(
                slim_items[j].get("item_description") or slim_items[j].get("description") or ""
            )
        if not section2.strip():
            section2 = nullable_str(r.get("description") or "")
        q_sig = _cross_match_description_for_similarity(section2)
        want_tag = normalized_item_key_from_input(clean_ccapr_item_no_input(ccapr_no)) if tagged else ""
        bench_uc = r.get("lowest_hist_unit_cost")

        best_i: Optional[int] = None
        best_s = -1.0
        for i in range(len(cross_df)):
            if tagged:
                tag_raw = cross_df.iloc[i].get(CROSS_CCAPR_ITEM_COL)
                if normalized_item_key_from_input(clean_ccapr_item_no_input(tag_raw)) != want_tag:
                    continue
            uc = _parse_reference_unit_cost_scalar(cross_df.iloc[i].get(COL_UNIT_COST))
            if uc is None or not np.isfinite(float(uc)):
                continue
            s = 0.0
            if q_sig and COL_DESC in cross_df.columns:
                qn = _normalize_cross_desc_for_match(q_sig)
                rn = _normalize_cross_desc_for_match(str(cross_df.iloc[i].get(COL_DESC) or ""))
                s = _cross_desc_similarity_score(qn, rn)
            if bench_uc is not None:
                try:
                    if abs(float(uc) - float(bench_uc)) <= 1e-4:
                        s += 0.25
                except (TypeError, ValueError):
                    pass
            if s > best_s:
                best_s = s
                best_i = i

        if best_i is None and not tagged and q_sig:
            best_i = _best_cross_reference_row_index(cross_df, section2, min_score=0.06)
        if best_i is None and tagged:
            best_i = _cross_match_enrich_pick_row_index(cross_df, {}, section2, ccapr_item_no=ccapr_no)

        if best_i is None:
            continue
        raw = cross_df.iloc[best_i].get(COL_ITEM_NO)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        ref_out = clean_ccapr_item_no_input(raw) or str(raw).strip()
        if ref_out:
            r["reference_item_no"] = ref_out

def _cross_match_reference_row_index_for_row(
    r: Dict[str, Any],
    cross_df: Optional[pd.DataFrame],
) -> Optional[int]:
    """
    Row index in ``cross_df`` for the same reference line as item no / description match.
    Prefer ``reference_item_no`` when set; if several ERP lines share that item, pick by description similarity.
    """
    if cross_df is None or cross_df.empty:
        return None
    ref_item = nullable_str(r.get("reference_item_no") or "")
    if ref_item and COL_ITEM_NO in cross_df.columns:
        want = normalized_item_key_from_input(ref_item)
        if want:
            hits: List[int] = []
            for i in range(len(cross_df)):
                raw = cross_df.iloc[i].get(COL_ITEM_NO)
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    continue
                if normalized_item_key_from_input(raw) == want:
                    hits.append(i)
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1 and COL_DESC in cross_df.columns:
                q = nullable_str(r.get("description") or "")
                if not q.strip():
                    return hits[0]
                qn = _normalize_cross_desc_for_match(q)
                desc_series = cross_df[COL_DESC].fillna("").astype(str)
                best_i: Optional[int] = None
                best_s = 0.0
                for i in hits:
                    rn = _normalize_cross_desc_for_match(desc_series.iloc[i])
                    s = _cross_desc_similarity_score(qn, rn)
                    if s > best_s:
                        best_s = s
                        best_i = i
                return best_i if best_i is not None else hits[0]
            if len(hits) > 1:
                return hits[0]
    q = nullable_str(r.get("description") or "")
    if q.strip():
        return _best_cross_reference_row_index(cross_df, q)
    return None

def _sync_cross_reference_po_date_from_matched_row(
    r: Dict[str, Any],
    cross_df: Optional[pd.DataFrame],
    *,
    vendor_key: str,
) -> None:
    """Set benchmark PO date from ``COL_PO_DATE`` on the matched reference row (not model text)."""
    idx = _cross_match_reference_row_index_for_row(r, cross_df)
    if idx is None or cross_df is None or COL_PO_DATE not in cross_df.columns:
        return
    h = cross_df.iloc[idx]
    iso = _po_date_iso_from_cell(h.get(COL_PO_DATE))
    if not iso:
        return
    r["lowest_benchmark_po_date"] = iso
    ref_vendor = nullable_str(h.get(COL_VENDOR)) if COL_VENDOR in cross_df.columns else ""
    if vendor_key and ref_vendor and _vendor_keys_match(_normalize_vendor_key_local(ref_vendor), vendor_key):
        r["same_vendor_benchmark_po_date"] = iso

def _apply_one_cross_match_to_row(
    r: Dict[str, Any],
    m: Dict[str, Any],
    *,
    vendor_key: str,
) -> bool:
    if not isinstance(m, dict):
        return False
    uc, ref_vendor, ref_unit, ref_po, ref_date, ref_site = _reference_fields_from_cross_match(m)
    if uc is None:
        return False
    # Apply whenever we have a numeric cost from the model (copied from TSV per prompt). Models often set
    # matched=false while still returning a candidate cost; strict matched-only would drop valid benchmarks.
    mf = _cross_ai_matched_flag(m)
    ref_default_po = "Ref. desc match (review)" if mf is False else "Ref. desc match"

    ref_vendor = ref_vendor or nullable_str(_dict_first(m, "reference_vendor", "referenceVendor", "vendor"))
    ref_unit = ref_unit or nullable_str(_dict_first(m, "reference_unit", "referenceUnit", "unit"))
    ref_po = ref_po or nullable_str(_dict_first(m, "reference_po_number", "referencePoNumber", "po_number", "poNumber"))
    ref_date = ref_date or nullable_str(_dict_first(m, "reference_po_date", "referencePoDate", "po_date"))
    if not ref_site:
        ref_site = nullable_str(
            _dict_first(m, "reference_site", "referenceSite", "site")
            or _dict_first(m, "matched_description_snippet", "matchedDescriptionSnippet")
        )

    r["has_history"] = True
    r["lowest_hist_unit_cost"] = uc
    r["lowest_benchmark_vendor"] = ref_vendor
    r["lowest_benchmark_po_number"] = ref_po or ref_default_po
    r["lowest_benchmark_po_date"] = ref_date or None
    r["lowest_benchmark_hist_unit"] = ref_unit
    r["lowest_benchmark_site_name"] = ref_site

    if vendor_key and _vendor_keys_match(_normalize_vendor_key_local(ref_vendor), vendor_key):
        r["same_vendor_latest_hist_unit_cost"] = uc
        r["same_vendor_benchmark_vendor"] = ref_vendor
        r["same_vendor_benchmark_po_number"] = ref_po
        r["same_vendor_benchmark_po_date"] = ref_date or None
        r["same_vendor_benchmark_site_name"] = ref_site
        r["same_vendor_hist_unit"] = ref_unit
    cpct = _cross_match_confidence_pct_from_dict(m)
    if cpct is not None:
        r["cross_search_confidence_pct"] = cpct
        r["cross_search_ai_confidence_pct"] = float(round(min(100.0, max(0.0, float(cpct))), 1))
    r["cross_erp_description_match"] = True
    return True

def _apply_cross_description_matches(
    rows: List[Dict[str, Any]],
    *,
    matches: List[Dict[str, Any]],
    ccapr_vendor: str,
    cross_df: Optional[pd.DataFrame] = None,
    section2_desc_by_item_key: Optional[Dict[str, str]] = None,
) -> int:
    if not rows or not matches:
        return 0
    by_item: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        if not isinstance(m, dict):
            continue
        item_raw = _dict_first(m, "item_no", "itemNo", "item_number", "ItemNo")
        k = normalized_item_key_from_input(item_raw or "")
        if not k:
            continue
        by_item[k] = m

    vendor_key = _normalize_vendor_key_local(ccapr_vendor)
    applied = 0
    for r in rows:
        k = normalized_item_key_from_input(r.get("item_no") or "")
        m = by_item.get(k)
        if not m:
            continue
        m_use = _cross_match_enrich_from_reference_df(
            m,
            cross_df,
            nullable_str(r.get("description") or ""),
            ccapr_item_no=nullable_str(r.get("item_no") or ""),
        )
        if _apply_one_cross_match_to_row(r, m_use, vendor_key=vendor_key):
            sec2 = (section2_desc_by_item_key or {}).get(k) if section2_desc_by_item_key else None
            _set_cross_reference_item_no_on_row(r, m_use, cross_df, section2_description=sec2)
            _sync_cross_reference_po_date_from_matched_row(r, cross_df, vendor_key=vendor_key)
            ai_pct_val = _cross_match_confidence_pct_from_dict(m_use)
            ai_picked = nullable_str(
                _dict_first(m_use, "reference_item_no", "referenceItemNo", "ref_item_no")
            )
            # Phase 3c: pull the per-line cross-encoder score + abstain flag out
            # of the matching cross_df row so the UI can surface "no confident
            # match" without inspecting the audit object.
            ce_score, abstain_flag = _lookup_cross_encoder_state_for_row(cross_df, m_use)
            _set_cross_search_audit(
                r,
                provenance="ai_tagged",
                ai_pct=ai_pct_val,
                ai_picked_item_no=ai_picked,
                cross_encoder_score=ce_score,
                abstain=abstain_flag,
            )
            applied += 1

    # Model often echoes the *reference sheet* item code in `item_no` instead of the CCAPR line.
    # Pair by index only when array lengths match exactly and every match carries a unit cost — otherwise
    # partial pairing misaligns CCAPR lines to the wrong AI objects.
    if applied == 0 and rows and matches:
        if len(rows) != len(matches):
            logger.warning(
                "Cross AI: index-pairing skipped — %s CCAPR rows vs %s matches (counts must match exactly for positional apply)",
                len(rows),
                len(matches),
            )
        else:
            enriched = [
                _cross_match_enrich_from_reference_df(
                    matches[i],
                    cross_df,
                    nullable_str(rows[i].get("description") or ""),
                    ccapr_item_no=nullable_str(rows[i].get("item_no") or ""),
                )
                for i in range(len(rows))
            ]
            missing_cost = sum(1 for em in enriched if _cross_match_effective_unit_cost(em) is None)
            if missing_cost:
                logger.warning(
                    "Cross AI: index-pairing skipped — %s/%s matches still missing unit cost after ERP lookup; falling back to local similarity only",
                    missing_cost,
                    len(matches),
                )
            else:
                mismatch_pairs: List[Tuple[str, str]] = []
                for i in range(len(rows)):
                    ai_item = normalized_item_key_from_input(
                        _dict_first(matches[i], "item_no", "itemNo", "item_number", "ItemNo") or ""
                    )
                    ccapr_item = normalized_item_key_from_input(rows[i].get("item_no") or "")
                    if ai_item and ccapr_item and ai_item != ccapr_item:
                        mismatch_pairs.append((ccapr_item, ai_item))
                if mismatch_pairs:
                    logger.warning(
                        "Cross AI: positional pairing applied but %s row(s) have item_no mismatch "
                        "(AI echoed reference keys, not CCAPR keys): %s",
                        len(mismatch_pairs),
                        mismatch_pairs[:5],
                    )
                pos = 0
                for i in range(len(rows)):
                    if _apply_one_cross_match_to_row(rows[i], enriched[i], vendor_key=vendor_key):
                        ik = normalized_item_key_from_input(rows[i].get("item_no") or "")
                        sec2 = (section2_desc_by_item_key or {}).get(ik) if section2_desc_by_item_key else None
                        _set_cross_reference_item_no_on_row(
                            rows[i], enriched[i], cross_df, section2_description=sec2
                        )
                        _sync_cross_reference_po_date_from_matched_row(
                            rows[i], cross_df, vendor_key=vendor_key
                        )
                        ai_pct_val = _cross_match_confidence_pct_from_dict(enriched[i])
                        ai_picked = nullable_str(
                            _dict_first(enriched[i], "reference_item_no", "referenceItemNo", "ref_item_no")
                        )
                        _set_cross_search_audit(
                            rows[i],
                            provenance="ai_positional_pairing",
                            ai_pct=ai_pct_val,
                            ai_picked_item_no=ai_picked,
                        )
                        pos += 1
                if pos:
                    applied = pos
                    logger.info("Cross AI: applied %s match(es) by row order", pos)
    return applied

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


def _strip_col(c: Any) -> str:
    return str(c).strip()


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

def _cross_match_expand_query(query: str) -> List[str]:
    """Phase 4A — domain-aware query expansion.

    Returns ``[query, paraphrase1, paraphrase2, ...]`` (always non-empty when
    the input is non-empty, with the original query as the first element).

    Each call is a Redis cache lookup; on a miss, one Haiku call. Results are
    cached forever per canonical-description hash so the steady-state cost is
    a single Redis ``GET`` (~0.3 ms).

    Hard fail-safe: any failure (Redis down, no API key, LLM error, parse
    error) returns ``[query]`` so the rest of the pipeline runs unchanged.
    The caller passing ``[query]`` to the shortlist builder is exactly the
    pre-Phase-4 baseline behaviour, so this helper can never *reduce* recall.
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        from query_expansion import expand_query, is_query_expansion_enabled
    except Exception:
        return [q]
    if not is_query_expansion_enabled():
        return [q]
    try:
        from session_store import get_redis_client_for_cache_use

        redis_client = get_redis_client_for_cache_use()
    except Exception:
        redis_client = None
    try:
        out = expand_query(q, redis_client=redis_client)
    except Exception:
        return [q]
    return out or [q]

def _cross_match_bm25_shortlist_index_labels(
    store: Dict[str, Any],
    tab: str,
    hist: pd.DataFrame,
    work: pd.DataFrame,
    query_texts: List[str],
    *,
    per_query_top_k: int = 50,
    max_total_rows: int = 500,
) -> Optional[List[Any]]:
    """
    BM25 shortlist over the full ERP tab: top ``per_query_top_k`` rows per non-empty query,
    merged by max score, capped at ``max_total_rows``. Returns index labels valid for ``work.loc``.
    """
    fp = nullable_str(store.get("erp_file_sha256"))
    if not fp:
        return None
    nonempty_queries = [nullable_str(q) for q in query_texts if nullable_str(q)]
    if not nonempty_queries:
        return None
    try:
        bm25_ix = get_or_build_erp_bm25_index(fp, tab, hist)
    except Exception:
        return None
    if bm25_ix.bm25 is None:
        return None
    work_index = work.index
    aggregate_max: Dict[Any, float] = {}
    for q in nonempty_queries:
        toks = tokenize_cross_desc_for_bm25(q)
        if not toks:
            continue
        scores = bm25_ix.scores_for_query_tokens(toks)
        if scores.size == 0:
            continue
        top_pos = np.argsort(-scores)[:per_query_top_k]
        for pos in top_pos:
            pos_i = int(pos)
            if pos_i < 0 or pos_i >= len(bm25_ix.row_index_labels):
                continue
            label = bm25_ix.row_index_labels[pos_i]
            if label not in work_index:
                continue
            s = float(scores[pos_i])
            prev = aggregate_max.get(label)
            if prev is None or s > prev:
                aggregate_max[label] = s
    if not aggregate_max:
        return None
    ordered = sorted(aggregate_max.keys(), key=lambda lb: aggregate_max[lb], reverse=True)
    ordered = [lb for lb in ordered if lb in work_index]
    return ordered[:max_total_rows]

def _cross_match_hybrid_shortlist_index_labels(
    store: Dict[str, Any],
    tab: str,
    hist: pd.DataFrame,
    work: pd.DataFrame,
    query_texts: List[str],
    *,
    bm25_per_query_top_k: int = 30,
    embedding_per_query_top_k: int = 30,
    max_total_rows: int = 50,
) -> Optional[List[Any]]:
    """
    Phase-2 hybrid shortlist: BM25 top-N ∪ embedding top-N, fused via reciprocal-rank.

    Falls back transparently to the BM25-only path when the company has no
    embedding index, or when the global / per-company embedding flag is off,
    or when any embedding step fails. This preserves the **rollback** acceptance
    criterion: ``CCAPR_ENABLE_EMBEDDINGS=0`` returns identical results to Phase 1.
    """
    bm25_labels = _cross_match_bm25_shortlist_index_labels(
        store,
        tab,
        hist,
        work,
        query_texts,
        per_query_top_k=bm25_per_query_top_k,
        max_total_rows=max_total_rows * 5,  # leave headroom for the union/dedupe
    ) or []

    if not _embeddings_hybrid_available_for_tab(tab):
        return bm25_labels[:max_total_rows] if bm25_labels else None

    nonempty_queries = [nullable_str(q) for q in query_texts if nullable_str(q)]
    if not nonempty_queries:
        return bm25_labels[:max_total_rows] if bm25_labels else None

    try:
        from embeddings.hybrid import (
            embedding_top_labels_for_query,
            hybrid_shortlist_labels,
        )

        from session_store import get_redis_client_for_cache_use

        redis_client = get_redis_client_for_cache_use()
        emb_label_scores: List[Tuple[Any, float]] = []
        for q in nonempty_queries:
            emb_label_scores.extend(
                embedding_top_labels_for_query(
                    tab,
                    work,
                    q,
                    top_k=embedding_per_query_top_k,
                    redis_client=redis_client,
                )
            )
        fused = hybrid_shortlist_labels(
            tab,
            work,
            bm25_labels,
            emb_label_scores,
            union_max=max_total_rows,
        )
        if fused:
            return fused
    except Exception:
        logging.getLogger(__name__).exception(
            "Hybrid embedding shortlist failed for tab %s; falling back to BM25 only", tab
        )
    return bm25_labels[:max_total_rows] if bm25_labels else None

def _embeddings_hybrid_available_for_tab(tab: str) -> bool:
    """Cheap pre-check — never imports the embedder unless the flags say yes."""
    try:
        from embeddings.hybrid import (
            get_company_store,
            is_embeddings_enabled_for_company,
        )
    except Exception:
        return False
    if not is_embeddings_enabled_for_company(tab):
        return False
    store = get_company_store(tab)
    return store is not None and store.size > 0

def _cross_match_candidates_per_line() -> int:
    explicit = os.environ.get("CCAPR_CROSS_CANDIDATES_PER_LINE")
    if explicit:
        try:
            return max(1, int(explicit))
        except ValueError:
            pass
    try:
        from cross_encoder.reranker import is_cross_encoder_enabled
        if is_cross_encoder_enabled():
            return 3
    except Exception:
        pass
    return 6

def _cross_exact_code_min_lex() -> float:
    try:
        v = float((os.environ.get("CCAPR_CROSS_EXACT_CODE_MIN_LEX") or "0.6").strip())
    except (TypeError, ValueError):
        return 0.6
    if not (0.0 <= v <= 1.0):
        return 0.6
    return v

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

def _cross_match_work_df_two_per_line(
    store: Dict[str, Any],
    tab_raw: Optional[str],
    line_items: List[Dict[str, Any]],
    *,
    candidates_per_line: Optional[int] = None,
    exclude_erp_item_nos_by_ccapr: Optional[Dict[str, Set[str]]] = None,
) -> Optional[pd.DataFrame]:
    """
    For each CCAPR line: BM25 shortlist for that line's description only, lexical rerank, keep up to
    ``candidates_per_line`` ERP rows. Concatenate with a leading ``CROSS_CCAPR_ITEM_COL``
    so Haiku restricts choices per line.

    When ``exclude_erp_item_nos_by_ccapr`` maps a normalized CCAPR item key to normalized ERP item keys,
    those ERP rows are dropped from the shortlist when possible (re-match: try another historical line).
    """
    # Phase 3c: when CE is on, default to 3 candidates per line (a sharper
    # rerank means the LLM gets a cleaner, smaller choice set). When CE is off,
    # fall back to the legacy 6.
    cap = max(
        1,
        int(candidates_per_line) if candidates_per_line is not None else _cross_match_candidates_per_line(),
    )
    prepared = _cross_match_prepare_work_df(store, tab_raw)
    if prepared is None:
        return None
    work, hist, tab = prepared
    fp = nullable_str(store.get("erp_file_sha256"))
    chunks: List[pd.DataFrame] = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        ccapr_no = clean_ccapr_item_no_input(item.get("item_no"))
        if not ccapr_no:
            continue
        ccapr_key = normalized_item_key_from_input(ccapr_no)
        excluded_norms: Set[str] = set()
        if exclude_erp_item_nos_by_ccapr:
            excluded_norms = set(exclude_erp_item_nos_by_ccapr.get(ccapr_key, set()))
        q = nullable_str(item.get("item_description") or item.get("description"))
        labels: Optional[List[Any]] = None
        # Phase 4A — query expansion. Domain-aware paraphrases (Haiku-cached)
        # widen the BM25 ∪ embedding shortlist to recover synonym /
        # abbreviation / paraphrase matches that pure lexical+vector lookup
        # against the original query misses. The first element is always the
        # original ``q`` so expansion can never *reduce* recall vs the
        # pre-Phase-4 baseline. Falls back transparently when the LLM is
        # unavailable / the env flag is off. Defined at outer scope so the
        # downstream rerank can see the expanded set even if the shortlist
        # call later fails for some reason.
        query_texts: List[str] = [q] if q else []
        if q and fp:
            query_texts = _cross_match_expand_query(q)
            try:
                labels = _cross_match_hybrid_shortlist_index_labels(
                    store,
                    tab,
                    hist,
                    work,
                    query_texts,
                    bm25_per_query_top_k=30,
                    embedding_per_query_top_k=30,
                    max_total_rows=_CROSS_MATCH_BM25_SHORTLIST_MAX,
                )
            except Exception:
                labels = None
        if labels:
            # Phase 3a: hard-filter clearly mismatched candidates and capture
            # soft penalties for the remaining ones. This shrinks the shortlist
            # *before* the rerank and weights down low-conf mismatches that
            # would otherwise drift into the candidate window.
            kept_labels, penalty_by_label, _filter_reasons = (
                _apply_attribute_filter_to_shortlist(work, labels, q or "")
            )
            if not kept_labels:
                # Edge: the filter dropped *every* candidate. Surfacing nothing
                # would tell the user "no match" — but a hard-mismatch on every
                # candidate is a measurement signal, not an intent signal. Fall
                # back to the unfiltered shortlist so the LLM still sees a context.
                kept_labels = labels
                penalty_by_label = {}
            w50 = work.loc[kept_labels].copy()
        else:
            w50 = work.head(min(_CROSS_MATCH_BM25_SHORTLIST_MAX, len(work))).copy()
            penalty_by_label = {}
        wide_pool = bool(excluded_norms)
        pool_n = min(
            len(w50),
            max(50, cap * 10) if wide_pool else max(12, cap * 5),
        )
        # Phase 3b: try the cross-encoder rerank first (semantically-aware
        # joint scoring). Falls back to the Phase 2 hybrid reranker (which
        # itself falls back to lexical-only) if the CE is disabled / fails.
        # Phase 3a soft penalties are passed through both code paths so
        # low-conf size/voltage/pack mismatches drop out of the top.
        # Phase 3c: capture the abstain state so we can surface it on each row.
        abstain_state: Dict[str, Any] = {"score": None, "abstain": False}

        def _on_abstain(score: float, top_label: Any, top_text: str, abstain: bool) -> None:
            abstain_state["score"] = score
            abstain_state["abstain"] = abstain

        # Phase 3 inv. — selective cross-encoder.
        # Always compute the hybrid (lex+emb) result first; it's the cheap
        # baseline (~50 ms). If the hybrid top-1 score AND its margin to top-2
        # are both above thresholds, the answer is unambiguous and CE adds no
        # value — skip the ~2-4s CE call entirely. The diagnostic showed CE
        # only "wins" on ambiguous shortlists; it actively *hurt* on confident
        # hybrid hits (1.8% real regression rate). This guard preserves the
        # CE precision win without paying its latency on the easy 70-80% of
        # queries that hybrid already nails.
        hybrid_score: Dict[str, float] = {}
        # Phase 4A: rerank against the *expanded* query set so candidates
        # pulled in by paraphrases get scored against those paraphrases too.
        # ``query_texts`` already contains [original, paraphrase1, ...] when QE
        # ran, or just [original] when it didn't.
        w_hybrid = _cross_match_hybrid_rerank_shortlist(
            w50,
            query_texts,
            tab,
            top_n=pool_n,
            penalty_by_label=penalty_by_label,
            out_top_score=hybrid_score,
        )
        ce_skip_top1 = float(os.environ.get("CCAPR_CE_SKIP_HYBRID_TOP1", "0.85"))
        ce_skip_margin = float(os.environ.get("CCAPR_CE_SKIP_HYBRID_MARGIN", "0.20"))
        hyb_top1 = float(hybrid_score.get("top1", 0.0))
        hyb_margin = float(hybrid_score.get("margin", 0.0))
        if hyb_top1 >= ce_skip_top1 and hyb_margin >= ce_skip_margin:
            # Confident hybrid result — keep it, skip CE.
            w_ranked = w_hybrid
        else:
            w_ranked = _cross_match_cross_encoder_rerank(
                w50,
                [q] if q else [],
                top_n=pool_n,
                penalty_by_label=penalty_by_label,
                abstain_callback=_on_abstain,
            )
            if w_ranked is None:
                # CE unavailable — fall back to the hybrid result we already computed.
                w_ranked = w_hybrid
        if w_ranked.empty or COL_ITEM_NO not in w_ranked.columns:
            continue
        key_series = w_ranked[COL_ITEM_NO].map(
            lambda v: normalized_item_key_from_input(clean_ccapr_item_no_input(v))
        )
        if excluded_norms:
            w_filtered = w_ranked.loc[~key_series.isin(excluded_norms)].copy()
            if w_filtered.empty:
                w_filtered = w_ranked.copy()
        else:
            w_filtered = w_ranked.copy()
        w_pick = w_filtered.head(cap).copy()
        if w_pick.empty:
            continue
        if CROSS_CCAPR_ITEM_COL not in w_pick.columns:
            w_pick.insert(0, CROSS_CCAPR_ITEM_COL, ccapr_no)
        # Phase 3c: tag every row in this CCAPR line's chunk with the abstain
        # state and the cross-encoder top-1 score. Downstream UI can hide the
        # selection if abstain=True; downstream LLM prompt can short-circuit to
        # "no confident match" when the score is below threshold for all rows.
        if abstain_state.get("score") is not None:
            w_pick[CROSS_CE_TOP_SCORE_COL] = float(abstain_state["score"])
            w_pick[CROSS_ABSTAIN_COL] = bool(abstain_state["abstain"])
        chunks.append(w_pick)
    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True)

def cross_match_pipeline_diagnose(
    store: Dict[str, Any],
    tab_raw: Optional[str],
    bm25_query_texts: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Golden-set / ops: surface BM25 shortlist item numbers (≤50), reranked top-10 item numbers,
    and final pipeline output item numbers (same as ``_cross_match_reference_work_df``).

    If the BM25 path is skipped (no fingerprint / index / queries), ``legacy_head_500_fallback`` is True
    and ``bm25_shortlist_item_nos`` is empty.
    """
    prepared = _cross_match_prepare_work_df(store, tab_raw)
    if prepared is None:
        return None
    work, hist, tab = prepared
    out: Dict[str, Any] = {
        "tab": tab,
        "legacy_head_500_fallback": False,
        "bm25_shortlist_item_nos": [],
        "rerank_top_item_nos": [],
        "pipeline_output_item_nos": [],
    }
    labels: Optional[List[Any]] = None
    if bm25_query_texts is not None:
        labels = _cross_match_hybrid_shortlist_index_labels(
            store,
            tab,
            hist,
            work,
            list(bm25_query_texts or []),
            bm25_per_query_top_k=30,
            embedding_per_query_top_k=30,
            max_total_rows=_CROSS_MATCH_BM25_SHORTLIST_MAX,
        )
    if labels:
        w50 = work.loc[labels].copy()
        # Field name kept for backward-compat; with embeddings on this is the
        # hybrid (BM25 ∪ embedding) shortlist.
        out["bm25_shortlist_item_nos"] = w50[COL_ITEM_NO].astype(str).str.strip().tolist()
        out["shortlist_kind"] = (
            "hybrid_bm25_emb"
            if _embeddings_hybrid_available_for_tab(tab)
            else "bm25_only"
        )
        w10 = _cross_match_rerank_shortlist_by_similarity(
            w50,
            list(bm25_query_texts or []),
            top_n=_CROSS_MATCH_RERANK_TOP_N,
        )
        out["rerank_top_item_nos"] = w10[COL_ITEM_NO].astype(str).str.strip().tolist()
    else:
        out["legacy_head_500_fallback"] = True
        legacy_cap = 500
        w = work.head(legacy_cap).copy() if len(work) > legacy_cap else work.copy()
        out["rerank_top_item_nos"] = w[COL_ITEM_NO].astype(str).str.strip().tolist()[:_CROSS_MATCH_RERANK_TOP_N]

    pipe = _cross_match_reference_work_df(store, tab_raw, bm25_query_texts=bm25_query_texts)
    if pipe is not None and not pipe.empty and COL_ITEM_NO in pipe.columns:
        out["pipeline_output_item_nos"] = pipe[COL_ITEM_NO].astype(str).str.strip().tolist()
    return out

def _cross_match_reference_work_df(
    store: Dict[str, Any],
    tab_raw: Optional[str],
    *,
    bm25_query_texts: Optional[List[str]] = None,
    bm25_line_items: Optional[List[Dict[str, Any]]] = None,
    line_candidates_per_line: Optional[int] = None,
    exclude_erp_item_nos_by_ccapr: Optional[Dict[str, Set[str]]] = None,
) -> Optional[pd.DataFrame]:
    """Same rows/columns as the cross-match TSV, for AI text and for local description fallback.

    When ``COL_PO_DATE`` is present, rows are sorted by parsed PO date **newest first** before
    applying the row cap (legacy) or BM25 shortlist.

    If ``bm25_line_items`` is set (cross-company Compare): ERP candidates per CCAPR line (default **two**),
    BM25+r rerank **for that line only**, with a leading ``CROSS_CCAPR_ITEM_COL`` tagging the line.

    Else if ``bm25_query_texts`` is set (tests / diagnostics): merged BM25 top-50 across all queries,
    then rerank to **top 10** (legacy batch behaviour).

    Otherwise: first **500** rows after PO-date sort.
    """
    if bm25_line_items:
        cap = line_candidates_per_line if line_candidates_per_line is not None else _cross_match_candidates_per_line()
        return _cross_match_work_df_two_per_line(
            store,
            tab_raw,
            bm25_line_items,
            candidates_per_line=cap,
            exclude_erp_item_nos_by_ccapr=exclude_erp_item_nos_by_ccapr,
        )

    prepared = _cross_match_prepare_work_df(store, tab_raw)
    if prepared is None:
        return None
    work, hist, tab = prepared
    legacy_cap = 500

    use_bm25 = bm25_query_texts is not None
    labels: Optional[List[Any]] = None
    if use_bm25:
        labels = _cross_match_hybrid_shortlist_index_labels(
            store,
            tab,
            hist,
            work,
            list(bm25_query_texts or []),
            bm25_per_query_top_k=30,
            embedding_per_query_top_k=30,
            max_total_rows=_CROSS_MATCH_BM25_SHORTLIST_MAX,
        )
    if labels:
        work = work.loc[labels].copy()
        work = _cross_match_rerank_shortlist_by_similarity(
            work,
            list(bm25_query_texts or []),
            top_n=_CROSS_MATCH_RERANK_TOP_N,
        )
    else:
        if len(work) > legacy_cap:
            work = work.head(legacy_cap).copy()
        else:
            work = work.copy()
    return work.reset_index(drop=True)


_CROSS_DESC_SKU_HYPHEN_RE = re.compile(r"(?<=[a-z0-9])-(?=[a-z0-9])")
_CROSS_DESC_DIGIT_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s+(mm|cm|m|km|kg|g|mg|ton|tons|in|inch|inches|ft|feet|sqm|cbm|pcs|nos|no|set|sets|pkt|box|boxes|ltr|liter|liters|gal|amp|amps|volt|volts|watt|watts|hp|kva|kw|psi|bar)\b",
    re.IGNORECASE,
)
_CROSS_DESC_LIGHT_STEM_RE = re.compile(r"(?:ies|sses|ses|ing|ed|s)$")
_CROSS_DESC_BM25_SPLIT_PATTERN = re.compile(r"[^\w]+")
_CROSS_DESC_NON_WORD_SPLIT = _CROSS_DESC_BM25_SPLIT_PATTERN


def _normalize_cross_desc_for_match(raw: str) -> str:
    """
    Step **1** of the cross-match / BM25 tokenizer contract (see module note below ``_CROSS_DESC_BM25_SPLIT_PATTERN``).

    Normalizes raw ERP or CCAPR description text: NFKC, collapse whitespace, strip, ASCII-lowercase
    for matching. Then performs two structure-preserving fusions before the splitter sees it:

    - **SKU hyphens**: ``pd-100039641`` → ``pd100039641`` (one token instead of two).
    - **Digit + unit**: ``12 mm`` → ``12mm`` (matches the way ERP rows usually write specs).

    Output is the string fed to the delimiter split in step 2 — do not apply a different normalization for BM25.
    """
    s = unicodedata.normalize("NFKC", raw or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    if not s:
        return s
    # Fuse digit + unit (`12 mm` -> `12mm`). Run twice because a token like `12 mm pipe`
    # may have two unit candidates after the first pass.
    for _ in range(2):
        s_new = _CROSS_DESC_DIGIT_UNIT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}", s)
        if s_new == s:
            break
        s = s_new
    # Fuse SKU hyphens between alphanumeric runs (preserves codes through ``[^\w]+`` split).
    s = _CROSS_DESC_SKU_HYPHEN_RE.sub("", s)
    return s

def _cross_desc_apply_light_stem(token: str) -> str:
    """
    Strip a small set of English suffixes from tokens long enough that the result is still informative.

    Conservative on purpose:
    - Only English-style suffixes (Arabic and other scripts left intact).
    - Only when token length >= 5 *after* stripping (so ``ed``, ``id`` are not wrecked into ``''``).
    - Pure-digit tokens are left alone (so SKUs like ``100039641`` survive).
    """
    if not token or not token.isascii() or token.isdigit():
        return token
    if len(token) < 6:
        return token
    if not re.search(r"[a-z]", token):
        return token
    stripped = _CROSS_DESC_LIGHT_STEM_RE.sub("", token)
    if len(stripped) >= 4:
        return stripped
    return token

def _cross_desc_token_list_from_normalized(norm: str) -> List[str]:
    """Step **2** of the contract: split *already-normalized* text. ``norm`` must come only from ``_normalize_cross_desc_for_match``."""
    if not norm:
        return []
    return [
        _cross_desc_apply_light_stem(t)
        for t in _CROSS_DESC_BM25_SPLIT_PATTERN.split(norm)
        if t
    ]

def tokenize_cross_desc_for_bm25(raw: str) -> List[str]:
    """
    Canonical tokenizer for BM25 **and** any other lexical stage that must stay aligned with cross-match.

    Use this for both corpus documents (ERP lines) and queries (CCAPR / snippet text) at index time and query time.
    """
    return _cross_desc_token_list_from_normalized(_normalize_cross_desc_for_match(raw))

def _cross_desc_tokens(norm: str, *, min_len: int) -> Set[str]:
    """Token set with minimum length; ``norm`` must be from ``_normalize_cross_desc_for_match`` only."""
    return {t for t in _cross_desc_token_list_from_normalized(norm) if len(t) >= min_len}

def _cross_desc_similarity_score(query_norm: str, row_norm: str) -> float:
    """
    Blend sequence similarity with token Jaccard, then boost when all longer query tokens appear
    in the candidate (keyword coverage). Never below ``SequenceMatcher`` ratio when token sets exist,
    so near-identical strings are not penalized by sparse Jaccard.
    """
    if len(query_norm) < 2 or len(row_norm) < 2:
        return 0.0
    seq = difflib.SequenceMatcher(None, query_norm, row_norm).ratio()
    tq = _cross_desc_tokens(query_norm, min_len=3)
    td = _cross_desc_tokens(row_norm, min_len=3)
    if not tq or not td:
        return seq
    jacc = len(tq & td) / max(1, len(tq | td))
    base = 0.42 * seq + 0.58 * jacc
    key_tokens = _cross_desc_tokens(query_norm, min_len=4)
    if key_tokens:
        coverage = sum(1 for w in key_tokens if w in td) / len(key_tokens)
        # 0.68: full coverage recovers score when union-Jaccard is low on noisy long references.
        boosted = base + (1.0 - base) * coverage * 0.68
    else:
        boosted = base
    score = min(1.0, max(0.0, boosted))
    return float(max(seq, score))

def _cross_match_rerank_shortlist_by_similarity(
    work: pd.DataFrame,
    query_texts: List[str],
    *,
    top_n: int = _CROSS_MATCH_RERANK_TOP_N,
) -> pd.DataFrame:
    """
    Second stage after BM25: score each row with ``_cross_desc_similarity_score`` against every
    non-empty query (best score wins), sort descending, keep the top ``top_n`` rows.
    """
    if work.empty or COL_DESC not in work.columns:
        return work
    nonempty = [nullable_str(q) for q in query_texts if nullable_str(q)]
    q_norms = [_normalize_cross_desc_for_match(q) for q in nonempty]
    q_norms = [qn for qn in q_norms if len(qn) >= 2]
    if not q_norms:
        return work.head(top_n).copy() if len(work) > top_n else work.copy()
    desc_series = work[COL_DESC].fillna("").astype(str)
    scored: List[Tuple[float, int]] = []
    for i in range(len(work)):
        rn = _normalize_cross_desc_for_match(desc_series.iloc[i])
        best = 0.0
        for qn in q_norms:
            best = max(best, _cross_desc_similarity_score(qn, rn))
        scored.append((best, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    take = min(top_n, len(scored))
    order = [scored[j][1] for j in range(take)]
    return work.iloc[order].copy()

def _attribute_filter_enabled() -> bool:
    return (os.environ.get("CCAPR_USE_STRUCTURED_FILTERS") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

def _apply_attribute_filter_to_shortlist(
    work: pd.DataFrame,
    shortlist_labels: List[Any],
    query_desc: str,
) -> Tuple[List[Any], Dict[Any, float], Dict[Any, str]]:
    """Phase 3a: drop hard-mismatched candidates and compute soft penalties.

    Returns ``(kept_labels_in_input_order, penalty_by_label, reason_by_label)``.
    Empty / missing inputs ⇒ no-op (returns the input unchanged).

    The function tolerates *any* failure of the extractor / filter (missing
    Redis, malformed cache, import error) by returning the input unchanged so
    the cross-search pipeline stays operational even if Phase 3 is broken.
    """
    if not shortlist_labels or not query_desc or not _attribute_filter_enabled():
        return list(shortlist_labels), {}, {}
    try:
        from attributes.extractor import get_extractor
        from attributes.filters import attribute_filter_score
        from session_store import get_redis_client_for_cache_use
    except Exception:
        return list(shortlist_labels), {}, {}

    if work is None or work.empty or COL_DESC not in work.columns:
        return list(shortlist_labels), {}, {}

    redis_client = get_redis_client_for_cache_use()
    extractor = get_extractor()

    try:
        query_attrs = extractor.extract(query_desc, redis_client=redis_client)
    except Exception:
        return list(shortlist_labels), {}, {}

    try:
        candidate_descs = [
            str(work.loc[lb][COL_DESC]) if lb in work.index else "" for lb in shortlist_labels
        ]
        candidate_attrs_list = extractor.extract_many(
            candidate_descs, redis_client=redis_client
        )
    except Exception:
        return list(shortlist_labels), {}, {}

    kept: List[Any] = []
    penalty_by_label: Dict[Any, float] = {}
    reason_by_label: Dict[Any, str] = {}
    for lb, cattrs in zip(shortlist_labels, candidate_attrs_list):
        try:
            decision = attribute_filter_score(query_attrs, cattrs)
        except Exception:
            kept.append(lb)
            continue
        if not decision.keep:
            reason_by_label[lb] = decision.reason
            continue
        kept.append(lb)
        if decision.penalty > 0.0:
            penalty_by_label[lb] = decision.penalty
            reason_by_label[lb] = decision.reason
    return kept, penalty_by_label, reason_by_label

def _abstain_threshold() -> float:
    try:
        return float(os.environ.get("CCAPR_ABSTAIN_THRESHOLD", "0.0"))
    except ValueError:
        return 0.0

def _cross_match_cross_encoder_rerank(
    work: pd.DataFrame,
    query_texts: List[str],
    *,
    top_n: int,
    penalty_by_label: Optional[Dict[Any, float]] = None,
    abstain_callback: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """Phase 3b: cross-encoder rerank of the work-frame candidates.

    Returns ``None`` (so callers fall back to the Phase-2 hybrid lex+emb
    reranker) when:
    - the cross-encoder kill switch is off (``CCAPR_USE_CROSS_ENCODER=0``),
    - the work-frame is empty / lacks the description column,
    - or the cross-encoder import / inference fails for any reason.

    Phase 3c: when ``abstain_callback`` is provided and the top-1 score is
    below the calibrated threshold, the callback is invoked with
    ``(top_score, top_label, top_text)`` so callers can surface the
    no-confident-match state to the UI.

    The cross-encoder is the precision lever in Phase 3 — it scores the
    (query, candidate) pair jointly so paraphrase / synonym wins from
    embeddings finally survive into the candidate window.
    """
    if work.empty or COL_DESC not in work.columns:
        return None
    nonempty = [nullable_str(q) for q in query_texts if nullable_str(q)]
    if not nonempty:
        return None
    try:
        from cross_encoder.pipeline import cross_encode_and_rank
        from cross_encoder.reranker import is_cross_encoder_enabled
        from session_store import get_redis_client_for_cache_use
    except Exception:
        return None
    if not is_cross_encoder_enabled():
        return None

    desc_series = work[COL_DESC].fillna("").astype(str)
    candidates = list(zip(work.index.tolist(), desc_series.tolist()))
    try:
        ranked = cross_encode_and_rank(
            nonempty[0],
            candidates,
            redis_client=get_redis_client_for_cache_use(),
            penalty_by_label=penalty_by_label,
            top_n=int(top_n),
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Cross-encoder rerank failed; falling back to Phase 2 hybrid rerank"
        )
        return None
    if not ranked:
        return None

    if abstain_callback is not None:
        top_label, top_text, top_score = ranked[0]
        try:
            abstain_callback(
                float(top_score),
                top_label,
                top_text,
                bool(float(top_score) < _abstain_threshold()),
            )
        except Exception:
            pass  # never let an audit hook break the pipeline

    ordered_labels = [lb for lb, _, _ in ranked]
    return work.loc[ordered_labels].copy()

def _cross_match_hybrid_rerank_shortlist(
    work: pd.DataFrame,
    query_texts: List[str],
    company: str,
    *,
    top_n: int = _CROSS_MATCH_RERANK_TOP_N,
    penalty_by_label: Optional[Dict[Any, float]] = None,
    out_top_score: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Phase 2 reranker: combines lexical similarity (``_cross_desc_similarity_score``)
    with embedding cosine similarity (per row's canonical description vs the query
    embedding) using a fixed convex weight.

    Phase 3 addition: ``penalty_by_label`` (label → soft penalty in [0, 1]) is
    *subtracted* from the rerank score for that row. This is how the structured
    attribute filter discounts mismatched-but-not-clearly-wrong candidates.

    Falls back to the lexical-only reranker when:
    - embeddings are disabled (or unavailable) for ``company``,
    - the query embedding cannot be produced,
    - or the work-frame is empty / lacks the description column.

    This is the "missing piece" that lets the embedding signal survive into the
    candidate window the LLM actually sees — without it, lexical rerank discards
    paraphrase / synonym wins from the BM25-∪-embedding shortlist.
    """
    if work.empty or COL_DESC not in work.columns:
        return work
    penalty_by_label = penalty_by_label or {}

    nonempty = [nullable_str(q) for q in query_texts if nullable_str(q)]
    q_norms = [_normalize_cross_desc_for_match(q) for q in nonempty]
    q_norms = [qn for qn in q_norms if len(qn) >= 2]
    if not q_norms:
        return work.head(top_n).copy() if len(work) > top_n else work.copy()

    # Try to enable embedding rerank. Any failure ⇒ silent fallback to lexical.
    emb_query_vec = None
    company_store = None
    try:
        from embeddings.builder import embed_query
        from embeddings.hybrid import (
            get_company_store,
            is_embeddings_enabled_for_company,
        )
        from ingest.canonical_desc import canonicalize_description
        from session_store import get_redis_client_for_cache_use

        if is_embeddings_enabled_for_company(company):
            company_store = get_company_store(company)
            if company_store is not None and company_store.size > 0:
                # Phase 4A — embed *every* query (orig + paraphrases) so the
                # embedding rerank scores each candidate against the closest of
                # the queries we have. Without this, the lexical max-over-queries
                # below works (line 3386) but the embedding score only reflects
                # the original query — which is exactly the failure mode that
                # nullifies query-expansion's shortlist gains in the candidate
                # window. Embedding lookup is Redis-cached (~0.5 ms each), so
                # the per-call overhead is < 2 ms even cold.
                rc = get_redis_client_for_cache_use()
                emb_query_vecs: List[np.ndarray] = []
                for q in nonempty:
                    try:
                        v = embed_query(q, redis_client=rc)
                    except Exception:
                        v = None
                    if v is not None:
                        emb_query_vecs.append(np.asarray(v, dtype=np.float32))
                emb_query_vec = emb_query_vecs[0] if emb_query_vecs else None
    except Exception:
        emb_query_vec = None
        company_store = None
        emb_query_vecs = []

    if emb_query_vec is None or company_store is None or company_store.size == 0:
        return _cross_match_rerank_shortlist_by_similarity(
            work, query_texts, top_n=top_n
        )

    # Score the candidates (we look up each row's canonical-desc vector lazily).
    from ingest.canonical_desc import canonicalize_description

    desc_series = work[COL_DESC].fillna("").astype(str)
    w = _HYBRID_RERANK_LEX_WEIGHT
    one_minus_w = 1.0 - w
    # Pre-normalise *every* query embedding so the per-row scoring is just N
    # cheap dot products (N ≤ 4 with default QE budget).
    qvec_units: List[np.ndarray] = []
    for v in emb_query_vecs:
        n = float(np.linalg.norm(v)) or 1.0
        qvec_units.append(v / n)

    # Build a tiny look-up of canonical-desc -> stored vector index.
    # Cheap because shortlist size is ~50 rows.
    canonical_descs = [canonicalize_description(s) for s in desc_series.tolist()]
    label_by_desc = company_store._id_by_desc  # type: ignore[attr-defined]

    work_index_list = list(work.index)
    scored: List[Tuple[float, int]] = []
    for i, (raw_desc, canon) in enumerate(zip(desc_series.tolist(), canonical_descs)):
        rn = _normalize_cross_desc_for_match(raw_desc)
        lex = 0.0
        for qn in q_norms:
            lex = max(lex, _cross_desc_similarity_score(qn, rn))
        emb_sim = 0.0
        label = label_by_desc.get(canon) if canon else None
        if label is not None:
            try:
                row_vec = company_store._index.get(int(label))  # type: ignore[attr-defined]
                if row_vec is not None:
                    rv = np.asarray(row_vec, dtype=np.float32).reshape(-1)
                    rv_norm = float(np.linalg.norm(rv)) or 1.0
                    rv_unit = rv / rv_norm
                    # Phase 4A: max cosine similarity over orig + paraphrases.
                    # This is the rerank-side counterpart of the shortlist-side
                    # query expansion union — without it, candidates pulled in
                    # by a paraphrase would lose to original-query matches in
                    # the rerank stage.
                    for qu in qvec_units:
                        s = float(np.dot(rv_unit, qu))
                        if s > emb_sim:
                            emb_sim = s
                    if emb_sim < 0.0:
                        emb_sim = 0.0  # avoid penalising unrelated rows below lex-only
            except Exception:
                emb_sim = 0.0
        score = w * lex + one_minus_w * emb_sim
        # Phase 3a: subtract the soft attribute-filter penalty (e.g., -0.4 for
        # low-conf size mismatch). Hard-filter drops happened earlier in the
        # pipeline so any label still here only carries a soft penalty.
        row_label = work_index_list[i]
        score -= float(penalty_by_label.get(row_label, 0.0))
        scored.append((score, i))

    scored.sort(key=lambda x: (-x[0], x[1]))
    take = min(top_n, len(scored))
    order = [scored[j][1] for j in range(take)]
    # Phase 3 inv. — surface the top-1 score so callers can decide whether the
    # hybrid result is confident enough to skip the more expensive CE rerank.
    if out_top_score is not None and scored:
        out_top_score["top1"] = float(scored[0][0])
        if len(scored) >= 2:
            out_top_score["margin"] = float(scored[0][0] - scored[1][0])
    return work.iloc[order].copy()

def _best_cross_reference_row_index(cross_df: pd.DataFrame, query_desc: str, *, min_score: float = 0.22) -> Optional[int]:
    if COL_DESC not in cross_df.columns or COL_UNIT_COST not in cross_df.columns:
        return None
    qn = _normalize_cross_desc_for_match(query_desc)
    if len(qn) < 3:
        return None
    desc_series = cross_df[COL_DESC].fillna("").astype(str)
    best_i: Optional[int] = None
    best_s = 0.0
    for i in range(len(cross_df)):
        uc = _parse_reference_unit_cost_scalar(cross_df.iloc[i].get(COL_UNIT_COST))
        if uc is None or not np.isfinite(float(uc)):
            continue
        rn = _normalize_cross_desc_for_match(desc_series.iloc[i])
        s = _cross_desc_similarity_score(qn, rn)
        if s > best_s:
            best_s = s
            best_i = i
    if best_i is None or best_s < min_score:
        return None
    return best_i

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

def _apply_cross_benchmark_from_reference_row(
    r: Dict[str, Any],
    cross_df: pd.DataFrame,
    idx: int,
    ccapr_vendor: str,
    query_desc: str,
    *,
    po_default: str,
) -> bool:
    """Copy benchmark fields from ``cross_df.iloc[idx]``; set description-match confidence vs ``query_desc``."""
    h = cross_df.iloc[idx]
    uc = _parse_reference_unit_cost_scalar(h.get(COL_UNIT_COST))
    if uc is None or not np.isfinite(float(uc)):
        return False
    vendor_key = _normalize_vendor_key_local(ccapr_vendor)
    ref_vendor = nullable_str(h.get(COL_VENDOR))
    ref_unit = nullable_str(h.get(COL_UNIT))
    ref_po = nullable_str(h.get(COL_PO))
    ref_date = _po_date_iso_from_cell(h.get(COL_PO_DATE))
    ref_site = nullable_str(h.get(COL_PROJECT))

    r["has_history"] = True
    r["lowest_hist_unit_cost"] = float(uc)
    r["lowest_benchmark_vendor"] = ref_vendor
    r["lowest_benchmark_po_number"] = ref_po or po_default
    r["lowest_benchmark_po_date"] = ref_date
    r["lowest_benchmark_hist_unit"] = ref_unit
    r["lowest_benchmark_site_name"] = ref_site

    if vendor_key and _vendor_keys_match(_normalize_vendor_key_local(ref_vendor), vendor_key):
        r["same_vendor_latest_hist_unit_cost"] = float(uc)
        r["same_vendor_benchmark_vendor"] = ref_vendor
        r["same_vendor_benchmark_po_number"] = ref_po
        r["same_vendor_benchmark_po_date"] = ref_date
        r["same_vendor_benchmark_site_name"] = ref_site
        r["same_vendor_hist_unit"] = ref_unit
    ref_no = clean_ccapr_item_no_input(h.get(COL_ITEM_NO))
    if ref_no:
        r["reference_item_no"] = ref_no
    ref_desc = nullable_str(str(h.get(COL_DESC) or ""))
    if ref_desc:
        r["reference_cross_description"] = ref_desc
    qn = _normalize_cross_desc_for_match(query_desc)
    rn = _normalize_cross_desc_for_match(str(h.get(COL_DESC) or ""))
    sim = _cross_desc_similarity_score(qn, rn)
    r["cross_search_confidence_pct"] = round(max(0.0, min(1.0, sim)) * 100.0, 1)
    r["cross_erp_description_match"] = True
    return True

def _reconcile_cross_rows_to_best_reference_description(
    rows: List[Dict[str, Any]],
    slim_items: List[Dict[str, Any]],
    cross_df: pd.DataFrame,
    _ccapr_vendor: str,
) -> None:
    """
    Computes ``cross_search_lexical_confidence_pct`` for every cross-matched row and re-runs
    ``_blend_cross_search_confidence`` so the displayed ``cross_search_confidence_pct`` reflects
    the documented formula (0.55 * lexical + 0.45 * AI). Does **not** change unit cost,
    PO, or ``reference_item_no``. Skips rows already settled by ``exact_code`` (always 100%).
    """
    if not rows or not slim_items or cross_df is None or cross_df.empty:
        return
    if COL_DESC not in cross_df.columns:
        return
    desc_series = cross_df[COL_DESC].fillna("").astype(str)
    n = min(len(rows), len(slim_items))
    for j in range(n):
        r = rows[j]
        if not r.get("cross_erp_description_match"):
            continue
        if str(r.get("match_provenance") or "") == "exact_code":
            continue
        q = nullable_str(slim_items[j].get("item_description") or slim_items[j].get("description") or "")
        if not q.strip():
            continue
        current_idx = _cross_match_reference_row_index_for_row(r, cross_df)
        if current_idx is None:
            continue
        if not (0 <= current_idx < len(cross_df)):
            continue
        qn = _normalize_cross_desc_for_match(q)
        rn = _normalize_cross_desc_for_match(str(desc_series.iloc[current_idx]))
        lexical_sim = round(max(0.0, min(1.0, _cross_desc_similarity_score(qn, rn))) * 100.0, 1)
        _set_cross_search_audit(
            r,
            provenance=str(r.get("match_provenance") or "ai_tagged"),
            lexical_pct=lexical_sim,
            ai_pct=r.get("cross_search_ai_confidence_pct"),
        )

def _fallback_cross_description_benchmarks(
    rows: List[Dict[str, Any]],
    slim_items: List[Dict[str, Any]],
    cross_df: pd.DataFrame,
    ccapr_vendor: str,
    *,
    store: Optional[Dict[str, Any]] = None,
    tab_raw: Optional[str] = None,
) -> int:
    """
    When the model omits reference_unit_cost, pick the best historical row by string/description
    similarity and copy Unit Cost (incl. MBL ``Price`` mapped into this frame) from the dataframe.
    """
    if not rows or not slim_items:
        return 0
    n = min(len(rows), len(slim_items))
    patched = 0
    for j in range(n):
        r = rows[j]
        if r.get("lowest_hist_unit_cost") is not None:
            continue
        q = nullable_str(slim_items[j].get("item_description") or slim_items[j].get("description") or "")
        if not q.strip():
            continue
        idx: Optional[int] = None
        lookup_df: Optional[pd.DataFrame] = None
        if cross_df is not None and not cross_df.empty and COL_DESC in cross_df.columns and COL_UNIT_COST in cross_df.columns:
            idx = _best_cross_reference_row_index(cross_df, q)
            lookup_df = cross_df
        if idx is None and store is not None:
            prepared = _cross_match_prepare_work_df(store, tab_raw)
            if prepared is not None:
                work_full, _, _ = prepared
                idx = _best_cross_reference_row_index(work_full, q)
                if idx is not None:
                    lookup_df = work_full
        if idx is None or lookup_df is None:
            continue
        if not _apply_cross_benchmark_from_reference_row(
            r, lookup_df, idx, ccapr_vendor, q, po_default="Ref. desc match (local)"
        ):
            continue
        patched += 1
        _sync_cross_reference_po_date_from_matched_row(
            r, lookup_df, vendor_key=_normalize_vendor_key_local(ccapr_vendor)
        )
        # Lexical fallback: confidence is purely a description-similarity number.
        _set_cross_search_audit(
            r,
            provenance="lexical_fallback",
            lexical_pct=r.get("cross_search_confidence_pct"),
        )
    return patched

def _slim_items_for_cross_match(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slim_items: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        no = clean_ccapr_item_no_input(it.get("itemNo"))
        if not no:
            continue
        slim_items.append(
            {
                "item_no": no,
                "item_description": nullable_str(it.get("itemDescription") or it.get("description") or ""),
            }
        )
    return slim_items

def _section2_description_lookup_by_ccapr_item(items: List[Dict[str, Any]]) -> Dict[str, str]:
    """Normalized Section 2 Item No. → manual item description (for cross reference resolution)."""
    out: Dict[str, str] = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        no = clean_ccapr_item_no_input(it.get("itemNo"))
        key = normalized_item_key_from_input(no)
        if not key:
            continue
        d = nullable_str(it.get("itemDescription") or it.get("description") or "")
        if d.strip():
            out[key] = d
    return out

def _cross_reference_item_no_is_ccapr_echo(ref: str, ccapr_item_no: str) -> bool:
    """True when ``reference_item_no`` is just the CCAPR / Section 2 code (not a distinct ERP material no.)."""
    if not ref or not ccapr_item_no:
        return False
    return normalized_item_key_from_input(ref) == normalized_item_key_from_input(ccapr_item_no)

def _blend_cross_search_confidence(lexical_pct: Optional[float], ai_pct: Optional[float]) -> Optional[float]:
    have_l = lexical_pct is not None and np.isfinite(lexical_pct)
    have_a = ai_pct is not None and np.isfinite(ai_pct)
    if have_l and have_a:
        return float(round(min(100.0, 0.55 * float(lexical_pct) + 0.45 * float(ai_pct)), 1))
    if have_l:
        return float(round(min(100.0, float(lexical_pct)), 1))
    if have_a:
        return float(round(min(100.0, float(ai_pct)), 1))
    return None

def _lookup_cross_encoder_state_for_row(
    cross_df: Optional[pd.DataFrame], m_use: Dict[str, Any]
) -> Tuple[Optional[float], Optional[bool]]:
    """Phase 3c — find the matching ``cross_df`` row and read its CE score + abstain flag.

    Returns ``(score, abstain)``; either may be ``None`` if absent.
    """
    if cross_df is None or cross_df.empty:
        return None, None
    if CROSS_CE_TOP_SCORE_COL not in cross_df.columns:
        return None, None
    item_raw = _dict_first(m_use, "item_no", "itemNo", "item_number", "ItemNo") or ""
    key = normalized_item_key_from_input(item_raw)
    if not key or CROSS_CCAPR_ITEM_COL not in cross_df.columns:
        return None, None
    try:
        mask = cross_df[CROSS_CCAPR_ITEM_COL].astype(str).map(
            lambda v: normalized_item_key_from_input(clean_ccapr_item_no_input(v))
        ) == key
        sub = cross_df.loc[mask]
        if sub.empty:
            return None, None
        first = sub.iloc[0]
        score = first.get(CROSS_CE_TOP_SCORE_COL)
        abstain = first.get(CROSS_ABSTAIN_COL) if CROSS_ABSTAIN_COL in cross_df.columns else None
        return (
            float(score) if score is not None and pd.notna(score) else None,
            bool(abstain) if abstain is not None and pd.notna(abstain) else None,
        )
    except Exception:
        return None, None

def _set_cross_search_audit(
    row: Dict[str, Any],
    *,
    provenance: str,
    bm25_score: Optional[float] = None,
    lexical_pct: Optional[float] = None,
    ai_pct: Optional[float] = None,
    bm25_top_item_nos: Optional[List[str]] = None,
    rerank_top_item_nos: Optional[List[str]] = None,
    ai_picked_item_no: Optional[str] = None,
    cross_encoder_score: Optional[float] = None,
    abstain: Optional[bool] = None,
) -> None:
    """Persist provenance + scores onto a compare row and refresh the blended confidence."""
    row["match_provenance"] = provenance
    if bm25_score is not None:
        try:
            row["cross_search_bm25_score"] = float(round(float(bm25_score), 3))
        except (TypeError, ValueError):
            pass
    if lexical_pct is not None:
        try:
            row["cross_search_lexical_confidence_pct"] = float(round(float(lexical_pct), 1))
        except (TypeError, ValueError):
            pass
    if ai_pct is not None:
        try:
            row["cross_search_ai_confidence_pct"] = float(round(float(ai_pct), 1))
        except (TypeError, ValueError):
            pass
    blended = _blend_cross_search_confidence(
        row.get("cross_search_lexical_confidence_pct"),
        row.get("cross_search_ai_confidence_pct"),
    )
    if blended is not None:
        row["cross_search_confidence_pct"] = blended
    audit = row.get("cross_search_audit") if isinstance(row.get("cross_search_audit"), dict) else {}
    audit = dict(audit) if audit else {}
    audit["provenance"] = provenance
    if bm25_top_item_nos is not None:
        audit["bm25_top_item_nos"] = list(bm25_top_item_nos)[:10]
    if rerank_top_item_nos is not None:
        audit["rerank_top_item_nos"] = list(rerank_top_item_nos)[:10]
    if ai_picked_item_no is not None:
        audit["ai_picked_item_no"] = str(ai_picked_item_no or "")
    if bm25_score is not None:
        audit["bm25_score"] = row.get("cross_search_bm25_score")
    if lexical_pct is not None:
        audit["lexical_pct"] = row.get("cross_search_lexical_confidence_pct")
    if ai_pct is not None:
        audit["ai_pct"] = row.get("cross_search_ai_confidence_pct")
    if blended is not None:
        audit["blended_pct"] = blended
    # Phase 3 — cross-encoder + abstention surfacing.
    if cross_encoder_score is not None:
        try:
            row["cross_search_cross_encoder_score"] = float(round(float(cross_encoder_score), 3))
            audit["cross_encoder_score"] = row["cross_search_cross_encoder_score"]
        except (TypeError, ValueError):
            pass
    if abstain is not None:
        row["cross_search_abstain"] = bool(abstain)
        audit["abstain"] = bool(abstain)
    row["cross_search_audit"] = audit

def _resolve_cross_exact_code_matches(
    store: Dict[str, Any],
    rows: List[Dict[str, Any]],
    slim_items: List[Dict[str, Any]],
    new_po_tab: Optional[str],
    ccapr_vendor: str,
) -> Set[str]:
    """
    Tier-1 #1: when a CCAPR Item No. exists *verbatim* in the target tab, copy the most-recent
    benchmark row directly (skipping BM25 + Haiku). Returns the set of CCAPR item keys that
    were resolved this way so the caller can exclude them from the slow path.
    """
    if not slim_items or not rows:
        return set()
    prepared = _cross_match_prepare_work_df(store, new_po_tab)
    if prepared is None:
        return set()
    work, _hist, _tab = prepared
    if work.empty or COL_ITEM_NO not in work.columns or COL_UNIT_COST not in work.columns:
        return set()
    erp_keys_series = work[COL_ITEM_NO].map(
        lambda v: normalized_item_key_from_input(clean_ccapr_item_no_input(v))
    )
    if erp_keys_series.empty:
        return set()
    by_row: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rk = normalized_item_key_from_input(r.get("item_no") or "")
        if rk:
            by_row[rk] = r
    resolved: Set[str] = set()
    min_lex = _cross_exact_code_min_lex()
    has_desc_col = COL_DESC in work.columns
    for it in slim_items:
        ccapr_no = clean_ccapr_item_no_input(it.get("item_no"))
        if not ccapr_no:
            continue
        ck = normalized_item_key_from_input(ccapr_no)
        if not ck or ck not in by_row:
            continue
        # First row whose normalized item key matches AND has a parseable unit cost.
        candidate_idx: Optional[int] = None
        for i, k in enumerate(erp_keys_series.tolist()):
            if k != ck:
                continue
            uc = _parse_reference_unit_cost_scalar(work.iloc[i].get(COL_UNIT_COST))
            if uc is not None and np.isfinite(float(uc)):
                candidate_idx = i
                break
        if candidate_idx is None:
            continue
        target_row = by_row[ck]
        query_desc = nullable_str(it.get("item_description") or it.get("description") or "")
        # Description gate: reject same-code-but-clearly-different-product collisions
        # so they flow through BM25 + Haiku instead of getting a silent 100 % stamp.
        # Empty-on-both-sides is treated as "no signal" and still short-circuits
        # (preserves legacy behaviour for rows that genuinely lack descriptions).
        candidate_desc = ""
        if has_desc_col:
            candidate_desc = nullable_str(work.iloc[candidate_idx].get(COL_DESC) or "")
        sim_score: Optional[float] = None
        if query_desc and candidate_desc:
            qn = _normalize_cross_desc_for_match(query_desc)
            dn = _normalize_cross_desc_for_match(candidate_desc)
            sim_score = _cross_desc_similarity_score(qn, dn)
            if sim_score < min_lex:
                logger.info(
                    "Cross exact-code shortcut declined for %s (sim=%.3f < %.2f); routing to AI path",
                    ccapr_no,
                    sim_score,
                    min_lex,
                )
                continue
        elif bool(query_desc) ^ bool(candidate_desc):
            logger.info(
                "Cross exact-code shortcut declined for %s (one-sided description); routing to AI path",
                ccapr_no,
            )
            continue
        applied = _apply_cross_benchmark_from_reference_row(
            target_row,
            work,
            candidate_idx,
            ccapr_vendor,
            query_desc,
            po_default="Ref. exact code match",
        )
        if not applied:
            continue
        # `_apply_cross_benchmark_from_reference_row` writes a description-similarity confidence;
        # an exact code match that passed the lex gate is treated as a perfect match —
        # promote the score and tag provenance. The audit retains the real `lexical_pct`
        # so reviewers can see *why* the shortcut fired.
        gate_pct = 100.0 if sim_score is None else float(round(min(100.0, max(0.0, sim_score * 100.0)), 1))
        _set_cross_search_audit(
            target_row,
            provenance="exact_code",
            lexical_pct=gate_pct,
            ai_pct=None,
        )
        target_row["cross_search_confidence_pct"] = 100.0
        resolved.add(ck)
    return resolved

def _apply_cross_company_match_pipeline(
    store: Dict[str, Any],
    rows: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    new_po_tab: Optional[str],
    ccapr_vendor: str,
    *,
    line_candidates_per_line: Optional[int] = None,
    exclude_erp_item_nos_by_ccapr: Optional[Dict[str, Set[str]]] = None,
    rematch_lexical_polish: bool = False,
) -> None:
    """BM25 shortlist + Haiku + local fallback for cross-company description matching (mutates ``rows``)."""
    slim_items_full = _slim_items_for_cross_match(items)
    section2_desc_by_key = _section2_description_lookup_by_ccapr_item(items)

    # Tier-1 #1 exact-code shortcut (skip BM25 + Haiku for verbatim hits).
    exact_resolved_keys = _resolve_cross_exact_code_matches(
        store, rows, slim_items_full, new_po_tab, ccapr_vendor
    )
    slim_items = [
        it
        for it in slim_items_full
        if normalized_item_key_from_input(clean_ccapr_item_no_input(it.get("item_no")))
        not in exact_resolved_keys
    ]
    if exact_resolved_keys:
        logger.info(
            "Cross exact-code shortcut resolved %s/%s line(s)",
            len(exact_resolved_keys),
            len(slim_items_full),
        )
    cross_df = _cross_match_reference_work_df(
        store,
        new_po_tab,
        bm25_line_items=slim_items,
        line_candidates_per_line=line_candidates_per_line,
        exclude_erp_item_nos_by_ccapr=exclude_erp_item_nos_by_ccapr,
    )
    n_applied = 0
    matches: List[Dict[str, Any]] = []
    if slim_items and cross_df is not None and not cross_df.empty:
        cross_df_llm = cross_df.copy()
        per_line_max = _cross_df_per_line_candidate_max(cross_df_llm)
        try:
            if external_ai_enabled() and _AI_SERVICE.available():
                # Phase 3c: keep CE/abstain columns on cross_df_llm for the
                # audit lookup (_lookup_cross_encoder_state_for_row) but drop
                # them before serializing the LLM prompt — those columns are
                # operational signals, not LLM-relevant facts.
                cross_df_for_tsv = cross_df_llm.drop(
                    columns=[c for c in (CROSS_ABSTAIN_COL, CROSS_CE_TOP_SCORE_COL) if c in cross_df_llm.columns],
                    errors="ignore",
                )
                tsv = cross_df_for_tsv.to_csv(sep="\t", index=False)
                if len(tsv) > 120_000:
                    tsv = tsv[:120_000]
                if tsv.strip():
                    ai = _AI_SERVICE.match_cross_company_descriptions(
                        reference_tsv=tsv,
                        items=slim_items,
                        per_line_candidate_max=per_line_max,
                    )
                    if ai.get("error"):
                        logger.warning("Cross description AI failed: %s", ai.get("error"))
                    else:
                        parsed = ai.get("result") or {}
                        matches = _extract_cross_matches_from_parsed(parsed)
                        n_applied = _apply_cross_description_matches(
                            rows,
                            matches=matches,
                            ccapr_vendor=ccapr_vendor,
                            cross_df=cross_df_llm,
                            section2_desc_by_item_key=section2_desc_by_key,
                        )
                        if n_applied:
                            logger.info("Cross description matches applied to %s compare row(s)", n_applied)
        except Exception as exc:
            logger.warning("Cross description AI skipped: %s", exc)
        try:
            n_fb = _fallback_cross_description_benchmarks(
                rows, slim_items, cross_df_llm, ccapr_vendor, store=store, tab_raw=new_po_tab
            )
            if n_fb:
                logger.info("Cross description local fallback applied to %s compare row(s)", n_fb)
            if matches and n_applied == 0 and n_fb == 0:
                logger.warning(
                    "Cross AI returned %s match row(s) but applied 0 (%s)",
                    len(matches),
                    _cross_match_apply_failure_hint(matches),
                )
        except Exception as exc:
            logger.warning("Cross description local fallback failed: %s", exc)
        try:
            _reconcile_cross_rows_to_best_reference_description(
                rows, slim_items, cross_df_llm, ccapr_vendor
            )
        except Exception as exc:
            logger.debug("Cross description best-row reconcile skipped: %s", exc)
        try:
            _backfill_cross_search_confidence_from_df(rows, cross_df_llm)
        except Exception as exc:
            logger.debug("Cross search confidence backfill skipped: %s", exc)
        try:
            _backfill_cross_reference_item_nos(rows, slim_items, cross_df_llm)
        except Exception as exc:
            logger.debug("Cross reference item no backfill skipped: %s", exc)
        try:
            _ensure_reference_cross_descriptions(rows, cross_df_llm)
        except Exception as exc:
            logger.debug("Cross reference description fill skipped: %s", exc)
        if rematch_lexical_polish:
            try:
                _rematch_lexical_best_among_tagged_candidates(
                    rows, slim_items, cross_df_llm, ccapr_vendor
                )
            except Exception as exc:
                logger.debug("Re-match lexical polish skipped: %s", exc)
            try:
                _reconcile_cross_rows_to_best_reference_description(
                    rows, slim_items, cross_df_llm, ccapr_vendor
                )
            except Exception as exc:
                logger.debug("Cross reconcile after re-match polish skipped: %s", exc)
            try:
                _backfill_cross_reference_item_nos(rows, slim_items, cross_df_llm)
            except Exception as exc:
                logger.debug("Cross reference item no backfill after polish skipped: %s", exc)
            try:
                _ensure_reference_cross_descriptions(rows, cross_df_llm)
            except Exception as exc:
                logger.debug("Cross reference description after polish skipped: %s", exc)

    tab = _normalize_new_po_source_tab(new_po_tab)
    for r in rows:
        r["compare_kind"] = "cross"
        if r.get("cross_erp_description_match") or str(r.get("match_provenance") or ""):
            r["benchmark_erp_tab"] = tab
        else:
            r["benchmark_erp_tab"] = ""
