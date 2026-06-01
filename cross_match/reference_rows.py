"""Resolve cross-match reference rows in candidate DataFrames."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from comparison_engine import (
    COL_DESC,
    COL_ITEM_NO,
    COL_UNIT_COST,
    clean_ccapr_item_no_input,
    normalized_item_key_from_input,
    nullable_str,
)

from .ai_match import (
    _cross_match_inferred_reference_item_no,
    _cross_match_lookup_query_for_df,
    _reference_item_no_from_cross_match_dict,
)
from .parsing import _parse_reference_unit_cost_scalar
from .text import _cross_desc_similarity_score, _normalize_cross_desc_for_match

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

