"""Apply cross-match results to compare rows."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from comparison_engine import (
    COL_DESC,
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

from .ai_match import (
    _cross_ai_matched_flag,
    _cross_match_confidence_pct_from_dict,
    _cross_match_description_for_similarity,
    _cross_match_effective_unit_cost,
    _cross_reference_item_no_is_ccapr_echo,
    _dict_first,
    _reference_fields_from_cross_match,
    _reference_item_no_from_cross_match_dict,
)
from .audit import _lookup_cross_encoder_state_for_row, _set_cross_search_audit
from .constants import CROSS_CCAPR_ITEM_COL
from .parsing import (
    _normalize_vendor_key_local,
    _parse_reference_unit_cost_scalar,
    _po_date_iso_from_cell,
    _vendor_keys_match,
)
from .reference_rows import (
    _best_cross_reference_row_index,
    _cross_match_enrich_from_reference_df,
    _cross_match_enrich_pick_row_index,
    _cross_match_reference_row_index_for_row,
    _reference_item_no_from_cross_df,
)
from .text import _cross_desc_similarity_score, _normalize_cross_desc_for_match
from .workframe import _cross_match_prepare_work_df

logger = logging.getLogger(__name__)

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

