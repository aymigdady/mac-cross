"""Top-level cross-company match pipeline orchestration."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from cross_ai import AIService, external_ai_enabled
from comparison_engine import (
    COL_DESC,
    COL_ITEM_NO,
    COL_UNIT_COST,
    clean_ccapr_item_no_input,
    normalized_item_key_from_input,
    nullable_str,
)

from .ai_match import _cross_match_apply_failure_hint, _extract_cross_matches_from_parsed
from .apply import (
    _apply_cross_benchmark_from_reference_row,
    _apply_cross_description_matches,
    _backfill_cross_reference_item_nos,
    _backfill_cross_search_confidence_from_df,
    _ensure_reference_cross_descriptions,
    _fallback_cross_description_benchmarks,
    _reconcile_cross_rows_to_best_reference_description,
    _rematch_lexical_best_among_tagged_candidates,
)
from .audit import _set_cross_search_audit
from .candidates import _cross_df_per_line_candidate_max, _cross_match_reference_work_df
from .constants import CROSS_ABSTAIN_COL, CROSS_CE_TOP_SCORE_COL, _cross_exact_code_min_lex
from .parsing import _normalize_new_po_source_tab, _parse_reference_unit_cost_scalar
from .text import _cross_desc_similarity_score, _normalize_cross_desc_for_match
from .workframe import _cross_match_prepare_work_df

logger = logging.getLogger(__name__)
_AI_SERVICE = AIService()

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

