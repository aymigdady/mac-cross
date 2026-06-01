"""Cross-search audit trail and confidence blending."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from comparison_engine import clean_ccapr_item_no_input, normalized_item_key_from_input

from .ai_match import _dict_first
from .constants import CROSS_ABSTAIN_COL, CROSS_CCAPR_ITEM_COL, CROSS_CE_TOP_SCORE_COL

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

