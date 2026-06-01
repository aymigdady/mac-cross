"""Cross-match pipeline constants and env-knob readers."""
from __future__ import annotations

import os

_CROSS_MATCH_BM25_SHORTLIST_MAX = 50

_CROSS_MATCH_RERANK_TOP_N = 10

_CROSS_MATCH_CANDIDATES_PER_LINE = int(os.environ.get("CCAPR_CROSS_CANDIDATES_PER_LINE", "6"))

_CROSS_MATCH_REMATCH_CANDIDATES_PER_LINE = int(os.environ.get("CCAPR_CROSS_REMATCH_CANDIDATES_PER_LINE", "8"))

_HYBRID_RERANK_LEX_WEIGHT = float(os.environ.get("CCAPR_HYBRID_RERANK_LEX_WEIGHT", "0.5"))

CROSS_CCAPR_ITEM_COL = "CCAPR Item No."

CROSS_ABSTAIN_COL = "_CCAPR_Abstain"

CROSS_CE_TOP_SCORE_COL = "_CCAPR_CE_TopScore"

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

def _abstain_threshold() -> float:
    try:
        return float(os.environ.get("CCAPR_ABSTAIN_THRESHOLD", "0.0"))
    except ValueError:
        return 0.0

def _attribute_filter_enabled() -> bool:
    return (os.environ.get("CCAPR_USE_STRUCTURED_FILTERS") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

