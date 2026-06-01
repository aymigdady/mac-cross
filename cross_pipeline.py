"""
Cross-company match pipeline (extracted from cost-control-app app.py).

Backward-compatible facade — implementation lives in cross_match/.
"""
from __future__ import annotations

from cross_match.ai_match import (
    _cross_match_apply_failure_hint,
    _cross_reference_item_no_is_ccapr_echo,
    _extract_cross_matches_from_parsed,
)
from cross_match.apply import (
    _apply_cross_benchmark_from_reference_row,
    _apply_cross_description_matches,
    _apply_one_cross_match_to_row,
    _backfill_cross_reference_item_nos,
    _backfill_cross_search_confidence_from_df,
    _ensure_reference_cross_descriptions,
    _fallback_cross_description_benchmarks,
    _reconcile_cross_rows_to_best_reference_description,
    _rematch_lexical_best_among_tagged_candidates,
    _set_cross_reference_item_no_on_row,
    _sync_cross_reference_po_date_from_matched_row,
)
from cross_match.audit import (
    _blend_cross_search_confidence,
    _lookup_cross_encoder_state_for_row,
    _set_cross_search_audit,
)
from cross_match.candidates import (
    _cross_df_per_line_candidate_max,
    _cross_match_reference_tsv,
    _cross_match_reference_work_df,
    _cross_match_work_df_two_per_line,
    cross_match_pipeline_diagnose,
)
from cross_match.columns import (
    _cross_match_ensure_unit_cost_column,
    _cross_match_guess_price_column,
)
from cross_match.constants import (
    CROSS_ABSTAIN_COL,
    CROSS_CCAPR_ITEM_COL,
    CROSS_CE_TOP_SCORE_COL,
    _CROSS_MATCH_BM25_SHORTLIST_MAX,
    _CROSS_MATCH_CANDIDATES_PER_LINE,
    _CROSS_MATCH_REMATCH_CANDIDATES_PER_LINE,
    _CROSS_MATCH_RERANK_TOP_N,
    _HYBRID_RERANK_LEX_WEIGHT,
)
from cross_match.parsing import (
    _dict_first,
    _normalize_new_po_source_tab,
    _normalize_vendor_key_local,
    _parse_reference_unit_cost_scalar,
    _po_date_iso_from_cell,
    _vendor_keys_match,
)
from cross_match.pipeline import (
    _apply_cross_company_match_pipeline,
    _resolve_cross_exact_code_matches,
    _section2_description_lookup_by_ccapr_item,
    _slim_items_for_cross_match,
)
from cross_match.reference_rows import (
    _best_cross_reference_row_index,
    _cross_df_resolve_row_index_for_match,
    _cross_match_enrich_from_reference_df,
    _cross_match_enrich_pick_row_index,
    _cross_match_reference_row_index_for_row,
    _first_cross_df_row_with_cost_for_item,
    _reference_item_no_from_cross_df,
)
from cross_match.rerank import (
    _apply_attribute_filter_to_shortlist,
    _cross_match_cross_encoder_rerank,
    _cross_match_hybrid_rerank_shortlist,
    _cross_match_rerank_shortlist_by_similarity,
)
from cross_match.shortlist import (
    _cross_match_bm25_shortlist_index_labels,
    _cross_match_expand_query,
    _cross_match_hybrid_shortlist_index_labels,
    _embeddings_hybrid_available_for_tab,
)
from cross_match.text import (
    _cross_desc_similarity_score,
    _normalize_cross_desc_for_match,
    tokenize_cross_desc_for_bm25,
)
from cross_match.workframe import (
    _cross_match_dedupe_receipt_rows,
    _cross_match_drop_unpriceable_rows,
    _cross_match_prepare_work_df,
    _effective_historical_store,
)

__all__ = [
    "CROSS_ABSTAIN_COL",
    "CROSS_CCAPR_ITEM_COL",
    "CROSS_CE_TOP_SCORE_COL",
    "_CROSS_MATCH_BM25_SHORTLIST_MAX",
    "_CROSS_MATCH_CANDIDATES_PER_LINE",
    "_CROSS_MATCH_REMATCH_CANDIDATES_PER_LINE",
    "_CROSS_MATCH_RERANK_TOP_N",
    "_HYBRID_RERANK_LEX_WEIGHT",
    "_apply_attribute_filter_to_shortlist",
    "_apply_cross_benchmark_from_reference_row",
    "_apply_cross_company_match_pipeline",
    "_apply_cross_description_matches",
    "_apply_one_cross_match_to_row",
    "_backfill_cross_reference_item_nos",
    "_backfill_cross_search_confidence_from_df",
    "_best_cross_reference_row_index",
    "_blend_cross_search_confidence",
    "_cross_desc_similarity_score",
    "_cross_df_per_line_candidate_max",
    "_cross_df_resolve_row_index_for_match",
    "_cross_match_apply_failure_hint",
    "_cross_match_bm25_shortlist_index_labels",
    "_cross_match_cross_encoder_rerank",
    "_cross_match_dedupe_receipt_rows",
    "_cross_match_drop_unpriceable_rows",
    "_cross_match_ensure_unit_cost_column",
    "_cross_match_enrich_from_reference_df",
    "_cross_match_enrich_pick_row_index",
    "_cross_match_expand_query",
    "_cross_match_guess_price_column",
    "_cross_match_hybrid_rerank_shortlist",
    "_cross_match_hybrid_shortlist_index_labels",
    "_cross_match_prepare_work_df",
    "_cross_match_reference_row_index_for_row",
    "_cross_match_reference_tsv",
    "_cross_match_reference_work_df",
    "_cross_match_rerank_shortlist_by_similarity",
    "_cross_match_work_df_two_per_line",
    "_dict_first",
    "_effective_historical_store",
    "_embeddings_hybrid_available_for_tab",
    "_ensure_reference_cross_descriptions",
    "_extract_cross_matches_from_parsed",
    "_fallback_cross_description_benchmarks",
    "_first_cross_df_row_with_cost_for_item",
    "_lookup_cross_encoder_state_for_row",
    "_normalize_cross_desc_for_match",
    "_normalize_new_po_source_tab",
    "_normalize_vendor_key_local",
    "_parse_reference_unit_cost_scalar",
    "_po_date_iso_from_cell",
    "_reconcile_cross_rows_to_best_reference_description",
    "_reference_item_no_from_cross_df",
    "_rematch_lexical_best_among_tagged_candidates",
    "_resolve_cross_exact_code_matches",
    "_section2_description_lookup_by_ccapr_item",
    "_set_cross_reference_item_no_on_row",
    "_set_cross_search_audit",
    "_slim_items_for_cross_match",
    "_sync_cross_reference_po_date_from_matched_row",
    "_vendor_keys_match",
    "cross_match_pipeline_diagnose",
    "tokenize_cross_desc_for_bm25",
]
