"""Build per-line candidate DataFrames and diagnostic helpers."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from comparison_engine import COL_DESC, COL_ITEM_NO, clean_ccapr_item_no_input, normalized_item_key_from_input, nullable_str

from .constants import (
    CROSS_ABSTAIN_COL,
    CROSS_CCAPR_ITEM_COL,
    CROSS_CE_TOP_SCORE_COL,
    _CROSS_MATCH_BM25_SHORTLIST_MAX,
    _CROSS_MATCH_RERANK_TOP_N,
    _cross_match_candidates_per_line,
)
from .rerank import (
    _apply_attribute_filter_to_shortlist,
    _cross_match_cross_encoder_rerank,
    _cross_match_hybrid_rerank_shortlist,
    _cross_match_rerank_shortlist_by_similarity,
)
from .shortlist import (
    _cross_match_expand_query,
    _cross_match_hybrid_shortlist_index_labels,
    _embeddings_hybrid_available_for_tab,
)
from .workframe import _cross_match_prepare_work_df

logger = logging.getLogger(__name__)

def _cross_df_per_line_candidate_max(cross_df: Optional[pd.DataFrame]) -> int:
    """Largest number of candidate rows for any single CCAPR line (for Haiku prompt sizing)."""
    if cross_df is None or cross_df.empty or CROSS_CCAPR_ITEM_COL not in cross_df.columns:
        return 2
    try:
        g = cross_df.groupby(cross_df[CROSS_CCAPR_ITEM_COL].astype(str).str.strip()).size()
        return max(2, min(int(g.max()), 12))
    except Exception:
        return 2

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

def _cross_match_reference_tsv(store: Dict[str, Any], tab_raw: Optional[str]) -> str:
    df = _cross_match_reference_work_df(store, tab_raw)
    if df is None or df.empty:
        return ""
    tsv = df.to_csv(sep="\t", index=False)
    max_chars = 120_000
    return tsv[:max_chars] if len(tsv) > max_chars else tsv

