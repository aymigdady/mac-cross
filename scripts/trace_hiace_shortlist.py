#!/usr/bin/env python3
"""Trace BM25 shortlist for HiAce line inside MAC-CROSS container."""
from __future__ import annotations

import json
import os
import sys
import uuid

sys.path.insert(0, "/app")

import app as app_module  # mac-cross app if exists — else load erp via requests session

ITEM_NO = "SE000015722"
DESC = "Rental HiAce (With Driver)"
TARGET = "MBL"

# Load ERP the same way cost-control does — mac-cross reads session from Redis
from session_store import DATA_STORE
from cross_pipeline import (
    _best_cross_reference_row_index,
    _cross_match_prepare_work_df,
    _cross_match_reference_work_df,
    _normalize_cross_desc_for_match,
    _resolve_cross_exact_code_matches,
    _slim_items_for_cross_match,
    _cross_desc_similarity_score,
)
from comparison_engine import COL_DESC, COL_ITEM_NO, COL_UNIT_COST, clean_ccapr_item_no_input


def main() -> None:
    session_id = os.environ.get("TRACE_SESSION_ID", "").strip()
    if not session_id:
        session_id = str(uuid.uuid4())
        print("No TRACE_SESSION_ID — loading ERP locally")
        DATA_STORE[session_id] = {}
        # mac-cross may not have _try_load_default_erp_blocking; skip if missing
        if hasattr(app_module, "_try_load_default_erp_blocking"):
            app_module._try_load_default_erp_blocking(session_id, app_module._default_erp_fetch_urls())

    store = DATA_STORE.get(session_id)
    if not store:
        print("Session not in Redis:", session_id)
        print("Run trace from cost-control first and pass TRACE_SESSION_ID")
        sys.exit(1)

    slim = [{"item_no": ITEM_NO, "item_description": DESC}]
    items_api = [{"itemNo": ITEM_NO, "itemDescription": DESC}]

    print("=== Exact-code shortcut ===")
    rows = [{"item_no": ITEM_NO, "description": DESC}]
    resolved = _resolve_cross_exact_code_matches(store, rows, slim, TARGET, "Test Vendor")
    print("resolved keys:", resolved)
    print("row after exact:", json.dumps({k: rows[0].get(k) for k in ("reference_item_no", "lowest_hist_unit_cost", "cross_search_confidence_pct")}, indent=2))

    print("\n=== MBL work df (prepare) ===")
    prep = _cross_match_prepare_work_df(store, TARGET)
    if prep:
        work, _, tab = prep
        print("tab", tab, "rows", len(work))
        m = work[COL_ITEM_NO].astype(str).str.contains("506000000000", na=False)
        print("506000000000 in full MBL work:", int(m.sum()))
        if m.any():
            print(work.loc[m, [COL_ITEM_NO, COL_DESC, COL_UNIT_COST]].head(1).to_string())
        m2 = work[COL_DESC].fillna("").astype(str).str.contains("hiace", case=False, na=False)
        print("hiace rows in full MBL work:", int(m2.sum()))
        if m2.any():
            print(work.loc[m2, [COL_ITEM_NO, COL_DESC]].drop_duplicates(COL_ITEM_NO).head(3).to_string())

    print("\n=== BM25 shortlist (cross_df) ===")
    cross_df = _cross_match_reference_work_df(store, TARGET, bm25_line_items=slim)
    if cross_df is None or cross_df.empty:
        print("cross_df EMPTY")
    else:
        print("cross_df rows:", len(cross_df), "cols:", list(cross_df.columns)[:8])
        if "CCAPR Item No." in cross_df.columns:
            tagged = cross_df[cross_df["CCAPR Item No."].astype(str).str.contains(ITEM_NO, na=False)]
            print("tagged for", ITEM_NO, ":", len(tagged))
            if not tagged.empty and COL_DESC in tagged.columns:
                for _, row in tagged.head(10).iterrows():
                    print(" ", row.get(COL_ITEM_NO), "|", str(row.get(COL_DESC))[:60])
        hiace = cross_df[COL_DESC].fillna("").astype(str).str.contains("hiace", case=False, na=False)
        print("hiace in shortlist:", int(hiace.sum()))

    print("\n=== Local fallback index (full cross_df scan) ===")
    if cross_df is not None and not cross_df.empty:
        idx = _best_cross_reference_row_index(cross_df, DESC)
        print("best idx:", idx)
        if idx is not None:
            r = cross_df.iloc[idx]
            print("best match:", r.get(COL_ITEM_NO), "|", r.get(COL_DESC))
            qn = _normalize_cross_desc_for_match(DESC)
            rn = _normalize_cross_desc_for_match(str(r.get(COL_DESC) or ""))
            print("similarity:", round(_cross_desc_similarity_score(qn, rn), 4))


if __name__ == "__main__":
    main()
