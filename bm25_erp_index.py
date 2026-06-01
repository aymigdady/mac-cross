"""
BM25 index layout for ERP line items (Phase 4).

Design (one index object per company tab — MBL, IFAS, MSB — never mix tabs):

- **One BM25 document per ERP row** in the DataFrame passed to the builder. That frame
  must already be the slice for a single tab (e.g. only MBL rows), so a query built for
  MBL never scores IFAS/MSB rows.

- **Document text** is primarily ``Item Description``. Optionally **Item No.** is
  prepended as plain text (same tokenizer as descriptions) so sparse descriptions still
  carry lexical signal from the material code.

- **Alignment**: corpus position ``i`` maps to ``row_index_labels[i]``, the **pandas index
  label** of that row in the same DataFrame. After ranking, use ``df.loc[label]`` (or
  ``df.loc[labels[i]]``) to read Unit Cost, Vendor, PO, dates, etc.

Tokenization must match cross-match: ``tokenize_cross_desc_for_bm25`` from ``app``
(imported lazily inside the builder to avoid a circular import at module load).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from comparison_engine import COL_DESC, COL_ITEM_NO, _item_no_series_from_df

COMPANY_TABS = frozenset({"MBL", "IFAS", "MSB"})


def normalize_company_tab(raw: Optional[str]) -> str:
    s = (raw or "").strip().upper()
    if s in COMPANY_TABS:
        return s
    return "MBL"


def _description_series_from_df(df: pd.DataFrame) -> pd.Series:
    if COL_DESC not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    s = df[COL_DESC]
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s


def erp_row_document_text(
    item_no: Any,
    description: Any,
    *,
    include_item_no: bool,
) -> str:
    """Plain-text document for BM25: description, optionally prefixed with item number."""
    desc = str(description if description is not None else "").strip()
    code = str(item_no if item_no is not None else "").strip()
    if include_item_no and code:
        out = f"{code} {desc}".strip()
        return out
    return desc


@dataclass(frozen=True)
class ErpBm25Index:
    """
    BM25 corpus aligned 1:1 with ``df.iloc[0]..iloc[n-1]`` row **labels** in ``row_index_labels``.

    ``bm25`` is None only when the corpus has no tokens at all (cannot initialize BM25 safely).
    """

    company_tab: str
    bm25: Optional[BM25Okapi]
    row_index_labels: Tuple[Any, ...]
    k1: float
    b: float

    def __post_init__(self) -> None:
        if self.bm25 is not None and self.bm25.corpus_size != len(self.row_index_labels):
            raise ValueError("BM25 corpus size must match row_index_labels length")

    @property
    def size(self) -> int:
        return len(self.row_index_labels)

    def scores_for_query_tokens(self, query_tokens: Sequence[str]) -> np.ndarray:
        """BM25 score per corpus row; same order as ``row_index_labels``."""
        n = len(self.row_index_labels)
        if n == 0:
            return np.array([], dtype=np.float64)
        if self.bm25 is None:
            return np.zeros(n, dtype=np.float64)
        q = list(query_tokens)
        return np.asarray(self.bm25.get_scores(q), dtype=np.float64)

    def ranked_positions(self, query_tokens: Sequence[str]) -> List[int]:
        """Corpus positions sorted by descending BM25 score (stable tie-break not guaranteed)."""
        s = self.scores_for_query_tokens(query_tokens)
        if s.size == 0:
            return []
        return [int(i) for i in np.argsort(-s)]


def build_erp_bm25_index_for_dataframe(
    df: pd.DataFrame,
    company_tab: str,
    *,
    include_item_no_in_document: bool = True,
    k1: float = 1.5,
    b: float = 0.75,
) -> ErpBm25Index:
    """
    Build a tab-scoped BM25 index over ``df`` (must be one company's ERP slice only).

    Rows are visited in ``iloc`` order; ``row_index_labels[i]`` is ``df.index[pos]`` for
    that row so callers can join back with ``df.loc[label]``.
    """
    tab = normalize_company_tab(company_tab)
    if df.empty:
        return ErpBm25Index(company_tab=tab, bm25=None, row_index_labels=tuple(), k1=k1, b=b)

    from cross_pipeline import tokenize_cross_desc_for_bm25

    item_s = _item_no_series_from_df(df)
    desc_s = _description_series_from_df(df)

    labels: List[Any] = []
    tokenized: List[List[str]] = []
    for pos in range(len(df)):
        labels.append(df.index[pos])
        doc = erp_row_document_text(
            item_s.iloc[pos],
            desc_s.iloc[pos],
            include_item_no=include_item_no_in_document,
        )
        tokenized.append(tokenize_cross_desc_for_bm25(doc))

    if not any(tokenized):
        return ErpBm25Index(
            company_tab=tab,
            bm25=None,
            row_index_labels=tuple(labels),
            k1=k1,
            b=b,
        )

    bm25 = BM25Okapi(tokenized, k1=k1, b=b)
    return ErpBm25Index(
        company_tab=tab,
        bm25=bm25,
        row_index_labels=tuple(labels),
        k1=k1,
        b=b,
    )
