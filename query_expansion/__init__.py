"""Phase-4A query expansion.

Public surface:

- :class:`QueryExpander` — the expander itself (regex normalisation + cached
  LLM-driven paraphrase generation).
- :func:`get_query_expander` — process-wide singleton.
- :func:`expand_query` — convenience helper that uses the singleton.
- :func:`is_query_expansion_enabled` — env-flag gate.

The expander is wired into ``app.py``'s per-line cross-search callsite so the
shortlist stage (BM25 ∪ embedding) sees the **union** of candidates produced
for the original query *and* its paraphrases. Reranking and abstention happen
unchanged downstream.
"""

from __future__ import annotations

from .expander import (  # noqa: F401
    EXPANSION_VERSION,
    EXPAND_REDIS_PREFIX,
    QueryExpander,
    expand_query,
    get_query_expander,
    is_query_expansion_enabled,
    set_query_expander,
)
