"""Phase 3b — cross-encoder rerank for cross-search candidate selection.

Public surface:

- :class:`reranker.CrossEncoderReranker` (and the singleton :func:`reranker.get_reranker`) —
  wraps :class:`sentence_transformers.CrossEncoder` with a Redis pair-level
  score cache.
- :func:`pipeline.cross_encode_and_rank` — convenience entry point that takes
  ``(query, [(candidate_id, candidate_text), ...])`` and returns the candidates
  sorted by cross-encoder score.

The module deliberately has zero Flask dependencies so it can be exercised
from the harness, from a CLI, and from unit tests without booting the web layer.
"""
