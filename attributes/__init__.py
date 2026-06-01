"""Phase 3 — structured attribute extraction + size/voltage/pack hard-filter.

Public surface:

- :class:`extractor.AttributeExtractor` — regex-first extractor with optional
  Haiku fallback, returning :class:`extractor.ExtractedAttributes` (every field
  has both a value and a confidence in ``[0, 1]``).
- :func:`extractor.extract_attributes` — convenience wrapper around the
  process-level extractor singleton.
- :func:`filters.attribute_filter_score` — hard/soft filter that returns
  ``(keep: bool, penalty: float, reason: str)`` for one (query, candidate) pair.

The module deliberately has zero Flask dependencies so it can be exercised from
the harness, from a CLI, and from unit tests without booting the web layer.
"""
