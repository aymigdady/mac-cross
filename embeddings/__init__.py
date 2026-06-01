"""Phase 2 — semantic recall via bge-m3 embeddings.

Public surface:

- :class:`embedder.Embedder` (and the singleton :func:`embedder.get_embedder`) —
  wraps :mod:`sentence_transformers` with a Redis description-level cache.
- :class:`hnsw_store.HnswStore` — disk-persisted usearch HNSW index, one per
  company, with append-only writes.
- :func:`builder.ensure_company_embedding_index` — idempotent build/load that
  computes embeddings for any canonical descriptions not already in the index.

The pipeline is deliberately decoupled from ``app.py`` (no Flask imports here)
so the embedder + index can be exercised from the harness, from a CLI, and from
unit tests without spinning up the web layer.
"""
