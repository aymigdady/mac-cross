"""
Disk-persisted HNSW index for canonical descriptions, one per company.

The index uses :mod:`usearch` (C++ HNSW with Python bindings, no external service).
We store **canonical description ⇒ embedding vector** with an integer label that
maps back to the canonical description string via a small companion file. This
gives us:

- Sub-15 ms top-30 queries on 30k vectors.
- Append-only writes — adding 50 new descriptions a week is O(50 log N).
- Multi-process safe via mmap reads (each Gunicorn worker mmaps the same file).
- A clean separation between *index* (vectors + topology) and *labels* (dict),
  so the canonical-desc registry remains the single source of truth for which
  item codes belong to a given description.

Layout on disk (one set per ``(company, version)``)::

    <root>/master-po-mbl.v1.usearch        ← usearch binary index
    <root>/master-po-mbl.v1.labels.json    ← label_id ⇒ canonical_desc mapping
    <root>/master-po-mbl.v1.meta.json      ← version, dim, count, fingerprint, built_at

The ``.labels.json`` file is small (~30k entries × ~50 bytes ≈ 1.5 MB) and
loaded into memory once per worker; the index itself is memory-mapped.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


HNSW_FILE_VERSION = 1


def _basename_for_company(company: str, version: int) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", (company or "").lower()).strip("-") or "unknown"
    return f"master-po-{safe}.v{version}"


def hnsw_cache_root() -> Optional[str]:
    """Resolve the cache directory (mirrors ``ingest.parquet_store.parquet_cache_root``).

    Honors ``CCAPR_HNSW_CACHE_DIR`` (explicit), then ``MASTER_PO_LINES_DIR/.ccapr-cache/hnsw``,
    then ``/tmp/ccapr-hnsw-cache``. Returns ``None`` if nothing is writable.
    """
    candidates: List[str] = []
    explicit = (os.environ.get("CCAPR_HNSW_CACHE_DIR") or "").strip()
    if explicit:
        candidates.append(explicit)
    root = (os.environ.get("MASTER_PO_LINES_DIR") or "").strip()
    if root:
        candidates.append(os.path.join(root, ".ccapr-cache", "hnsw"))
    candidates.append(os.path.join(tempfile.gettempdir(), "ccapr-hnsw-cache"))
    for path in candidates:
        try:
            os.makedirs(path, mode=0o755, exist_ok=True)
            probe = os.path.join(path, ".write-probe")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.unlink(probe)
            return path
        except OSError as exc:
            logger.debug("HNSW cache root %s unusable: %s", path, exc)
    return None


def cache_paths_for_company(
    company: str, *, version: int = HNSW_FILE_VERSION
) -> Optional[Tuple[str, str, str]]:
    """Return ``(index_path, labels_path, meta_path)`` or None if no writable root."""
    root = hnsw_cache_root()
    if root is None:
        return None
    base = _basename_for_company(company, version)
    return (
        os.path.join(root, f"{base}.usearch"),
        os.path.join(root, f"{base}.labels.json"),
        os.path.join(root, f"{base}.meta.json"),
    )


@dataclass
class HnswSearchHit:
    """A single nearest-neighbour result: canonical description + similarity in ``[-1, 1]``."""

    canonical_desc: str
    similarity: float
    label: int


class HnswStore:
    """Append-only HNSW index keyed by canonical description.

    Use :meth:`open_or_create` for the standard "build-on-miss" entry point; use
    :meth:`add_descriptions` to incrementally add new descriptions; use
    :meth:`search` to query.

    The store is **per-company-per-version**: bumping ``EMBEDDING_INDEX_VERSION``
    (in :mod:`embeddings.embedder`) creates a fresh ``.v2`` set without touching
    the previous one — useful for safe rollbacks.
    """

    def __init__(
        self,
        company: str,
        dim: int,
        *,
        version: int = HNSW_FILE_VERSION,
    ) -> None:
        self.company = company
        self.dim = int(dim)
        self.version = int(version)
        self._lock = threading.RLock()
        # ``label -> canonical_desc`` and inverse for fast collision detection.
        self._labels_by_id: Dict[int, str] = {}
        self._id_by_desc: Dict[str, int] = {}
        self._next_label: int = 0
        self._index = None  # type: ignore[assignment]
        self._paths: Optional[Tuple[str, str, str]] = cache_paths_for_company(
            company, version=version
        )
        self._load_existing_or_init()

    # --- Public API -----------------------------------------------------------

    @classmethod
    def open_or_create(cls, company: str, dim: int) -> "HnswStore":
        return cls(company, dim)

    @property
    def size(self) -> int:
        return len(self._labels_by_id)

    @property
    def index_path(self) -> Optional[str]:
        return self._paths[0] if self._paths else None

    def known_descriptions(self) -> List[str]:
        return list(self._labels_by_id.values())

    def descriptions_missing_from(self, candidates: Iterable[str]) -> List[str]:
        """Filter ``candidates`` down to the ones not already in the index.

        Used by the index builder to embed only the delta.
        """
        out: List[str] = []
        seen: set = set()
        for d in candidates:
            if not d or d in seen:
                continue
            seen.add(d)
            if d not in self._id_by_desc:
                out.append(d)
        return out

    def add_descriptions(
        self, canonical_descs: List[str], vectors: np.ndarray
    ) -> int:
        """Append new (description, vector) pairs. Returns the number actually added.

        Idempotent: if a description is already present its vector is left as-is
        (we never overwrite — a re-embed would suggest a model/version change,
        which is what ``EMBEDDING_INDEX_VERSION`` exists for).
        """
        if len(canonical_descs) != int(vectors.shape[0]):
            raise ValueError(
                f"add_descriptions: {len(canonical_descs)} descs vs {vectors.shape[0]} vectors"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"vectors have dim {vectors.shape[1]}, store expects {self.dim}"
            )
        with self._lock:
            self._ensure_index()
            new_labels: List[int] = []
            new_vectors: List[np.ndarray] = []
            for desc, vec in zip(canonical_descs, vectors):
                if not desc or desc in self._id_by_desc:
                    continue
                label = self._next_label
                self._next_label += 1
                self._labels_by_id[label] = desc
                self._id_by_desc[desc] = label
                new_labels.append(label)
                new_vectors.append(np.asarray(vec, dtype=np.float32))
            if new_labels:
                arr = np.stack(new_vectors, axis=0)
                self._index.add(np.array(new_labels, dtype=np.int64), arr)
            return len(new_labels)

    def search(self, query_vector: np.ndarray, top_k: int = 30) -> List[HnswSearchHit]:
        """Return the ``top_k`` nearest canonical descriptions for ``query_vector``."""
        if self._index is None or self.size == 0:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector[None, :]
        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"query has dim {query_vector.shape[1]}, store expects {self.dim}"
            )
        with self._lock:
            matches = self._index.search(query_vector, count=int(top_k))
        out: List[HnswSearchHit] = []
        # usearch returns either a Matches object (for batch queries) or BatchMatches;
        # both expose ``.keys`` and ``.distances`` arrays. We always pass a single row
        # so we read row 0.
        try:
            keys = matches.keys
            distances = matches.distances
            if hasattr(keys, "shape") and len(keys.shape) > 1:
                keys = keys[0]
                distances = distances[0]
        except AttributeError:
            # Fallback: iterable of (key, distance) pairs
            keys = [m.key for m in matches]
            distances = [m.distance for m in matches]
        for label, distance in zip(keys, distances):
            label = int(label)
            desc = self._labels_by_id.get(label)
            if desc is None:
                continue
            # usearch with cosine metric returns ``1 - cos_similarity`` as the distance,
            # so similarity = 1 - distance. Clamp to handle FP rounding.
            sim = float(1.0 - float(distance))
            if sim > 1.0:
                sim = 1.0
            if sim < -1.0:
                sim = -1.0
            out.append(HnswSearchHit(canonical_desc=desc, similarity=sim, label=label))
        return out

    def persist(
        self,
        *,
        fingerprint: Optional[str] = None,
        extra_meta: Optional[dict] = None,
    ) -> bool:
        """Write the index, labels and meta to disk atomically. Returns success."""
        if self._paths is None:
            return False
        index_path, labels_path, meta_path = self._paths
        with self._lock:
            try:
                if self._index is None:
                    return False
                tmp_index = index_path + ".tmp"
                self._index.save(tmp_index)
                os.replace(tmp_index, index_path)
                _atomic_write_text(
                    labels_path,
                    json.dumps(
                        {str(k): v for k, v in self._labels_by_id.items()},
                        ensure_ascii=False,
                    ),
                )
                meta = {
                    "file_version": HNSW_FILE_VERSION,
                    "embedding_index_version": self.version,
                    "company": self.company,
                    "dim": self.dim,
                    "count": self.size,
                    "next_label": self._next_label,
                    "built_at_utc": time.time(),
                    "fingerprint": fingerprint or "",
                }
                if extra_meta:
                    meta.update(extra_meta)
                _atomic_write_text(meta_path, json.dumps(meta, indent=2))
                logger.info(
                    "HNSW persisted (%s v%d): %d descriptions -> %s",
                    self.company,
                    self.version,
                    self.size,
                    index_path,
                )
                return True
            except Exception as exc:
                logger.warning("HNSW persist failed for %s: %s", self.company, exc)
                return False

    # --- Internal -------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        from usearch.index import Index

        self._index = Index(
            ndim=self.dim,
            metric="cos",
            dtype="f32",
            connectivity=16,
            expansion_add=128,
            expansion_search=64,
        )

    def _load_existing_or_init(self) -> None:
        if self._paths is None:
            return
        index_path, labels_path, meta_path = self._paths
        if not (os.path.isfile(index_path) and os.path.isfile(labels_path)):
            return
        try:
            with open(labels_path, "r", encoding="utf-8") as fh:
                labels_obj = json.load(fh)
            self._labels_by_id = {int(k): str(v) for k, v in labels_obj.items()}
            self._id_by_desc = {v: k for k, v in self._labels_by_id.items()}
            self._next_label = (max(self._labels_by_id) + 1) if self._labels_by_id else 0
            self._ensure_index()
            self._index.load(index_path)
            logger.info(
                "HNSW loaded (%s v%d): %d descriptions from %s",
                self.company,
                self.version,
                self.size,
                index_path,
            )
        except Exception as exc:
            logger.warning(
                "Could not load existing HNSW for %s (%s); will rebuild on demand",
                self.company,
                exc,
            )
            self._labels_by_id = {}
            self._id_by_desc = {}
            self._next_label = 0
            self._index = None


def _atomic_write_text(path: str, text: str) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def invalidate_company_index(company: str, *, version: int = HNSW_FILE_VERSION) -> None:
    """Delete index/labels/meta for one company. Idempotent."""
    paths = cache_paths_for_company(company, version=version)
    if paths is None:
        return
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass
