"""
GCS upload/download for per-company HNSW index files (.usearch, .labels.json, .meta.json).

When ``CCAPR_GCS_BUCKET`` is unset or ``google-cloud-storage`` is unavailable, all
functions no-op gracefully (never raise).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

from .hnsw_store import HNSW_FILE_VERSION, _basename_for_company

logger = logging.getLogger(__name__)

_UPLOAD_LOCKS: Dict[str, threading.Lock] = {}
_UPLOAD_LOCKS_GUARD = threading.Lock()


def _bucket_name() -> str:
    return (os.environ.get("CCAPR_GCS_BUCKET") or "").strip()


def _storage_client():
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        return None
    try:
        return storage.Client()
    except Exception as exc:
        logger.warning("GCS client unavailable: %s", exc)
        return None


def _local_paths(company: str, local_dir: str) -> Optional[Tuple[str, str, str]]:
    if not local_dir:
        return None
    base = _basename_for_company(company, HNSW_FILE_VERSION)
    return (
        os.path.join(local_dir, f"{base}.usearch"),
        os.path.join(local_dir, f"{base}.labels.json"),
        os.path.join(local_dir, f"{base}.meta.json"),
    )


def _gcs_object_name(company: str, filename: str) -> str:
    company_key = (company or "").strip().lower() or "unknown"
    return f"hnsw/{company_key}/{filename}"


def _filenames_for_company(company: str) -> List[str]:
    base = _basename_for_company(company, HNSW_FILE_VERSION)
    return [f"{base}.usearch", f"{base}.labels.json", f"{base}.meta.json"]


def download_if_missing(company: str, local_dir: str) -> bool:
    """Pull index files from GCS when local .usearch is absent. Returns True if usable."""
    bucket = _bucket_name()
    if not bucket:
        return False
    paths = _local_paths(company, local_dir)
    if paths is None:
        return False
    index_path, labels_path, meta_path = paths
    if os.path.isfile(index_path):
        return True
    client = _storage_client()
    if client is None:
        return False
    try:
        os.makedirs(local_dir, mode=0o755, exist_ok=True)
        bkt = client.bucket(bucket)
        got_index = False
        for fname, dest in zip(
            _filenames_for_company(company),
            (index_path, labels_path, meta_path),
        ):
            blob_name = _gcs_object_name(company, fname)
            try:
                blob = bkt.blob(blob_name)
                if not blob.exists():
                    logger.debug("GCS object missing: gs://%s/%s", bucket, blob_name)
                    continue
                blob.download_to_filename(dest)
                if fname.endswith(".usearch"):
                    got_index = True
                logger.info("GCS downloaded gs://%s/%s -> %s", bucket, blob_name, dest)
            except Exception as exc:
                logger.warning("GCS download failed for %s: %s", blob_name, exc)
        return got_index or os.path.isfile(index_path)
    except Exception as exc:
        logger.warning("GCS download_if_missing failed for %s: %s", company, exc)
        return False


def gcs_meta_fingerprint(company: str, local_dir: str) -> str:
    """Return fingerprint from local meta.json, or empty string."""
    paths = _local_paths(company, local_dir)
    if paths is None:
        return ""
    _index, _labels, meta_path = paths
    if not os.path.isfile(meta_path):
        return ""
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        fp = meta.get("fingerprint") if isinstance(meta, dict) else ""
        return str(fp or "").strip()
    except Exception as exc:
        logger.debug("Could not read meta fingerprint for %s: %s", company, exc)
        return ""


def _upload_company_sync(company: str, local_dir: str) -> None:
    bucket = _bucket_name()
    if not bucket:
        return
    paths = _local_paths(company, local_dir)
    if paths is None:
        return
    client = _storage_client()
    if client is None:
        return
    try:
        bkt = client.bucket(bucket)
        for fname, local_path in zip(_filenames_for_company(company), paths):
            if not os.path.isfile(local_path):
                continue
            blob_name = _gcs_object_name(company, fname)
            try:
                bkt.blob(blob_name).upload_from_filename(local_path)
                logger.info("GCS uploaded %s -> gs://%s/%s", local_path, bucket, blob_name)
            except Exception as exc:
                logger.warning("GCS upload failed for %s: %s", blob_name, exc)
    except Exception as exc:
        logger.warning("GCS upload sync failed for %s: %s", company, exc)


def _upload_lock_for(company: str) -> threading.Lock:
    with _UPLOAD_LOCKS_GUARD:
        if company not in _UPLOAD_LOCKS:
            _UPLOAD_LOCKS[company] = threading.Lock()
        return _UPLOAD_LOCKS[company]


def upload_async(company: str, local_dir: str) -> None:
    """Upload index files to GCS in a background daemon thread (never raises)."""
    if not _bucket_name():
        return

    def _worker() -> None:
        lock = _upload_lock_for(company)
        if not lock.acquire(blocking=False):
            logger.info("GCS upload for %s already in progress, skipping", company)
            return
        try:
            _upload_company_sync(company, local_dir)
        finally:
            lock.release()

    threading.Thread(
        target=_worker,
        name=f"gcs-hnsw-upload-{company}",
        daemon=True,
    ).start()
