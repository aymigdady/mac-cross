"""Tests for GCS HNSW backend (mocked — no real GCS calls)."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from embeddings import gcs_hnsw_backend as gcs


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "hnsw"
    d.mkdir()
    return str(d)


def test_download_if_missing_no_bucket(cache_dir):
    with patch.dict(os.environ, {"CCAPR_GCS_BUCKET": ""}, clear=False):
        assert gcs.download_if_missing("MBL", cache_dir) is False


def test_download_if_missing_skips_when_local_exists(cache_dir):
    index_path, _, _ = gcs._local_paths("MBL", cache_dir)
    assert index_path
    with open(index_path, "wb") as fh:
        fh.write(b"fake")
    with patch.dict(os.environ, {"CCAPR_GCS_BUCKET": "test-bucket"}, clear=False):
        with patch.object(gcs, "_storage_client") as mock_client:
            assert gcs.download_if_missing("MBL", cache_dir) is True
            mock_client.assert_not_called()


def test_gcs_meta_fingerprint_reads_meta(cache_dir):
    _, _, meta_path = gcs._local_paths("MBL", cache_dir)
    assert meta_path
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": "abc123"}, fh)
    assert gcs.gcs_meta_fingerprint("MBL", cache_dir) == "abc123"


def test_upload_async_no_bucket(cache_dir):
    index_path, labels_path, meta_path = gcs._local_paths("MBL", cache_dir)
    for p in (index_path, labels_path, meta_path):
        with open(p, "wb") as fh:
            fh.write(b"x")
    with patch.dict(os.environ, {"CCAPR_GCS_BUCKET": ""}, clear=False):
        gcs.upload_async("MBL", cache_dir)


def test_download_pulls_from_gcs(cache_dir):
    with patch.dict(os.environ, {"CCAPR_GCS_BUCKET": "test-bucket"}, clear=False):
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_to_filename.side_effect = lambda dest: open(dest, "wb").write(b"idx")

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch.object(gcs, "_storage_client", return_value=mock_client):
            assert gcs.download_if_missing("MBL", cache_dir) is True
            assert mock_blob.download_to_filename.call_count == 3
