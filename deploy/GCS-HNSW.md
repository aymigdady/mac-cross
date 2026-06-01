# GCS HNSW index persistence

MAC-CROSS persists per-company usearch indexes to GCS so Cloud Run cold starts avoid full embedding rebuilds.

## Architecture

- **Redis (Memorystore):** per-description bge-m3 vector cache (`ccapr:emb:m3:*`)
- **Local disk (`CCAPR_HNSW_CACHE_DIR`):** live mmap index for search
- **GCS (`CCAPR_GCS_BUCKET`):** durable copy of the three files per company

Object layout: `hnsw/{company_lower}/{base}.{ext}` where `base` is e.g. `master-po-mbl.v1`.

## Code

| Module | Role |
|--------|------|
| `embeddings/gcs_hnsw_backend.py` | `download_if_missing`, `upload_async`, `gcs_meta_fingerprint` |
| `embeddings/builder.py` | Download before open; fingerprint skip; upload after final persist |
| `embeddings/hybrid.py` | Download before `get_company_store` open (query path) |
| `cross_routes.py` | Passes `erp_file_sha256` as fingerprint on rebuild |

When `CCAPR_GCS_BUCKET` is unset, all GCS functions no-op — behavior matches pre-GCS code.

## Manual smoke

1. Set `CCAPR_GCS_BUCKET` and deploy mac-cross with `storage.objectAdmin` on the bucket.
2. Upload ERP via MAC → triggers `/api/embeddings/rebuild`.
3. Verify objects: `gcloud storage ls gs://mac-cost-intellegence-ccapr-artifacts/hnsw/MBL/`
4. Redeploy mac-cross (new revision) → cross-search should work without full rebuild if fingerprint matches.
