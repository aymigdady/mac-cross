# MAC-CROSS

Cross-company PO search service for the MAC cost-control app: BM25, bge-m3 embeddings, HNSW recall, cross-encoder rerank, and Haiku description matching.

## Run locally (standalone)

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY for AI cross-match
docker compose up --build
```

Service: http://localhost:8081

## Run with MAC (cost-control-app-main-4)

Clone the main app and this repo as siblings:

```bash
git clone https://github.com/aymigdady/cost-control-app-main-4.git
git clone https://github.com/aymigdady/mac-cross.git MAC-CROSS
```

In the MAC repo `docker-compose.yml`, `mac-cross` is wired when `CCAPR_CROSS_SERVICE_URL=http://mac-cross:8080` and both stacks share the same Redis + `ccapr-emb-cache` volume.

Set the same `CCAPR_CROSS_SERVICE_TOKEN` in both services.

## GCS HNSW persistence (production)

MAC-CROSS is the **single writer** to GCS for HNSW index files (`.usearch`, `.labels.json`, `.meta.json`).

| Env var | Purpose |
|---------|---------|
| `CCAPR_GCS_BUCKET` | GCS bucket name (unset = GCS disabled, local-only behavior) |
| `CCAPR_HNSW_CACHE_DIR` | Writable local cache (e.g. `/tmp/ccapr-hnsw-cache` on Cloud Run) |
| `REDIS_URL` | Shared Memorystore (embedding vector cache) |

**Flow:** On cold start, `download_if_missing` pulls indexes from `gs://{bucket}/hnsw/{company}/`. After rebuild, `upload_async` pushes updated files. Fingerprint in `meta.json` (from session `erp_file_sha256`) skips rebuild when ERP unchanged.

### GCP setup (run manually)

```bash
PROJECT=mac-cost-intellegence
BUCKET=mac-cost-intellegence-ccapr-artifacts
REGION=us-central1
SA=$(gcloud projects describe $PROJECT --format='value(projectNumber)')-compute@developer.gserviceaccount.com

gcloud storage buckets create gs://$BUCKET \
  --project=$PROJECT --location=$REGION \
  --uniform-bucket-level-access --public-access-prevention

# Lifecycle: delete hnsw/ objects older than 90 days
cat > /tmp/ccapr-hnsw-lifecycle.json <<'EOF'
{"rule":[{"action":{"type":"Delete"},"condition":{"age":90,"matchesPrefix":["hnsw/"]}}]}
EOF
gcloud storage buckets update gs://$BUCKET --lifecycle-file=/tmp/ccapr-hnsw-lifecycle.json

gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
  --member="serviceAccount:$SA" --role="roles/storage.objectAdmin"
```

**mac-cross Cloud Run env:**

```text
CCAPR_GCS_BUCKET=mac-cost-intellegence-ccapr-artifacts
CCAPR_HNSW_CACHE_DIR=/tmp/ccapr-hnsw-cache
REDIS_URL=<Memorystore secret>
```

**cost-control-app** (no GCS upload): `CCAPR_CROSS_SERVICE_URL` + `CCAPR_SKIP_ML=1`.

See also [deploy/GCS-HNSW.md](deploy/GCS-HNSW.md).

## Cloud Run (production)

Deploy as a **separate Cloud Run service** (`mac-cross`) in project `mac-cost-intellegence`. The main app (`cost-control-app`) calls it via `CCAPR_CROSS_SERVICE_URL` + shared Bearer token.

### One-time setup

```bash
# Prerequisites: cost-control setup-cloud-run-persistence.sh (Redis + VPC connector)
./deploy/setup-cloud-run.sh
./deploy/setup-gcb-autodeploy.sh   # optional: auto-deploy on push to main

# Add Anthropic API key (required for cross-match):
#   GCP Console → Secret Manager → ccapr-anthropic-api-key → New version
```

### Deploy

```bash
gcloud builds triggers run mac-cross-deploy --branch=main --region=us-central1
# Then redeploy cost-control so it picks up mac-cross URL:
# gcloud builds triggers run cost-control-app-main-4-deploy --branch=main
```

Or push to `main` on `aymigdady/mac-cross` if the Cloud Build trigger is configured.

### Cloud Run resources

- 4Gi memory, 2 CPU, 300s timeout, max-instances 1
- Secrets: `REDIS_URL`, `ANTHROPIC_API_KEY`, `CCAPR_CROSS_SERVICE_TOKEN`
- Env: `CCAPR_GCS_BUCKET`, embedding/cross-search tuning (see `cloudbuild.yaml`)

### Verify

```bash
URL=$(gcloud run services describe mac-cross --region=us-central1 --format='value(status.url)')
curl -s "$URL/health"
curl -s -H "Authorization: Bearer $(gcloud secrets versions access latest --secret=ccapr-cross-service-token)" \
  "$URL/ready"
```

After cross compare in the app, check HNSW uploads:

```bash
gcloud storage ls gs://mac-cost-intellegence-ccapr-artifacts/hnsw/
```

## API (Bearer token if `CCAPR_CROSS_SERVICE_TOKEN` is set)

| Method | Path |
|--------|------|
| GET | `/health` |
| GET | `/ready` |
| GET | `/api/embeddings/status?sessionId=` |
| POST | `/api/embeddings/rebuild` `{ "sessionId": "..." }` |
| POST | `/api/cross-company/match` — body: sessionId, rows, items, newPoSourceTab, ccaprVendor, … |

## Manual smoke

1. Start MAC + MAC-CROSS via MAC `docker compose up`
2. Sign in, upload ERP, run cross-company compare (issuing company ≠ target tab)
3. Compare rows should include `cross_search_confidence_pct`
