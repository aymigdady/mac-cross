#!/usr/bin/env bash
# One-time (idempotent) GCP setup for mac-cross on Cloud Run:
#   - Secret Manager: cross-service token, anthropic API key placeholder
#   - GCS bucket IAM for HNSW index persistence
#   - IAM: Cloud Run + Cloud Build can read secrets
#
# Prerequisites: run cost-control deploy/setup-cloud-run-persistence.sh first (Redis + VPC connector).
#
# Usage:
#   ./deploy/setup-cloud-run.sh
#   GCP_PROJECT=mac-cost-intellegence REGION=us-central1 ./deploy/setup-cloud-run.sh
#
# After this:
#   1. Add your Anthropic key in Secret Manager → ccapr-anthropic-api-key
#   2. gcloud builds triggers run mac-cross-deploy --branch=main
set -euo pipefail

PROJECT="${GCP_PROJECT:-mac-cost-intellegence}"
REGION="${REGION:-us-central1}"
SERVICE="${CLOUD_RUN_SERVICE:-mac-cross}"
CONNECTOR_NAME="${VPC_CONNECTOR_NAME:-ccapr-run-connector}"
CROSS_TOKEN_SECRET="${CROSS_TOKEN_SECRET_NAME:-ccapr-cross-service-token}"
ANTHROPIC_SECRET="${ANTHROPIC_SECRET_NAME:-ccapr-anthropic-api-key}"
REDIS_SECRET="${REDIS_SECRET_NAME:-ccapr-redis-url}"
GCS_BUCKET="${CCAPR_GCS_BUCKET:-mac-cost-intellegence-ccapr-artifacts}"

echo "==> Project: $PROJECT  Region: $REGION"
gcloud config set project "$PROJECT" >/dev/null

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo "==> Enabling APIs (idempotent)..."
gcloud services enable \
  run.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project="$PROJECT" \
  --quiet

grant_secret() {
  local secret="$1"
  local member="$2"
  gcloud secrets add-iam-policy-binding "$secret" \
    --project="$PROJECT" \
    --member="$member" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null 2>&1 || true
}

echo "==> Secret: $CROSS_TOKEN_SECRET (shared with cost-control-app)"
if gcloud secrets describe "$CROSS_TOKEN_SECRET" --project="$PROJECT" &>/dev/null; then
  echo "    Already exists (not rotated)."
else
  CROSS_TOKEN="$(openssl rand -hex 32)"
  echo -n "$CROSS_TOKEN" | gcloud secrets create "$CROSS_TOKEN_SECRET" \
    --data-file=- --replication-policy=automatic --project="$PROJECT"
  echo "    Created. Same token must be mounted on cost-control-app and mac-cross."
fi

echo "==> Secret: $ANTHROPIC_SECRET (placeholder — add key in Console)"
if gcloud secrets describe "$ANTHROPIC_SECRET" --project="$PROJECT" &>/dev/null; then
  echo "    Already exists. Add/update key:"
  echo "      gcloud secrets versions add $ANTHROPIC_SECRET --data-file=- --project=$PROJECT"
else
  echo -n "REPLACE_ME" | gcloud secrets create "$ANTHROPIC_SECRET" \
    --data-file=- --replication-policy=automatic --project="$PROJECT"
  echo "    Created placeholder. Replace with real key:"
  echo "      echo -n 'sk-ant-...' | gcloud secrets versions add $ANTHROPIC_SECRET --data-file=- --project=$PROJECT"
fi

echo "==> IAM: secret accessor for Cloud Run + Cloud Build"
for SEC in "$CROSS_TOKEN_SECRET" "$ANTHROPIC_SECRET" "$REDIS_SECRET"; do
  if gcloud secrets describe "$SEC" --project="$PROJECT" &>/dev/null; then
    grant_secret "$SEC" "serviceAccount:${RUN_SA}"
    grant_secret "$SEC" "serviceAccount:${BUILD_SA}"
  else
    echo "    Skip $SEC (not found — run setup-cloud-run-persistence.sh for Redis secret)"
  fi
done

echo "==> GCS bucket: gs://$GCS_BUCKET (HNSW index persistence)"
if gcloud storage buckets describe "gs://${GCS_BUCKET}" --project="$PROJECT" &>/dev/null; then
  echo "    Bucket exists."
else
  echo "    Creating bucket..."
  gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --project="$PROJECT" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi

echo "==> IAM: Cloud Run SA objectAdmin on gs://${GCS_BUCKET}/hnsw/*"
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/storage.objectAdmin" \
  --project="$PROJECT" \
  --quiet >/dev/null 2>&1 || true

echo ""
echo "Done (mac-cross infra)."
echo "  Cross token secret: $CROSS_TOKEN_SECRET"
echo "  Anthropic secret:   $ANTHROPIC_SECRET (add real key before cross-match)"
echo "  GCS bucket:         gs://${GCS_BUCKET}"
echo ""
echo "Next:"
echo "  1. Add Anthropic API key to Secret Manager"
echo "  2. ./deploy/setup-gcb-autodeploy.sh   # or push to main on aymigdady/mac-cross"
echo "  3. gcloud builds triggers run mac-cross-deploy --branch=main --region=$REGION"
