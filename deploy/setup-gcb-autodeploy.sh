#!/usr/bin/env bash
# Create Cloud Build trigger: auto build + deploy mac-cross on every push to main.
# Repo: aymigdady/mac-cross  |  Config: cloudbuild.yaml (repo root)
set -euo pipefail

PROJECT="${GCP_PROJECT:-mac-cost-intellegence}"
REGION="${GCP_REGION:-us-central1}"
TRIGGER_NAME="${GCB_TRIGGER_NAME:-mac-cross-deploy}"
REPO_OWNER="${GITHUB_REPO_OWNER:-aymigdady}"
REPO_NAME="${GITHUB_REPO_NAME:-mac-cross}"
BRANCH_PATTERN='^main$'

echo "==> Project: $PROJECT  Region: $REGION"
gcloud config set project "$PROJECT" >/dev/null

BILLING=$(gcloud billing projects describe "$PROJECT" --format='value(billingEnabled)' 2>/dev/null || echo "unknown")
if [[ "$BILLING" != "True" ]]; then
  echo "ERROR: Billing is not enabled on $PROJECT."
  exit 1
fi

if gcloud builds triggers describe "$TRIGGER_NAME" --region="$REGION" --format='value(name)' &>/dev/null; then
  echo "==> Trigger '$TRIGGER_NAME' already exists."
else
  echo "==> Creating trigger '$TRIGGER_NAME' for $REPO_OWNER/$REPO_NAME (branch $BRANCH_PATTERN)..."
  gcloud builds triggers create github \
    --name="$TRIGGER_NAME" \
    --repo-owner="$REPO_OWNER" \
    --repo-name="$REPO_NAME" \
    --branch-pattern="$BRANCH_PATTERN" \
    --build-config=cloudbuild.yaml \
    --region="$REGION" \
    --description="Auto build and deploy mac-cross to Cloud Run on push to main"
fi

echo ""
echo "Done. Every push to main will deploy Cloud Run service: mac-cross ($REGION)"
echo ""
echo "Test manually:"
echo "  gcloud builds triggers run $TRIGGER_NAME --branch=main --region=$REGION"
