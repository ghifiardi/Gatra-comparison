#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gatra-prd-c335}"

echo "[deploy_queue] project=${PROJECT_ID}"
echo "[deploy_queue] deploying Top-200 queue table..."
bq --project_id="${PROJECT_ID}" query --use_legacy_sql=false < sql/10_deploy_prod_queue_top200.sql

echo "[deploy_queue] deploying safe Streamlit view..."
bq --project_id="${PROJECT_ID}" query --use_legacy_sql=false < sql/30_deploy_safe_view.sql

echo "[deploy_queue] running verification queries..."
bq --project_id="${PROJECT_ID}" query --use_legacy_sql=false < sql/20_verify_prod_queue.sql

echo "[deploy_queue] done."
