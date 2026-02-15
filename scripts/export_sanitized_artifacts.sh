#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gatra-prd-c335}"
DATASET="${DATASET:-gatra_database}"
SAFE_VIEW="${SAFE_VIEW:-vw_ada_queue_streamlit_safe}"
OUT_ROOT="${OUT_ROOT:-reports/public_exports}"
TOP_K="${TOP_K:-200}"
SNAPSHOT_DT="${SNAPSHOT_DT:-}"
PUBLISH_BUCKET="${PUBLISH_BUCKET:-}"

query() {
  bq --project_id="${PROJECT_ID}" query --use_legacy_sql=false --quiet "$@"
}

if [[ -z "${SNAPSHOT_DT}" ]]; then
  SNAPSHOT_DT="$(query --format=csv \
    "SELECT CAST(MAX(snapshot_dt) AS STRING) AS snapshot_dt FROM \`${PROJECT_ID}.${DATASET}.${SAFE_VIEW}\`" \
    | tail -n 1)"
fi

if [[ -z "${SNAPSHOT_DT}" || "${SNAPSHOT_DT}" == "NULL" ]]; then
  echo "No snapshot date found in ${PROJECT_ID}.${DATASET}.${SAFE_VIEW}" >&2
  exit 1
fi

OUT_DIR="${OUT_ROOT}/${SNAPSHOT_DT}"
mkdir -p "${OUT_DIR}"

WORKLIST_SQL="
SELECT
  snapshot_dt,
  rk,
  alarm_key_hash,
  predicted_y,
  prob_y0,
  prob_y1,
  scored_at
FROM \`${PROJECT_ID}.${DATASET}.${SAFE_VIEW}\`
WHERE snapshot_dt = DATE('${SNAPSHOT_DT}')
ORDER BY rk ASC
LIMIT ${TOP_K}
"

KPI_SQL="
SELECT
  snapshot_dt,
  COUNT(*) AS queue_size,
  AVG(prob_y0) AS avg_prob_y0,
  MIN(prob_y0) AS min_prob_y0,
  MAX(prob_y0) AS max_prob_y0
FROM \`${PROJECT_ID}.${DATASET}.${SAFE_VIEW}\`
WHERE snapshot_dt >= DATE_SUB(DATE('${SNAPSHOT_DT}'), INTERVAL 30 DAY)
GROUP BY snapshot_dt
ORDER BY snapshot_dt DESC
"

query --format=csv "${WORKLIST_SQL}" > "${OUT_DIR}/worklist_${SNAPSHOT_DT}.csv"
query --format=json "${WORKLIST_SQL}" > "${OUT_DIR}/worklist_${SNAPSHOT_DT}.json"
query --format=csv "${KPI_SQL}" > "${OUT_DIR}/daily_kpi_30d_${SNAPSHOT_DT}.csv"
query --format=json "${KPI_SQL}" > "${OUT_DIR}/daily_kpi_30d_${SNAPSHOT_DT}.json"

cat > "${OUT_DIR}/manifest.json" <<EOF
{
  "project_id": "${PROJECT_ID}",
  "dataset": "${DATASET}",
  "safe_view": "${SAFE_VIEW}",
  "snapshot_dt": "${SNAPSHOT_DT}",
  "top_k": ${TOP_K},
  "files": [
    "worklist_${SNAPSHOT_DT}.csv",
    "worklist_${SNAPSHOT_DT}.json",
    "daily_kpi_30d_${SNAPSHOT_DT}.csv",
    "daily_kpi_30d_${SNAPSHOT_DT}.json"
  ]
}
EOF

if [[ -n "${PUBLISH_BUCKET}" ]]; then
  gsutil -m cp "${OUT_DIR}"/* "gs://${PUBLISH_BUCKET}/ada-topk/${SNAPSHOT_DT}/"
  echo "Published to gs://${PUBLISH_BUCKET}/ada-topk/${SNAPSHOT_DT}/"
fi

echo "Exported sanitized artifacts to ${OUT_DIR}"
