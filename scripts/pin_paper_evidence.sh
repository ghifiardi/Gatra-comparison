#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-}"
OUT_ROOT="${OUT_ROOT:-reports/paper_artifacts}"

if [[ -z "${RUN_ID}" ]]; then
  echo "RUN_ID is required (example: RUN_ID=20260216T083354Z)" >&2
  exit 1
fi

RUN_DIR="reports/runs/${RUN_ID}"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"

required=(
  "statistical_analysis.json"
  "table1_statistical.tex"
  "morl_selected_test.json"
  "classical_test.json"
)

mkdir -p "${OUT_DIR}"

for rel in "${required[@]}"; do
  src="${RUN_DIR}/${rel}"
  if [[ ! -f "${src}" ]]; then
    echo "Missing required artifact: ${src}" >&2
    exit 1
  fi
  cp "${src}" "${OUT_DIR}/${rel}"
done

{
  echo "{"
  echo "  \"run_id\": \"${RUN_ID}\","
  echo "  \"source_run_dir\": \"${RUN_DIR}\","
  echo "  \"files\": ["
  for i in "${!required[@]}"; do
    sep=","
    if [[ "${i}" -eq "$(( ${#required[@]} - 1 ))" ]]; then
      sep=""
    fi
    echo "    \"${required[$i]}\"${sep}"
  done
  echo "  ]"
  echo "}"
} > "${OUT_DIR}/evidence_manifest.json"

echo "Pinned evidence artifacts to: ${OUT_DIR}"
