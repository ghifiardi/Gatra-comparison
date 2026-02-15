#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/paper_matrix.sh [--csv] [--bq] [--all] [--dry-run] [--seeds LIST] [--bq-seeds LIST] [--index-out PATH] [--append-index]

Options:
  --csv                 Run CSV conditions (default if no mode is provided)
  --bq                  Run BigQuery conditions
  --all                 Run both CSV and BigQuery conditions
  --dry-run             Print commands and index rows without executing
  --seeds LIST          CSV seeds as comma-separated list (default: 42,1337,2026)
  --bq-seeds LIST       BigQuery seeds as comma-separated list (default: 42)
  --index-out PATH      Index CSV output (default: reports/paper_results/week1_run_index.csv)
  --append-index        Append rows to existing index file (write header only if file is new)
  --index PATH          Backward-compatible alias for --index-out
  -h, --help            Show this help
USAGE
}

RUN_CSV=0
RUN_BQ=0
DRY_RUN=0
CSV_SEEDS="42,1337,2026"
BQ_SEEDS="42"
INDEX_OUT="reports/paper_results/week1_run_index.csv"
APPEND_INDEX=0
INDEX_HEADER_WITH_GROUP='timestamp,run_group,condition,seed,backend,runner,data_config,morl_config,meta_config,robustness_config,meta_stability_config,run_dir,status,command'
INDEX_HEADER_BASE='timestamp,condition,seed,backend,runner,data_config,morl_config,meta_config,robustness_config,meta_stability_config,run_dir,status,command'
INDEX_HAS_RUN_GROUP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)
      RUN_CSV=1
      shift
      ;;
    --bq)
      RUN_BQ=1
      shift
      ;;
    --all)
      RUN_CSV=1
      RUN_BQ=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --seeds)
      CSV_SEEDS="$2"
      shift 2
      ;;
    --bq-seeds)
      BQ_SEEDS="$2"
      shift 2
      ;;
    --index-out|--index)
      INDEX_OUT="$2"
      shift 2
      ;;
    --append-index)
      APPEND_INDEX=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ $RUN_CSV -eq 0 && $RUN_BQ -eq 0 ]]; then
  RUN_CSV=1
fi

PY="${PY:-python3.11}"
OUT_ROOT="${OUT_ROOT:-reports/runs}"
CONTRACT_CACHE_ROOT="${CONTRACT_CACHE_ROOT:-reports/contracts_cache}"

DATA_CSV="${DATA_CSV:-configs/data_local_gatra_prd_c335.yaml}"
DATA_BQ="${DATA_BQ:-configs/data_bigquery_gatra_prd_c335.yaml}"
MORL_NORMALIZED="${MORL_NORMALIZED:-configs/morl_realdata_normalized.yaml}"
MORL_NORM_OFF="${MORL_NORM_OFF:-configs/morl_realdata.yaml}"
META_RELAXED="${META_RELAXED:-configs/meta_controller_relaxed.yaml}"
META_STRICT="${META_STRICT:-configs/meta_controller.yaml}"
ROBUST_REALISM_OFF="${ROBUST_REALISM_OFF:-configs/robustness_realism_off.yaml}"
ROBUST_R1="${ROBUST_R1:-configs/robustness_label_delay_7d_unknown.yaml}"
ROBUST_R2="${ROBUST_R2:-configs/robustness_label_delay_12h_benign.yaml}"
META_STABILITY_CFG="${META_STABILITY_CFG:-configs/meta_stability.yaml}"
PPO_CONFIG_BASE="${PPO_CONFIG_BASE:-configs/ppo.yaml}"
IFOREST_CONFIG_BASE="${IFOREST_CONFIG_BASE:-configs/iforest.yaml}"

init_index() {
  mkdir -p "$(dirname "$INDEX_OUT")"
  if [[ $APPEND_INDEX -eq 1 && -s "$INDEX_OUT" ]]; then
    local header
    header="$(head -n1 "$INDEX_OUT")"
    if [[ "$header" == "$INDEX_HEADER_WITH_GROUP" ]]; then
      INDEX_HAS_RUN_GROUP=1
    elif [[ "$header" == "$INDEX_HEADER_BASE" ]]; then
      INDEX_HAS_RUN_GROUP=0
    else
      echo "Unsupported index header in $INDEX_OUT: $header" >&2
      exit 2
    fi
    return
  fi

  INDEX_HAS_RUN_GROUP=1
  printf '%s\n' "$INDEX_HEADER_WITH_GROUP" > "$INDEX_OUT"
}

append_index_row() {
  local run_group="$1"
  local condition="$2"
  local seed="$3"
  local backend="$4"
  local runner="$5"
  local data_config="$6"
  local morl_config="$7"
  local meta_config="$8"
  local robustness_config="$9"
  local meta_stability_config="${10}"
  local run_dir="${11}"
  local status="${12}"
  local cmd_str="${13}"

  if [[ $INDEX_HAS_RUN_GROUP -eq 1 ]]; then
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$(csv_escape "$run_group")" \
      "$(csv_escape "$condition")" \
      "$(csv_escape "$seed")" \
      "$(csv_escape "$backend")" \
      "$(csv_escape "$runner")" \
      "$(csv_escape "$data_config")" \
      "$(csv_escape "$morl_config")" \
      "$(csv_escape "$meta_config")" \
      "$(csv_escape "$robustness_config")" \
      "$(csv_escape "$meta_stability_config")" \
      "$(csv_escape "$run_dir")" \
      "$(csv_escape "$status")" \
      "$(csv_escape "$cmd_str")" >> "$INDEX_OUT"
  else
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$(csv_escape "$condition")" \
      "$(csv_escape "$seed")" \
      "$(csv_escape "$backend")" \
      "$(csv_escape "$runner")" \
      "$(csv_escape "$data_config")" \
      "$(csv_escape "$morl_config")" \
      "$(csv_escape "$meta_config")" \
      "$(csv_escape "$robustness_config")" \
      "$(csv_escape "$meta_stability_config")" \
      "$(csv_escape "$run_dir")" \
      "$(csv_escape "$status")" \
      "$(csv_escape "$cmd_str")" >> "$INDEX_OUT"
  fi
}

TMP_CFG_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/paper_matrix_cfg.XXXXXX")"
cleanup() {
  rm -rf "$TMP_CFG_ROOT"
}
trap cleanup EXIT

csv_escape() {
  local s="$1"
  s=${s//\"/\"\"}
  printf '"%s"' "$s"
}

create_seeded_configs() {
  local seed="$1"
  local out_dir="$TMP_CFG_ROOT/seed_${seed}_$RANDOM"
  mkdir -p "$out_dir"
  local out_ppo="$out_dir/ppo.yaml"
  local out_if="$out_dir/iforest.yaml"

  "$PY" - "$seed" "$PPO_CONFIG_BASE" "$IFOREST_CONFIG_BASE" "$out_ppo" "$out_if" <<'PY'
import sys
from typing import Any

import yaml

seed = int(sys.argv[1])
ppo_in = sys.argv[2]
if_in = sys.argv[3]
ppo_out = sys.argv[4]
if_out = sys.argv[5]

with open(ppo_in, "r") as f:
    ppo_cfg: dict[str, Any] = yaml.safe_load(f)

ppo_cfg.setdefault("rl", {})["seed"] = seed

with open(ppo_out, "w") as f:
    yaml.safe_dump(ppo_cfg, f, sort_keys=False)

with open(if_in, "r") as f:
    if_cfg: dict[str, Any] = yaml.safe_load(f)

if_cfg.setdefault("model", {})["random_state"] = seed

with open(if_out, "w") as f:
    yaml.safe_dump(if_cfg, f, sort_keys=False)
PY

  echo "$out_ppo|$out_if"
}

run_and_record() {
  local condition="$1"
  local seed="$2"
  local backend="$3"
  local runner="$4"
  local data_config="$5"
  local morl_config="$6"
  local meta_config="$7"
  local robustness_config="$8"
  local meta_stability_config="$9"
  shift 9

  local -a cmd=("$@")
  local cmd_str
  cmd_str="$(printf '%q ' "${cmd[@]}")"
  local run_dir=""
  local status="planned"
  local run_group="$backend"
  if [[ "$backend" == "bigquery" ]]; then
    run_group="bq"
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] $condition seed=$seed backend=$backend"
    echo "  $cmd_str"
  else
    echo "[RUN] $condition seed=$seed backend=$backend"
    local log_file
    log_file="$(mktemp "${TMPDIR:-/tmp}/paper_matrix_run.XXXXXX.log")"
    if "${cmd[@]}" 2>&1 | tee "$log_file"; then
      status="ok"
      run_dir="$(grep -Eo 'Run complete -> reports/runs/[^[:space:]]+' "$log_file" | tail -n1 | awk '{print $4}')"
      if [[ -z "$run_dir" ]]; then
        status="parse_error"
      fi
    else
      status="failed"
    fi
    rm -f "$log_file"
    if [[ "$status" != "ok" ]]; then
      echo "Condition failed: $condition seed=$seed status=$status" >&2
      append_index_row \
        "$run_group" \
        "$condition" \
        "$seed" \
        "$backend" \
        "$runner" \
        "$data_config" \
        "$morl_config" \
        "$meta_config" \
        "$robustness_config" \
        "$meta_stability_config" \
        "$run_dir" \
        "$status" \
        "$cmd_str"
      exit 1
    fi
  fi

  append_index_row \
    "$run_group" \
    "$condition" \
    "$seed" \
    "$backend" \
    "$runner" \
    "$data_config" \
    "$morl_config" \
    "$meta_config" \
    "$robustness_config" \
    "$meta_stability_config" \
    "$run_dir" \
    "$status" \
    "$cmd_str"
}

run_csv_matrix() {
  local seed="$1"
  local cfg_pair
  cfg_pair="$(create_seeded_configs "$seed")"
  local ppo_cfg="${cfg_pair%%|*}"
  local if_cfg="${cfg_pair##*|}"

  run_and_record "A1_csv_option2_default" "$seed" "csv" "make" \
    "$DATA_CSV" "$MORL_NORMALIZED" "$META_RELAXED" "" "" \
    make run_morl_policy_quick \
      "DATA_CONFIG=$DATA_CSV" \
      "MORL_CONFIG=$MORL_NORMALIZED" \
      "META_CONFIG=$META_RELAXED" \
      "IFOREST_CONFIG=$if_cfg" \
      "PPO_CONFIG=$ppo_cfg" \
      "CONTRACT_CACHE_ROOT=$CONTRACT_CACHE_ROOT" \
      "OUT_ROOT=$OUT_ROOT"

  run_and_record "B1_csv_normalization_off" "$seed" "csv" "make" \
    "$DATA_CSV" "$MORL_NORM_OFF" "$META_RELAXED" "" "" \
    make run_morl_policy_quick \
      "DATA_CONFIG=$DATA_CSV" \
      "MORL_CONFIG=$MORL_NORM_OFF" \
      "META_CONFIG=$META_RELAXED" \
      "IFOREST_CONFIG=$if_cfg" \
      "PPO_CONFIG=$ppo_cfg" \
      "CONTRACT_CACHE_ROOT=$CONTRACT_CACHE_ROOT" \
      "OUT_ROOT=$OUT_ROOT"

  run_and_record "B2_csv_meta_strict" "$seed" "csv" "make" \
    "$DATA_CSV" "$MORL_NORMALIZED" "$META_STRICT" "" "" \
    make run_morl_policy_quick \
      "DATA_CONFIG=$DATA_CSV" \
      "MORL_CONFIG=$MORL_NORMALIZED" \
      "META_CONFIG=$META_STRICT" \
      "IFOREST_CONFIG=$if_cfg" \
      "PPO_CONFIG=$ppo_cfg" \
      "CONTRACT_CACHE_ROOT=$CONTRACT_CACHE_ROOT" \
      "OUT_ROOT=$OUT_ROOT"

  run_and_record "B3_csv_realism_off" "$seed" "csv" "make" \
    "$DATA_CSV" "$MORL_NORMALIZED" "$META_RELAXED" "$ROBUST_REALISM_OFF" "" \
    make run_morl_policy_robust_quick \
      "DATA_CONFIG=$DATA_CSV" \
      "MORL_CONFIG=$MORL_NORMALIZED" \
      "META_CONFIG=$META_RELAXED" \
      "ROBUSTNESS_CONFIG=$ROBUST_REALISM_OFF" \
      "IFOREST_CONFIG=$if_cfg" \
      "PPO_CONFIG=$ppo_cfg" \
      "CONTRACT_CACHE_ROOT=$CONTRACT_CACHE_ROOT" \
      "OUT_ROOT=$OUT_ROOT"

  run_and_record "R1_csv_label_delay_7d_unknown" "$seed" "csv" "make" \
    "$DATA_CSV" "$MORL_NORMALIZED" "$META_RELAXED" "$ROBUST_R1" "" \
    make run_morl_policy_robust_quick \
      "DATA_CONFIG=$DATA_CSV" \
      "MORL_CONFIG=$MORL_NORMALIZED" \
      "META_CONFIG=$META_RELAXED" \
      "ROBUSTNESS_CONFIG=$ROBUST_R1" \
      "IFOREST_CONFIG=$if_cfg" \
      "PPO_CONFIG=$ppo_cfg" \
      "CONTRACT_CACHE_ROOT=$CONTRACT_CACHE_ROOT" \
      "OUT_ROOT=$OUT_ROOT"

  run_and_record "R2_csv_label_delay_12h_benign" "$seed" "csv" "make" \
    "$DATA_CSV" "$MORL_NORMALIZED" "$META_RELAXED" "$ROBUST_R2" "" \
    make run_morl_policy_robust_quick \
      "DATA_CONFIG=$DATA_CSV" \
      "MORL_CONFIG=$MORL_NORMALIZED" \
      "META_CONFIG=$META_RELAXED" \
      "ROBUSTNESS_CONFIG=$ROBUST_R2" \
      "IFOREST_CONFIG=$if_cfg" \
      "PPO_CONFIG=$ppo_cfg" \
      "CONTRACT_CACHE_ROOT=$CONTRACT_CACHE_ROOT" \
      "OUT_ROOT=$OUT_ROOT"

  run_and_record "C1_csv_meta_stability" "$seed" "csv" "make" \
    "$DATA_CSV" "$MORL_NORMALIZED" "$META_RELAXED" "" "$META_STABILITY_CFG" \
    make run_meta_stability_quick \
      "DATA_CONFIG=$DATA_CSV" \
      "MORL_CONFIG=$MORL_NORMALIZED" \
      "META_CONFIG=$META_RELAXED" \
      "META_STABILITY_CONFIG=$META_STABILITY_CFG" \
      "IFOREST_CONFIG=$if_cfg" \
      "PPO_CONFIG=$ppo_cfg" \
      "CONTRACT_CACHE_ROOT=$CONTRACT_CACHE_ROOT" \
      "OUT_ROOT=$OUT_ROOT"
}

run_bq_matrix() {
  local seed="$1"
  local cfg_pair
  cfg_pair="$(create_seeded_configs "$seed")"
  local ppo_cfg="${cfg_pair%%|*}"
  local if_cfg="${cfg_pair##*|}"

  run_and_record "A2_bq_option2_replication" "$seed" "bigquery" "make" \
    "$DATA_BQ" "$MORL_NORMALIZED" "$META_RELAXED" "" "" \
    make run_morl_policy_quick \
      "DATA_CONFIG=$DATA_BQ" \
      "MORL_CONFIG=$MORL_NORMALIZED" \
      "META_CONFIG=$META_RELAXED" \
      "IFOREST_CONFIG=$if_cfg" \
      "PPO_CONFIG=$ppo_cfg" \
      "CONTRACT_CACHE_ROOT=$CONTRACT_CACHE_ROOT" \
      "OUT_ROOT=$OUT_ROOT"
}

IFS=',' read -r -a CSV_SEED_ARR <<< "$CSV_SEEDS"
IFS=',' read -r -a BQ_SEED_ARR <<< "$BQ_SEEDS"
init_index

if [[ $RUN_CSV -eq 1 ]]; then
  for seed in "${CSV_SEED_ARR[@]}"; do
    run_csv_matrix "$seed"
  done
fi

if [[ $RUN_BQ -eq 1 ]]; then
  for seed in "${BQ_SEED_ARR[@]}"; do
    run_bq_matrix "$seed"
  done
fi

echo "Wrote index: $INDEX_OUT"
