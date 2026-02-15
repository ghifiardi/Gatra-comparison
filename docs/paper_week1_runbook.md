# v0.11 Week-1 Paper Runbook

This runbook defines a deterministic, copy-paste experiment matrix for paper-ready Week-1 runs.

## Scope

- Option 2 production-default behavior
  - relaxed meta-controller
  - contract cache
  - artifact contract outputs
- Backends:
  - CSV main runs
  - BigQuery replication runs
- Reproducibility:
  - 3 seeds for CSV conditions
  - 1 seed for BigQuery by default (optional 3-seed mode)

## Prerequisites

From repo root:

```bash
poetry install -E bigquery
make lint
make test
```

BigQuery auth (required only for BQ runs):

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project gatra-prd-c335
```

## Canonical Configs

- CSV data: `configs/data_local_gatra_prd_c335.yaml`
- BigQuery data: `configs/data_bigquery_gatra_prd_c335.yaml`
- MORL normalized: `configs/morl_realdata_normalized.yaml`
- MORL norm off: `configs/morl_realdata.yaml`
- META relaxed: `configs/meta_controller_relaxed.yaml`
- META strict: `configs/meta_controller.yaml`
- Robustness (base): `configs/robustness.yaml`
- Meta-stability: `configs/meta_stability.yaml`
- Cache root default: `reports/contracts_cache`

Week-1 helper robustness configs:

- Realism OFF (disable created-at label gating): `configs/robustness_realism_off.yaml`
- R1 label delay variant (7d unknown): `configs/robustness_label_delay_7d_unknown.yaml`
- R2 label delay variant (12h benign): `configs/robustness_label_delay_12h_benign.yaml`

## Experiment Matrix

- A1 CSV Option2 default (Primary)
  - normalized + relaxed + realism ON
- A2 BQ Option2 default (Primary replication)
  - normalized + relaxed + realism ON
- B1 CSV normalization OFF (Ablation)
- B2 CSV meta strict (Ablation)
- B3 CSV realism OFF (Ablation)
  - created-at label availability gating disabled via robustness config flag
- R1 CSV robustness label-delay variant (7d unknown)
- R2 CSV robustness label-delay variant (12h benign)
- C1 CSV meta-stability suite

## Run Policy

- CSV conditions: seeds `42,1337,2026`
- BQ conditions: seed `42` by default
- Optional BQ 3-seed mode: `42,1337,2026`

## One-Command Runner

Dry-run (print planned matrix):

```bash
bash scripts/paper_matrix.sh --csv --dry-run --index-out reports/paper_results/week1_run_index_csv.csv
```

Run full CSV matrix (3 seeds):

```bash
bash scripts/paper_matrix.sh --csv --seeds 42,1337,2026 --index-out reports/paper_results/week1_run_index_csv.csv
```

Run BigQuery replication (1 seed):

```bash
bash scripts/paper_matrix.sh --bq --bq-seeds 42 --index-out reports/paper_results/week1_run_index_bq.csv
```

Run both:

```bash
bash scripts/paper_matrix.sh --all --seeds 42,1337,2026 --bq-seeds 42 --index-out reports/paper_results/week1_run_index.csv
```

The script writes an index:

- `reports/paper_results/week1_run_index.csv`
- Recommended split indices for separate runs:
  - `reports/paper_results/week1_run_index_csv.csv`
  - `reports/paper_results/week1_run_index_bq.csv`

If you need to append to an existing index:

```bash
bash scripts/paper_matrix.sh --bq --bq-seeds 42 \
  --index-out reports/paper_results/week1_run_index_csv.csv \
  --append-index
```

## Direct Canonical Commands (single run)

A1 CSV main:

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata_normalized.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

A2 BigQuery replication:

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_bigquery_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata_normalized.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

B1 normalization OFF:

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

B2 strict meta:

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata_normalized.yaml \
  META_CONFIG=configs/meta_controller.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

B3 realism OFF (created-at gating disabled):

```bash
make run_morl_policy_robust_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata_normalized.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  ROBUSTNESS_CONFIG=configs/robustness_realism_off.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

R1 robustness delay 7d unknown:

```bash
make run_morl_policy_robust_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata_normalized.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  ROBUSTNESS_CONFIG=configs/robustness_label_delay_7d_unknown.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

R2 robustness delay 12h benign:

```bash
make run_morl_policy_robust_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata_normalized.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  ROBUSTNESS_CONFIG=configs/robustness_label_delay_12h_benign.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

C1 meta-stability:

```bash
make run_meta_stability_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata_normalized.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  META_STABILITY_CONFIG=configs/meta_stability.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

## Artifact Contract (Minimum)

For each main/ablation run directory `reports/runs/<run_id>/`:

- `eval/morl/meta_selection.json`
- `eval/morl/meta_feasibility.json`
- `eval/morl/morl_selected_test.json`
- `report/run_manifest.json`

For C1 additionally:

- `eval/meta_stability/meta_stability.json`
- `eval/meta_stability/meta_stability_table.csv`
- `eval/meta_stability/meta_stability.md`

## Collect Results to Single CSV

Using separate index files generated by CSV/BQ runners:

```bash
python scripts/collect_paper_results.py \
  --index reports/paper_results/week1_run_index_csv.csv \
  --index reports/paper_results/week1_run_index_bq.csv \
  --out reports/paper_results/paper_week1_results.csv
```

Glob mode is also supported:

```bash
python scripts/collect_paper_results.py \
  --index-glob "reports/paper_results/week1_run_index*.csv" \
  --out reports/paper_results/paper_week1_results.csv
```

Output:

- `reports/paper_results/paper_week1_results.csv`

## Notes

- Single-class test windows can produce `null`/N/A AUC values. Collector preserves these as empty fields; it does not fail.
- Contract cache behavior is expected:
  - first run (new key): cache miss
  - repeat run (same key): cache hit
