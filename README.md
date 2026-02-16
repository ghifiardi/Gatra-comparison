# gatra-comparison

![CI](../../actions/workflows/ci.yml/badge.svg)

Head-to-head evaluation of:
- Architecture B: Isolation Forest baseline
- Architecture A: Actor-Critic RL (PPO) policy

## Quickstart
```bash
poetry install
make test
make train_b
make train_a
make eval
make serve
```

## MORL (v0.5)

Run the multi-objective PPO pipeline (contract-only):

```bash
make run_morl_quick
```

Detailed MORL usage and policy-selection guidance:
- `docs/morl.md`

Operational policy evaluation + join diagnostics quick run:
```bash
make run_morl_policy_quick
```

## Production Default (Option 2)

Production-default path uses relaxed meta-controller fallback and contract cache.

Operator quickstart (local CSV, no BigQuery auth required):

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

Operator quickstart (BigQuery source, requires ADC/service-account auth):

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_bigquery_gatra_prd_c335.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

Strict override (for explicit hard-constraint behavior):

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  META_CONFIG=configs/meta_controller.yaml
```

Artifact contract (minimum expected outputs for `run_morl_policy_quick`):
- `reports/runs/<run_id>/eval/morl/meta_selection.json`
- `reports/runs/<run_id>/eval/morl/meta_feasibility.json`
- `reports/runs/<run_id>/eval/morl/morl_selected_test.json`
- `reports/runs/<run_id>/report/run_manifest.json`
- `reports/runs/<run_id>/morl_selected_test.json`
- `reports/runs/<run_id>/classical_test.json`
- `reports/runs/<run_id>/statistical_analysis.json`
- `reports/runs/<run_id>/table1_statistical.tex`

If these are missing, likely causes:
- The run used a different `run_id` than expected.
- The run exited before meta-selection (check terminal output for first error).
- `--morl-config` / `--meta-config` was overridden to disable meta-controller behavior.

Additional guidance:
- `docs/v0.10_option2_production_default.md`

Run statistical significance analysis (paired test + bootstrap CI + correction):

```bash
make run_statistical_analysis RUN_ID=<run_id>
```

Pin canonical paper evidence artifacts for a run:

```bash
make paper_pin_evidence RUN_ID=<run_id>
```

Generate paper tables bundle for a run (statistical table + flattened results row):

```bash
make paper_tables RUN_ID=<run_id>
```

Generate paper-ready Results paragraph:

```bash
make paper_results_paragraph RUN_ID=<run_id>
```

Generate K-sweep artifacts (defaults: `50,100,200`):

```bash
make paper_k_sweep RUN_ID=<run_id>
```

## Queue Deploy (BigQuery)

Repeatable SQL deployment is versioned under `sql/`:
- `sql/10_deploy_prod_queue_top200.sql`
- `sql/20_verify_prod_queue.sql`
- `sql/30_deploy_safe_view.sql`

Canonical commands:

```bash
make deploy_queue BQ_PROJECT=gatra-prd-c335
make deploy_safe_view BQ_PROJECT=gatra-prd-c335
make verify_queue BQ_PROJECT=gatra-prd-c335
```

One-shot deploy + verify:

```bash
PROJECT_ID=gatra-prd-c335 bash scripts/deploy_queue.sh
```

## External Share (Static Sanitized Exports)

Generate daily sanitized CSV/JSON artifacts from the safe BigQuery view (no live backend required):

```bash
make export_sanitized_artifacts BQ_PROJECT=gatra-prd-c335
```

Output directory:
- `reports/public_exports/<snapshot_dt>/worklist_<snapshot_dt>.csv`
- `reports/public_exports/<snapshot_dt>/worklist_<snapshot_dt>.json`
- `reports/public_exports/<snapshot_dt>/daily_kpi_30d_<snapshot_dt>.csv`
- `reports/public_exports/<snapshot_dt>/daily_kpi_30d_<snapshot_dt>.json`
- `reports/public_exports/<snapshot_dt>/manifest.json`

Optional publish to GCS:

```bash
PROJECT_ID=gatra-prd-c335 PUBLISH_BUCKET=<bucket-name> bash scripts/export_sanitized_artifacts.sh
```

By default, uses a toy dataset generator (configs/data.yaml: dataset.source=toy).
Replace data loader in data/loaders.py to read parquet/csv/bq.

## BigQuery authentication (BigQuery-first mode)

This repo uses Application Default Credentials (ADC).

### Option A — ADC via gcloud (local dev)
```bash
gcloud auth application-default login
```

### Option B — Service account key (CI / servers)
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

### Smoke test (BigQuery access)
```bash
python -c "from google.cloud import bigquery; print(bigquery.Client().project)"
```

---
