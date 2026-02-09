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
For local CSV data (no BigQuery auth required), run:

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

Strict override (for explicit hard-constraint behavior):

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  META_CONFIG=configs/meta_controller.yaml
```

Inspect these artifacts when fallback triggers:
- `reports/runs/<run_id>/eval/morl/meta_selection.json`
- `reports/runs/<run_id>/eval/morl/meta_feasibility.json`
- `reports/runs/<run_id>/eval/morl/morl_selected_test.json`

Additional guidance:
- `docs/v0.10_option2_production_default.md`

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
