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
