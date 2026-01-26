# gatra-comparison

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

---
