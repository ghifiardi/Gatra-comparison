# v0.5 MORL (MO-PPO) Usage Guide

This repo now supports a contract-only multi-objective RL path for Architecture A.

## What it optimizes

`configs/morl.yaml` defines three default objectives:

- `detect`: reward for true positive alerts and penalty for missed threats.
- `fp_cost`: penalty for false positive alerts.
- `analyst_cost`: per-alert workload penalty.

MO-PPO trains one preference-conditioned policy. Input to the actor/critic is
`[state_v128 || preference_weights]`, where the weights live on a simplex.

## How to run

Quick run:

```bash
make run_morl_quick
```

Full run:

```bash
make train_morl
```

Direct CLI equivalent:

```bash
PYTHONPATH=. python3.11 -m runs.cli \
  --data-config configs/data.yaml \
  --iforest-config configs/iforest.yaml \
  --ppo-config configs/ppo.yaml \
  --eval-config configs/eval.yaml \
  --morl-config configs/morl.yaml \
  --out-root reports/runs
```

## Outputs

For run id `<run_id>`, MORL artifacts are written to:

- Model: `reports/runs/<run_id>/models/morl/`
- Evaluation: `reports/runs/<run_id>/eval/morl/`

Key files:

- `morl_results.json`: metrics/objective means per weight vector.
- `morl_table.csv`: flat table for analysis.
- `morl.md`: summary with Pareto candidates.
- `hypervolume.json`: dominated-volume scalar over chosen primary metrics.

## Interpreting Pareto and Hypervolume

Primary metrics in default config:

- maximize: `pr_auc`, `f1`
- minimize: `alerts_per_1k` (internally converted to utility for Pareto/HV)

A Pareto candidate is a weight setting where no other setting is better on all
primary metrics simultaneously.

Hypervolume gives a single scalar quality summary over the evaluated set
(relative to configured reference point). Larger is better.

## Recommended policy-selection rules

- High recall priority: select Pareto point with highest `recall` / `pr_auc`.
- Alert budget priority: select Pareto point minimizing `alerts_per_1k` while
  keeping `f1` above the operational floor.
- Balanced operations: select point near the Pareto "knee" where reducing
  alerts further causes a steep drop in detection metrics.

## Constraints

- MORL training/evaluation is contract-only.
- No BigQuery calls are required by MORL tests or evaluation.
- Existing PPO path remains available and unchanged by default.
