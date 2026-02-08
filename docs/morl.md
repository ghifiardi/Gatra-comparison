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

## Meta-controller (v0.5.1)

The meta-controller selects a single weight vector `w*` from candidate weights
using **VAL** results and then evaluates `w*` on **TEST**.

Quick meta-selection run:

```bash
make run_meta_quick
```

Full meta-selection run:

```bash
make meta_select
```

Direct CLI equivalent:

```bash
PYTHONPATH=. python3.11 -m runs.cli \
  --data-config configs/data.yaml \
  --iforest-config configs/iforest.yaml \
  --ppo-config configs/ppo.yaml \
  --eval-config configs/eval.yaml \
  --morl-config configs/morl.yaml \
  --meta-config configs/meta_controller.yaml \
  --out-root reports/runs
```

Selection methods in `configs/meta_controller.yaml`:

- `greedy`: pick best feasible candidate by primary metric; tie-break by lower
  `alerts_per_1k`, then lexicographic `w`.
- `bandit_ucb`: deterministic seeded UCB with pseudo-noisy observations.
- `bandit_thompson`: deterministic seeded Gaussian Thompson sampling.

## Real-data objectives (v0.6)

`configs/morl_realdata.yaml` adds Level-1 objectives derived from contract arrays:

- `time_to_triage_seconds` (TTT): per-session `max(ts)-min(ts)` and minimized in reward.
- `detection_coverage`: novelty of `(page, action)` within session and maximized in reward.

The MORL path remains contract-only. If required split arrays are missing, training/eval
falls back to synthetic objective behavior.

Real-data quick run with local CSV export (data files are intentionally not committed):

```bash
make run_morl_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata.yaml
```

Direct CLI equivalent:

```bash
PYTHONPATH=. python3.11 -m runs.cli \
  --data-config configs/data_local_gatra_prd_c335.yaml \
  --iforest-config configs/iforest.yaml \
  --ppo-config configs/ppo.yaml \
  --eval-config configs/eval.yaml \
  --morl-config configs/morl_realdata.yaml \
  --meta-config configs/meta_controller.yaml \
  --out-root reports/runs \
  --quick
```

## Objective normalization controls (v0.6.1)

v0.6.1 adds objective-level normalization/capping controls for real-data MORL.
This keeps TTT (seconds) and coverage (counts/rates) on comparable scales.

Baseline (unnormalized) config:

```bash
make run_morl_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata.yaml
```

Normalized config:

```bash
make run_morl_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  MORL_CONFIG=configs/morl_realdata_normalized.yaml
```

Normalization behavior in `configs/morl_realdata_normalized.yaml`:

- objective-level `norm` and `cap_pctl` controls
- reference stats from `val` split only
- same stats applied to configured splits (`val`, `test`)
- optional persistence to `objectives_norm.json`

The MORL path remains contract-only with no BigQuery runtime calls.

## Outputs

For run id `<run_id>`, MORL artifacts are written to:

- Model: `reports/runs/<run_id>/models/morl/`
- Evaluation: `reports/runs/<run_id>/eval/morl/`

Key files:

- `morl_results_val.json`: sweep results on VAL (used for meta selection).
- `morl_results_test.json`: sweep results on TEST.
- `morl_selected_test.json`: single TEST evaluation for selected `w*`.
- `meta_selection.json`: selected weight + method + constraints + trace.
- `meta_selection.md`: human-readable selection report.
- `morl_table_test.csv` / `morl_test.md`: TEST sweep tabular/markdown summaries.
- `../contract/objectives_{train,val,test}.npz`: per-row real-data objective signals.
- `../contract/objectives_meta.json`: objective definitions + normalization stats.
- `../contract/objectives_norm.json`: persisted reference normalization statistics.

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
- Existing MORL sweep without meta-controller remains available when
  `--meta-config` is not provided.
