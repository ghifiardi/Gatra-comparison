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
  --meta-config configs/meta_controller_relaxed.yaml \
  --out-root reports/runs
```

Selection methods in `configs/meta_controller_relaxed.yaml` (default via Makefile):

- `greedy`: pick best feasible candidate by primary metric; tie-break by lower
  `alerts_per_1k`, then lexicographic `w`.
- `bandit_ucb`: deterministic seeded UCB with pseudo-noisy observations.
- `bandit_thompson`: deterministic seeded Gaussian Thompson sampling.

Strict override is available with `configs/meta_controller.yaml` when you want hard
constraint behavior without the relaxed fallback schedule.

### When constraints are infeasible (v0.5.2)

On small or heavily imbalanced datasets, strict constraints can produce zero feasible
weights. v0.5.2 adds deterministic fallback handling so runs do not fail by default:

- `constraints.mode: hard | soft`
- `relaxation.enabled: true | false`
- ordered `relaxation.schedule` steps to progressively relax constraints
- `fail_on_infeasible: false` (default) keeps pipeline running and emits diagnostics

Behavior:

1. Try original constraints.
2. If none feasible, apply relaxation steps in order.
3. If still none feasible, select best-effort candidate using penalty-based soft scoring.

Artifacts written under `eval/morl/`:

- `meta_selection.json`
- `meta_feasibility.json`
- `meta_selection.md`

`meta_feasibility.json` includes feasible counts per relax step, fallback mode, final
constraints used, and violated-constraint summary for auditability.

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
  --meta-config configs/meta_controller_relaxed.yaml \
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

Statistical comparison artifacts (MORL selected vs classical IF baseline):

- `../../morl_selected_test.json`: selected MORL TEST payload with bootstrap `metric_samples`.
- `../../classical_test.json`: baseline TEST payload with bootstrap `metric_samples`.
- `../../statistical_analysis.json`: paired test summary (`p_value`, `p_adj`, CI, effect size, power).
- `../../table1_statistical.tex`: paper-ready LaTeX table.

Manual rerun command:

```bash
make run_statistical_analysis RUN_ID=<run_id>
```

Paper packaging commands:

```bash
make paper_pin_evidence RUN_ID=<run_id>
make paper_tables RUN_ID=<run_id>
```

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

## Operational policy evaluation (v0.7)

v0.7 adds policy evaluation based on VAL threshold tuning and TEST replay.

Quick run with constraints:

```bash
make policy_eval
```

Combined quick run (MORL + meta selection + join diagnostics + policy eval):

```bash
make run_morl_policy_quick
```

Production default (Option 2) for local CSV:

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml
```

Production default (Option 2) for BigQuery source (requires ADC/service-account auth):

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_bigquery_gatra_prd_c335.yaml \
  META_CONFIG=configs/meta_controller_relaxed.yaml \
  CONTRACT_CACHE_ROOT=reports/contracts_cache
```

Strict override:

```bash
make run_morl_policy_quick \
  DATA_CONFIG=configs/data_local_gatra_prd_c335.yaml \
  META_CONFIG=configs/meta_controller.yaml
```

Policy outputs:

- `reports/runs/<run_id>/eval/policy/policy_eval.json`
- `reports/runs/<run_id>/eval/policy/policy_eval.md`

Modes in `configs/policy_eval.yaml`:

- `alert_budget`: maximize primary metric under `alerts_per_1k_max`.
- `min_recall`: maximize primary metric with `recall_min`.
- `cost_aware`: minimize triage time while respecting configured constraints.

The report includes both:

- selected policy-constrained threshold
- best-F1 threshold baseline

so operational trade-offs are explicit.

## Label availability realism + join diagnostics (v0.7)

Robustness label delay now supports created-time gating using:

- duration delays (`"7d"`, `"12h"`, `"30m"`)
- policy `treat_as_unknown` (marks labels as `-1`) or `treat_as_benign` (`0`)

When created-time gating is active, artifacts are written under robustness eval:

- `labels_available_mask_test.npy`
- `y_test_available.npy`
- `label_delay_meta.json`

Join diagnostics are config-driven via `configs/join.yaml` and write:

- `reports/runs/<run_id>/eval/join/join_map.npz`
- `reports/runs/<run_id>/eval/join/join_meta.json`

Default join priority is `alarm_id > row_key > time_window` with time-window
fallback disabled unless explicitly enabled.

### Data reality note (single-class TEST splits)

- If TEST has only one class, ROC-AUC/PR metrics can be undefined and emit warnings.
- This is expected for narrow windows or aggressive delay gating, not a pipeline failure.
- Prefer policy-budget metrics and objective summaries for operational comparisons.
- If needed, adjust split windows or label availability policy to increase positives in TEST.

## Constraints

- MORL training/evaluation is contract-only.
- No BigQuery calls are required by MORL tests or evaluation.
- Existing PPO path remains available and unchanged by default.
- Existing MORL sweep without meta-controller remains available when
  `--meta-config` is not provided.
