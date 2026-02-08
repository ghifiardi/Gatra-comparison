from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose
from typing import Any, Literal, Sequence, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from .preferences import normalize_weight_grid

MethodName = Literal["greedy", "bandit_ucb", "bandit_thompson"]


@dataclass(frozen=True)
class ConstraintConfig:
    alerts_per_1k_max: float | None
    recall_min: float | None
    precision_min: float | None
    pr_auc_min: float | None


@dataclass(frozen=True)
class ObjectiveConfig:
    primary: str
    tie_breaker: str
    utility: dict[str, float]


@dataclass(frozen=True)
class MethodConfig:
    name: MethodName
    rounds: int
    explore_frac: float
    ucb_c: float
    thompson_sigma: float


@dataclass(frozen=True)
class MetaControllerConfig:
    enabled: bool
    selection_split: str
    candidate_source: str
    candidate_explicit: list[list[float]]
    objective: ObjectiveConfig
    constraints: ConstraintConfig
    method: MethodConfig


@dataclass(frozen=True)
class CandidateRow:
    w: list[float]
    metrics: dict[str, float]
    objective_means: dict[str, float]
    meta: dict[str, Any]


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _to_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _to_float(value)


def _weight_key(w: Sequence[float]) -> tuple[float, ...]:
    return tuple(round(float(v), 6) for v in w)


def _load_yaml_mapping(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in YAML: {path}")
    return cast(dict[str, Any], raw)


def _load_meta_config(path: str) -> MetaControllerConfig:
    root = _load_yaml_mapping(path)
    mc = cast(dict[str, Any], root.get("meta_controller", {}))
    method_raw = cast(dict[str, Any], mc.get("method", {}))
    params = cast(dict[str, Any], method_raw.get("params", {}))
    constraints_raw = cast(dict[str, Any], mc.get("constraints", {}))
    objective_raw = cast(dict[str, Any], mc.get("objective", {}))
    candidates_raw = cast(dict[str, Any], mc.get("candidates", {}))

    name = str(method_raw.get("name", "greedy"))
    if name not in {"greedy", "bandit_ucb", "bandit_thompson"}:
        raise ValueError(f"Unsupported meta-controller method: {name}")

    return MetaControllerConfig(
        enabled=bool(mc.get("enabled", False)),
        selection_split=str(mc.get("selection_split", "val")),
        candidate_source=str(candidates_raw.get("source", "morl_config_grid")),
        candidate_explicit=[
            [float(v) for v in row]
            for row in cast(list[list[float]], candidates_raw.get("explicit", []))
        ],
        objective=ObjectiveConfig(
            primary=str(objective_raw.get("primary", "pr_auc")),
            tie_breaker=str(objective_raw.get("tie_breaker", "alerts_per_1k")),
            utility={
                str(k): _to_float(v)
                for k, v in cast(dict[str, object], objective_raw.get("utility", {})).items()
            },
        ),
        constraints=ConstraintConfig(
            alerts_per_1k_max=_to_optional_float(constraints_raw.get("alerts_per_1k_max")),
            recall_min=_to_optional_float(constraints_raw.get("recall_min")),
            precision_min=_to_optional_float(constraints_raw.get("precision_min")),
            pr_auc_min=_to_optional_float(constraints_raw.get("pr_auc_min")),
        ),
        method=MethodConfig(
            name=cast(MethodName, name),
            rounds=int(params.get("rounds", 20)),
            explore_frac=float(params.get("explore_frac", 0.2)),
            ucb_c=float(params.get("ucb_c", 1.0)),
            thompson_sigma=float(params.get("thompson_sigma", 0.05)),
        ),
    )


def _load_morl_grid(morl_cfg_path: str, k_objectives: int) -> list[list[float]]:
    cfg = _load_yaml_mapping(morl_cfg_path)
    morl = cast(dict[str, Any], cfg.get("morl", {}))
    eval_cfg = cast(dict[str, Any], morl.get("eval", {}))
    sweep_cfg = cast(dict[str, Any], eval_cfg.get("weight_sweep", {}))
    grid = cast(list[list[float]], sweep_cfg.get("grid", []))
    normalized = normalize_weight_grid(grid, k_objectives)
    return [[float(v) for v in row.tolist()] for row in normalized]


def _load_result_rows(val_results_path: str) -> list[CandidateRow]:
    with open(val_results_path, "r") as f:
        raw = cast(dict[str, Any], json.load(f))
    raw_rows = cast(list[dict[str, Any]], raw.get("results", []))

    rows: list[CandidateRow] = []
    for row in raw_rows:
        w_raw = row.get("w", row.get("weights"))
        if not isinstance(w_raw, list):
            raise ValueError("Each result row must include list field 'w' (or legacy 'weights')")
        w = [float(v) for v in cast(list[float], w_raw)]

        metrics_raw = row.get("metrics")
        if not isinstance(metrics_raw, dict):
            raise ValueError("Each result row must include 'metrics' mapping")
        metrics = {str(k): _to_float(v) for k, v in metrics_raw.items()}

        obj_raw = row.get("objective_means")
        if not isinstance(obj_raw, dict):
            raise ValueError("Each result row must include 'objective_means' mapping")
        objective_means = {str(k): _to_float(v) for k, v in obj_raw.items()}

        meta_raw = row.get("meta", {})
        meta = cast(dict[str, Any], meta_raw if isinstance(meta_raw, dict) else {})
        rows.append(CandidateRow(w=w, metrics=metrics, objective_means=objective_means, meta=meta))

    return rows


def score_candidate(metrics: dict[str, float], objective: ObjectiveConfig) -> float:
    if objective.primary == "utility":
        total = 0.0
        for metric_name, weight in objective.utility.items():
            total += float(weight) * float(metrics.get(metric_name, 0.0))
        return float(total)
    return float(metrics.get(objective.primary, float("-inf")))


def _constraint_violation_score(row: CandidateRow, c: ConstraintConfig) -> float:
    m = row.metrics
    score = 0.0
    if c.alerts_per_1k_max is not None:
        score += max(0.0, m.get("alerts_per_1k", float("inf")) - c.alerts_per_1k_max)
    if c.recall_min is not None:
        score += max(0.0, c.recall_min - m.get("recall", 0.0))
    if c.precision_min is not None:
        score += max(0.0, c.precision_min - m.get("precision", 0.0))
    if c.pr_auc_min is not None:
        score += max(0.0, c.pr_auc_min - m.get("pr_auc", 0.0))
    return float(score)


def _is_feasible(row: CandidateRow, c: ConstraintConfig) -> bool:
    m = row.metrics
    if (
        c.alerts_per_1k_max is not None
        and m.get("alerts_per_1k", float("inf")) > c.alerts_per_1k_max
    ):
        return False
    if c.recall_min is not None and m.get("recall", 0.0) < c.recall_min:
        return False
    if c.precision_min is not None and m.get("precision", 0.0) < c.precision_min:
        return False
    if c.pr_auc_min is not None and m.get("pr_auc", 0.0) < c.pr_auc_min:
        return False
    return True


def filter_candidates(
    rows: Sequence[CandidateRow],
    constraints: ConstraintConfig,
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    feasible: list[CandidateRow] = []
    rejected: list[CandidateRow] = []
    for row in rows:
        if _is_feasible(row, constraints):
            feasible.append(row)
        else:
            rejected.append(row)
    return feasible, rejected


def _tie_break_tuple(row: CandidateRow) -> tuple[float, tuple[float, ...]]:
    return (float(row.metrics.get("alerts_per_1k", float("inf"))), _weight_key(row.w))


def _argmax_with_tiebreak(values: NDArray[np.float64], rows: Sequence[CandidateRow]) -> int:
    best = 0
    for i in range(1, len(rows)):
        if values[i] > values[best]:
            best = i
            continue
        if isclose(float(values[i]), float(values[best]), rel_tol=0.0, abs_tol=1e-12):
            if _tie_break_tuple(rows[i]) < _tie_break_tuple(rows[best]):
                best = i
    return best


@dataclass(frozen=True)
class SelectionOutcome:
    selected_index: int
    trace: list[dict[str, Any]]


class GreedySelector:
    def select(self, rows: Sequence[CandidateRow], objective: ObjectiveConfig) -> SelectionOutcome:
        scored = [score_candidate(r.metrics, objective) for r in rows]
        scored_arr = np.asarray(scored, dtype=np.float64)
        best = _argmax_with_tiebreak(scored_arr, rows)
        trace = [
            {
                "round": 0,
                "arm_index": i,
                "w": row.w,
                "score_true": float(scored_arr[i]),
                "alerts_per_1k": float(row.metrics.get("alerts_per_1k", float("inf"))),
            }
            for i, row in enumerate(rows)
        ]
        return SelectionOutcome(selected_index=best, trace=trace)


class UCBSelector:
    def __init__(self, rounds: int, ucb_c: float, sigma: float):
        self.rounds = rounds
        self.ucb_c = ucb_c
        self.sigma = sigma

    def select(
        self,
        rows: Sequence[CandidateRow],
        objective: ObjectiveConfig,
        seed: int,
    ) -> SelectionOutcome:
        true_rewards = np.asarray(
            [score_candidate(r.metrics, objective) for r in rows], dtype=np.float64
        )
        n_arms = len(rows)
        rounds = max(self.rounds, n_arms)
        rng = np.random.default_rng(seed)

        counts = np.zeros((n_arms,), dtype=np.int_)
        sums = np.zeros((n_arms,), dtype=np.float64)
        trace: list[dict[str, Any]] = []

        # Initialization: one pull each arm.
        for arm in range(n_arms):
            observed = float(true_rewards[arm] + rng.normal(0.0, self.sigma))
            counts[arm] += 1
            sums[arm] += observed
            trace.append(
                {
                    "round": arm,
                    "arm_index": arm,
                    "w": rows[arm].w,
                    "reward_true": float(true_rewards[arm]),
                    "reward_observed": observed,
                    "rule": "init",
                }
            )

        for t in range(n_arms, rounds):
            means = sums / np.maximum(counts, 1)
            bonus = self.ucb_c * np.sqrt(np.log(float(t + 1)) / np.maximum(counts, 1))
            chosen = _argmax_with_tiebreak(means + bonus, rows)
            observed = float(true_rewards[chosen] + rng.normal(0.0, self.sigma))
            counts[chosen] += 1
            sums[chosen] += observed
            trace.append(
                {
                    "round": t,
                    "arm_index": chosen,
                    "w": rows[chosen].w,
                    "reward_true": float(true_rewards[chosen]),
                    "reward_observed": observed,
                    "rule": "ucb",
                }
            )

        means = sums / np.maximum(counts, 1)
        selected = _argmax_with_tiebreak(means, rows)
        return SelectionOutcome(selected_index=selected, trace=trace)


class ThompsonSelector:
    def __init__(self, rounds: int, sigma: float):
        self.rounds = rounds
        self.sigma = sigma

    def select(
        self,
        rows: Sequence[CandidateRow],
        objective: ObjectiveConfig,
        seed: int,
    ) -> SelectionOutcome:
        true_rewards = np.asarray(
            [score_candidate(r.metrics, objective) for r in rows], dtype=np.float64
        )
        n_arms = len(rows)
        rounds = max(self.rounds, n_arms)
        rng = np.random.default_rng(seed)

        mu = np.zeros((n_arms,), dtype=np.float64)
        var = np.ones((n_arms,), dtype=np.float64)
        obs_var = max(self.sigma * self.sigma, 1e-12)
        trace: list[dict[str, Any]] = []

        for t in range(rounds):
            sampled_theta = rng.normal(mu, np.sqrt(var))
            chosen = _argmax_with_tiebreak(sampled_theta, rows)

            observed = float(true_rewards[chosen] + rng.normal(0.0, self.sigma))
            prior_mu = mu[chosen]
            prior_var = var[chosen]

            post_var = 1.0 / ((1.0 / prior_var) + (1.0 / obs_var))
            post_mu = post_var * ((prior_mu / prior_var) + (observed / obs_var))
            mu[chosen] = post_mu
            var[chosen] = post_var

            trace.append(
                {
                    "round": t,
                    "arm_index": chosen,
                    "w": rows[chosen].w,
                    "reward_true": float(true_rewards[chosen]),
                    "reward_observed": observed,
                    "posterior_mean": float(mu[chosen]),
                    "posterior_var": float(var[chosen]),
                    "rule": "thompson",
                }
            )

        selected = _argmax_with_tiebreak(mu, rows)
        return SelectionOutcome(selected_index=selected, trace=trace)


def _raise_no_feasible(rows: Sequence[CandidateRow], constraints: ConstraintConfig) -> None:
    ranked = sorted(
        rows,
        key=lambda r: (_constraint_violation_score(r, constraints), _tie_break_tuple(r)),
    )
    preview = [
        {
            "w": row.w,
            "violation_score": _constraint_violation_score(row, constraints),
            "metrics": row.metrics,
        }
        for row in ranked[:5]
    ]
    raise ValueError(
        "No feasible candidates satisfy constraints. Top-5 closest candidates:\n"
        + json.dumps(preview, indent=2)
    )


def select_weight(
    meta_cfg_path: str,
    morl_cfg_path: str,
    val_results_path: str,
    seed: int,
) -> dict[str, Any]:
    cfg = _load_meta_config(meta_cfg_path)
    if not cfg.enabled:
        raise ValueError("Meta-controller disabled; set meta_controller.enabled=true")
    if cfg.selection_split != "val":
        raise ValueError("v0.5.1 requires meta_controller.selection_split=val")

    rows = _load_result_rows(val_results_path)
    if not rows:
        raise ValueError(f"No candidate rows found in {val_results_path}")
    k = len(rows[0].w)

    if cfg.candidate_source == "morl_config_grid":
        candidate_weights = _load_morl_grid(morl_cfg_path, k)
    elif cfg.candidate_source == "explicit":
        candidate_weights = [
            [float(v) for v in row.tolist()]
            for row in normalize_weight_grid(cfg.candidate_explicit, k)
        ]
    else:
        raise ValueError(f"Unsupported candidates.source: {cfg.candidate_source}")

    row_by_w = {_weight_key(r.w): r for r in rows}
    candidate_rows: list[CandidateRow] = []
    missing: list[list[float]] = []
    for w in candidate_weights:
        row = row_by_w.get(_weight_key(w))
        if row is None:
            missing.append(w)
            continue
        candidate_rows.append(row)
    if missing:
        raise ValueError(f"Missing candidate weights in val results: {missing}")
    if not candidate_rows:
        raise ValueError("No candidate rows available after weight matching")

    feasible_rows, _ = filter_candidates(candidate_rows, cfg.constraints)
    if not feasible_rows:
        _raise_no_feasible(candidate_rows, cfg.constraints)

    if cfg.method.name == "greedy":
        outcome = GreedySelector().select(feasible_rows, cfg.objective)
    elif cfg.method.name == "bandit_ucb":
        outcome = UCBSelector(
            rounds=cfg.method.rounds,
            ucb_c=cfg.method.ucb_c,
            sigma=cfg.method.thompson_sigma,
        ).select(feasible_rows, cfg.objective, seed=seed)
    elif cfg.method.name == "bandit_thompson":
        outcome = ThompsonSelector(
            rounds=cfg.method.rounds,
            sigma=cfg.method.thompson_sigma,
        ).select(feasible_rows, cfg.objective, seed=seed)
    else:
        raise ValueError(f"Unsupported method name: {cfg.method.name}")

    selected = feasible_rows[outcome.selected_index]
    return {
        "selected_weight": [float(v) for v in selected.w],
        "method": {
            "name": cfg.method.name,
            "params": {
                "rounds": cfg.method.rounds,
                "explore_frac": cfg.method.explore_frac,
                "ucb_c": cfg.method.ucb_c,
                "thompson_sigma": cfg.method.thompson_sigma,
            },
        },
        "constraints": {
            "alerts_per_1k_max": cfg.constraints.alerts_per_1k_max,
            "recall_min": cfg.constraints.recall_min,
            "precision_min": cfg.constraints.precision_min,
            "pr_auc_min": cfg.constraints.pr_auc_min,
        },
        "feasible_count": len(feasible_rows),
        "candidate_count": len(candidate_rows),
        "selected_val_metrics": selected.metrics,
        "selected_val_objective_means": selected.objective_means,
        "selection_trace": outcome.trace,
    }
