from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from math import isclose
from typing import Any, Sequence, cast

import numpy as np
import yaml

from architecture_a_rl.morl.meta_controller import (
    CandidateRow,
    ConstraintConfig,
    ObjectiveConfig,
    RelaxationConfig,
    SoftPenaltyConfig,
    _load_result_rows,
    _parse_constraint_mode,
    _to_float,
    _to_optional_float,
    score_candidate,
    select_with_constraints,
)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    kind: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ConditionResult:
    name: str
    kind: str
    selected_weight: list[float]
    selected_metrics: dict[str, float]
    fallback_used: bool
    fallback_mode: str
    relax_trace_length: int
    regret: float
    best_utility: float
    selected_utility: float
    feasible_count: int
    skipped: bool
    skip_reason: str | None


@dataclass(frozen=True)
class StabilityConfig:
    conditions: list[ConditionSpec]
    utility: dict[str, float]
    objective_primary: str
    objective_tie_breaker: str
    base_constraints: ConstraintConfig
    base_relaxation: RelaxationConfig
    fail_on_infeasible: bool
    method_name: str
    method_params: dict[str, Any]


def _load_stability_config(path: str) -> StabilityConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    stab = cast(dict[str, Any], raw.get("meta_stability", {}))

    conditions_raw = cast(list[dict[str, Any]], stab.get("conditions", []))
    conditions: list[ConditionSpec] = []
    for c in conditions_raw:
        conditions.append(
            ConditionSpec(
                name=str(c.get("name", "unknown")),
                kind=str(c.get("kind", "custom")),
                overrides=cast(dict[str, Any], c.get("overrides", {})),
            )
        )

    utility = {
        str(k): _to_float(v) for k, v in cast(dict[str, object], stab.get("utility", {})).items()
    }

    constraints_raw = cast(dict[str, Any], stab.get("constraints", {}))
    penalty_raw = cast(dict[str, Any], constraints_raw.get("soft_penalty", {}))
    relaxation_raw = cast(dict[str, Any], stab.get("relaxation", {}))
    schedule_raw = cast(list[dict[str, object]], relaxation_raw.get("schedule", []))

    from architecture_a_rl.morl.meta_controller import RelaxationStep

    schedule = [
        RelaxationStep(
            alerts_per_1k_max=_to_optional_float(item.get("alerts_per_1k_max")),
            recall_min=_to_optional_float(item.get("recall_min")),
            precision_min=_to_optional_float(item.get("precision_min")),
            pr_auc_min=_to_optional_float(item.get("pr_auc_min")),
        )
        for item in schedule_raw
    ]

    method_raw = cast(dict[str, Any], stab.get("method", {}))
    method_params = cast(dict[str, Any], method_raw.get("params", {}))

    return StabilityConfig(
        conditions=conditions,
        utility=utility,
        objective_primary=str(stab.get("objective_primary", "pr_auc")),
        objective_tie_breaker=str(stab.get("objective_tie_breaker", "alerts_per_1k")),
        base_constraints=ConstraintConfig(
            mode=_parse_constraint_mode(constraints_raw.get("mode", "hard")),
            alerts_per_1k_max=_to_optional_float(constraints_raw.get("alerts_per_1k_max")),
            recall_min=_to_optional_float(constraints_raw.get("recall_min")),
            precision_min=_to_optional_float(constraints_raw.get("precision_min")),
            pr_auc_min=_to_optional_float(constraints_raw.get("pr_auc_min")),
            soft_penalty=SoftPenaltyConfig(
                lambda_recall=float(penalty_raw.get("lambda_recall", 5.0)),
                lambda_alerts=float(penalty_raw.get("lambda_alerts", 0.05)),
                lambda_precision=float(penalty_raw.get("lambda_precision", 1.0)),
                lambda_pr_auc=float(penalty_raw.get("lambda_pr_auc", 1.0)),
            ),
        ),
        base_relaxation=RelaxationConfig(
            enabled=bool(relaxation_raw.get("enabled", True)),
            schedule=schedule,
            stop_on_first_feasible=bool(relaxation_raw.get("stop_on_first_feasible", True)),
        ),
        fail_on_infeasible=bool(stab.get("fail_on_infeasible", False)),
        method_name=str(method_raw.get("name", "greedy")),
        method_params=method_params,
    )


def _apply_condition_overrides(
    candidates: Sequence[CandidateRow],
    condition: ConditionSpec,
) -> list[CandidateRow]:
    overrides = condition.overrides
    noise_sigma = float(overrides.get("noise_sigma", 0.0))
    recall_scale = float(overrides.get("recall_scale", 1.0))
    precision_scale = float(overrides.get("precision_scale", 1.0))
    alerts_scale = float(overrides.get("alerts_scale", 1.0))
    pr_auc_scale = float(overrides.get("pr_auc_scale", 1.0))
    seed = int(overrides.get("seed", 0))

    rng = np.random.default_rng(seed) if noise_sigma > 0.0 else None

    perturbed: list[CandidateRow] = []
    for row in candidates:
        m = dict(row.metrics)
        if recall_scale != 1.0 and "recall" in m:
            m["recall"] = float(np.clip(m["recall"] * recall_scale, 0.0, 1.0))
        if precision_scale != 1.0 and "precision" in m:
            m["precision"] = float(np.clip(m["precision"] * precision_scale, 0.0, 1.0))
        if alerts_scale != 1.0 and "alerts_per_1k" in m:
            m["alerts_per_1k"] = max(0.0, m["alerts_per_1k"] * alerts_scale)
        if pr_auc_scale != 1.0 and "pr_auc" in m:
            m["pr_auc"] = float(np.clip(m["pr_auc"] * pr_auc_scale, 0.0, 1.0))
        if rng is not None:
            for k in m:
                m[k] = float(m[k] + rng.normal(0.0, noise_sigma))
        perturbed.append(
            CandidateRow(
                w=row.w,
                metrics=m,
                objective_means=row.objective_means,
                meta=row.meta,
            )
        )
    return perturbed


def _build_objective(cfg: StabilityConfig) -> ObjectiveConfig:
    return ObjectiveConfig(
        primary=cfg.objective_primary,
        tie_breaker=cfg.objective_tie_breaker,
        utility=cfg.utility,
    )


def _build_method_config(cfg: StabilityConfig) -> Any:
    from architecture_a_rl.morl.meta_controller import MethodConfig

    name = cfg.method_name
    if name not in {"greedy", "bandit_ucb", "bandit_thompson"}:
        raise ValueError(f"Unsupported method: {name}")
    return MethodConfig(
        name=cast(Any, name),
        rounds=int(cfg.method_params.get("rounds", 20)),
        explore_frac=float(cfg.method_params.get("explore_frac", 0.2)),
        ucb_c=float(cfg.method_params.get("ucb_c", 1.0)),
        thompson_sigma=float(cfg.method_params.get("thompson_sigma", 0.05)),
    )


def compute_selection_change_rate(
    baseline_w: list[float],
    condition_weights: list[list[float]],
) -> float:
    if not condition_weights:
        return 0.0
    changes = sum(1 for w in condition_weights if w != baseline_w)
    return float(changes) / float(len(condition_weights))


def compute_weight_l1_distance(w1: Sequence[float], w2: Sequence[float]) -> float:
    return float(sum(abs(a - b) for a, b in zip(w1, w2)))


def compute_regret(
    candidates: Sequence[CandidateRow],
    selected: CandidateRow,
    objective: ObjectiveConfig,
) -> tuple[float, float, float]:
    best_utility = max(score_candidate(r.metrics, objective) for r in candidates)
    selected_utility = score_candidate(selected.metrics, objective)
    regret = best_utility - selected_utility
    return regret, best_utility, selected_utility


def run_stability_suite(
    val_results_path: str,
    stability_cfg_path: str,
    seed: int,
) -> dict[str, Any]:
    cfg = _load_stability_config(stability_cfg_path)
    candidates = _load_result_rows(val_results_path)
    if not candidates:
        raise ValueError(f"No candidate rows in {val_results_path}")

    objective = _build_objective(cfg)
    method = _build_method_config(cfg)
    results: list[ConditionResult] = []

    baseline_selected, _, baseline_diag = select_with_constraints(
        candidates=candidates,
        constraints=cfg.base_constraints,
        method=method,
        objective=objective,
        relaxation=cfg.base_relaxation,
        fail_on_infeasible=cfg.fail_on_infeasible,
        seed=seed,
    )
    baseline_regret, baseline_best_u, baseline_sel_u = compute_regret(
        candidates, baseline_selected, objective
    )
    results.append(
        ConditionResult(
            name="baseline",
            kind="baseline",
            selected_weight=baseline_selected.w,
            selected_metrics=baseline_selected.metrics,
            fallback_used=bool(baseline_diag.get("fallback_used", False)),
            fallback_mode=str(baseline_diag.get("fallback_mode", "none")),
            relax_trace_length=len(baseline_diag.get("relaxation_trace", [])),
            regret=baseline_regret,
            best_utility=baseline_best_u,
            selected_utility=baseline_sel_u,
            feasible_count=int(baseline_diag.get("feasible_count_initial", 0)),
            skipped=False,
            skip_reason=None,
        )
    )

    for cond in cfg.conditions:
        try:
            perturbed = _apply_condition_overrides(candidates, cond)
            selected, _, diag = select_with_constraints(
                candidates=perturbed,
                constraints=cfg.base_constraints,
                method=method,
                objective=objective,
                relaxation=cfg.base_relaxation,
                fail_on_infeasible=cfg.fail_on_infeasible,
                seed=seed,
            )
            regret, best_u, sel_u = compute_regret(perturbed, selected, objective)
            results.append(
                ConditionResult(
                    name=cond.name,
                    kind=cond.kind,
                    selected_weight=selected.w,
                    selected_metrics=selected.metrics,
                    fallback_used=bool(diag.get("fallback_used", False)),
                    fallback_mode=str(diag.get("fallback_mode", "none")),
                    relax_trace_length=len(diag.get("relaxation_trace", [])),
                    regret=regret,
                    best_utility=best_u,
                    selected_utility=sel_u,
                    feasible_count=int(diag.get("feasible_count_initial", 0)),
                    skipped=False,
                    skip_reason=None,
                )
            )
        except Exception as exc:
            results.append(
                ConditionResult(
                    name=cond.name,
                    kind=cond.kind,
                    selected_weight=[],
                    selected_metrics={},
                    fallback_used=False,
                    fallback_mode="none",
                    relax_trace_length=0,
                    regret=0.0,
                    best_utility=0.0,
                    selected_utility=0.0,
                    feasible_count=0,
                    skipped=True,
                    skip_reason=str(exc),
                )
            )

    non_skipped = [r for r in results if not r.skipped and r.name != "baseline"]
    condition_weights = [r.selected_weight for r in non_skipped]

    selection_change_rate = compute_selection_change_rate(baseline_selected.w, condition_weights)
    weight_distances = [
        compute_weight_l1_distance(baseline_selected.w, w) for w in condition_weights
    ]
    avg_weight_l1_distance = float(np.mean(weight_distances)) if weight_distances else 0.0
    regrets = [r.regret for r in non_skipped]
    avg_regret = float(np.mean(regrets)) if regrets else 0.0
    constraint_violations = sum(1 for r in non_skipped if r.fallback_used)
    constraint_violation_rate = (
        float(constraint_violations) / float(len(non_skipped)) if non_skipped else 0.0
    )
    worst_regret = max(regrets) if regrets else 0.0
    worst_regret_condition = ""
    for r in non_skipped:
        if isclose(r.regret, worst_regret, rel_tol=0.0, abs_tol=1e-12):
            worst_regret_condition = r.name
            break

    return {
        "conditions": [
            {
                "name": r.name,
                "kind": r.kind,
                "selected_weight": r.selected_weight,
                "selected_metrics": r.selected_metrics,
                "fallback_used": r.fallback_used,
                "fallback_mode": r.fallback_mode,
                "relax_trace_length": r.relax_trace_length,
                "regret": r.regret,
                "best_utility": r.best_utility,
                "selected_utility": r.selected_utility,
                "feasible_count": r.feasible_count,
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
            }
            for r in results
        ],
        "aggregate": {
            "selection_change_rate": selection_change_rate,
            "avg_weight_l1_distance": avg_weight_l1_distance,
            "constraint_violation_rate": constraint_violation_rate,
            "avg_regret": avg_regret,
            "worst_regret": worst_regret,
            "worst_regret_condition": worst_regret_condition,
            "n_conditions": len(non_skipped),
            "n_skipped": sum(1 for r in results if r.skipped),
        },
        "baseline": {
            "selected_weight": baseline_selected.w,
            "selected_metrics": baseline_selected.metrics,
            "regret": baseline_regret,
        },
    }


def write_stability_artifacts(
    out_dir: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "meta_stability.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    conditions = payload.get("conditions", [])
    csv_path = os.path.join(out_dir, "meta_stability_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "kind",
                "selected_weight",
                "fallback_used",
                "fallback_mode",
                "regret",
                "best_utility",
                "selected_utility",
                "feasible_count",
                "skipped",
            ]
        )
        for row in conditions:
            writer.writerow(
                [
                    row.get("name"),
                    row.get("kind"),
                    row.get("selected_weight"),
                    row.get("fallback_used"),
                    row.get("fallback_mode"),
                    f"{row.get('regret', 0.0):.6f}",
                    f"{row.get('best_utility', 0.0):.6f}",
                    f"{row.get('selected_utility', 0.0):.6f}",
                    row.get("feasible_count"),
                    row.get("skipped"),
                ]
            )

    agg = payload.get("aggregate", {})
    baseline = payload.get("baseline", {})
    lines = [
        "# Meta-Selection Stability Report",
        "",
        "## Baseline",
        f"- Selected weight: `{baseline.get('selected_weight')}`",
        f"- Regret: {baseline.get('regret', 0.0):.6f}",
        "",
        "## Conditions",
        "",
        "| Condition | Kind | Selected Weight | Fallback | Regret | Feasible |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in conditions:
        if row.get("skipped"):
            lines.append(f"| {row['name']} | {row['kind']} | SKIPPED | - | - | - |")
        else:
            lines.append(
                f"| {row['name']} | {row['kind']} |"
                f" `{row['selected_weight']}` |"
                f" {row['fallback_mode']} |"
                f" {row['regret']:.6f} |"
                f" {row['feasible_count']} |"
            )
    lines.extend(
        [
            "",
            "## Volatility Summary",
            f"- Selection change rate: {agg.get('selection_change_rate', 0.0):.4f}",
            f"- Avg weight L1 distance: {agg.get('avg_weight_l1_distance', 0.0):.6f}",
            f"- Constraint violation rate: {agg.get('constraint_violation_rate', 0.0):.4f}",
            f"- Avg regret: {agg.get('avg_regret', 0.0):.6f}",
            "",
            "## Worst-Case Regret",
            f"- Worst regret: {agg.get('worst_regret', 0.0):.6f}",
            f"- Caused by: {agg.get('worst_regret_condition', 'n/a')}",
            "",
        ]
    )

    md_path = os.path.join(out_dir, "meta_stability.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return {
        "json": json_path,
        "csv": csv_path,
        "md": md_path,
    }
