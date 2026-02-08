from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from evaluation.metrics import classification_metrics


@dataclass(frozen=True)
class PolicyEvalConfig:
    mode: str
    primary_metric: str
    tie_breaker: str
    alerts_per_1k_max: float | None
    recall_min: float | None
    triage_seconds_per_alert: float


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return cast(dict[str, Any], payload)


def _parse_cfg(path: str) -> PolicyEvalConfig:
    payload = _load_yaml(path)
    cfg = cast(dict[str, Any], payload.get("policy_eval", {}))
    constraints = cast(dict[str, Any], cfg.get("constraints", {}))
    return PolicyEvalConfig(
        mode=str(cfg.get("mode", "alert_budget")),
        primary_metric=str(cfg.get("primary_metric", "f1")),
        tie_breaker=str(cfg.get("tie_breaker", "alerts_per_1k")),
        alerts_per_1k_max=(
            float(constraints["alerts_per_1k_max"])
            if constraints.get("alerts_per_1k_max") is not None
            else None
        ),
        recall_min=(
            float(constraints["recall_min"]) if constraints.get("recall_min") is not None else None
        ),
        triage_seconds_per_alert=float(cfg.get("triage_seconds_per_alert", 45.0)),
    )


def _candidate_thresholds(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    if scores.size == 0:
        return np.asarray([0.5], dtype=np.float64)
    uniq = cast(NDArray[np.float64], np.unique(scores).astype(np.float64))
    upper = np.nextafter(float(np.max(uniq)), float("inf"))
    lower = np.nextafter(float(np.min(uniq)), float("-inf"))
    all_vals = np.concatenate(([upper], uniq, [lower])).astype(np.float64)
    return cast(NDArray[np.float64], np.unique(all_vals)[::-1].astype(np.float64))


def _evaluate_threshold(
    y_true: NDArray[np.int_],
    scores: NDArray[np.float64],
    threshold: float,
    triage_seconds_per_alert: float,
) -> dict[str, float]:
    metrics = classification_metrics(y_true, scores, threshold=threshold)
    alerts_per_1k = float(1000.0 * np.mean(scores >= threshold)) if scores.size else 0.0
    metrics["alerts_per_1k"] = alerts_per_1k
    metrics["triage_seconds_per_1k"] = alerts_per_1k * triage_seconds_per_alert
    metrics["threshold"] = float(threshold)
    return metrics


def _is_feasible(metrics: dict[str, float], cfg: PolicyEvalConfig) -> bool:
    if cfg.alerts_per_1k_max is not None and metrics["alerts_per_1k"] > cfg.alerts_per_1k_max:
        return False
    if cfg.recall_min is not None and metrics["recall"] < cfg.recall_min:
        return False
    return True


def _selection_key(metrics: dict[str, float], cfg: PolicyEvalConfig) -> tuple[float, float, float]:
    if cfg.mode == "cost_aware":
        return (
            -metrics["triage_seconds_per_1k"],
            metrics.get(cfg.primary_metric, 0.0),
            -metrics["threshold"],
        )
    return (
        metrics.get(cfg.primary_metric, 0.0),
        -metrics.get(cfg.tie_breaker, metrics["alerts_per_1k"]),
        metrics["threshold"],
    )


def select_threshold_under_policy(
    y_val: NDArray[np.int_],
    scores_val: NDArray[np.float64],
    cfg: PolicyEvalConfig,
) -> dict[str, Any]:
    candidates = _candidate_thresholds(scores_val)
    feasible: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
    for thr in candidates:
        row = _evaluate_threshold(y_val, scores_val, float(thr), cfg.triage_seconds_per_alert)
        all_rows.append(row)
        if _is_feasible(row, cfg):
            feasible.append(row)

    if not feasible:
        ranked = sorted(
            all_rows,
            key=lambda r: (
                abs(r["alerts_per_1k"] - (cfg.alerts_per_1k_max or r["alerts_per_1k"])),
                abs((cfg.recall_min or r["recall"]) - r["recall"]),
            ),
        )
        preview = ranked[:5]
        raise ValueError(
            "No feasible policy threshold found for configured constraints. "
            f"Closest candidates: {preview}"
        )

    selected = max(feasible, key=lambda r: _selection_key(r, cfg))
    best_f1 = max(all_rows, key=lambda r: (r["f1"], -r["alerts_per_1k"], r["threshold"]))
    return {
        "selected_val": selected,
        "best_f1_val": best_f1,
        "num_candidates": int(candidates.shape[0]),
        "num_feasible": int(len(feasible)),
    }


def run_policy_eval(
    *,
    policy_cfg_path: str,
    y_val: NDArray[np.int_],
    y_test: NDArray[np.int_],
    val_scores_by_model: dict[str, NDArray[np.float64]],
    test_scores_by_model: dict[str, NDArray[np.float64]],
    out_dir: str,
) -> dict[str, Any]:
    cfg = _parse_cfg(policy_cfg_path)
    payload: dict[str, Any] = {
        "mode": cfg.mode,
        "primary_metric": cfg.primary_metric,
        "tie_breaker": cfg.tie_breaker,
        "constraints": {
            "alerts_per_1k_max": cfg.alerts_per_1k_max,
            "recall_min": cfg.recall_min,
        },
        "models": {},
    }

    for model_name, val_scores in val_scores_by_model.items():
        test_scores = test_scores_by_model[model_name]
        selected = select_threshold_under_policy(y_val, val_scores, cfg)
        threshold = float(selected["selected_val"]["threshold"])
        selected_test = _evaluate_threshold(
            y_test, test_scores, threshold, cfg.triage_seconds_per_alert
        )
        best_f1_threshold = float(selected["best_f1_val"]["threshold"])
        best_f1_test = _evaluate_threshold(
            y_test, test_scores, best_f1_threshold, cfg.triage_seconds_per_alert
        )
        payload["models"][model_name] = {
            **selected,
            "selected_test": selected_test,
            "best_f1_test": best_f1_test,
        }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "policy_eval.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Policy Evaluation",
        "",
        f"- Mode: {cfg.mode}",
        f"- Primary metric: {cfg.primary_metric}",
        f"- Alerts budget per 1k: {cfg.alerts_per_1k_max}",
        f"- Recall minimum: {cfg.recall_min}",
        "",
        "## Constraint outcomes on TEST",
    ]
    for model_name, result in cast(dict[str, dict[str, Any]], payload["models"]).items():
        selected_test = cast(dict[str, float], result["selected_test"])
        best_f1_test = cast(dict[str, float], result["best_f1_test"])
        lines.extend(
            [
                f"### {model_name}",
                f"- selected_threshold: {float(result['selected_val']['threshold']):.6f}",
                f"- selected_test_f1: {selected_test['f1']:.6f}",
                f"- selected_test_recall: {selected_test['recall']:.6f}",
                f"- selected_test_alerts_per_1k: {selected_test['alerts_per_1k']:.3f}",
                f"- selected_test_triage_seconds_per_1k: {selected_test['triage_seconds_per_1k']:.3f}",
                f"- best_f1_threshold: {float(result['best_f1_val']['threshold']):.6f}",
                f"- best_f1_test_f1: {best_f1_test['f1']:.6f}",
                f"- best_f1_test_alerts_per_1k: {best_f1_test['alerts_per_1k']:.3f}",
                "",
            ]
        )

    md_path = os.path.join(out_dir, "policy_eval.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    payload["paths"] = {"json": json_path, "md": md_path}
    return payload
