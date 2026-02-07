from __future__ import annotations

import csv
import json
import os
from typing import Any, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from architecture_a_rl.morl.networks import (
    PreferenceConditionedActor,
)
from architecture_a_rl.morl.objectives import compute_reward_matrix, parse_objectives
from architecture_a_rl.morl.preferences import normalize_weight_grid
from evaluation.metrics import classification_metrics
from evaluation.pareto import hypervolume_from_rows, pareto_filter


def _load_morl_cfg(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = cast(dict[str, Any], yaml.safe_load(f))
    if "morl" not in cfg:
        raise ValueError("Missing 'morl' root in MORL config")
    return cast(dict[str, Any], cfg["morl"])


def _load_contract_test(contract_dir: str) -> tuple[NDArray[np.float32], NDArray[np.int_]]:
    x = np.load(os.path.join(contract_dir, "features_v128_test.npy")).astype(np.float32)
    y = np.load(os.path.join(contract_dir, "y_true.npy")).astype(np.int_)
    return x, y


def _safe_metric_value(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


def _write_csv(path: str, rows: list[dict[str, Any]], objective_names: list[str]) -> None:
    cols = [
        "weights",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "alerts_per_1k",
    ] + [f"objective_{name}" for name in objective_names]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            metrics = cast(dict[str, float], row["metrics"])
            obj = cast(dict[str, float], row["objective_means"])
            out: dict[str, Any] = {
                "weights": json.dumps(row["weights"]),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "alerts_per_1k": metrics["alerts_per_1k"],
            }
            for name in objective_names:
                out[f"objective_{name}"] = obj[name]
            writer.writerow(out)


def _write_markdown(
    path: str,
    rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    primary_metrics: list[str],
    hypervolume: float | None,
) -> None:
    lines = [
        "# MORL Evaluation Summary",
        "",
        f"- Evaluated weights: {len(rows)}",
        f"- Pareto candidates: {len(pareto_rows)}",
        f"- Primary metrics: {', '.join(primary_metrics)}",
    ]
    if hypervolume is not None:
        lines.append(f"- Hypervolume: {hypervolume:.6f}")
    lines.extend(
        [
            "",
            "## Practitioner guidance",
            "- Pick weights on the Pareto set to avoid dominated operating points.",
            "- Use higher detect weight when recall is prioritized over analyst load.",
            "- Use higher analyst_cost weight when alert budget is constrained.",
            "",
            "## Pareto candidates",
            "| weights | pr_auc | f1 | alerts_per_1k |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in pareto_rows:
        m = cast(dict[str, float], row["metrics"])
        lines.append(
            f"| `{row['weights']}` | {m['pr_auc']:.4f} | {m['f1']:.4f} | {m['alerts_per_1k']:.2f} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_morl_weight_sweep(
    contract_dir: str,
    morl_model_dir: str,
    morl_cfg_path: str,
    out_dir: str,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    morl_cfg = _load_morl_cfg(morl_cfg_path)
    k = int(morl_cfg.get("k_objectives", 3))
    objectives = parse_objectives(cast(list[dict[str, object]], morl_cfg.get("objectives", [])))
    if len(objectives) != k:
        raise ValueError(f"Expected {k} objectives, got {len(objectives)}")

    train_cfg = cast(dict[str, Any], morl_cfg.get("training", {}))
    hidden = [int(v) for v in cast(list[int], train_cfg.get("hidden", [256, 128, 64]))]

    actor = PreferenceConditionedActor(state_dim=128, k_objectives=k, hidden=hidden, action_dim=2)
    actor_path = os.path.join(morl_model_dir, "actor.pt")
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    actor.eval()

    eval_cfg = cast(dict[str, Any], morl_cfg.get("eval", {}))
    sweep_cfg = cast(dict[str, Any], eval_cfg.get("weight_sweep", {}))
    weight_grid = normalize_weight_grid(cast(list[list[float]], sweep_cfg.get("grid", [])), k)

    x_test, y_test = _load_contract_test(contract_dir)

    rows: list[dict[str, Any]] = []
    objective_names = [o.name for o in objectives]

    with torch.no_grad():
        states = torch.tensor(x_test, dtype=torch.float32)
        for w in weight_grid:
            w_rep = np.repeat(w[None, :], x_test.shape[0], axis=0).astype(np.float32)
            w_t = torch.tensor(w_rep, dtype=torch.float32)
            xw = torch.cat([states, w_t], dim=1)
            probs = actor(xw).cpu().numpy()
            y_score = probs[:, 1].astype(np.float64)
            actions = (y_score >= 0.5).astype(np.int_)

            base_metrics = classification_metrics(y_test, y_score, threshold=0.5)
            metrics = {
                "precision": _safe_metric_value(base_metrics.get("precision", 0.0)),
                "recall": _safe_metric_value(base_metrics.get("recall", 0.0)),
                "f1": _safe_metric_value(base_metrics.get("f1", 0.0)),
                "roc_auc": _safe_metric_value(base_metrics.get("roc_auc", 0.0)),
                "pr_auc": _safe_metric_value(base_metrics.get("pr_auc", 0.0)),
                "alerts_per_1k": float(1000.0 * np.mean(actions == 1)),
            }

            reward_matrix = compute_reward_matrix(y_test, actions, objectives)
            objective_means = {
                objective_names[i]: float(np.mean(reward_matrix[:, i])) for i in range(k)
            }

            rows.append(
                {
                    "weights": [float(v) for v in w.tolist()],
                    "metrics": metrics,
                    "objective_means": objective_means,
                }
            )

    pareto_cfg = cast(dict[str, Any], eval_cfg.get("pareto", {}))
    primary_metrics = [str(v) for v in cast(list[str], pareto_cfg.get("primary_metrics", []))]
    if not primary_metrics:
        primary_metrics = ["pr_auc", "f1", "alerts_per_1k"]

    pareto_rows = (
        pareto_filter(rows, primary_metrics) if bool(pareto_cfg.get("enabled", True)) else []
    )

    hv_cfg = cast(dict[str, Any], eval_cfg.get("hypervolume", {}))
    hv_value: float | None = None
    if bool(hv_cfg.get("enabled", True)):
        reference = cast(list[float], hv_cfg.get("reference", [0.0, 0.0, 1000.0]))
        hv_value = hypervolume_from_rows(rows, primary_metrics, reference)

    payload = {
        "k_objectives": k,
        "objective_names": objective_names,
        "primary_metrics": primary_metrics,
        "results": rows,
        "pareto": pareto_rows,
        "hypervolume": hv_value,
    }

    results_path = os.path.join(out_dir, "morl_results.json")
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)

    _write_csv(os.path.join(out_dir, "morl_table.csv"), rows, objective_names)
    _write_markdown(
        os.path.join(out_dir, "morl.md"),
        rows=rows,
        pareto_rows=pareto_rows,
        primary_metrics=primary_metrics,
        hypervolume=hv_value,
    )

    if hv_value is not None:
        with open(os.path.join(out_dir, "hypervolume.json"), "w") as f:
            json.dump(
                {
                    "primary_metrics": primary_metrics,
                    "value": hv_value,
                },
                f,
                indent=2,
            )

    return payload
