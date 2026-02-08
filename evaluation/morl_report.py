from __future__ import annotations

import csv
from dataclasses import asdict
import json
import os
from typing import Any, Sequence, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from architecture_a_rl.morl.networks import PreferenceConditionedActor
from architecture_a_rl.morl.objectives import parse_objectives
from architecture_a_rl.morl.objectives_realdata import compute_reward_matrix_realdata_aware
from architecture_a_rl.morl.preferences import normalize_weight_grid
from evaluation.metrics import classification_metrics
from evaluation.pareto import hypervolume_from_rows, pareto_filter


def _load_morl_cfg(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = cast(dict[str, Any], yaml.safe_load(f))
    if "morl" not in cfg:
        raise ValueError("Missing 'morl' root in MORL config")
    return cast(dict[str, Any], cfg["morl"])


def _load_contract_split(
    contract_dir: str, split: str
) -> tuple[NDArray[np.float32], NDArray[np.int_]]:
    x = np.load(os.path.join(contract_dir, f"features_v128_{split}.npy")).astype(np.float32)
    if split == "test":
        y = np.load(os.path.join(contract_dir, "y_true.npy")).astype(np.int_)
    else:
        y = np.load(os.path.join(contract_dir, f"y_{split}.npy")).astype(np.int_)
    return x, y


def _safe_metric_value(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _write_csv(path: str, rows: list[dict[str, Any]], objective_names: list[str]) -> None:
    cols = [
        "w",
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
                "w": json.dumps(row["w"]),
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
    objective_specs: list[dict[str, object]],
    primary_metrics: list[str],
    hypervolume: float | None,
    split: str,
    objective_source: str,
    level1_stats: dict[str, float],
    normalization_summary: dict[str, Any],
) -> None:
    normalization_applied = bool(normalization_summary.get("applied", False))
    signals_used = "normalized objective signals" if normalization_applied else "raw objective signals"
    lines = [
        "# MORL Evaluation Summary",
        "",
        f"- Split: {split}",
        f"- Evaluated weights: {len(rows)}",
        f"- Pareto candidates: {len(pareto_rows)}",
        f"- Primary metrics: {', '.join(primary_metrics)}",
        f"- Objective source: {objective_source}",
        f"- Reward signals used: {signals_used}",
    ]
    if hypervolume is not None:
        lines.append(f"- Hypervolume: {hypervolume:.6f}")
    lines.extend(["", "## Objectives"])
    for spec in objective_specs:
        name = str(spec.get("name", "unknown"))
        kind = str(spec.get("type", "unknown"))
        if kind == "classification":
            details = (
                f"tp={_to_float(spec.get('tp', 0.0))}, "
                f"fn={_to_float(spec.get('fn', 0.0))}, "
                f"fp={_to_float(spec.get('fp', 0.0))}, "
                f"tn={_to_float(spec.get('tn', 0.0))}"
            )
        elif kind == "fp_penalty":
            details = f"fp_penalty={_to_float(spec.get('fp_penalty', 0.0))}"
        elif kind == "per_alert_cost":
            details = f"per_alert_penalty={_to_float(spec.get('per_alert_penalty', 0.0))}"
        else:
            details = "n/a"
        lines.append(f"- `{name}` ({kind}): {details}")
    if normalization_summary:
        lines.extend(["", "## Normalization"])
        lines.append(f"- applied: {normalization_applied}")
        lines.append(f"- reference_split: {normalization_summary.get('reference_split')}")
        lines.append(f"- apply_to: {normalization_summary.get('apply_to')}")
        lines.append(f"- clip_z: {normalization_summary.get('clip_z')}")
        lines.append(f"- eps: {normalization_summary.get('eps')}")
        objectives_stats = normalization_summary.get("objectives", {})
        if isinstance(objectives_stats, dict) and objectives_stats:
            lines.extend(
                [
                    "",
                    "| objective | norm | direction | cap_pctl | mean | std | clip_z |",
                    "| --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for name, stat_raw in objectives_stats.items():
                if not isinstance(stat_raw, dict):
                    continue
                stat = cast(dict[str, Any], stat_raw)
                lines.append(
                    "| "
                    f"{name} | {stat.get('norm')} | {stat.get('direction')} |"
                    f" {stat.get('cap_pctl')} | {float(stat.get('mean', 0.0)):.6f} |"
                    f" {float(stat.get('std', 0.0)):.6f} | {float(stat.get('clip_z', 0.0)):.2f} |"
                )
    if objective_source == "level1_realdata":
        lines.extend(
            [
                "",
                "## Level-1 objective definitions",
                "- `time_to_triage_seconds`: per-session max(timestamp)-min(timestamp); minimized in reward.",
                "- `detection_coverage`: novelty of `(page, action)` interactions within session; maximized in reward.",
            ]
        )
        if level1_stats:
            lines.extend(
                [
                    "",
                    "## Level-1 raw stats",
                    f"- ttt_raw_median_seconds: {level1_stats.get('ttt_raw_median_seconds', 0.0):.3f}",
                    f"- ttt_raw_p90_seconds: {level1_stats.get('ttt_raw_p90_seconds', 0.0):.3f}",
                    f"- coverage_raw_novelty_rate: {level1_stats.get('coverage_raw_novelty_rate', 0.0):.6f}",
                    f"- coverage_raw_rate_per_1k: {level1_stats.get('coverage_raw_rate_per_1k', 0.0):.3f}",
                    "",
                    "## Level-1 normalized stats (z-space)",
                    f"- ttt_norm_median_z: {level1_stats.get('ttt_norm_median_z', 0.0):.6f}",
                    f"- ttt_norm_p90_z: {level1_stats.get('ttt_norm_p90_z', 0.0):.6f}",
                    f"- coverage_norm_median_z: {level1_stats.get('coverage_norm_median_z', 0.0):.6f}",
                    f"- coverage_norm_p90_z: {level1_stats.get('coverage_norm_p90_z', 0.0):.6f}",
                ]
            )
    lines.extend(
        [
            "",
            "## Practitioner guidance",
            "- Pick weights on the Pareto set to avoid dominated operating points.",
            "- Use higher detect weight when recall is prioritized over analyst load.",
            "- Use higher analyst_cost weight when alert budget is constrained.",
            "",
            "## Pareto candidates",
            "| w | pr_auc | f1 | alerts_per_1k |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in pareto_rows:
        m = cast(dict[str, float], row["metrics"])
        lines.append(
            f"| `{row['w']}` | {m['pr_auc']:.4f} | {m['f1']:.4f} | {m['alerts_per_1k']:.2f} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _load_actor(
    morl_model_dir: str,
    hidden: list[int],
    k_objectives: int,
) -> tuple[PreferenceConditionedActor, int]:
    actor = PreferenceConditionedActor(
        state_dim=128, k_objectives=k_objectives, hidden=hidden, action_dim=2
    )
    actor_path = os.path.join(morl_model_dir, "actor.pt")
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    actor.eval()

    seed = 0
    meta_path = os.path.join(morl_model_dir, "morl_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = cast(dict[str, Any], json.load(f))
        seed = int(meta.get("seed", 0))
    return actor, seed


def evaluate_morl_weight_on_split(
    contract_dir: str,
    morl_model_dir: str,
    morl_cfg_path: str,
    split: str,
    w: Sequence[float],
) -> dict[str, Any]:
    morl_cfg = _load_morl_cfg(morl_cfg_path)
    k = int(morl_cfg.get("k_objectives", 3))
    objectives = parse_objectives(cast(list[dict[str, object]], morl_cfg.get("objectives", [])))
    if len(objectives) != k:
        raise ValueError(f"Expected {k} objectives, got {len(objectives)}")

    train_cfg = cast(dict[str, Any], morl_cfg.get("training", {}))
    hidden = [int(v) for v in cast(list[int], train_cfg.get("hidden", [256, 128, 64]))]
    realdata_cfg = cast(dict[str, Any], morl_cfg.get("realdata_objectives", {}))
    legacy_default_norm = str(realdata_cfg.get("normalization", "none"))
    normalization_cfg = cast(dict[str, Any], morl_cfg.get("normalization", {}))
    actor, seed = _load_actor(morl_model_dir=morl_model_dir, hidden=hidden, k_objectives=k)

    x_split, y_split = _load_contract_split(contract_dir, split)
    w_arr = normalize_weight_grid([list(w)], k)[0]

    with torch.no_grad():
        states = torch.tensor(x_split, dtype=torch.float32)
        w_rep = np.repeat(w_arr[None, :], x_split.shape[0], axis=0).astype(np.float32)
        w_t = torch.tensor(w_rep, dtype=torch.float32)
        xw = torch.cat([states, w_t], dim=1)
        probs = actor(xw).cpu().numpy()
        y_score = probs[:, 1].astype(np.float64)
        actions = (y_score >= 0.5).astype(np.int_)

    base_metrics = classification_metrics(y_split, y_score, threshold=0.5)
    metrics = {
        "precision": _safe_metric_value(base_metrics.get("precision", 0.0)),
        "recall": _safe_metric_value(base_metrics.get("recall", 0.0)),
        "f1": _safe_metric_value(base_metrics.get("f1", 0.0)),
        "roc_auc": _safe_metric_value(base_metrics.get("roc_auc", 0.0)),
        "pr_auc": _safe_metric_value(base_metrics.get("pr_auc", 0.0)),
        "alerts_per_1k": float(1000.0 * np.mean(actions == 1)),
    }
    objective_names = [o.name for o in objectives]
    reward_result = compute_reward_matrix_realdata_aware(
        y_true=y_split,
        actions=actions,
        objectives=objectives,
        contract_dir=contract_dir,
        split=split,
        normalization_cfg=normalization_cfg,
        legacy_default_norm=legacy_default_norm,
    )
    reward_matrix = reward_result.reward_matrix
    objective_means = {objective_names[i]: float(np.mean(reward_matrix[:, i])) for i in range(k)}

    row = {
        "w": [float(v) for v in w_arr.tolist()],
        "weights": [float(v) for v in w_arr.tolist()],
        "metrics": metrics,
        "objective_means": objective_means,
        "meta": {
            "seed": seed,
            "objective_source": reward_result.source,
            "level1_stats": reward_result.stats,
            "normalization": reward_result.normalization,
        },
    }
    return row


def run_morl_weight_sweep(
    contract_dir: str,
    morl_model_dir: str,
    morl_cfg_path: str,
    out_dir: str,
    split: str = "test",
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split for MORL sweep: {split}")

    morl_cfg = _load_morl_cfg(morl_cfg_path)
    k = int(morl_cfg.get("k_objectives", 3))
    objectives = parse_objectives(cast(list[dict[str, object]], morl_cfg.get("objectives", [])))
    if len(objectives) != k:
        raise ValueError(f"Expected {k} objectives, got {len(objectives)}")

    eval_cfg = cast(dict[str, Any], morl_cfg.get("eval", {}))
    sweep_cfg = cast(dict[str, Any], eval_cfg.get("weight_sweep", {}))
    weight_grid = normalize_weight_grid(cast(list[list[float]], sweep_cfg.get("grid", [])), k)

    rows: list[dict[str, Any]] = []
    for w in weight_grid:
        rows.append(
            evaluate_morl_weight_on_split(
                contract_dir=contract_dir,
                morl_model_dir=morl_model_dir,
                morl_cfg_path=morl_cfg_path,
                split=split,
                w=[float(v) for v in w.tolist()],
            )
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

    objective_source = "fallback_synthetic"
    level1_stats: dict[str, float] = {}
    normalization_summary: dict[str, Any] = {}
    if rows:
        first_meta = cast(dict[str, Any], rows[0].get("meta", {}))
        objective_source = str(first_meta.get("objective_source", "fallback_synthetic"))
        stats_raw = first_meta.get("level1_stats", {})
        if isinstance(stats_raw, dict):
            level1_stats = {str(k): float(v) for k, v in stats_raw.items()}
        norm_raw = first_meta.get("normalization", {})
        if isinstance(norm_raw, dict):
            normalization_summary = cast(dict[str, Any], norm_raw)

    payload = {
        "split": split,
        "k_objectives": k,
        "objective_names": [o.name for o in objectives],
        "objective_source": objective_source,
        "level1_stats": level1_stats,
        "normalization_summary": normalization_summary,
        "primary_metrics": primary_metrics,
        "results": rows,
        "pareto": pareto_rows,
        "hypervolume": hv_value,
    }

    results_path = os.path.join(out_dir, f"morl_results_{split}.json")
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)

    _write_csv(
        os.path.join(out_dir, f"morl_table_{split}.csv"),
        rows,
        [o.name for o in objectives],
    )
    _write_markdown(
        os.path.join(out_dir, f"morl_{split}.md"),
        rows=rows,
        pareto_rows=pareto_rows,
        objective_specs=[cast(dict[str, object], asdict(o)) for o in objectives],
        primary_metrics=primary_metrics,
        hypervolume=hv_value,
        split=split,
        objective_source=objective_source,
        level1_stats=level1_stats,
        normalization_summary=normalization_summary,
    )

    if hv_value is not None:
        with open(os.path.join(out_dir, f"hypervolume_{split}.json"), "w") as f:
            json.dump(
                {"split": split, "primary_metrics": primary_metrics, "value": hv_value}, f, indent=2
            )

    # Backward compatibility for existing v0.5.0 consumers expecting test filenames.
    if split == "test":
        with open(os.path.join(out_dir, "morl_results.json"), "w") as f:
            json.dump(payload, f, indent=2)
        _write_csv(os.path.join(out_dir, "morl_table.csv"), rows, [o.name for o in objectives])
        _write_markdown(
            os.path.join(out_dir, "morl.md"),
            rows=rows,
            pareto_rows=pareto_rows,
            objective_specs=[cast(dict[str, object], asdict(o)) for o in objectives],
            primary_metrics=primary_metrics,
            hypervolume=hv_value,
            split=split,
            objective_source=objective_source,
            level1_stats=level1_stats,
            normalization_summary=normalization_summary,
        )
        if hv_value is not None:
            with open(os.path.join(out_dir, "hypervolume.json"), "w") as f:
                json.dump(
                    {"split": split, "primary_metrics": primary_metrics, "value": hv_value},
                    f,
                    indent=2,
                )

    return payload
