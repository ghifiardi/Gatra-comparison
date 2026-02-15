from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _format_weight(raw: Any) -> str:
    if not isinstance(raw, list):
        return ""
    vals: list[str] = []
    for v in raw:
        if isinstance(v, (int, float)):
            vals.append(f"{float(v):.6g}")
        else:
            vals.append(str(v))
    return ";".join(vals)


def _run_id_from_dir(run_dir: Path) -> str:
    return run_dir.name


def _flatten_metric(metrics: dict[str, Any], key: str) -> Any:
    val = metrics.get(key)
    if isinstance(val, (int, float, str)) or val is None:
        return val
    return ""


def _extract_row(run_dir: Path, meta: dict[str, str]) -> dict[str, Any]:
    morl_dir = run_dir / "eval" / "morl"
    policy_dir = run_dir / "eval" / "policy"
    report_manifest = run_dir / "report" / "run_manifest.json"
    meta_stability_json = run_dir / "eval" / "meta_stability" / "meta_stability.json"

    meta_selection = _read_json(morl_dir / "meta_selection.json")
    meta_feasibility = _read_json(morl_dir / "meta_feasibility.json")
    morl_selected = _read_json(morl_dir / "morl_selected_test.json")
    policy_eval = _read_json(policy_dir / "policy_eval.json")
    manifest = _read_json(report_manifest)
    stability = _read_json(meta_stability_json)

    selected_metrics = morl_selected.get("metrics", {})
    if not isinstance(selected_metrics, dict):
        selected_metrics = {}

    objective_means = morl_selected.get("objective_means", {})
    if not isinstance(objective_means, dict):
        objective_means = {}

    ppo_sel_test = (
        policy_eval.get("models", {}).get("ppo", {}).get("selected_test", {})
        if isinstance(policy_eval.get("models", {}), dict)
        else {}
    )
    if not isinstance(ppo_sel_test, dict):
        ppo_sel_test = {}

    if_sel_test = (
        policy_eval.get("models", {}).get("iforest", {}).get("selected_test", {})
        if isinstance(policy_eval.get("models", {}), dict)
        else {}
    )
    if not isinstance(if_sel_test, dict):
        if_sel_test = {}

    agg = stability.get("aggregate", {})
    if not isinstance(agg, dict):
        agg = {}

    config_snapshot = manifest.get("config_snapshot", {})
    if not isinstance(config_snapshot, dict):
        config_snapshot = {}

    morl_manifest = manifest.get("morl", {})
    if not isinstance(morl_manifest, dict):
        morl_manifest = {}

    return {
        "run_id": _run_id_from_dir(run_dir),
        "run_dir": str(run_dir),
        "run_group": meta.get("run_group", ""),
        "condition": meta.get("condition", ""),
        "seed": meta.get("seed", ""),
        "backend": meta.get("backend", ""),
        "status": meta.get("status", ""),
        "selected_weight": _format_weight(meta_selection.get("selected_weight")),
        "fallback_used": meta_feasibility.get("fallback_used"),
        "fallback_mode": meta_feasibility.get("fallback_mode"),
        "feasible_count_initial": meta_feasibility.get("feasible_count_initial"),
        "relax_trace_len": len(meta_feasibility.get("relaxation_trace", []))
        if isinstance(meta_feasibility.get("relaxation_trace", []), list)
        else "",
        "selection_rationale": meta_feasibility.get("selection_rationale"),
        "morl_precision": _flatten_metric(selected_metrics, "precision"),
        "morl_recall": _flatten_metric(selected_metrics, "recall"),
        "morl_f1": _flatten_metric(selected_metrics, "f1"),
        "morl_pr_auc": _flatten_metric(selected_metrics, "pr_auc"),
        "morl_roc_auc": _flatten_metric(selected_metrics, "roc_auc"),
        "morl_alerts_per_1k": _flatten_metric(selected_metrics, "alerts_per_1k"),
        "morl_objective_means_json": json.dumps(objective_means, sort_keys=True),
        "policy_ppo_precision": _flatten_metric(ppo_sel_test, "precision"),
        "policy_ppo_recall": _flatten_metric(ppo_sel_test, "recall"),
        "policy_ppo_f1": _flatten_metric(ppo_sel_test, "f1"),
        "policy_ppo_pr_auc": _flatten_metric(ppo_sel_test, "pr_auc"),
        "policy_ppo_alerts_per_1k": _flatten_metric(ppo_sel_test, "alerts_per_1k"),
        "policy_ppo_threshold": _flatten_metric(ppo_sel_test, "threshold"),
        "policy_iforest_precision": _flatten_metric(if_sel_test, "precision"),
        "policy_iforest_recall": _flatten_metric(if_sel_test, "recall"),
        "policy_iforest_f1": _flatten_metric(if_sel_test, "f1"),
        "policy_iforest_pr_auc": _flatten_metric(if_sel_test, "pr_auc"),
        "policy_iforest_alerts_per_1k": _flatten_metric(if_sel_test, "alerts_per_1k"),
        "policy_iforest_threshold": _flatten_metric(if_sel_test, "threshold"),
        "objective_source": morl_manifest.get("objective_source"),
        "normalization_applied": (morl_manifest.get("normalization_summary") or {}).get("applied")
        if isinstance(morl_manifest.get("normalization_summary"), dict)
        else "",
        "data_config_path": config_snapshot.get("data"),
        "morl_config_path": config_snapshot.get("morl"),
        "meta_config_path": config_snapshot.get("meta_controller"),
        "manifest_path": str(report_manifest),
        "meta_stability_selection_change_rate": agg.get("selection_change_rate"),
        "meta_stability_avg_weight_l1_distance": agg.get("avg_weight_l1_distance"),
        "meta_stability_constraint_violation_rate": agg.get("constraint_violation_rate"),
        "meta_stability_avg_regret": agg.get("avg_regret"),
        "meta_stability_worst_regret": agg.get("worst_regret"),
        "meta_stability_worst_regret_condition": agg.get("worst_regret_condition"),
    }


def _rows_from_index(index_path: Path) -> list[tuple[Path, dict[str, str]]]:
    rows: list[tuple[Path, dict[str, str]]] = []
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    with index_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            run_dir = (rec.get("run_dir") or "").strip()
            if not run_dir:
                continue
            rows.append((Path(run_dir), rec))
    return rows


def _expand_index_paths(indices: list[Path], index_globs: list[str]) -> list[Path]:
    merged: list[Path] = []
    for idx in indices:
        merged.append(idx)
    for pattern in index_globs:
        for raw in sorted(glob.glob(pattern)):
            merged.append(Path(raw))
    deduped: list[Path] = []
    seen: set[str] = set()
    for p in merged:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def collect_rows(
    indices: list[Path], index_globs: list[str], run_dirs: list[Path]
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    outputs: list[dict[str, Any]] = []

    indexed: list[tuple[Path, dict[str, str]]] = []
    for idx in _expand_index_paths(indices, index_globs):
        indexed.extend(_rows_from_index(idx))

    for rd in run_dirs:
        indexed.append((rd, {}))

    for run_dir, meta in indexed:
        key = (
            (meta.get("condition") or "").strip(),
            (meta.get("seed") or "").strip(),
            str(run_dir),
        )
        if key in seen:
            continue
        seen.add(key)
        if not run_dir.exists():
            outputs.append(
                {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "run_group": meta.get("run_group", ""),
                    "condition": meta.get("condition", ""),
                    "seed": meta.get("seed", ""),
                    "backend": meta.get("backend", ""),
                    "status": "missing_run_dir",
                }
            )
            continue
        outputs.append(_extract_row(run_dir, meta))
    return outputs


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_dir",
        "run_group",
        "condition",
        "seed",
        "backend",
        "status",
        "selected_weight",
        "fallback_used",
        "fallback_mode",
        "feasible_count_initial",
        "relax_trace_len",
        "selection_rationale",
        "morl_precision",
        "morl_recall",
        "morl_f1",
        "morl_pr_auc",
        "morl_roc_auc",
        "morl_alerts_per_1k",
        "morl_objective_means_json",
        "policy_ppo_precision",
        "policy_ppo_recall",
        "policy_ppo_f1",
        "policy_ppo_pr_auc",
        "policy_ppo_alerts_per_1k",
        "policy_ppo_threshold",
        "policy_iforest_precision",
        "policy_iforest_recall",
        "policy_iforest_f1",
        "policy_iforest_pr_auc",
        "policy_iforest_alerts_per_1k",
        "policy_iforest_threshold",
        "objective_source",
        "normalization_applied",
        "data_config_path",
        "morl_config_path",
        "meta_config_path",
        "manifest_path",
        "meta_stability_selection_change_rate",
        "meta_stability_avg_weight_l1_distance",
        "meta_stability_constraint_violation_rate",
        "meta_stability_avg_regret",
        "meta_stability_worst_regret",
        "meta_stability_worst_regret_condition",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect paper Week-1 run artifacts into one CSV")
    parser.add_argument(
        "--index",
        action="append",
        type=Path,
        default=[],
        help="CSV index from scripts/paper_matrix.sh (condition/seed/run_dir mapping)",
    )
    parser.add_argument(
        "--index-glob",
        action="append",
        default=[],
        help="Glob for index CSV paths (can be repeated)",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        type=Path,
        help="Run directory to collect (can be repeated)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/paper_results/paper_week1_results.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.index and not args.index_glob and not args.run_dir:
        raise SystemExit("Provide --index or at least one --run-dir")

    rows = collect_rows(args.index, args.index_glob, args.run_dir)
    write_csv(rows, args.out)
    print(f"Wrote: {args.out}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
