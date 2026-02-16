from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime
from typing import Any, cast

import yaml


def utc_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return cast(dict[str, Any], data)


def dump_yaml(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def build_run_manifest(
    run_id: str,
    created_at: str,
    git_commit: str,
    config_hashes: dict[str, str],
    config_snapshot: dict[str, str],
    data_cfg: dict[str, Any],
    schema_hash: str,
    poetry_lock_hash: str | None,
    contract_id: str,
    contract_meta: dict[str, Any],
    mode: str,
    seeds: dict[str, int],
    robustness: dict[str, Any] | None = None,
    morl: dict[str, Any] | None = None,
    meta_controller: dict[str, Any] | None = None,
    join_diagnostics: dict[str, Any] | None = None,
    policy_eval: dict[str, Any] | None = None,
    meta_stability: dict[str, Any] | None = None,
    statistical_analysis: dict[str, Any] | None = None,
    contract_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = data_cfg.get("dataset", {})
    labels = data_cfg.get("labels", {})
    source = dataset.get("source", "unknown")
    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "git_commit": git_commit,
        "schema_hash": schema_hash,
        "mode": mode,
        "config_hashes": config_hashes,
        "config_snapshot": config_snapshot,
        "poetry_lock_sha256": poetry_lock_hash,
        "seeds": seeds,
        "data_source": {
            "source": source,
            "bq_project": dataset.get("bq_project"),
            "bq_dataset": dataset.get("bq_dataset"),
            "bq_events_table": dataset.get("bq_events_table"),
            "bq_labels_table": labels.get("bq_labels_table") or dataset.get("bq_labels_table"),
            "path": dataset.get("path"),
            "label_path": labels.get("path") or dataset.get("label_path"),
        },
        "contract": {
            "id": contract_id,
            "counts": contract_meta.get("counts", {}),
            "label_pos_rate": contract_meta.get("label_pos_rate", {}),
            "splits": contract_meta.get("splits", {}),
            **(contract_cache or {}),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "robustness": robustness or {},
        "morl": morl or {},
        "meta_controller": meta_controller or {},
        "join_diagnostics": join_diagnostics or {},
        "policy_eval": policy_eval or {},
        "meta_stability": meta_stability or {},
        "statistical_analysis": statistical_analysis or {},
    }
    return manifest


def write_run_manifest(path: str, manifest: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def render_summary_md(
    run_id: str,
    git_commit: str,
    schema_hash: str,
    contract_meta: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    train_times: dict[str, float],
    iforest_cfg: dict[str, Any],
    ppo_cfg: dict[str, Any],
    contract_dir: str,
    run_root: str,
    mode: str,
    policy_eval: dict[str, Any] | None = None,
) -> str:
    counts = contract_meta.get("counts", {})
    pos_rates = contract_meta.get("label_pos_rate", {})
    split_lines = [
        f"- Train: n={counts.get('train', 0)}, pos_rate={pos_rates.get('train', 0.0):.4f}",
        f"- Val: n={counts.get('val', 0)}, pos_rate={pos_rates.get('val', 0.0):.4f}",
        f"- Test: n={counts.get('test', 0)}, pos_rate={pos_rates.get('test', 0.0):.4f}",
    ]

    def fmt_metric(value: Any) -> str:
        if isinstance(value, (int, float)):
            v = float(value)
            return "N/A" if math.isnan(v) else f"{v:.4f}"
        return "N/A"

    def row(model_key: str) -> str:
        m = metrics.get(model_key, {})
        return (
            f"| {model_key} | {fmt_metric(m.get('roc_auc', float('nan')))} |"
            f" {fmt_metric(m.get('pr_auc', float('nan')))} | {fmt_metric(m.get('precision', 0.0))} |"
            f" {fmt_metric(m.get('recall', 0.0))} | {fmt_metric(m.get('f1', 0.0))} |"
            f" {thresholds.get(model_key, 0.0):.3f} |"
        )

    splits = contract_meta.get("splits", {})
    train_window = splits.get("train", {})
    val_window = splits.get("val", {})
    test_window = splits.get("test", {})

    lines = [
        "# Run Summary",
        "",
        "## Run metadata",
        f"- Run ID: {run_id}",
        f"- Mode: {mode}",
        f"- Git commit: {git_commit}",
        f"- Schema hash: {schema_hash}",
        "",
        "## Dataset window + counts",
        f"- Train window: {train_window.get('start')} → {train_window.get('end')}",
        f"- Val window: {val_window.get('start')} → {val_window.get('end')}",
        f"- Test window: {test_window.get('start')} → {test_window.get('end')}",
        *split_lines,
        "",
        "## Models",
        f"- IF params: n_estimators={iforest_cfg.get('model', {}).get('n_estimators')}, "
        f"contamination={iforest_cfg.get('model', {}).get('contamination')}",
        f"- PPO params: epochs={ppo_cfg.get('train', {}).get('epochs')}, "
        f"batch_size={ppo_cfg.get('train', {}).get('batch_size')}, "
        f"hidden={ppo_cfg.get('networks', {}).get('hidden_sizes')}",
        "",
        "## Results",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        row("iforest"),
        row("ppo"),
        "",
    ]
    test_pos_rate = float(pos_rates.get("test", 0.0))
    if test_pos_rate <= 0.0 or test_pos_rate >= 1.0:
        lines.extend(
            [
                "## Metric Notes",
                "- ROC AUC omitted: only one class present in TEST labels.",
                "- PR AUC omitted: only one class present in TEST labels.",
                "",
            ]
        )
    if policy_eval:
        lines.extend(
            [
                "## Policy Evaluation",
                f"- Mode: {policy_eval.get('mode')}",
                f"- Primary metric: {policy_eval.get('primary_metric')}",
                f"- Constraints: {policy_eval.get('constraints')}",
                "",
            ]
        )
    lines.extend(
        [
            f"- Train time (IF): {train_times.get('iforest', 0.0):.2f}s",
            f"- Train time (PPO): {train_times.get('ppo', 0.0):.2f}s",
            "",
            "## Interpretation",
            "- Compare F1 and PR-AUC to decide which architecture better fits your operational goals.",
            "- Use thresholds to tune alert volume vs recall.",
            "",
            "## Reproduce",
            f"- Run dir: {run_root}",
            f"- Contract dir: {contract_dir}",
            "- Command:",
            f"  cd {os.path.dirname(run_root)}",
            "  python -m runs.cli \\",
            f"    --data-config {run_root}/config/data.yaml \\",
            f"    --iforest-config {run_root}/config/iforest.yaml \\",
            f"    --ppo-config {run_root}/config/ppo.yaml \\",
            f"    --eval-config {run_root}/config/eval.yaml \\",
            f"    --out-root {os.path.dirname(run_root)} \\",
            f"    --run-id {run_id} \\",
            "    --overwrite",
            "",
        ]
    )
    return "\n".join(lines)
