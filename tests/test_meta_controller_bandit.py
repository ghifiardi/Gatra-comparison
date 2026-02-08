from __future__ import annotations

import json
from pathlib import Path

import yaml

from architecture_a_rl.morl.meta_controller import select_weight


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_common_files(tmp_path: Path) -> tuple[Path, Path]:
    results_path = tmp_path / "morl_results_val.json"
    _write_json(
        results_path,
        {
            "results": [
                {
                    "w": [0.7, 0.2, 0.1],
                    "metrics": {"pr_auc": 0.90, "f1": 0.20, "precision": 0.50, "recall": 0.70, "roc_auc": 0.70, "alerts_per_1k": 35.0},
                    "objective_means": {"detect": 0.2, "fp_cost": -0.3, "analyst_cost": -0.3},
                    "meta": {"seed": 7},
                },
                {
                    "w": [0.5, 0.3, 0.2],
                    "metrics": {"pr_auc": 0.75, "f1": 0.10, "precision": 0.40, "recall": 0.60, "roc_auc": 0.60, "alerts_per_1k": 15.0},
                    "objective_means": {"detect": 0.1, "fp_cost": -0.2, "analyst_cost": -0.2},
                    "meta": {"seed": 7},
                },
                {
                    "w": [0.2, 0.2, 0.6],
                    "metrics": {"pr_auc": 0.65, "f1": 0.08, "precision": 0.30, "recall": 0.50, "roc_auc": 0.55, "alerts_per_1k": 8.0},
                    "objective_means": {"detect": 0.1, "fp_cost": -0.1, "analyst_cost": -0.1},
                    "meta": {"seed": 7},
                },
            ]
        },
    )

    morl_cfg_path = tmp_path / "morl.yaml"
    _write_yaml(morl_cfg_path, {"morl": {"k_objectives": 3}})
    return results_path, morl_cfg_path


def test_meta_controller_ucb_is_deterministic(tmp_path: Path) -> None:
    results_path, morl_cfg_path = _write_common_files(tmp_path)
    meta_cfg_path = tmp_path / "meta_ucb.yaml"
    _write_yaml(
        meta_cfg_path,
        {
            "meta_controller": {
                "enabled": True,
                "selection_split": "val",
                "candidates": {
                    "source": "explicit",
                    "explicit": [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.2, 0.2, 0.6]],
                },
                "objective": {
                    "primary": "pr_auc",
                    "tie_breaker": "alerts_per_1k",
                    "utility": {"pr_auc": 1.0, "alerts_per_1k": -0.01},
                },
                "constraints": {
                    "alerts_per_1k_max": None,
                    "recall_min": None,
                    "precision_min": None,
                    "pr_auc_min": None,
                },
                "method": {
                    "name": "bandit_ucb",
                    "params": {"rounds": 10, "explore_frac": 0.2, "ucb_c": 1.0, "thompson_sigma": 0.02},
                },
            }
        },
    )

    selected = select_weight(
        meta_cfg_path=str(meta_cfg_path),
        morl_cfg_path=str(morl_cfg_path),
        val_results_path=str(results_path),
        seed=7,
    )
    assert selected["selected_weight"] == [0.7, 0.2, 0.1]
    assert len(selected["selection_trace"]) == 10


def test_meta_controller_thompson_is_deterministic(tmp_path: Path) -> None:
    results_path, morl_cfg_path = _write_common_files(tmp_path)
    meta_cfg_path = tmp_path / "meta_thompson.yaml"
    _write_yaml(
        meta_cfg_path,
        {
            "meta_controller": {
                "enabled": True,
                "selection_split": "val",
                "candidates": {
                    "source": "explicit",
                    "explicit": [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.2, 0.2, 0.6]],
                },
                "objective": {
                    "primary": "pr_auc",
                    "tie_breaker": "alerts_per_1k",
                    "utility": {"pr_auc": 1.0, "alerts_per_1k": -0.01},
                },
                "constraints": {
                    "alerts_per_1k_max": None,
                    "recall_min": None,
                    "precision_min": None,
                    "pr_auc_min": None,
                },
                "method": {
                    "name": "bandit_thompson",
                    "params": {"rounds": 10, "explore_frac": 0.2, "ucb_c": 1.0, "thompson_sigma": 0.02},
                },
            }
        },
    )

    selected = select_weight(
        meta_cfg_path=str(meta_cfg_path),
        morl_cfg_path=str(morl_cfg_path),
        val_results_path=str(results_path),
        seed=7,
    )
    assert selected["selected_weight"] == [0.5, 0.3, 0.2]
    assert len(selected["selection_trace"]) == 10
