from __future__ import annotations

import json
from pathlib import Path

import yaml

from architecture_a_rl.morl.meta_controller import select_weight


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_meta_controller_greedy_constraints_and_tie_break(tmp_path: Path) -> None:
    results_path = tmp_path / "morl_results_val.json"
    _write_json(
        results_path,
        {
            "results": [
                {
                    "w": [0.7, 0.2, 0.1],
                    "metrics": {
                        "pr_auc": 0.80,
                        "f1": 0.10,
                        "precision": 0.40,
                        "recall": 0.70,
                        "roc_auc": 0.60,
                        "alerts_per_1k": 50.0,
                    },
                    "objective_means": {"detect": 0.1, "fp_cost": -0.2, "analyst_cost": -0.3},
                    "meta": {"seed": 42},
                },
                {
                    "w": [0.5, 0.3, 0.2],
                    "metrics": {
                        "pr_auc": 0.75,
                        "f1": 0.12,
                        "precision": 0.50,
                        "recall": 0.65,
                        "roc_auc": 0.62,
                        "alerts_per_1k": 25.0,
                    },
                    "objective_means": {"detect": 0.2, "fp_cost": -0.1, "analyst_cost": -0.2},
                    "meta": {"seed": 42},
                },
                {
                    "w": [0.4, 0.4, 0.2],
                    "metrics": {
                        "pr_auc": 0.75,
                        "f1": 0.11,
                        "precision": 0.49,
                        "recall": 0.68,
                        "roc_auc": 0.61,
                        "alerts_per_1k": 20.0,
                    },
                    "objective_means": {"detect": 0.2, "fp_cost": -0.2, "analyst_cost": -0.1},
                    "meta": {"seed": 42},
                },
                {
                    "w": [0.2, 0.2, 0.6],
                    "metrics": {
                        "pr_auc": 0.70,
                        "f1": 0.20,
                        "precision": 0.70,
                        "recall": 0.80,
                        "roc_auc": 0.66,
                        "alerts_per_1k": 10.0,
                    },
                    "objective_means": {"detect": 0.4, "fp_cost": -0.1, "analyst_cost": -0.1},
                    "meta": {"seed": 42},
                },
            ]
        },
    )

    meta_cfg_path = tmp_path / "meta.yaml"
    _write_yaml(
        meta_cfg_path,
        {
            "meta_controller": {
                "enabled": True,
                "selection_split": "val",
                "candidates": {
                    "source": "explicit",
                    "explicit": [
                        [0.7, 0.2, 0.1],
                        [0.5, 0.3, 0.2],
                        [0.4, 0.4, 0.2],
                        [0.2, 0.2, 0.6],
                    ],
                },
                "objective": {
                    "primary": "pr_auc",
                    "tie_breaker": "alerts_per_1k",
                    "utility": {"pr_auc": 1.0, "alerts_per_1k": -0.01},
                },
                "constraints": {
                    "alerts_per_1k_max": 30.0,
                    "recall_min": 0.60,
                    "precision_min": None,
                    "pr_auc_min": None,
                },
                "method": {
                    "name": "greedy",
                    "params": {
                        "rounds": 10,
                        "explore_frac": 0.2,
                        "ucb_c": 1.0,
                        "thompson_sigma": 0.05,
                    },
                },
            }
        },
    )

    # Unused for explicit candidates, but required by interface.
    morl_cfg_path = tmp_path / "morl.yaml"
    _write_yaml(morl_cfg_path, {"morl": {"k_objectives": 3}})

    selected = select_weight(
        meta_cfg_path=str(meta_cfg_path),
        morl_cfg_path=str(morl_cfg_path),
        val_results_path=str(results_path),
        seed=42,
    )

    assert selected["selected_weight"] == [0.4, 0.4, 0.2]
    assert selected["feasible_count"] == 3
    assert selected["candidate_count"] == 4
