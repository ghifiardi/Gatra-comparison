from __future__ import annotations

import json
from pathlib import Path

import yaml

from architecture_a_rl.morl.meta_controller import select_weight


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_meta_controller_autorelax_finds_feasible_step(tmp_path: Path) -> None:
    results_path = tmp_path / "morl_results_val.json"
    _write_json(
        results_path,
        {
            "results": [
                {
                    "w": [0.8, 0.2],
                    "metrics": {
                        "pr_auc": 0.90,
                        "f1": 0.30,
                        "precision": 0.40,
                        "recall": 0.58,
                        "roc_auc": 0.70,
                        "alerts_per_1k": 35.0,
                    },
                    "objective_means": {"ttt": -0.1, "coverage": 0.2},
                    "meta": {"seed": 42},
                },
                {
                    "w": [0.5, 0.5],
                    "metrics": {
                        "pr_auc": 0.70,
                        "f1": 0.20,
                        "precision": 0.50,
                        "recall": 0.52,
                        "roc_auc": 0.60,
                        "alerts_per_1k": 25.0,
                    },
                    "objective_means": {"ttt": -0.2, "coverage": 0.1},
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
                "fail_on_infeasible": False,
                "candidates": {
                    "source": "explicit",
                    "explicit": [[0.8, 0.2], [0.5, 0.5]],
                },
                "objective": {
                    "primary": "pr_auc",
                    "tie_breaker": "alerts_per_1k",
                    "utility": {"pr_auc": 1.0, "alerts_per_1k": -0.01},
                },
                "constraints": {
                    "mode": "hard",
                    "alerts_per_1k_max": 30.0,
                    "recall_min": 0.60,
                    "precision_min": None,
                    "pr_auc_min": None,
                    "soft_penalty": {
                        "lambda_recall": 5.0,
                        "lambda_alerts": 0.05,
                        "lambda_precision": 1.0,
                        "lambda_pr_auc": 1.0,
                    },
                },
                "relaxation": {
                    "enabled": True,
                    "stop_on_first_feasible": True,
                    "schedule": [
                        {"recall_min": 0.55, "alerts_per_1k_max": 40.0},
                        {"recall_min": 0.50, "alerts_per_1k_max": 80.0},
                    ],
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
    morl_cfg_path = tmp_path / "morl.yaml"
    _write_yaml(morl_cfg_path, {"morl": {"k_objectives": 2}})

    selected = select_weight(
        meta_cfg_path=str(meta_cfg_path),
        morl_cfg_path=str(morl_cfg_path),
        val_results_path=str(results_path),
        seed=42,
    )

    assert selected["selected_weight"] == [0.8, 0.2]
    assert selected["feasible_under_original_constraints"] is False
    feasibility = selected["feasibility"]
    assert feasibility["fallback_used"] is True
    assert feasibility["fallback_mode"] == "auto_relaxation"
    assert feasibility["relaxation_step_used"] == 0
    assert feasibility["final_constraints_used"]["recall_min"] == 0.55
