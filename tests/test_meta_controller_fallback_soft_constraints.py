from __future__ import annotations

import json
from pathlib import Path

import yaml

from architecture_a_rl.morl.meta_controller import select_weight
from evaluation.meta_selection_report import write_meta_selection_artifacts


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _build_soft_case(tmp_path: Path) -> tuple[Path, Path, Path]:
    results_path = tmp_path / "morl_results_val.json"
    _write_json(
        results_path,
        {
            "results": [
                {
                    "w": [0.9, 0.1],
                    "metrics": {
                        "pr_auc": 0.90,
                        "f1": 0.25,
                        "precision": 0.45,
                        "recall": 0.40,
                        "roc_auc": 0.70,
                        "alerts_per_1k": 60.0,
                    },
                    "objective_means": {"ttt": -0.2, "coverage": 0.2},
                    "meta": {"seed": 42},
                },
                {
                    "w": [0.4, 0.6],
                    "metrics": {
                        "pr_auc": 0.82,
                        "f1": 0.22,
                        "precision": 0.50,
                        "recall": 0.55,
                        "roc_auc": 0.65,
                        "alerts_per_1k": 35.0,
                    },
                    "objective_means": {"ttt": -0.1, "coverage": 0.15},
                    "meta": {"seed": 42},
                },
            ]
        },
    )

    meta_cfg_path = tmp_path / "meta_soft.yaml"
    _write_yaml(
        meta_cfg_path,
        {
            "meta_controller": {
                "enabled": True,
                "selection_split": "val",
                "fail_on_infeasible": False,
                "candidates": {
                    "source": "explicit",
                    "explicit": [[0.9, 0.1], [0.4, 0.6]],
                },
                "objective": {
                    "primary": "pr_auc",
                    "tie_breaker": "alerts_per_1k",
                    "utility": {"pr_auc": 1.0, "alerts_per_1k": -0.01},
                },
                "constraints": {
                    "mode": "soft",
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
                    "schedule": [],
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
    return results_path, meta_cfg_path, morl_cfg_path


def test_meta_controller_soft_constraints_is_deterministic_and_has_diagnostics(
    tmp_path: Path,
) -> None:
    results_path, meta_cfg_path, morl_cfg_path = _build_soft_case(tmp_path)

    s1 = select_weight(
        meta_cfg_path=str(meta_cfg_path),
        morl_cfg_path=str(morl_cfg_path),
        val_results_path=str(results_path),
        seed=42,
    )
    s2 = select_weight(
        meta_cfg_path=str(meta_cfg_path),
        morl_cfg_path=str(morl_cfg_path),
        val_results_path=str(results_path),
        seed=42,
    )

    assert s1["selected_weight"] == s2["selected_weight"] == [0.4, 0.6]
    assert s1["feasibility"]["fallback_mode"] == "soft_constraints"
    assert s1["penalty_breakdown"] is not None
    assert "final_score" in s1["penalty_breakdown"]
    assert "violated_constraints_summary" in s1["feasibility"]

    out_dir = tmp_path / "artifacts"
    selected_test = {"w": s1["selected_weight"], "metrics": {"f1": 0.1}}
    artifacts = write_meta_selection_artifacts(str(out_dir), s1, selected_test)
    assert Path(artifacts["meta_selection_json"]).exists()
    assert Path(artifacts["meta_feasibility_json"]).exists()

    payload = json.loads(Path(artifacts["meta_feasibility_json"]).read_text())
    assert payload["fallback_mode"] == "soft_constraints"
    assert "final_constraints_used" in payload
