from __future__ import annotations

import json
from pathlib import Path

import yaml

from evaluation.meta_stability import run_stability_suite, write_stability_artifacts


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_stability_suite_smoke(tmp_path: Path) -> None:
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
                        "recall": 0.65,
                        "roc_auc": 0.70,
                        "alerts_per_1k": 25.0,
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
                        "recall": 0.55,
                        "roc_auc": 0.60,
                        "alerts_per_1k": 20.0,
                    },
                    "objective_means": {"ttt": -0.2, "coverage": 0.1},
                    "meta": {"seed": 42},
                },
                {
                    "w": [0.3, 0.7],
                    "metrics": {
                        "pr_auc": 0.60,
                        "f1": 0.15,
                        "precision": 0.55,
                        "recall": 0.70,
                        "roc_auc": 0.55,
                        "alerts_per_1k": 15.0,
                    },
                    "objective_means": {"ttt": -0.3, "coverage": 0.05},
                    "meta": {"seed": 42},
                },
            ]
        },
    )

    stability_cfg_path = tmp_path / "meta_stability.yaml"
    _write_yaml(
        stability_cfg_path,
        {
            "meta_stability": {
                "objective_primary": "pr_auc",
                "objective_tie_breaker": "alerts_per_1k",
                "utility": {"pr_auc": 1.0, "alerts_per_1k": -0.01},
                "constraints": {
                    "mode": "hard",
                    "alerts_per_1k_max": 30.0,
                    "recall_min": 0.50,
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
                        {"recall_min": 0.40, "alerts_per_1k_max": 50.0},
                    ],
                },
                "fail_on_infeasible": False,
                "method": {
                    "name": "greedy",
                    "params": {
                        "rounds": 10,
                        "explore_frac": 0.2,
                        "ucb_c": 1.0,
                        "thompson_sigma": 0.05,
                    },
                },
                "conditions": [
                    {
                        "name": "noise_low",
                        "kind": "robustness",
                        "overrides": {"noise_sigma": 0.01, "seed": 200},
                    },
                    {
                        "name": "recall_drop",
                        "kind": "label_availability",
                        "overrides": {"recall_scale": 0.80, "seed": 201},
                    },
                    {
                        "name": "alert_budget",
                        "kind": "policy_regime",
                        "overrides": {"alerts_scale": 0.50, "seed": 202},
                    },
                ],
            }
        },
    )

    payload = run_stability_suite(
        val_results_path=str(results_path),
        stability_cfg_path=str(stability_cfg_path),
        seed=42,
    )

    assert "conditions" in payload
    assert "aggregate" in payload
    assert "baseline" in payload
    assert len(payload["conditions"]) == 4  # baseline + 3 conditions

    agg = payload["aggregate"]
    assert "selection_change_rate" in agg
    assert "avg_weight_l1_distance" in agg
    assert "constraint_violation_rate" in agg
    assert "avg_regret" in agg
    assert "worst_regret" in agg
    assert "worst_regret_condition" in agg
    assert agg["n_conditions"] == 3
    assert agg["n_skipped"] == 0

    for cond in payload["conditions"]:
        assert cond["regret"] >= 0.0
        assert cond["best_utility"] >= cond["selected_utility"]

    out_dir = tmp_path / "artifacts"
    artifacts = write_stability_artifacts(str(out_dir), payload)
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["csv"]).exists()
    assert Path(artifacts["md"]).exists()

    loaded = json.loads(Path(artifacts["json"]).read_text())
    assert loaded["aggregate"]["n_conditions"] == 3

    md_content = Path(artifacts["md"]).read_text()
    assert "Conditions" in md_content
    assert "Volatility Summary" in md_content
    assert "Worst-Case Regret" in md_content
    assert "noise_low" in md_content
    assert "recall_drop" in md_content
    assert "alert_budget" in md_content


def test_stability_suite_deterministic(tmp_path: Path) -> None:
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
                        "recall": 0.65,
                        "roc_auc": 0.70,
                        "alerts_per_1k": 25.0,
                    },
                    "objective_means": {},
                    "meta": {},
                },
                {
                    "w": [0.5, 0.5],
                    "metrics": {
                        "pr_auc": 0.70,
                        "f1": 0.20,
                        "precision": 0.50,
                        "recall": 0.55,
                        "roc_auc": 0.60,
                        "alerts_per_1k": 20.0,
                    },
                    "objective_means": {},
                    "meta": {},
                },
            ]
        },
    )

    cfg_path = tmp_path / "meta_stability.yaml"
    _write_yaml(
        cfg_path,
        {
            "meta_stability": {
                "objective_primary": "pr_auc",
                "objective_tie_breaker": "alerts_per_1k",
                "utility": {"pr_auc": 1.0},
                "constraints": {
                    "mode": "hard",
                    "alerts_per_1k_max": 30.0,
                    "recall_min": 0.50,
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
                    "enabled": False,
                    "stop_on_first_feasible": True,
                    "schedule": [],
                },
                "fail_on_infeasible": False,
                "method": {
                    "name": "greedy",
                    "params": {
                        "rounds": 10,
                        "explore_frac": 0.2,
                        "ucb_c": 1.0,
                        "thompson_sigma": 0.05,
                    },
                },
                "conditions": [
                    {
                        "name": "noise",
                        "kind": "robustness",
                        "overrides": {"noise_sigma": 0.02, "seed": 300},
                    },
                ],
            }
        },
    )

    r1 = run_stability_suite(str(results_path), str(cfg_path), seed=42)
    r2 = run_stability_suite(str(results_path), str(cfg_path), seed=42)

    assert r1["conditions"] == r2["conditions"]
    assert r1["aggregate"] == r2["aggregate"]
    assert r1["baseline"] == r2["baseline"]
