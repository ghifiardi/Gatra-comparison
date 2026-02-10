from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_collect_paper_results_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "reports" / "runs" / "20260101T000000Z"

    _write_json(
        run_dir / "eval" / "morl" / "meta_selection.json",
        {"selected_weight": [0.7, 0.2, 0.1]},
    )
    _write_json(
        run_dir / "eval" / "morl" / "meta_feasibility.json",
        {
            "fallback_used": True,
            "fallback_mode": "soft_constraints",
            "feasible_count_initial": 0,
            "relaxation_trace": [{"step": 0}],
            "selection_rationale": "soft_fallback",
        },
    )
    _write_json(
        run_dir / "eval" / "morl" / "morl_selected_test.json",
        {
            "metrics": {
                "precision": 0.2,
                "recall": 0.4,
                "f1": 0.26,
                "pr_auc": 0.11,
                "roc_auc": 0.52,
                "alerts_per_1k": 14.0,
            },
            "objective_means": {"detect": 0.1, "fp_cost": -0.2},
        },
    )
    _write_json(
        run_dir / "eval" / "policy" / "policy_eval.json",
        {
            "models": {
                "ppo": {
                    "selected_test": {
                        "precision": 0.3,
                        "recall": 0.5,
                        "f1": 0.37,
                        "pr_auc": 0.21,
                        "alerts_per_1k": 20.0,
                        "threshold": 0.4,
                    }
                },
                "iforest": {
                    "selected_test": {
                        "precision": 0.1,
                        "recall": 0.2,
                        "f1": 0.13,
                        "pr_auc": 0.07,
                        "alerts_per_1k": 8.0,
                        "threshold": 0.75,
                    }
                },
            }
        },
    )
    _write_json(
        run_dir / "report" / "run_manifest.json",
        {
            "config_snapshot": {
                "data": "config/data.yaml",
                "morl": "config/morl.yaml",
                "meta_controller": "config/meta_controller.yaml",
            },
            "morl": {
                "objective_source": "fallback_synthetic",
                "normalization_summary": {"applied": False},
            },
        },
    )
    _write_json(
        run_dir / "eval" / "meta_stability" / "meta_stability.json",
        {
            "aggregate": {
                "selection_change_rate": 0.1,
                "avg_weight_l1_distance": 0.2,
                "constraint_violation_rate": 0.3,
                "avg_regret": 0.01,
                "worst_regret": 0.05,
                "worst_regret_condition": "noise_medium",
            }
        },
    )

    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "condition",
                "seed",
                "backend",
                "runner",
                "data_config",
                "morl_config",
                "meta_config",
                "robustness_config",
                "meta_stability_config",
                "run_dir",
                "status",
                "command",
            ]
        )
        writer.writerow(
            [
                "2026-01-01T00:00:00Z",
                "A1_csv_option2_default",
                "42",
                "csv",
                "make",
                "configs/data_local_gatra_prd_c335.yaml",
                "configs/morl_realdata_normalized.yaml",
                "configs/meta_controller_relaxed.yaml",
                "",
                "",
                str(run_dir),
                "ok",
                "make run_morl_policy_quick ...",
            ]
        )

    out_csv = tmp_path / "paper_week1_results.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_paper_results.py",
            "--index",
            str(index_path),
            "--out",
            str(out_csv),
        ],
        check=True,
    )

    with out_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["condition"] == "A1_csv_option2_default"
    assert row["seed"] == "42"
    assert row["selected_weight"] == "0.7;0.2;0.1"
    assert row["fallback_mode"] == "soft_constraints"
    assert row["morl_f1"] == "0.26"
    assert row["policy_ppo_threshold"] == "0.4"
    assert row["meta_stability_worst_regret_condition"] == "noise_medium"
