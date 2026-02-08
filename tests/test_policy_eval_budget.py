from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.policy_eval import run_policy_eval


def test_policy_eval_selects_threshold_under_alert_budget(tmp_path: Path) -> None:
    cfg = tmp_path / "policy_eval.yaml"
    cfg.write_text(
        (
            "policy_eval:\n"
            "  mode: alert_budget\n"
            "  primary_metric: f1\n"
            "  tie_breaker: alerts_per_1k\n"
            "  constraints:\n"
            "    alerts_per_1k_max: 500.0\n"
            "    recall_min: null\n"
            "  triage_seconds_per_alert: 60.0\n"
        )
    )

    y_val = np.asarray([1, 0, 1, 0], dtype=np.int_)
    y_test = np.asarray([1, 0, 1, 0], dtype=np.int_)
    val_scores = np.asarray([0.9, 0.8, 0.4, 0.1], dtype=np.float64)
    test_scores = np.asarray([0.95, 0.85, 0.45, 0.05], dtype=np.float64)

    out = run_policy_eval(
        policy_cfg_path=str(cfg),
        y_val=y_val,
        y_test=y_test,
        val_scores_by_model={"ppo": val_scores},
        test_scores_by_model={"ppo": test_scores},
        out_dir=str(tmp_path / "eval"),
    )

    selected = out["models"]["ppo"]["selected_val"]
    assert selected["alerts_per_1k"] <= 500.0
    assert selected["threshold"] >= 0.8
    assert Path(out["paths"]["json"]).exists()
    assert Path(out["paths"]["md"]).exists()
