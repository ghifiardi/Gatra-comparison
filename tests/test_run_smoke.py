from __future__ import annotations

import json
from pathlib import Path

import pytest
from runs.cli import main as run_main


def _write_data_config(path: Path) -> None:
    path.write_text(
        """
dataset:
  source: "toy"
  n: 1000

schema:
  timestamp_field: "ts"
  id_field: "event_id"

splits:
  mode: "time"
  train:
    start: "2025-01-01"
    end: "2025-01-02"
  val:
    start: "2025-01-02"
    end: "2025-01-03"
  test:
    start: "2025-01-03"
    end: "2025-01-04"

features:
  v7:
    enabled: true
    fields:
      - duration
      - bytes_sent
      - bytes_received
      - port
      - protocol
      - hour_of_day
      - day_of_week
  v128:
    enabled: true
    history_window_minutes: 60
    dims: 128
""".strip()
    )


def test_run_cli_quick_smoke(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    data_cfg = tmp_path / "data.yaml"
    _write_data_config(data_cfg)

    out_root = tmp_path / "runs"
    run_id = "testrun"

    run_main(
        data_config=str(data_cfg),
        iforest_config="configs/iforest.yaml",
        ppo_config="configs/ppo.yaml",
        eval_config="configs/eval.yaml",
        out_root=str(out_root),
        quick=True,
        overwrite=True,
        run_id=run_id,
    )

    run_dir = out_root / run_id
    assert (run_dir / "contract" / "features_v7_train.npy").exists()
    assert (run_dir / "models" / "iforest" / "model.joblib").exists()
    assert (run_dir / "models" / "ppo" / "actor.pt").exists()
    assert (run_dir / "eval" / "metrics.json").exists()
    assert (run_dir / "report" / "summary.md").exists()

    with open(run_dir / "report" / "run_manifest.json", "r") as f:
        manifest = json.load(f)
    assert manifest["mode"] == "quick"
