from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from architecture_a_rl.train import train_ppo_from_arrays
from architecture_b_iforest.train import train_iforest_from_arrays
from data.contract_export import export_frozen_contract_to_dir
from evaluation.robustness import run_robustness_suite


def _write_data_config(path: Path) -> None:
    path.write_text(
        """
dataset:
  source: "toy"
  n: 600

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


def _write_iforest_config(path: Path) -> None:
    path.write_text(
        """
model:
  name: "isolation_forest"
  random_state: 42
  n_estimators: 25
  max_samples: "auto"
  contamination: "auto"
  max_features: 1.0
  bootstrap: false

scoring:
  threshold: 0.8
  score_calibration: "minmax"

io:
  output_dir: "./artifacts/iforest"
""".strip()
    )


def _write_ppo_config(path: Path) -> None:
    path.write_text(
        """
rl:
  algo: "ppo"
  seed: 42
  state_dim: 128
  action_dim: 4
  actions: ["escalate", "contain", "monitor", "dismiss"]

reward:
  tp_base: 10.0
  fp_base: -3.0
  fn_base: -15.0
  efficiency_bonus: 1.0
  action_cost:
    escalate: 3.0
    contain: 2.0
    monitor: 1.0
    dismiss: 0.5

networks:
  hidden_sizes: [64, 32]
  activation: "relu"

train:
  epochs: 1
  batch_size: 16
  lr: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
  log_every_steps: 200

io:
  output_dir: "./artifacts/ppo"
  run_dir: "./runs"
""".strip()
    )


def _write_robustness_config(path: Path) -> None:
    path.write_text(
        """
robustness:
  enabled: true
  seed: 42
  variants:
    - name: baseline
      kind: none
    - name: missing_mcar_05
      kind: missingness
      rate: 0.05
      strategy: mcar
      fill: zero
""".strip()
    )


def test_robustness_suite_smoke(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    data_cfg = tmp_path / "data.yaml"
    iforest_cfg = tmp_path / "iforest.yaml"
    ppo_cfg = tmp_path / "ppo.yaml"
    robustness_cfg = tmp_path / "robustness.yaml"
    _write_data_config(data_cfg)
    _write_iforest_config(iforest_cfg)
    _write_ppo_config(ppo_cfg)
    _write_robustness_config(robustness_cfg)

    contract_dir = tmp_path / "contract"
    export_frozen_contract_to_dir(
        data_cfg_path=str(data_cfg),
        out_dir=str(contract_dir),
        include_splits=True,
        contract_id="test",
    )

    X7_train = np.load(contract_dir / "features_v7_train.npy").astype(np.float32)
    X128_train = np.load(contract_dir / "features_v128_train.npy").astype(np.float32)
    y_train = np.load(contract_dir / "y_train.npy").astype(np.int8)
    X128_val = np.load(contract_dir / "features_v128_val.npy").astype(np.float32)
    y_val = np.load(contract_dir / "y_val.npy").astype(np.int8)

    iforest_dir = tmp_path / "iforest"
    ppo_dir = tmp_path / "ppo"
    train_iforest_from_arrays(str(iforest_cfg), X7_train, str(iforest_dir))
    train_ppo_from_arrays(
        str(ppo_cfg), X128_train, y_train, str(ppo_dir), x_val=X128_val, y_val=y_val
    )

    out_dir = tmp_path / "robustness"
    run_robustness_suite(
        contract_dir=str(contract_dir),
        iforest_model_dir=str(iforest_dir),
        ppo_model_dir=str(ppo_dir),
        ppo_config=str(ppo_cfg),
        robustness_cfg_path=str(robustness_cfg),
        out_dir=str(out_dir),
        quick=True,
    )

    assert (out_dir / "robustness_results.json").exists()
    assert (out_dir / "robustness_table.csv").exists()
    assert (out_dir / "robustness.md").exists()
