from __future__ import annotations

from pathlib import Path

import numpy as np

from architecture_a_rl.morl.objectives import parse_objectives
from architecture_a_rl.morl.objectives_realdata import compute_reward_matrix_realdata_aware


def _write_split_arrays(
    contract_dir: Path,
    split: str,
    session_id: list[str],
    timestamps: list[int],
    page: list[str],
    action: list[str],
) -> None:
    np.save(contract_dir / f"session_id_{split}.npy", np.asarray(session_id, dtype=np.str_))
    np.save(
        contract_dir / f"timestamps_epoch_s_{split}.npy", np.asarray(timestamps, dtype=np.int64)
    )
    np.save(contract_dir / f"page_{split}.npy", np.asarray(page, dtype=np.str_))
    np.save(contract_dir / f"action_{split}.npy", np.asarray(action, dtype=np.str_))


def test_normalization_stats_use_reference_split_only(tmp_path: Path) -> None:
    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()

    _write_split_arrays(
        contract_dir,
        "val",
        session_id=["a", "a", "b"],
        timestamps=[0, 10, 5],
        page=["p", "p", "q"],
        action=["x", "x", "y"],
    )
    _write_split_arrays(
        contract_dir,
        "test",
        session_id=["c", "c", "d"],
        timestamps=[0, 20, 5],
        page=["p", "p", "q"],
        action=["x", "x", "y"],
    )

    objectives = parse_objectives(
        [
            {
                "name": "time_to_triage_seconds",
                "type": "time_to_triage_seconds",
                "norm": "zscore",
                "direction": "minimize",
            }
        ]
    )
    norm_cfg = {"reference_split": "val", "apply_to": ["val", "test"], "persist": True}

    y_val = np.asarray([0, 0, 0], dtype=np.int_)
    a_val = np.asarray([1, 1, 1], dtype=np.int_)
    val_result = compute_reward_matrix_realdata_aware(
        y_true=y_val,
        actions=a_val,
        objectives=objectives,
        contract_dir=str(contract_dir),
        split="val",
        normalization_cfg=norm_cfg,
        legacy_default_norm="none",
    )
    assert val_result.normalization["reference_split"] == "val"

    y_test = np.asarray([0, 0, 0], dtype=np.int_)
    a_test = np.asarray([1, 1, 1], dtype=np.int_)
    test_result = compute_reward_matrix_realdata_aware(
        y_true=y_test,
        actions=a_test,
        objectives=objectives,
        contract_dir=str(contract_dir),
        split="test",
        normalization_cfg=norm_cfg,
        legacy_default_norm="none",
    )

    val_raw = np.asarray([10.0, 10.0, 0.0], dtype=np.float32)
    mean = float(np.mean(val_raw))
    std = float(np.std(val_raw))
    expected_test = ((np.asarray([20.0, 20.0, 0.0], dtype=np.float32) - mean) / std).astype(
        np.float32
    )

    got = test_result.objective_signals
    assert got is not None
    np.testing.assert_allclose(got[:, 0], expected_test, rtol=1e-5, atol=1e-5)
