from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from architecture_a_rl.morl.objectives import parse_objectives
from architecture_a_rl.morl.objectives_realdata import compute_reward_matrix_realdata_aware


def test_realdata_objective_alignment_error_is_clear(tmp_path: Path) -> None:
    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()
    np.save(contract_dir / "session_id_train.npy", np.asarray(["s1", "s2"], dtype=np.str_))
    np.save(contract_dir / "timestamps_epoch_s_train.npy", np.asarray([1, 2], dtype=np.int64))
    np.save(contract_dir / "page_train.npy", np.asarray(["p1", "p2"], dtype=np.str_))
    np.save(contract_dir / "action_train.npy", np.asarray(["a1", "a2"], dtype=np.str_))

    y = np.asarray([0, 1, 0], dtype=np.int_)
    actions = np.asarray([0, 1, 1], dtype=np.int_)
    objectives = parse_objectives(
        [
            {"name": "ttt", "type": "time_to_triage_seconds"},
            {"name": "coverage", "type": "detection_coverage"},
        ]
    )

    with pytest.raises(ValueError, match="Alignment error"):
        compute_reward_matrix_realdata_aware(
            y_true=y,
            actions=actions,
            objectives=objectives,
            contract_dir=str(contract_dir),
            split="train",
            normalization="minmax",
        )
