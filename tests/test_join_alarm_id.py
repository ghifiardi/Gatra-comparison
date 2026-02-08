from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.join import build_join_map


def _write_join_cfg(path: Path, *, collision_rule: str = "first") -> None:
    path.write_text(
        (
            "join:\n"
            "  split: test\n"
            '  priority: ["alarm_id", "row_key", "time_window"]\n'
            f"  collision_rule: {collision_rule}\n"
            "  time_window_fallback:\n"
            "    enabled: false\n"
            "    window_seconds: 3600\n"
        )
    )


def test_join_alarm_id_priority_and_collisions(tmp_path: Path) -> None:
    contract = tmp_path / "contract"
    contract.mkdir()
    np.save(contract / "row_key_test.npy", np.asarray(["rk1", "rk2", "rk3"], dtype=np.str_))
    np.save(contract / "alarm_id_test.npy", np.asarray(["a1", "", "a2"], dtype=np.str_))
    np.save(contract / "session_id_test.npy", np.asarray(["s1", "s2", "s3"], dtype=np.str_))
    np.save(contract / "user_test.npy", np.asarray(["u1", "u2", "u3"], dtype=np.str_))
    np.save(contract / "timestamps_epoch_s_test.npy", np.asarray([10, 20, 30], dtype=np.int64))

    np.save(contract / "label_row_key_test.npy", np.asarray(["", "", ""], dtype=np.str_))
    np.save(contract / "label_alarm_id_test.npy", np.asarray(["a1", "a2", "a2"], dtype=np.str_))
    np.save(
        contract / "label_created_at_epoch_s_test.npy", np.asarray([11, 21, 31], dtype=np.int64)
    )

    cfg = tmp_path / "join.yaml"
    _write_join_cfg(cfg, collision_rule="first")

    out = build_join_map(
        contract_dir=str(contract), join_cfg_path=str(cfg), out_dir=str(tmp_path / "out")
    )
    mapped = np.load(out["join_map"])
    assert mapped["label_index"].tolist() == [0, -1, 1]
    assert mapped["method"].tolist() == ["alarm_id", "unmatched", "alarm_id"]

    with open(out["join_meta"], "r") as f:
        meta = json.load(f)
    assert meta["matched_count"] == 2
    assert meta["unmatched_count"] == 1
    assert meta["ambiguity_count"] == 1


def test_join_no_match_scenario(tmp_path: Path) -> None:
    contract = tmp_path / "contract"
    contract.mkdir()
    np.save(contract / "row_key_test.npy", np.asarray(["rk1"], dtype=np.str_))
    np.save(contract / "alarm_id_test.npy", np.asarray([""], dtype=np.str_))
    np.save(contract / "session_id_test.npy", np.asarray(["s1"], dtype=np.str_))
    np.save(contract / "user_test.npy", np.asarray(["u1"], dtype=np.str_))
    np.save(contract / "timestamps_epoch_s_test.npy", np.asarray([10], dtype=np.int64))
    np.save(contract / "label_row_key_test.npy", np.asarray([""], dtype=np.str_))
    np.save(contract / "label_alarm_id_test.npy", np.asarray([""], dtype=np.str_))
    np.save(contract / "label_created_at_epoch_s_test.npy", np.asarray([100], dtype=np.int64))

    cfg = tmp_path / "join.yaml"
    _write_join_cfg(cfg)
    out = build_join_map(
        contract_dir=str(contract), join_cfg_path=str(cfg), out_dir=str(tmp_path / "out")
    )
    mapped = np.load(out["join_map"])
    assert mapped["label_index"].tolist() == [-1]
    assert mapped["method"].tolist() == ["unmatched"]
