from __future__ import annotations

import numpy as np

from architecture_a_rl.morl.objectives_realdata import compute_time_to_triage_seconds


def test_time_to_triage_seconds_is_constant_within_session() -> None:
    sessions = np.asarray(["s1", "s1", "s2", "s2", "s2"], dtype=np.str_)
    ts = np.asarray([10, 40, 5, 7, 15], dtype=np.int64)
    ttt = compute_time_to_triage_seconds(sessions, ts)
    np.testing.assert_allclose(ttt, np.asarray([30, 30, 10, 10, 10], dtype=np.float32))
