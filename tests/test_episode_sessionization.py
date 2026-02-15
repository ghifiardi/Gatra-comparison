from __future__ import annotations

import numpy as np

from data.contract_export import compute_episode_segments


def test_episode_sessionization_respects_session_and_gap() -> None:
    sessions = np.asarray(["s1", "s1", "s1", "s1", "s2", "s2"], dtype=np.str_)
    ts = np.asarray([0, 100, 2200, 2300, 2400, 4300], dtype=np.int64)

    episode_id, episode_start, episode_end = compute_episode_segments(
        sessions, ts, gap_seconds=1800
    )

    assert episode_id.tolist() == [0, 0, 1, 1, 2, 3]
    assert episode_start.tolist() == [0, 0, 2200, 2200, 2400, 4300]
    assert episode_end.tolist() == [100, 100, 2300, 2300, 2400, 4300]
    assert len(episode_id) == len(ts)
    assert len(episode_start) == len(ts)
    assert len(episode_end) == len(ts)
