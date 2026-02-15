from __future__ import annotations

import numpy as np

from architecture_a_rl.morl.objectives_realdata import compute_coverage_increments


def test_coverage_increments_track_session_novelty() -> None:
    sessions = np.asarray(["s1", "s1", "s1", "s2", "s2"], dtype=np.str_)
    pages = np.asarray(["home", "home", "activity", "home", "home"], dtype=np.str_)
    actions = np.asarray(["view", "view", "click", "view", "click"], dtype=np.str_)
    coverage = compute_coverage_increments(sessions, pages, actions)
    np.testing.assert_allclose(coverage, np.asarray([1, 0, 1, 1, 1], dtype=np.float32))
