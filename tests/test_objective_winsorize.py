from __future__ import annotations

import numpy as np

from architecture_a_rl.morl.objectives_normalize import winsorize


def test_winsorize_clamps_upper_outlier() -> None:
    x = np.concatenate([np.ones((100,), dtype=np.float32), np.asarray([100.0], dtype=np.float32)])
    out = winsorize(x, p=0.99)
    assert float(np.max(out)) <= 1.0
    assert float(out[-1]) <= 1.0
