from __future__ import annotations

import numpy as np

from architecture_a_rl.morl.objectives_normalize import apply_zscore


def test_apply_zscore_respects_clip_bounds() -> None:
    x = np.asarray([0.0, 10.0, -10.0], dtype=np.float32)
    out = apply_zscore(x, mean=0.0, std=1.0, clip_z=2.0)
    np.testing.assert_allclose(out, np.asarray([0.0, 2.0, -2.0], dtype=np.float32))
