from __future__ import annotations

import numpy as np

from architecture_a_rl.morl.objectives_normalize import rate_per_1k


def test_rate_per_1k_scaling() -> None:
    counts = np.asarray([1.0, 2.0, 0.0], dtype=np.float32)
    denom = np.asarray([10.0, 20.0, 5.0], dtype=np.float32)
    out = rate_per_1k(counts, denom)
    np.testing.assert_allclose(out, np.asarray([100.0, 100.0, 0.0], dtype=np.float32))
