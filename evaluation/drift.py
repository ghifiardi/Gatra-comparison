from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def drift_score(reference: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    if reference.size == 0 or current.size == 0:
        return float("nan")
    return float(np.abs(reference.mean() - current.mean()))
