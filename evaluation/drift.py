from __future__ import annotations
import numpy as np

def drift_score(reference: np.ndarray, current: np.ndarray) -> float:
    if reference.size == 0 or current.size == 0:
        return float("nan")
    return float(np.abs(reference.mean() - current.mean()))
