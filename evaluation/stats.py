from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class BootstrapResult:
    mean: float
    low: float
    high: float

def bootstrap_mean(values: np.ndarray, n: int = 1000, seed: int = 0) -> BootstrapResult:
    if values.size == 0:
        nan = float("nan")
        return BootstrapResult(mean=nan, low=nan, high=nan)

    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(sample.mean()))

    means_arr = np.array(means, dtype=float)
    return BootstrapResult(
        mean=float(values.mean()),
        low=float(np.percentile(means_arr, 2.5)),
        high=float(np.percentile(means_arr, 97.5)),
    )
