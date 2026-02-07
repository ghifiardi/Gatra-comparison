from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def validate_simplex_weights(
    weights: NDArray[np.float32],
    *,
    atol: float = 1e-6,
) -> NDArray[np.float32]:
    if weights.ndim != 1:
        raise ValueError("weights must be rank-1")
    if weights.shape[0] == 0:
        raise ValueError("weights cannot be empty")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.isclose(float(weights.sum()), 1.0, atol=atol):
        raise ValueError("weights must sum to 1.0")
    return weights.astype(np.float32)


def sample_dirichlet_weights(
    alpha: Sequence[float],
    n_samples: int,
    seed: int,
) -> NDArray[np.float32]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if alpha_arr.ndim != 1:
        raise ValueError("dirichlet alpha must be rank-1")
    if np.any(alpha_arr <= 0.0):
        raise ValueError("dirichlet alpha values must be > 0")

    rng = np.random.default_rng(seed)
    sampled = rng.dirichlet(alpha_arr.astype(np.float64), size=n_samples).astype(np.float32)
    for i in range(n_samples):
        validate_simplex_weights(sampled[i])
    return sampled


def normalize_weight_grid(
    grid: Sequence[Sequence[float]],
    k_objectives: int,
) -> NDArray[np.float32]:
    if not grid:
        raise ValueError("weight grid cannot be empty")

    out = np.zeros((len(grid), k_objectives), dtype=np.float32)
    for i, row in enumerate(grid):
        arr = np.asarray(row, dtype=np.float32)
        if arr.shape != (k_objectives,):
            raise ValueError(
                f"weight grid row {i} has invalid shape {arr.shape}; expected ({k_objectives},)"
            )
        out[i] = validate_simplex_weights(arr)
    return out
