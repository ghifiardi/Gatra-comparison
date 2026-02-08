from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: str
    norm: str
    cap_pctl: float | None


@dataclass(frozen=True)
class NormalizationSpec:
    reference_split: str
    apply_to: tuple[str, ...]
    eps: float
    clip_z: float
    persist: bool


@dataclass(frozen=True)
class NormalizationStats:
    name: str
    norm: str
    direction: str
    cap_pctl: float | None
    cap_value: float | None
    mean: float
    std: float
    clip_z: float
    eps: float


def winsorize(x: NDArray[np.float32], p: float) -> NDArray[np.float32]:
    if x.ndim != 1:
        raise ValueError("winsorize expects a rank-1 array")
    if not (0.0 < p <= 1.0):
        raise ValueError("winsorize percentile p must be in (0, 1]")
    if x.size == 0 or p >= 1.0:
        return x.astype(np.float32, copy=True)
    cap = float(np.quantile(x, p, method="linear"))
    return np.minimum(x, cap).astype(np.float32)


def compute_stats(x: NDArray[np.float32], eps: float) -> tuple[float, float]:
    if x.ndim != 1:
        raise ValueError("compute_stats expects a rank-1 array")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    mean = float(np.mean(x)) if x.size else 0.0
    std = float(np.std(x)) if x.size else 0.0
    if std < eps:
        std = eps
    return mean, std


def apply_zscore(
    x: NDArray[np.float32], mean: float, std: float, clip_z: float
) -> NDArray[np.float32]:
    if std <= 0.0:
        raise ValueError("std must be > 0")
    z = (x - np.float32(mean)) / np.float32(std)
    if clip_z > 0.0:
        z = np.clip(z, -clip_z, clip_z)
    return z.astype(np.float32)


def log1p_then_zscore(
    x: NDArray[np.float32], mean: float, std: float, clip_z: float
) -> NDArray[np.float32]:
    xp = np.log1p(np.maximum(x, np.float32(0.0))).astype(np.float32)
    return apply_zscore(xp, mean=mean, std=std, clip_z=clip_z)


def rate_per_1k(x_counts: NDArray[np.float32], denom_events: NDArray[np.float32]) -> NDArray[np.float32]:
    if x_counts.shape != denom_events.shape:
        raise ValueError("x_counts and denom_events must have identical shape")
    denom = np.maximum(denom_events, np.float32(1.0))
    return (np.float32(1000.0) * x_counts / denom).astype(np.float32)
