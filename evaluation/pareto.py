from __future__ import annotations

from itertools import combinations
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def _utility(metric_name: str, value: float) -> float:
    if not math.isfinite(value):
        return float("-inf")
    # Convert minimized metrics into maximize-utility axes.
    if metric_name == "alerts_per_1k":
        return -value
    return value


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _metrics_from_row(row: dict[str, object]) -> dict[str, float]:
    raw = row.get("metrics")
    if not isinstance(raw, dict):
        raise ValueError("Each row must include a metrics mapping")
    out: dict[str, float] = {}
    for key, value in raw.items():
        out[str(key)] = _to_float(value)
    return out


def primary_metric_vector(
    metrics: dict[str, float],
    primary_metrics: Sequence[str],
) -> NDArray[np.float64]:
    vals = [_utility(name, float(metrics[name])) for name in primary_metrics]
    return np.asarray(vals, dtype=np.float64)


def pareto_indices(points: NDArray[np.float64]) -> list[int]:
    if points.ndim != 2:
        raise ValueError("points must be rank-2")

    n = points.shape[0]
    keep: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            ge = np.all(points[j] >= points[i])
            gt = np.any(points[j] > points[i])
            if ge and gt:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def pareto_filter(
    rows: Sequence[dict[str, object]],
    primary_metrics: Sequence[str],
) -> list[dict[str, object]]:
    if not rows:
        return []
    points = np.stack(
        [primary_metric_vector(_metrics_from_row(row), primary_metrics) for row in rows],
        axis=0,
    )
    idx = set(pareto_indices(points))
    return [row for i, row in enumerate(rows) if i in idx]


def hypervolume_3d(
    points_utility: NDArray[np.float64],
    reference_utility: NDArray[np.float64],
) -> float:
    if points_utility.ndim != 2 or points_utility.shape[1] != 3:
        raise ValueError("points_utility must have shape (n,3)")
    if reference_utility.shape != (3,):
        raise ValueError("reference_utility must have shape (3,)")

    # Keep points that dominate the reference in utility space.
    valid = points_utility[np.all(points_utility >= reference_utility, axis=1)]
    n = valid.shape[0]
    if n == 0:
        return 0.0

    hv = 0.0
    for r in range(1, n + 1):
        sign = 1.0 if r % 2 == 1 else -1.0
        for idxs in combinations(range(n), r):
            upper = np.min(valid[list(idxs)], axis=0)
            edge = upper - reference_utility
            if np.any(edge <= 0.0):
                continue
            hv += sign * float(edge[0] * edge[1] * edge[2])
    return float(hv)


def hypervolume_from_rows(
    rows: Sequence[dict[str, object]],
    primary_metrics: Sequence[str],
    reference: Sequence[float],
) -> float:
    if len(primary_metrics) != 3:
        raise ValueError("hypervolume currently supports exactly 3 primary metrics")
    points = np.stack(
        [primary_metric_vector(_metrics_from_row(row), primary_metrics) for row in rows],
        axis=0,
    )
    ref_raw = np.asarray(reference, dtype=np.float64)
    if ref_raw.shape != (3,):
        raise ValueError("reference must have shape (3,)")
    ref = np.asarray(
        [_utility(primary_metrics[i], float(ref_raw[i])) for i in range(3)],
        dtype=np.float64,
    )
    return hypervolume_3d(points, ref)
