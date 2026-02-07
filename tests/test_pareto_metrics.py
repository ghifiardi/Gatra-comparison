from __future__ import annotations

import numpy as np
from typing import cast

from evaluation.pareto import hypervolume_3d, hypervolume_from_rows, pareto_filter


def test_pareto_filter_drops_dominated_point() -> None:
    rows: list[dict[str, object]] = [
        {
            "weights": [0.7, 0.2, 0.1],
            "metrics": {"pr_auc": 0.80, "f1": 0.70, "alerts_per_1k": 100.0},
        },
        {
            "weights": [0.6, 0.3, 0.1],
            "metrics": {"pr_auc": 0.70, "f1": 0.60, "alerts_per_1k": 200.0},
        },
        {
            "weights": [0.4, 0.4, 0.2],
            "metrics": {"pr_auc": 0.75, "f1": 0.80, "alerts_per_1k": 300.0},
        },
    ]

    out = pareto_filter(rows, ["pr_auc", "f1", "alerts_per_1k"])
    weights: set[tuple[float, ...]] = {tuple(cast(list[float], r["weights"])) for r in out}

    assert (0.6, 0.3, 0.1) not in weights
    assert (0.7, 0.2, 0.1) in weights
    assert (0.4, 0.4, 0.2) in weights


def test_hypervolume_is_deterministic() -> None:
    points = np.array(
        [
            [1.0, 1.0, -5.0],
            [2.0, 1.0, -4.0],
        ],
        dtype=np.float64,
    )
    ref = np.array([0.0, 0.0, -10.0], dtype=np.float64)
    hv = hypervolume_3d(points, ref)
    assert hv == 12.0

    rows: list[dict[str, object]] = [
        {"weights": [0.5, 0.3, 0.2], "metrics": {"pr_auc": 1.0, "f1": 1.0, "alerts_per_1k": 5.0}},
        {"weights": [0.7, 0.2, 0.1], "metrics": {"pr_auc": 2.0, "f1": 1.0, "alerts_per_1k": 4.0}},
    ]
    hv_rows = hypervolume_from_rows(rows, ["pr_auc", "f1", "alerts_per_1k"], [0.0, 0.0, 10.0])
    assert hv_rows == 12.0
