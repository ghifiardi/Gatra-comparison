from __future__ import annotations

import numpy as np

from architecture_a_rl.morl.objectives import (
    compute_reward_matrix,
    compute_reward_vector,
    parse_objectives,
    scalarize_matrix,
)


def _default_objectives() -> list[dict[str, object]]:
    return [
        {
            "name": "detect",
            "type": "classification",
            "tp": 1.0,
            "fn": -1.0,
            "fp": 0.0,
            "tn": 0.0,
        },
        {
            "name": "fp_cost",
            "type": "fp_penalty",
            "fp_penalty": -0.2,
        },
        {
            "name": "analyst_cost",
            "type": "per_alert_cost",
            "per_alert_penalty": -0.05,
        },
    ]


def test_reward_vector_cases() -> None:
    objectives = parse_objectives(_default_objectives())

    tp = compute_reward_vector(y_true=1, action=1, objectives=objectives)
    fn = compute_reward_vector(y_true=1, action=0, objectives=objectives)
    fp = compute_reward_vector(y_true=0, action=1, objectives=objectives)
    tn = compute_reward_vector(y_true=0, action=0, objectives=objectives)

    np.testing.assert_allclose(tp, np.array([1.0, 0.0, -0.05], dtype=np.float32))
    np.testing.assert_allclose(fn, np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(fp, np.array([0.0, -0.2, -0.05], dtype=np.float32))
    np.testing.assert_allclose(tn, np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_scalarization_matches_dot_product() -> None:
    objectives = parse_objectives(_default_objectives())
    y = np.array([1, 1, 0, 0], dtype=np.int_)
    a = np.array([1, 0, 1, 0], dtype=np.int_)

    reward_matrix = compute_reward_matrix(y, a, objectives)
    w = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    scalar = scalarize_matrix(reward_matrix, w)

    expected = reward_matrix @ w
    np.testing.assert_allclose(scalar, expected.astype(np.float32))
