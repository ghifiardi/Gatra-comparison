from __future__ import annotations

from architecture_a_rl.morl.meta_controller import CandidateRow, ObjectiveConfig
from evaluation.meta_stability import (
    compute_regret,
    compute_selection_change_rate,
    compute_weight_l1_distance,
)


def _make_row(w: list[float], pr_auc: float, alerts: float) -> CandidateRow:
    return CandidateRow(
        w=w,
        metrics={"pr_auc": pr_auc, "alerts_per_1k": alerts},
        objective_means={},
        meta={},
    )


def test_selection_change_rate_all_same() -> None:
    baseline = [0.8, 0.2]
    conditions = [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]
    assert compute_selection_change_rate(baseline, conditions) == 0.0


def test_selection_change_rate_all_different() -> None:
    baseline = [0.8, 0.2]
    conditions = [[0.5, 0.5], [0.3, 0.7]]
    assert compute_selection_change_rate(baseline, conditions) == 1.0


def test_selection_change_rate_mixed() -> None:
    baseline = [0.8, 0.2]
    conditions = [[0.8, 0.2], [0.5, 0.5], [0.8, 0.2], [0.3, 0.7]]
    rate = compute_selection_change_rate(baseline, conditions)
    assert abs(rate - 0.5) < 1e-9


def test_selection_change_rate_empty() -> None:
    assert compute_selection_change_rate([0.8, 0.2], []) == 0.0


def test_weight_l1_distance_identical() -> None:
    assert compute_weight_l1_distance([0.8, 0.2], [0.8, 0.2]) == 0.0


def test_weight_l1_distance_known() -> None:
    d = compute_weight_l1_distance([0.8, 0.2], [0.5, 0.5])
    assert abs(d - 0.6) < 1e-9


def test_regret_sign_convention() -> None:
    """regret = utility(best) - utility(selected), always >= 0."""
    obj = ObjectiveConfig(
        primary="pr_auc",
        tie_breaker="alerts_per_1k",
        utility={"pr_auc": 1.0, "alerts_per_1k": -0.01},
    )
    best = _make_row([0.8, 0.2], pr_auc=0.95, alerts=10.0)
    worse = _make_row([0.5, 0.5], pr_auc=0.80, alerts=20.0)

    regret, best_u, sel_u = compute_regret([best, worse], worse, obj)
    assert regret >= 0.0
    assert best_u >= sel_u
    assert abs(best_u - 0.95) < 1e-9
    assert abs(sel_u - 0.80) < 1e-9
    assert abs(regret - 0.15) < 1e-9


def test_regret_zero_when_best_selected() -> None:
    obj = ObjectiveConfig(
        primary="pr_auc",
        tie_breaker="alerts_per_1k",
        utility={},
    )
    best = _make_row([0.8, 0.2], pr_auc=0.95, alerts=10.0)
    other = _make_row([0.5, 0.5], pr_auc=0.80, alerts=20.0)

    regret, best_u, sel_u = compute_regret([best, other], best, obj)
    assert regret == 0.0
    assert best_u == sel_u


def test_regret_with_utility_mode() -> None:
    """Regret works when primary='utility' with composite weights."""
    obj = ObjectiveConfig(
        primary="utility",
        tie_breaker="alerts_per_1k",
        utility={"pr_auc": 1.0, "alerts_per_1k": -0.01},
    )
    best = _make_row([0.8, 0.2], pr_auc=0.95, alerts=10.0)
    worse = _make_row([0.5, 0.5], pr_auc=0.80, alerts=50.0)

    regret, best_u, sel_u = compute_regret([best, worse], worse, obj)
    assert regret >= 0.0
    expected_best = 0.95 + (-0.01 * 10.0)
    expected_worse = 0.80 + (-0.01 * 50.0)
    assert abs(best_u - expected_best) < 1e-9
    assert abs(sel_u - expected_worse) < 1e-9
    assert abs(regret - (expected_best - expected_worse)) < 1e-9
