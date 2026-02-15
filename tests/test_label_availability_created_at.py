from __future__ import annotations

import numpy as np

from evaluation.label_availability import apply_label_availability, parse_duration_to_seconds
from evaluation.metrics import classification_metrics


def test_created_at_gating_treat_as_unknown_marks_unavailable() -> None:
    y = np.asarray([1, 0, 1], dtype=np.int_)
    event_ts = np.asarray([100, 200, 300], dtype=np.int64)
    label_created = np.asarray([90, 150, 400], dtype=np.int64)

    y_out, available, meta = apply_label_availability(
        y_true=y,
        event_timestamps_epoch_s=event_ts,
        label_created_at_epoch_s=label_created,
        delay="30m",
        policy="treat_as_unknown",
        eval_time_epoch_s=300,
    )

    assert available.tolist() == [False, False, False]
    assert y_out.tolist() == [-1, -1, -1]
    assert meta["delay_seconds"] == 1800
    assert meta["unavailable_pos"] == 2


def test_created_at_gating_treat_as_benign_and_metrics_ignore_unknown() -> None:
    y = np.asarray([1, 0, 1, 1], dtype=np.int_)
    event_ts = np.asarray([100, 200, 300, 400], dtype=np.int64)
    label_created = np.asarray([90, 190, 350, 800], dtype=np.int64)

    y_unknown, _, _ = apply_label_availability(
        y_true=y,
        event_timestamps_epoch_s=event_ts,
        label_created_at_epoch_s=label_created,
        delay=0,
        policy="treat_as_unknown",
        eval_time_epoch_s=400,
    )
    scores = np.asarray([0.9, 0.1, 0.8, 0.7], dtype=np.float64)
    m_unknown = classification_metrics(y_unknown.astype(np.int_), scores, threshold=0.5)
    m_trimmed = classification_metrics(y[:3], scores[:3], threshold=0.5)
    assert m_unknown["f1"] == m_trimmed["f1"]
    assert m_unknown["pr_auc"] == m_trimmed["pr_auc"]

    y_benign, _, _ = apply_label_availability(
        y_true=y,
        event_timestamps_epoch_s=event_ts,
        label_created_at_epoch_s=label_created,
        delay=0,
        policy="treat_as_benign",
        eval_time_epoch_s=400,
    )
    assert y_benign.tolist() == [1, 0, 1, 0]


def test_parse_duration_variants() -> None:
    assert parse_duration_to_seconds("7d") == 7 * 86400
    assert parse_duration_to_seconds("12h") == 12 * 3600
    assert parse_duration_to_seconds("30m") == 30 * 60
