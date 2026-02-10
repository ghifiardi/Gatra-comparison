from __future__ import annotations

from datetime import datetime, timezone
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from data.bq_loader import load_events_labels_from_bigquery, row_to_raw_event


def _mock_bq(event_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> MagicMock:
    mock_bq = MagicMock()
    call_count = {"n": 0}

    def query_side_effect(_: str) -> MagicMock:
        result_mock = MagicMock()
        if call_count["n"] == 0:
            result_mock.result.return_value = iter(label_rows)
        else:
            result_mock.result.return_value = iter(event_rows)
        call_count["n"] += 1
        return result_mock

    mock_bq.Client.return_value.query.side_effect = query_side_effect
    return mock_bq


def _run_loader_with_mock(
    cfg: dict[str, Any], event_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> tuple[list[Any], list[Any]]:
    mock_bq = _mock_bq(event_rows, label_rows)
    mock_google = MagicMock()
    mock_google_cloud = MagicMock()
    mock_google_cloud.bigquery = mock_bq

    saved = {
        k: sys.modules[k]
        for k in ("google", "google.cloud", "google.cloud.bigquery")
        if k in sys.modules
    }
    try:
        sys.modules["google"] = mock_google
        sys.modules["google.cloud"] = mock_google_cloud
        sys.modules["google.cloud.bigquery"] = mock_bq
        return load_events_labels_from_bigquery(cfg, "dummy.yaml")
    finally:
        for key in ("google", "google.cloud", "google.cloud.bigquery"):
            if key in saved:
                sys.modules[key] = saved[key]
            else:
                sys.modules.pop(key, None)


def _event_mapping() -> dict[str, Any]:
    return {
        "event_ts_col": "event_timestamp",
        "session_id_col": "session_id",
        "user_col": "user",
        "action_col": "action",
        "page_col": "page",
        "details_col": "details",
        "row_key_mode": "derived_sha1",
        "row_key_cols": ["session_id", "event_timestamp", "user", "action", "page", "details"],
    }


def test_bq_loader_derives_row_key_when_missing() -> None:
    row = {
        "event_timestamp": datetime(2025, 12, 8, 2, 13, 46, tzinfo=timezone.utc),
        "session_id": "s1",
        "user": "u1",
        "action": "login_success",
        "page": "unknown",
        "details": "{}",
    }
    e = row_to_raw_event(row, _event_mapping())
    assert e.row_key is not None
    assert len(e.row_key) == 40
    assert e.event_id == e.row_key
    assert e.event_timestamp_epoch_s == 1765160026


def test_bq_loader_parses_datetime_timestamp() -> None:
    row = {
        "event_timestamp": datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    }
    mapping = _event_mapping()
    e = row_to_raw_event(row, mapping)
    assert e.event_timestamp_epoch_s == 1735689601


def test_bq_loader_parses_iso_timestamp() -> None:
    row = {
        "event_timestamp": "2025-01-01T00:00:02Z",
    }
    mapping = _event_mapping()
    e = row_to_raw_event(row, mapping)
    assert e.event_timestamp_epoch_s == 1735689602


def test_bq_loader_derived_key_is_deterministic() -> None:
    row = {
        "event_timestamp": "2025-01-01T00:00:02Z",
        "session_id": "sess",
        "user": "alice",
        "action": "login",
        "page": "overview",
        "details": "{}",
    }
    mapping = _event_mapping()
    key_a = row_to_raw_event(row, mapping).row_key
    key_b = row_to_raw_event(row, mapping).row_key
    row_changed = dict(row)
    row_changed["page"] = "profile"
    key_c = row_to_raw_event(row_changed, mapping).row_key
    assert key_a == key_b
    assert key_a != key_c


def test_bq_loader_config_missing_cols_errors_cleanly() -> None:
    row = {"session_id": "only-session"}
    mapping = dict(_event_mapping())
    mapping["event_ts_col"] = "missing_col"
    with pytest.raises(ValueError, match="missing or invalid timestamp"):
        row_to_raw_event(row, mapping)


def test_bigquery_loader_mock_returns_required_fields() -> None:
    cfg: dict[str, Any] = {
        "dataset": {
            "source": "bigquery",
            "bq_project": "proj",
            "bq_dataset": "ds",
            "bq_events_table": "activity_logs",
            "bq_labels_table": "ada_feedback",
            "events": {
                "event_ts_col": "event_timestamp",
                "session_id_col": "session_id",
                "user_col": "user",
                "action_col": "action",
                "page_col": "page",
                "details_col": "details",
                "row_key": {
                    "mode": "derived_sha1",
                    "cols": [
                        "session_id",
                        "event_timestamp",
                        "user",
                        "action",
                        "page",
                        "details",
                    ],
                },
            },
            "labels": {
                "alarm_id_col": "alarm_id",
                "case_class_col": "case_class",
                "created_at_col": "created_at",
                "event_id_col": "id",
            },
        }
    }
    label_rows = [
        {
            "id": "1",
            "alarm_id": "a-1",
            "case_class": "investigated",
            "created_at": "2025-01-01T00:00:00Z",
        }
    ]
    event_rows = [
        {
            "event_timestamp": "2025-01-01T00:00:00Z",
            "session_id": "s1",
            "user": "u1",
            "action": "login",
            "page": "home",
            "details": "{}",
        }
    ]
    events, labels = _run_loader_with_mock(cfg, event_rows, label_rows)
    assert len(events) == 1
    assert len(labels) == 1
    assert events[0].row_key is not None
    assert events[0].event_timestamp_epoch_s == 1735689600
