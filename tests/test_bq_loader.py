from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from data.bq_loader import (
    load_events_labels_from_bigquery,
    row_to_label,
    row_to_raw_event,
)


# ── test_import_guard ────────────────────────────────────────────────
def test_import_guard() -> None:
    """data.bq_loader can be imported without google.cloud installed."""
    import data.bq_loader as mod

    assert hasattr(mod, "load_events_labels_from_bigquery")
    assert hasattr(mod, "row_to_raw_event")
    assert hasattr(mod, "row_to_label")


# ── test_missing_required_keys ───────────────────────────────────────
@pytest.mark.parametrize(
    "missing_key",
    ["bq_project", "bq_dataset", "bq_events_table"],
)
def test_missing_required_keys(missing_key: str) -> None:
    base: dict[str, Any] = {
        "bq_project": "proj",
        "bq_dataset": "ds",
        "bq_events_table": "events",
    }
    base.pop(missing_key)
    cfg: dict[str, Any] = {"dataset": base}
    with pytest.raises(ValueError, match=f"dataset.{missing_key}"):
        load_events_labels_from_bigquery(cfg, "dummy.yaml")


# ── test_row_to_raw_event ────────────────────────────────────────────
def test_row_to_raw_event_basic() -> None:
    row: dict[str, Any] = {
        "row_key": "rk_001",
        "event_timestamp_epoch_s": 1700000000,
        "session_id": "sess_1",
        "user": "alice",
        "action": "login",
        "page": "Overview",
        "details": "ok",
        "alarm_id": "alm_1",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "port": 443,
        "protocol": "TCP",
        "duration": 1.5,
        "bytes_sent": 1024.0,
        "bytes_received": 2048.0,
    }
    e = row_to_raw_event(row)
    assert e.event_id == "rk_001"
    assert e.row_key == "rk_001"
    assert e.event_timestamp_epoch_s == 1700000000
    assert e.session_id == "sess_1"
    assert e.user == "alice"
    assert e.user_id == "alice"
    assert e.action == "login"
    assert e.page == "Overview"
    assert e.alarm_id == "alm_1"
    assert e.src_ip == "10.0.0.1"
    assert e.port == 443
    assert e.bytes_sent == 1024.0


def test_row_to_raw_event_missing_optional_fields() -> None:
    row: dict[str, Any] = {"row_key": "rk_002", "event_timestamp_epoch_s": 1700000000}
    e = row_to_raw_event(row)
    assert e.event_id == "rk_002"
    assert e.session_id is None
    assert e.alarm_id is None
    assert e.src_ip is None
    assert e.port is None


def test_row_to_raw_event_uses_event_id_fallback() -> None:
    row: dict[str, Any] = {"event_id": "eid_003", "event_timestamp_epoch_s": 1700000000}
    e = row_to_raw_event(row)
    assert e.event_id == "eid_003"


def test_row_to_raw_event_raises_on_missing_id() -> None:
    with pytest.raises(ValueError, match="missing both row_key and event_id"):
        row_to_raw_event({})


# ── test_row_to_label ────────────────────────────────────────────────
def test_row_to_label_threat() -> None:
    row: dict[str, Any] = {
        "event_id": "eid_1",
        "row_key": "rk_1",
        "alarm_id": "alm_1",
        "label": "threat",
        "created_at": 1700000000,
    }
    lb = row_to_label(row)
    assert lb.event_id == "eid_1"
    assert lb.label == "threat"
    assert lb.severity == 1.0
    assert lb.source == "bigquery_feedback"
    assert lb.row_key == "rk_1"
    assert lb.alarm_id == "alm_1"
    assert lb.label_created_at_epoch_s == 1700000000


def test_row_to_label_unknown_label_text() -> None:
    row: dict[str, Any] = {"event_id": "eid_2", "label": "something_else"}
    lb = row_to_label(row)
    assert lb.label == "unknown"
    assert lb.severity == 0.0


def test_row_to_label_iso_datetime() -> None:
    row: dict[str, Any] = {"event_id": "eid_3", "created_at": "2024-01-01T00:00:00Z"}
    lb = row_to_label(row)
    assert lb.label_created_at_epoch_s == 1704067200


def test_row_to_label_missing_everything() -> None:
    lb = row_to_label({})
    assert lb.event_id == "__unmatched__"
    assert lb.label == "unknown"


# ── test_label_matching_priority ─────────────────────────────────────
def _make_mock_bq_module(
    event_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> MagicMock:
    """Build a mock google.cloud.bigquery module with a Client that returns rows."""
    mock_bq = MagicMock()
    call_count = {"n": 0}

    def mock_query(sql: str) -> MagicMock:
        result_mock = MagicMock()
        if call_count["n"] == 0:
            result_mock.result.return_value = iter(label_rows)
        else:
            result_mock.result.return_value = iter(event_rows)
        call_count["n"] += 1
        return result_mock

    mock_bq.Client.return_value.query.side_effect = mock_query
    return mock_bq


def _run_with_mock_bq(
    cfg: dict[str, Any],
    event_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> tuple[list[Any], list[Any]]:
    """Run load_events_labels_from_bigquery with a mocked BigQuery module."""
    mock_bq = _make_mock_bq_module(event_rows, label_rows)
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
        events, labels = load_events_labels_from_bigquery(cfg, "dummy.yaml")
    finally:
        for k in ("google", "google.cloud", "google.cloud.bigquery"):
            if k in saved:
                sys.modules[k] = saved[k]
            else:
                sys.modules.pop(k, None)
    return events, labels


def test_label_matching_prefers_row_key() -> None:
    label_rows = [
        {
            "event_id": "eid_1",
            "row_key": "rk_1",
            "alarm_id": "alm_1",
            "label": "threat",
            "created_at": 1700000000,
        },
    ]
    event_rows = [
        {"row_key": "rk_1", "event_timestamp_epoch_s": 1700000000, "alarm_id": "alm_1"},
    ]
    cfg: dict[str, Any] = {
        "dataset": {
            "bq_project": "proj",
            "bq_dataset": "ds",
            "bq_events_table": "events",
            "bq_labels_table": "labels",
        }
    }
    events, labels = _run_with_mock_bq(cfg, event_rows, label_rows)
    assert len(events) == 1
    assert len(labels) == 1
    assert labels[0].label == "threat"
    assert labels[0].source == "bigquery_feedback"


def test_label_matching_falls_back_to_alarm_id() -> None:
    label_rows = [
        {
            "event_id": "other",
            "row_key": "other_rk",
            "alarm_id": "alm_shared",
            "label": "threat",
            "created_at": 1700000000,
        },
    ]
    event_rows = [
        {"row_key": "rk_no_match", "event_timestamp_epoch_s": 1700000000, "alarm_id": "alm_shared"},
    ]
    cfg: dict[str, Any] = {
        "dataset": {
            "bq_project": "proj",
            "bq_dataset": "ds",
            "bq_events_table": "events",
            "bq_labels_table": "labels",
        }
    }
    events, labels = _run_with_mock_bq(cfg, event_rows, label_rows)
    assert labels[0].label == "threat"


def test_label_matching_falls_back_to_event_id() -> None:
    label_rows = [
        {
            "event_id": "rk_evt",
            "row_key": "other",
            "alarm_id": "other_alm",
            "label": "threat",
            "created_at": 1700000000,
        },
    ]
    event_rows = [
        {"row_key": "rk_evt", "event_timestamp_epoch_s": 1700000000},
    ]
    cfg: dict[str, Any] = {
        "dataset": {
            "bq_project": "proj",
            "bq_dataset": "ds",
            "bq_events_table": "events",
            "bq_labels_table": "labels",
        }
    }
    events, labels = _run_with_mock_bq(cfg, event_rows, label_rows)
    # event_id = row_key = "rk_evt", matched via labels_by_event_id
    assert labels[0].label == "threat"


def test_label_matching_default_when_no_match() -> None:
    label_rows = [
        {
            "event_id": "no_match",
            "row_key": "no_match",
            "alarm_id": "no_match",
            "label": "threat",
            "created_at": 1700000000,
        },
    ]
    event_rows = [
        {
            "row_key": "rk_unmatched",
            "event_timestamp_epoch_s": 1700000000,
            "alarm_id": "alm_unmatched",
        },
    ]
    cfg: dict[str, Any] = {
        "dataset": {
            "bq_project": "proj",
            "bq_dataset": "ds",
            "bq_events_table": "events",
            "bq_labels_table": "labels",
        }
    }
    events, labels = _run_with_mock_bq(cfg, event_rows, label_rows)
    assert labels[0].label == "benign"
    assert labels[0].source == "bigquery_default"
