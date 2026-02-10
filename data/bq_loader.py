from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

from .schemas import Label, LabelType, RawEvent


def _require_cfg_key(dataset_cfg: dict[str, Any], key: str) -> str:
    val = dataset_cfg.get(key)
    if val is None or str(val).strip() == "":
        raise ValueError(f"BigQuery config missing required key: dataset.{key}")
    return str(val).strip()


def _label_from_text(raw: str | None) -> LabelType:
    normalized = (raw or "").strip().lower()
    if normalized in {"threat", "benign", "unknown"}:
        return cast(LabelType, normalized)
    return "unknown"


def _parse_epoch_or_datetime(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp())


def _row_get(row: Any, key: str) -> Any:
    """Safely get a value from a BigQuery Row or dict, returning None on missing."""
    try:
        return row[key]
    except (KeyError, IndexError, AttributeError):
        return None


def _row_get_str(row: Any, key: str) -> str | None:
    val = _row_get(row, key)
    if val is None:
        return None
    text = str(val).strip()
    return text or None


def _row_get_int(row: Any, key: str) -> int | None:
    val = _row_get(row, key)
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _row_get_float(row: Any, key: str) -> float | None:
    val = _row_get(row, key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def row_to_raw_event(row: dict[str, Any]) -> RawEvent:
    """Convert a BigQuery row (or dict) to a RawEvent.

    Exported for unit testing.
    """
    row_key = str(row.get("row_key") or row.get("event_id") or "")
    if not row_key:
        raise ValueError("BigQuery row missing both row_key and event_id")

    epoch_s_raw = _row_get(row, "event_timestamp_epoch_s")
    if epoch_s_raw is not None:
        epoch_s = int(float(epoch_s_raw))
    else:
        epoch_s = None

    if epoch_s is not None:
        ts = datetime.utcfromtimestamp(epoch_s)
    else:
        ts = datetime(1970, 1, 1)

    return RawEvent(
        event_id=row_key,
        ts=ts,
        row_key=_row_get_str(row, "row_key"),
        event_timestamp_epoch_s=epoch_s,
        session_id=_row_get_str(row, "session_id"),
        user=_row_get_str(row, "user"),
        user_id=_row_get_str(row, "user_id") or _row_get_str(row, "user"),
        action=_row_get_str(row, "action"),
        page=_row_get_str(row, "page"),
        details=_row_get_str(row, "details"),
        alarm_id=_row_get_str(row, "alarm_id"),
        src_ip=_row_get_str(row, "src_ip"),
        dst_ip=_row_get_str(row, "dst_ip"),
        port=_row_get_int(row, "port"),
        protocol=_row_get_str(row, "protocol"),
        duration=_row_get_float(row, "duration"),
        bytes_sent=_row_get_float(row, "bytes_sent"),
        bytes_received=_row_get_float(row, "bytes_received"),
    )


def row_to_label(row: dict[str, Any]) -> Label:
    """Convert a BigQuery label/feedback row (or dict) to a Label.

    Exported for unit testing.
    """
    event_id = _row_get_str(row, "event_id")
    row_key = _row_get_str(row, "row_key")
    alarm_id = _row_get_str(row, "alarm_id")
    label_text = _label_from_text(_row_get_str(row, "label"))
    created_at_epoch_s = _parse_epoch_or_datetime(_row_get(row, "created_at"))

    return Label(
        event_id=event_id or row_key or alarm_id or "__unmatched__",
        label=label_text,
        severity=1.0 if label_text == "threat" else 0.0,
        source="bigquery_feedback",
        alarm_id=alarm_id,
        row_key=row_key,
        label_created_at_epoch_s=created_at_epoch_s,
    )


def load_events_labels_from_bigquery(
    cfg: dict[str, Any], data_cfg_path: str
) -> tuple[list[RawEvent], list[Label]]:
    """Load events and labels from BigQuery tables.

    The google.cloud.bigquery import is lazy so that the module can be imported
    in CI environments without GCP dependencies installed.
    """
    dataset_cfg = cfg.get("dataset", {})

    # Validate config before importing google.cloud (fail-fast without GCP SDK).
    bq_project = _require_cfg_key(dataset_cfg, "bq_project")
    bq_dataset = _require_cfg_key(dataset_cfg, "bq_dataset")
    bq_events_table = _require_cfg_key(dataset_cfg, "bq_events_table")
    bq_labels_table = _row_get_str(dataset_cfg, "bq_labels_table")

    limit_raw = dataset_cfg.get("limit")
    limit = int(limit_raw) if limit_raw is not None else None
    if limit is not None and limit <= 0:
        limit = None

    from google.cloud import bigquery  # lazy import — keeps CI clean

    client = bigquery.Client(project=bq_project)

    # ── Pass 1: load labels ───────────────────────────────────────────
    labels_by_row_key: dict[str, Label] = {}
    labels_by_alarm_id: dict[str, Label] = {}
    labels_by_event_id: dict[str, Label] = {}

    if bq_labels_table:
        labels_sql = f"SELECT * FROM `{bq_project}.{bq_dataset}.{bq_labels_table}`"
        for bq_row in client.query(labels_sql).result():
            row_dict = dict(bq_row)
            lb = row_to_label(row_dict)

            if lb.row_key:
                labels_by_row_key.setdefault(lb.row_key, lb)
            if lb.alarm_id:
                labels_by_alarm_id.setdefault(lb.alarm_id, lb)
            event_id_ref = _row_get_str(row_dict, "event_id")
            if event_id_ref:
                labels_by_event_id.setdefault(event_id_ref, lb)

    # ── Pass 2: load events and match labels ──────────────────────────
    events_sql = f"SELECT * FROM `{bq_project}.{bq_dataset}.{bq_events_table}`"
    if limit:
        events_sql += f" LIMIT {limit}"

    events: list[RawEvent] = []
    labels: list[Label] = []

    for bq_row in client.query(events_sql).result():
        row_dict = dict(bq_row)
        e = row_to_raw_event(row_dict)
        events.append(e)

        # Label matching: row_key → alarm_id → event_id
        matched = labels_by_row_key.get(e.row_key or "")
        if matched is None and e.alarm_id:
            matched = labels_by_alarm_id.get(e.alarm_id)
        if matched is None:
            matched = labels_by_event_id.get(e.event_id)

        if matched is None:
            labels.append(
                Label(
                    event_id=e.event_id,
                    label="benign",
                    severity=0.0,
                    source="bigquery_default",
                    row_key=e.row_key,
                    alarm_id=e.alarm_id,
                    label_created_at_epoch_s=e.event_timestamp_epoch_s,
                )
            )
        else:
            labels.append(
                Label(
                    event_id=e.event_id,
                    label=matched.label,
                    severity=matched.severity,
                    source=matched.source,
                    alarm_id=matched.alarm_id or e.alarm_id,
                    row_key=matched.row_key or e.row_key,
                    label_created_at_epoch_s=matched.label_created_at_epoch_s
                    or e.event_timestamp_epoch_s,
                )
            )

    return events, labels
