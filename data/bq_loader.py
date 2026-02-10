from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, cast

from .schemas import Label, LabelType, RawEvent


def _label_from_text(raw: str | None, case_class: str | None = None) -> LabelType:
    normalized = (raw or "").strip().lower()
    if normalized in {"threat", "benign", "unknown"}:
        return cast(LabelType, normalized)
    if (case_class or "").strip():
        return "threat"
    return "unknown"


def _parse_epoch_or_datetime(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp())
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
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


def _first_non_empty_str(*values: Any) -> str | None:
    for v in values:
        if v is None:
            continue
        text = str(v).strip()
        if text:
            return text
    return None


def _derive_row_key(row: dict[str, Any], cols: list[str]) -> str:
    parts: list[str] = []
    for col in cols:
        raw = _row_get(row, col)
        if raw is None:
            parts.append("")
        elif isinstance(raw, datetime):
            dt = raw
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            parts.append(dt.isoformat())
        else:
            parts.append(str(raw))
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _mapping_from(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    raw = cfg.get(key)
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    return {}


def _resolve_event_mapping(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = _mapping_from(cfg, "dataset")
    bq_cfg = _mapping_from(cfg, "bigquery")
    event_cfg = _mapping_from(dataset_cfg, "events")
    if not event_cfg:
        event_cfg = _mapping_from(cfg, "events")
    if not event_cfg:
        event_cfg = _mapping_from(bq_cfg, "events")
    row_key_cfg = _mapping_from(event_cfg, "row_key")

    event_ts_col = _first_non_empty_str(
        event_cfg.get("event_ts_col"),
        event_cfg.get("event_timestamp_col"),
        "event_timestamp_epoch_s",
    )
    if event_ts_col is None:
        event_ts_col = "event_timestamp_epoch_s"

    default_derived_cols = [
        str(event_cfg.get("session_id_col", "session_id")),
        str(event_ts_col),
        str(event_cfg.get("user_col", "user")),
        str(event_cfg.get("action_col", "action")),
        str(event_cfg.get("page_col", "page")),
        str(event_cfg.get("details_col", "details")),
    ]
    cols_raw = row_key_cfg.get("cols", default_derived_cols)
    cols = (
        [str(c).strip() for c in cols_raw if str(c).strip()] if isinstance(cols_raw, list) else []
    )
    if not cols:
        cols = default_derived_cols

    return {
        "row_key_col": str(event_cfg.get("row_key_col", "row_key")),
        "event_id_col": str(event_cfg.get("event_id_col", "event_id")),
        "event_ts_col": str(event_ts_col),
        "session_id_col": str(event_cfg.get("session_id_col", "session_id")),
        "user_col": str(event_cfg.get("user_col", "user")),
        "user_id_col": str(event_cfg.get("user_id_col", "user_id")),
        "action_col": str(event_cfg.get("action_col", "action")),
        "page_col": str(event_cfg.get("page_col", "page")),
        "details_col": str(event_cfg.get("details_col", "details")),
        "alarm_id_col": str(event_cfg.get("alarm_id_col", "alarm_id")),
        "row_key_mode": str(row_key_cfg.get("mode", "derived_sha1")),
        "row_key_cols": cols,
    }


def _resolve_label_mapping(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = _mapping_from(cfg, "dataset")
    bq_cfg = _mapping_from(cfg, "bigquery")
    label_cfg = _mapping_from(dataset_cfg, "labels")
    if not label_cfg:
        label_cfg = _mapping_from(cfg, "labels")
    if not label_cfg:
        label_cfg = _mapping_from(bq_cfg, "labels")
    return {
        "event_id_col": str(label_cfg.get("event_id_col", "event_id")),
        "row_key_col": str(label_cfg.get("row_key_col", "row_key")),
        "alarm_id_col": str(label_cfg.get("alarm_id_col", "alarm_id")),
        "label_col": str(label_cfg.get("label_col", "label")),
        "case_class_col": str(label_cfg.get("case_class_col", "case_class")),
        "created_at_col": str(label_cfg.get("created_at_col", "created_at")),
    }


def _resolve_bq_ids(cfg: dict[str, Any]) -> tuple[str, str, str, str | None, int | None]:
    dataset_cfg = _mapping_from(cfg, "dataset")
    bq_cfg = _mapping_from(cfg, "bigquery")
    bq_project = _first_non_empty_str(
        dataset_cfg.get("bq_project"), bq_cfg.get("project_id"), cfg.get("bq_project")
    )
    bq_dataset = _first_non_empty_str(
        dataset_cfg.get("bq_dataset"), bq_cfg.get("dataset"), cfg.get("bq_dataset")
    )
    bq_events_table = _first_non_empty_str(
        dataset_cfg.get("bq_events_table"),
        bq_cfg.get("activity_logs_table"),
        cfg.get("bq_events_table"),
    )
    bq_labels_table = _first_non_empty_str(
        dataset_cfg.get("bq_labels_table"),
        bq_cfg.get("ada_feedback_table"),
        cfg.get("bq_labels_table"),
    )
    if bq_project is None:
        raise ValueError("BigQuery config missing required key: dataset.bq_project")
    if bq_dataset is None:
        raise ValueError("BigQuery config missing required key: dataset.bq_dataset")
    if bq_events_table is None:
        raise ValueError("BigQuery config missing required key: dataset.bq_events_table")

    limit_raw = dataset_cfg.get("limit", bq_cfg.get("limit_rows"))
    limit = int(limit_raw) if limit_raw is not None else None
    if limit is not None and limit <= 0:
        limit = None

    return bq_project, bq_dataset, bq_events_table, bq_labels_table, limit


def row_to_raw_event(row: dict[str, Any], event_mapping: dict[str, Any] | None = None) -> RawEvent:
    """Convert a BigQuery row (or dict) to a RawEvent.

    Exported for unit testing.
    """
    mapping = event_mapping or {}
    row_key_col = str(mapping.get("row_key_col", "row_key"))
    event_id_col = str(mapping.get("event_id_col", "event_id"))
    event_ts_col = str(mapping.get("event_ts_col", "event_timestamp_epoch_s"))
    row_key_mode = str(mapping.get("row_key_mode", "strict"))
    row_key_cols_raw = mapping.get("row_key_cols", [])
    row_key_cols = (
        [str(c).strip() for c in row_key_cols_raw if str(c).strip()]
        if isinstance(row_key_cols_raw, list)
        else []
    )

    row_key = _row_get_str(row, row_key_col)
    event_id_ref = _row_get_str(row, event_id_col)
    if row_key is None and event_id_ref is not None:
        row_key = event_id_ref
    if row_key is None:
        if row_key_mode == "derived_sha1":
            row_key = _derive_row_key(row, row_key_cols)
        else:
            raise ValueError("BigQuery row missing both row_key and event_id")

    epoch_s = _parse_epoch_or_datetime(_row_get(row, event_ts_col))
    if epoch_s is None and event_ts_col != "event_timestamp_epoch_s":
        epoch_s = _parse_epoch_or_datetime(_row_get(row, "event_timestamp_epoch_s"))
    if epoch_s is None and event_ts_col != "event_timestamp":
        epoch_s = _parse_epoch_or_datetime(_row_get(row, "event_timestamp"))
    if epoch_s is None:
        raise ValueError(
            f"BigQuery row missing or invalid timestamp for configured column: {event_ts_col}"
        )

    if epoch_s is not None:
        ts = datetime.utcfromtimestamp(epoch_s)
    else:
        ts = datetime(1970, 1, 1)

    return RawEvent(
        event_id=event_id_ref or row_key,
        ts=ts,
        row_key=row_key,
        event_timestamp_epoch_s=epoch_s,
        session_id=_row_get_str(row, str(mapping.get("session_id_col", "session_id"))),
        user=_row_get_str(row, str(mapping.get("user_col", "user"))),
        user_id=_row_get_str(row, str(mapping.get("user_id_col", "user_id")))
        or _row_get_str(row, str(mapping.get("user_col", "user"))),
        action=_row_get_str(row, str(mapping.get("action_col", "action"))),
        page=_row_get_str(row, str(mapping.get("page_col", "page"))),
        details=_row_get_str(row, str(mapping.get("details_col", "details"))),
        alarm_id=_row_get_str(row, str(mapping.get("alarm_id_col", "alarm_id"))),
        src_ip=_row_get_str(row, "src_ip"),
        dst_ip=_row_get_str(row, "dst_ip"),
        port=_row_get_int(row, "port"),
        protocol=_row_get_str(row, "protocol"),
        duration=_row_get_float(row, "duration"),
        bytes_sent=_row_get_float(row, "bytes_sent"),
        bytes_received=_row_get_float(row, "bytes_received"),
    )


def row_to_label(row: dict[str, Any], label_mapping: dict[str, Any] | None = None) -> Label:
    """Convert a BigQuery label/feedback row (or dict) to a Label.

    Exported for unit testing.
    """
    mapping = label_mapping or {}
    event_id_col = str(mapping.get("event_id_col", "event_id"))
    row_key_col = str(mapping.get("row_key_col", "row_key"))
    alarm_id_col = str(mapping.get("alarm_id_col", "alarm_id"))
    label_col = str(mapping.get("label_col", "label"))
    case_class_col = str(mapping.get("case_class_col", "case_class"))
    created_at_col = str(mapping.get("created_at_col", "created_at"))

    event_id = _row_get_str(row, event_id_col)
    row_key = _row_get_str(row, row_key_col)
    alarm_id = _row_get_str(row, alarm_id_col)
    label_text = _label_from_text(_row_get_str(row, label_col), _row_get_str(row, case_class_col))
    created_at_epoch_s = _parse_epoch_or_datetime(_row_get(row, created_at_col))

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
    _ = data_cfg_path  # reserved for path-based resolution if needed later
    bq_project, bq_dataset, bq_events_table, bq_labels_table, limit = _resolve_bq_ids(cfg)
    event_mapping = _resolve_event_mapping(cfg)
    label_mapping = _resolve_label_mapping(cfg)

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
            lb = row_to_label(row_dict, label_mapping)

            if lb.row_key:
                labels_by_row_key.setdefault(lb.row_key, lb)
            if lb.alarm_id:
                labels_by_alarm_id.setdefault(lb.alarm_id, lb)
            event_id_ref = _row_get_str(row_dict, str(label_mapping["event_id_col"]))
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
        e = row_to_raw_event(row_dict, event_mapping)
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
