from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import Label, RawEvent


def _resolve_path(data_cfg_path: str, maybe_rel_path: str) -> Path:
    p = Path(maybe_rel_path)
    if p.is_absolute():
        return p
    # Prefer resolving relative to repo/process CWD (so config snapshots under
    # reports/runs/**/config still point at repo-local data/ paths).
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    # Fallback: resolve relative to the YAML config file location.
    return (Path(data_cfg_path).resolve().parent / p).resolve()


def _require_str(row: dict[str, str], key: str) -> str:
    if key not in row:
        raise ValueError(f"CSV is missing required column: {key}")
    val = (row.get(key) or "").strip()
    if not val:
        raise ValueError(f"CSV column is empty: {key}")
    return val


def _parse_epoch_s(raw: str, key_name: str) -> int:
    try:
        # Some exports may encode epoch as float-like strings.
        return int(float(raw))
    except Exception as e:
        raise ValueError(f"Invalid epoch seconds in column {key_name}: {raw!r}") from e


def _parse_epoch_or_datetime(raw: str, key_name: str) -> int:
    text = raw.strip()
    if not text:
        raise ValueError(f"Empty timestamp value for column {key_name}")
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


def _label_from_text(raw: str | None, case_class: str | None) -> str:
    normalized = (raw or "").strip().lower()
    if normalized in {"threat", "benign", "unknown"}:
        return normalized
    if (case_class or "").strip():
        return "threat"
    return "unknown"


def _resolve_label_path(cfg: dict[str, Any], data_cfg_path: str) -> Path | None:
    dataset_cfg = cfg.get("dataset", {})
    labels_cfg = cfg.get("labels", {})
    raw = dataset_cfg.get("label_path")
    if raw is None and isinstance(labels_cfg, dict):
        raw = labels_cfg.get("path")
    if not raw:
        return None
    return _resolve_path(data_cfg_path, str(raw))


def load_events_labels_from_csv(
    cfg: dict[str, Any], data_cfg_path: str
) -> tuple[list[RawEvent], list[Label]]:
    dataset_cfg = cfg.get("dataset", {})
    csv_path = _resolve_path(data_cfg_path, str(dataset_cfg.get("path", "")))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    row_key_field = str(dataset_cfg.get("row_key_field", "")).strip()
    timestamp_epoch_field = str(dataset_cfg.get("timestamp_epoch_field", "")).strip()
    if not row_key_field:
        raise ValueError("Missing dataset.row_key_field for CSV source")
    if not timestamp_epoch_field:
        raise ValueError("Missing dataset.timestamp_epoch_field for CSV source")

    columns_cfg = dataset_cfg.get("columns", {}) or {}
    if not isinstance(columns_cfg, dict):
        raise ValueError("dataset.columns must be a mapping")
    columns: dict[str, str] = {str(k): str(v) for k, v in columns_cfg.items()}

    limit_raw = dataset_cfg.get("limit")
    limit = int(limit_raw) if limit_raw is not None else None
    if limit is not None and limit <= 0:
        limit = None

    labels_by_row_key: dict[str, Label] = {}
    labels_by_alarm_id: dict[str, Label] = {}
    labels_by_event_id: dict[str, Label] = {}

    label_path = _resolve_label_path(cfg, data_cfg_path)
    if label_path is not None and label_path.exists():
        label_cfg_raw = dataset_cfg.get("label_columns", {})
        label_cfg = label_cfg_raw if isinstance(label_cfg_raw, dict) else {}
        label_field = str(label_cfg.get("label", "label"))
        case_class_field = str(label_cfg.get("case_class", "case_class"))
        event_id_field = str(label_cfg.get("event_id", "event_id"))
        row_key_ref_field = str(label_cfg.get("row_key", "row_key"))
        alarm_id_field = str(label_cfg.get("alarm_id", "alarm_id"))
        created_at_field = str(label_cfg.get("created_at", "created_at"))

        with label_path.open(newline="") as lf:
            reader = csv.DictReader(lf)
            if reader.fieldnames is not None:
                for row in reader:
                    row_key_ref = (row.get(row_key_ref_field) or "").strip() or None
                    alarm_id_ref = (row.get(alarm_id_field) or "").strip() or None
                    event_id_ref = (row.get(event_id_field) or "").strip() or None
                    case_class = (row.get(case_class_field) or "").strip() or None
                    label_text = _label_from_text(row.get(label_field), case_class)
                    created_raw = (row.get(created_at_field) or "").strip()
                    created_at_epoch_s = (
                        _parse_epoch_or_datetime(created_raw, created_at_field)
                        if created_raw
                        else None
                    )
                    lb = Label(
                        event_id=event_id_ref or row_key_ref or alarm_id_ref or "__unmatched__",
                        label=label_text,
                        severity=1.0 if label_text == "threat" else 0.0,
                        source="csv_feedback",
                        alarm_id=alarm_id_ref,
                        row_key=row_key_ref,
                        label_created_at_epoch_s=created_at_epoch_s,
                    )
                    if row_key_ref:
                        labels_by_row_key.setdefault(row_key_ref, lb)
                    if alarm_id_ref:
                        labels_by_alarm_id.setdefault(alarm_id_ref, lb)
                    if event_id_ref:
                        labels_by_event_id.setdefault(event_id_ref, lb)

    events: list[RawEvent] = []
    labels: list[Label] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break

            # Canonical keys
            row_key = _require_str(row, row_key_field)
            epoch_s_raw = _require_str(row, timestamp_epoch_field)
            epoch_s = _parse_epoch_s(epoch_s_raw, timestamp_epoch_field)

            # Canonical time axis: epoch seconds -> UTC datetime (naive).
            ts = datetime.utcfromtimestamp(epoch_s)

            # Optional pass-through fields (traceability)
            def get_opt(name: str) -> str | None:
                col = columns.get(name)
                if not col:
                    return None
                v = row.get(col)
                if v is None:
                    return None
                v = v.strip()
                return v or None

            session_id = get_opt("session_id")
            user = get_opt("user")
            action = get_opt("action")
            page = get_opt("page")
            details = get_opt("details")
            alarm_id = get_opt("alarm_id")

            e = RawEvent(
                event_id=row_key,  # preserve row_key as canonical id
                ts=ts,
                row_key=row_key,
                event_timestamp_epoch_s=epoch_s,
                session_id=session_id,
                user=user,
                user_id=user,  # keep compatibility with existing schema fields
                action=action,
                page=page,
                details=details,
                alarm_id=alarm_id,
            )
            events.append(e)

            matched = labels_by_row_key.get(row_key)
            if matched is None and alarm_id:
                matched = labels_by_alarm_id.get(alarm_id)
            if matched is None:
                matched = labels_by_event_id.get(row_key)

            if matched is None:
                labels.append(
                    Label(
                        event_id=e.event_id,
                        label="benign",
                        severity=0.0,
                        source="csv_default",
                        row_key=row_key,
                        alarm_id=alarm_id,
                        label_created_at_epoch_s=epoch_s,
                    )
                )
            else:
                labels.append(
                    Label(
                        event_id=e.event_id,
                        label=matched.label,
                        severity=matched.severity,
                        source=matched.source,
                        alarm_id=matched.alarm_id or alarm_id,
                        row_key=matched.row_key or row_key,
                        label_created_at_epoch_s=matched.label_created_at_epoch_s or epoch_s,
                    )
                )

    return events, labels
