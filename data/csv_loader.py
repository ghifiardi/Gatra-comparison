from __future__ import annotations

import csv
from datetime import datetime
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
            )
            events.append(e)

            # Default labels to keep pipeline runnable (until real label join wiring exists)
            labels.append(
                Label(event_id=e.event_id, label="benign", severity=0.0, source="csv_default")
            )

    return events, labels
