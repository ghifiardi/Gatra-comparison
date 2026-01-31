from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal, Optional, Mapping, cast

import pandas as pd
import yaml

from .bq import BQTableRef, bq_client, fetch_rows_timewindow, fetch_rows_all
from .schemas import RawEvent, Label, LabelType
from .toy import ToyDataset

SourceType = Literal["toy", "csv", "parquet", "bigquery"]


@dataclass
class LoadedData:
    events: list[RawEvent]
    labels: list[Label]


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return cast(dict[str, Any], data)


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return cast(datetime, value.to_pydatetime())
    return datetime.fromisoformat(str(value))

def _opt_str(row: Mapping[str, Any], key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    value = row.get(key)
    return None if value is None else str(value)


def _opt_int(row: Mapping[str, Any], key: Optional[str]) -> Optional[int]:
    if not key:
        return None
    value = row.get(key)
    return None if value is None else int(value)


def _opt_float(row: Mapping[str, Any], key: Optional[str]) -> Optional[float]:
    if not key:
        return None
    value = row.get(key)
    return None if value is None else float(value)


def _normalize_label(value: Any) -> LabelType:
    if isinstance(value, str):
        v = value.lower()
        if v in ("threat", "benign", "unknown"):
            return cast(LabelType, v)
    return "unknown"


def _map_events(rows: Iterable[Mapping[str, Any]], mapping: Mapping[str, str]) -> list[RawEvent]:
    events: list[RawEvent] = []
    src_key = mapping.get("src_ip")
    dst_key = mapping.get("dst_ip")
    port_key = mapping.get("port")
    protocol_key = mapping.get("protocol")
    duration_key = mapping.get("duration")
    bytes_sent_key = mapping.get("bytes_sent")
    bytes_received_key = mapping.get("bytes_received")
    user_key = mapping.get("user_id")
    host_key = mapping.get("host_id")
    for r in rows:
        events.append(
            RawEvent(
                event_id=str(r.get(mapping["event_id"])),
                ts=_parse_ts(r.get(mapping["ts"])),
                src_ip=_opt_str(r, src_key),
                dst_ip=_opt_str(r, dst_key),
                port=_opt_int(r, port_key),
                protocol=_opt_str(r, protocol_key),
                duration=_opt_float(r, duration_key),
                bytes_sent=_opt_float(r, bytes_sent_key),
                bytes_received=_opt_float(r, bytes_received_key),
                user_id=_opt_str(r, user_key),
                host_id=_opt_str(r, host_key),
            )
        )
    return events


def _map_labels(rows: Iterable[Mapping[str, Any]], mapping: Mapping[str, str]) -> list[Label]:
    labels: list[Label] = []
    label_key = mapping.get("label")
    severity_key = mapping.get("severity")
    source_key = mapping.get("source")
    for r in rows:
        label_value = r.get(label_key) if label_key else None
        severity_value = r.get(severity_key) if severity_key else None
        source_value = r.get(source_key) if source_key else None
        labels.append(
            Label(
                event_id=str(r.get(mapping["event_id"])),
                label=_normalize_label(label_value),
                severity=float(severity_value or 0.0),
                source=str(source_value or "unknown"),
            )
        )
    return labels


def load_parquet(events_path: str, labels_path: str) -> LoadedData:
    ev_df = pd.read_parquet(events_path)
    lb_df = pd.read_parquet(labels_path)
    events = [RawEvent(**row) for row in ev_df.to_dict(orient="records")]
    labels = [Label(**row) for row in lb_df.to_dict(orient="records")]
    return LoadedData(events=events, labels=labels)


def load_csv(events_path: str, labels_path: str) -> LoadedData:
    ev_df = pd.read_csv(events_path)
    lb_df = pd.read_csv(labels_path)
    events = [RawEvent(**row) for row in ev_df.to_dict(orient="records")]
    labels = [Label(**row) for row in lb_df.to_dict(orient="records")]
    return LoadedData(events=events, labels=labels)


def load_bigquery(cfg: dict[str, Any]) -> LoadedData:
    dataset_cfg = cfg["dataset"]
    labels_cfg = cfg.get("labels", {})
    mapping_cfg = cfg.get("mapping", {})
    ev_map = cast(dict[str, str], mapping_cfg.get("events", {}))
    lb_map = cast(dict[str, str], mapping_cfg.get("labels", {}))

    project = dataset_cfg["bq_project"]
    dataset = dataset_cfg["bq_dataset"]
    events_table = dataset_cfg["bq_events_table"]
    labels_table = labels_cfg.get("bq_labels_table", dataset_cfg.get("bq_labels_table"))

    if not events_table or not labels_table:
        raise ValueError("BigQuery events/labels table names must be set in config")

    limit = dataset_cfg.get("limit")

    splits = cfg["splits"]
    start = splits["train"]["start"]
    end = splits["test"]["end"]

    client = bq_client()

    for key in ("event_id", "ts"):
        if key not in ev_map:
            raise ValueError(f"Missing required events mapping key: {key}")
    if "event_id" not in lb_map:
        raise ValueError("Missing required labels mapping key: event_id")

    ev_rows = list(
        fetch_rows_timewindow(
            client=client,
            table=BQTableRef(project, dataset, events_table),
            ts_col=ev_map["ts"],
            start_iso=start,
            end_iso=end,
            limit=limit,
        )
    )

    lb_ts_col: Optional[str] = lb_map.get("ts")
    if lb_ts_col:
        lb_rows = list(
            fetch_rows_timewindow(
                client=client,
                table=BQTableRef(project, dataset, labels_table),
                ts_col=lb_ts_col,
                start_iso=start,
                end_iso=end,
                limit=limit,
            )
        )
    else:
        lb_rows = list(
            fetch_rows_all(
                client=client,
                table=BQTableRef(project, dataset, labels_table),
                limit=limit,
            )
        )

    events = _map_events(ev_rows, ev_map)
    labels = _map_labels(lb_rows, lb_map)

    return LoadedData(events=events, labels=labels)


def load_data(data_config_path: str) -> LoadedData:
    cfg = _load_yaml(data_config_path)
    source: SourceType = cfg["dataset"]["source"]

    if source == "toy":
        n = int(cfg["dataset"].get("n", 5000))
        ds = ToyDataset(n=n, seed=42)
        events, labels = ds.generate()
        return LoadedData(events=events, labels=labels)

    if source == "parquet":
        events_path = cfg["dataset"]["path"]
        labels_path = cfg.get("labels", {}).get("path", cfg["dataset"]["label_path"])
        return load_parquet(events_path, labels_path)

    if source == "csv":
        events_path = cfg["dataset"]["path"]
        labels_path = cfg.get("labels", {}).get("path", cfg["dataset"]["label_path"])
        return load_csv(events_path, labels_path)

    if source == "bigquery":
        return load_bigquery(cfg)

    raise NotImplementedError(f"Data source not implemented: {source}")
