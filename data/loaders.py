from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal, Optional

import pandas as pd
import yaml

from .bq import BQTableRef, bq_client, fetch_rows_timewindow, fetch_rows_all
from .schemas import RawEvent, Label
from .toy import ToyDataset

SourceType = Literal["toy", "csv", "parquet", "bigquery"]


@dataclass
class LoadedData:
    events: list[RawEvent]
    labels: list[Label]


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return datetime.fromisoformat(str(value))


def _map_events(rows: Iterable[dict[str, Any]], mapping: dict[str, str]) -> list[RawEvent]:
    events: list[RawEvent] = []
    for r in rows:
        events.append(
            RawEvent(
                event_id=str(r.get(mapping["event_id"])),
                ts=_parse_ts(r.get(mapping["ts"])),
                src_ip=str(r.get(mapping.get("src_ip"))) if mapping.get("src_ip") else None,
                dst_ip=str(r.get(mapping.get("dst_ip"))) if mapping.get("dst_ip") else None,
                port=int(r.get(mapping.get("port")))
                if r.get(mapping.get("port")) is not None
                else None,
                protocol=(
                    str(r.get(mapping.get("protocol")))
                    if r.get(mapping.get("protocol")) is not None
                    else None
                ),
                duration=(
                    float(r.get(mapping.get("duration")))
                    if r.get(mapping.get("duration")) is not None
                    else None
                ),
                bytes_sent=(
                    float(r.get(mapping.get("bytes_sent")))
                    if r.get(mapping.get("bytes_sent")) is not None
                    else None
                ),
                bytes_received=(
                    float(r.get(mapping.get("bytes_received")))
                    if r.get(mapping.get("bytes_received")) is not None
                    else None
                ),
                user_id=(
                    str(r.get(mapping.get("user_id")))
                    if r.get(mapping.get("user_id")) is not None
                    else None
                ),
                host_id=(
                    str(r.get(mapping.get("host_id")))
                    if r.get(mapping.get("host_id")) is not None
                    else None
                ),
            )
        )
    return events


def _map_labels(rows: Iterable[dict[str, Any]], mapping: dict[str, str]) -> list[Label]:
    labels: list[Label] = []
    for r in rows:
        labels.append(
            Label(
                event_id=str(r.get(mapping["event_id"])),
                label=str(r.get(mapping.get("label")) or "unknown"),
                severity=float(r.get(mapping.get("severity")) or 0.0),
                source=str(r.get(mapping.get("source")) or "unknown"),
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
    ev_map = mapping_cfg.get("events", {})
    lb_map = mapping_cfg.get("labels", {})

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
