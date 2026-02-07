from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, cast
import yaml
from .schemas import RawEvent, Label
from .toy import ToyDataset
from .csv_loader import load_events_labels_from_csv

SourceType = Literal["toy", "csv", "parquet", "bigquery"]


@dataclass
class LoadedData:
    events: list[RawEvent]
    labels: list[Label]


def load_data(data_config_path: str) -> LoadedData:
    with open(data_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    source: SourceType = cfg["dataset"]["source"]

    if source == "toy":
        dataset_cfg = cfg.get("dataset", {})
        n = int(dataset_cfg.get("n", 5000))
        seed = int(dataset_cfg.get("seed", 42))
        limit_raw = dataset_cfg.get("limit")
        limit = int(limit_raw) if limit_raw is not None else None
        if limit is not None and limit > 0:
            n = min(n, limit)
        ds = ToyDataset(n=n, seed=seed)
        events, labels = ds.generate()
        return LoadedData(events=events, labels=labels)

    if source == "csv":
        events, labels = load_events_labels_from_csv(cast(dict[str, Any], cfg), data_config_path)
        return LoadedData(events=events, labels=labels)

    # Stubs you can implement later
    raise NotImplementedError(f"Data source not implemented: {source}")
