from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import yaml
from .schemas import RawEvent, Label
from .toy import ToyDataset

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
        ds = ToyDataset(n=5000, seed=42)
        events, labels = ds.generate()
        return LoadedData(events=events, labels=labels)

    # Stubs you can implement later
    raise NotImplementedError(f"Data source not implemented: {source}")
