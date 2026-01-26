from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import yaml
from .schemas import RawEvent, Label

@dataclass
class SplitData:
    train: tuple[list[RawEvent], list[Label]]
    val: tuple[list[RawEvent], list[Label]]
    test: tuple[list[RawEvent], list[Label]]

def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def time_split(events: list[RawEvent], labels: list[Label], data_cfg_path: str) -> SplitData:
    with open(data_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    s = cfg["splits"]

    label_map = {lb.event_id: lb for lb in labels}

    def select(start: str, end: str) -> tuple[list[RawEvent], list[Label]]:
        t0, t1 = _parse_dt(start), _parse_dt(end)
        ev = [e for e in events if t0 <= e.ts <= t1]
        lb = [label_map[e.event_id] for e in ev if e.event_id in label_map]
        return ev, lb

    train = select(s["train"]["start"], s["train"]["end"])
    val = select(s["val"]["start"], s["val"]["end"])
    test = select(s["test"]["start"], s["test"]["end"])
    return SplitData(train=train, val=val, test=test)
