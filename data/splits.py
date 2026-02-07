from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import yaml
from .schemas import RawEvent, Label


@dataclass
class SplitData:
    train: tuple[list[RawEvent], list[Label]]
    val: tuple[list[RawEvent], list[Label]]
    test: tuple[list[RawEvent], list[Label]]


def _to_epoch_s(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp())


def _parse_window_boundary_epoch_s(raw: str, *, is_end: bool) -> int:
    dt = datetime.fromisoformat(raw)
    # For date-only strings (YYYY-MM-DD), treat end boundaries as inclusive
    # end-of-day to match split config intent.
    if len(raw) == 10 and is_end:
        dt = dt + timedelta(days=1) - timedelta(seconds=1)
    return _to_epoch_s(dt)


def _event_epoch_s(event: RawEvent) -> int:
    if event.event_timestamp_epoch_s is not None:
        return int(event.event_timestamp_epoch_s)
    return _to_epoch_s(event.ts)


def time_split(events: list[RawEvent], labels: list[Label], data_cfg_path: str) -> SplitData:
    with open(data_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    s = cfg["splits"]

    label_map = {lb.event_id: lb for lb in labels}

    def select(start: str, end: str) -> tuple[list[RawEvent], list[Label]]:
        t0 = _parse_window_boundary_epoch_s(start, is_end=False)
        t1 = _parse_window_boundary_epoch_s(end, is_end=True)
        ev = [e for e in events if t0 <= _event_epoch_s(e) <= t1]
        lb = [label_map[e.event_id] for e in ev if e.event_id in label_map]
        return ev, lb

    train = select(s["train"]["start"], s["train"]["end"])
    val = select(s["val"]["start"], s["val"]["end"])
    test = select(s["test"]["start"], s["test"]["end"])
    return SplitData(train=train, val=val, test=test)
