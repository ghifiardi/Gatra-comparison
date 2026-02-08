from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.contract_export import export_frozen_contract_to_dir
from data.loaders import load_data
from data.schemas import Label, RawEvent
from data.splits import time_split


def _write_split_cfg(path: Path) -> None:
    path.write_text(
        """
dataset:
  source: "toy"
splits:
  train:
    start: "2025-01-01"
    end: "2025-01-01"
  val:
    start: "2025-01-02"
    end: "2025-01-02"
  test:
    start: "2025-01-03"
    end: "2025-01-03"
""".strip()
    )


def test_time_split_uses_event_timestamp_epoch_and_date_end_is_inclusive(tmp_path: Path) -> None:
    cfg = tmp_path / "split.yaml"
    _write_split_cfg(cfg)

    train_epoch = int(datetime(2025, 1, 1, 23, 30, tzinfo=timezone.utc).timestamp())
    val_epoch = int(datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc).timestamp())
    test_epoch = int(datetime(2025, 1, 3, 0, 15, tzinfo=timezone.utc).timestamp())

    events = [
        RawEvent(
            event_id="evt_train",
            # Deliberately mismatched ts so split must use event_timestamp_epoch_s.
            ts=datetime(2030, 1, 1, 0, 0),
            event_timestamp_epoch_s=train_epoch,
        ),
        RawEvent(
            event_id="evt_val",
            ts=datetime(2025, 1, 2, 12, 0),
            event_timestamp_epoch_s=val_epoch,
        ),
        RawEvent(
            event_id="evt_test",
            ts=datetime(2025, 1, 3, 0, 15),
            event_timestamp_epoch_s=test_epoch,
        ),
    ]
    labels = [
        Label(event_id=e.event_id, label="benign", severity=0.0, source="test") for e in events
    ]

    splits = time_split(events, labels, str(cfg))
    assert [e.event_id for e in splits.train[0]] == ["evt_train"]
    assert [e.event_id for e in splits.val[0]] == ["evt_val"]
    assert [e.event_id for e in splits.test[0]] == ["evt_test"]


def test_csv_loader_preserves_row_key_and_contract_exports_timestamps(tmp_path: Path) -> None:
    events_csv = tmp_path / "events.csv"
    events_csv.write_text(
        "\n".join(
            [
                "row_key,event_timestamp_epoch_s,session_id,user,action,page,details",
                "rk_train,1735732800,s1,u1,login,Overview,d1",
                "rk_val,1735822800,s2,u2,click,Activity Logs,d2",
                "rk_test,1735912800,s3,u3,view,Human Feedback,d3",
            ]
        )
    )

    cfg = tmp_path / "data_csv.yaml"
    cfg.write_text(
        f"""
dataset:
  source: "csv"
  path: "{events_csv}"
  row_key_field: "row_key"
  timestamp_epoch_field: "event_timestamp_epoch_s"
  columns:
    session_id: "session_id"
    user: "user"
    action: "action"
    page: "page"
    details: "details"
splits:
  train:
    start: "2025-01-01"
    end: "2025-01-01"
  val:
    start: "2025-01-02"
    end: "2025-01-02"
  test:
    start: "2025-01-03"
    end: "2025-01-03"
""".strip()
    )

    loaded = load_data(str(cfg))
    assert loaded.events[0].event_id == "rk_train"
    assert loaded.events[0].row_key == "rk_train"
    assert loaded.events[0].event_timestamp_epoch_s == 1735732800

    out_dir = tmp_path / "contract"
    paths = export_frozen_contract_to_dir(
        data_cfg_path=str(cfg),
        out_dir=str(out_dir),
        include_splits=True,
        contract_id="test-csv",
    )

    ts_train = np.load(paths.timestamps_train_npy or "")
    ts_val = np.load(paths.timestamps_val_npy or "")
    ts_test = np.load(paths.timestamps_test_npy or "")
    session_train = np.load(paths.session_id_train_npy or "")
    session_val = np.load(paths.session_id_val_npy or "")
    session_test = np.load(paths.session_id_test_npy or "")
    assert ts_train.tolist() == [1735732800]
    assert ts_val.tolist() == [1735822800]
    assert ts_test.tolist() == [1735912800]
    assert session_train.tolist() == ["s1"]
    assert session_val.tolist() == ["s2"]
    assert session_test.tolist() == ["s3"]
    assert len(session_train) == len(ts_train)
    assert len(session_val) == len(ts_val)
    assert len(session_test) == len(ts_test)

    events_train_path = Path(paths.events_train_parquet or "")
    if events_train_path.suffix == ".parquet":
        events_train_df = pd.read_parquet(events_train_path)
    else:
        events_train_df = pd.read_csv(events_train_path)
    assert "row_key" in events_train_df.columns
    assert events_train_df["row_key"].iloc[0] == "rk_train"

    with open(paths.meta_json, "r") as f:
        meta = json.load(f)
    assert "episodes" in meta
