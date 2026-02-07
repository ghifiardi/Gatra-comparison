from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from .features import extract_features_v7, extract_features_v128, HistoryContext
from .loaders import load_data
from .schema_hash import schema_hash_from_config
from .splits import time_split


@dataclass(frozen=True)
class ContractPaths:
    root: str
    events_parquet: str
    labels_parquet: str
    v7_npy: str
    v128_npy: str
    y_true_npy: str
    meta_json: str
    schema_hash_txt: str
    v7_train_npy: str | None = None
    v128_train_npy: str | None = None
    y_train_npy: str | None = None
    v7_val_npy: str | None = None
    v128_val_npy: str | None = None
    y_val_npy: str | None = None
    v7_test_npy: str | None = None
    v128_test_npy: str | None = None
    events_train_parquet: str | None = None
    labels_train_parquet: str | None = None
    events_val_parquet: str | None = None
    labels_val_parquet: str | None = None
    events_test_parquet: str | None = None
    labels_test_parquet: str | None = None
    timestamps_train_npy: str | None = None
    timestamps_val_npy: str | None = None
    timestamps_test_npy: str | None = None


def _contract_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return cast(dict[str, Any], data)


def _filter_labeled(
    events: list[Any],
    labels: list[Any],
) -> tuple[list[Any], list[Any], NDArray[np.int8]]:
    label_map = {lb.event_id: lb for lb in labels}
    kept_events: list[Any] = []
    kept_labels: list[Any] = []
    y_vals: list[int] = []
    for e in events:
        lb = label_map.get(e.event_id)
        if lb is None or lb.label == "unknown":
            continue
        kept_events.append(e)
        kept_labels.append(lb)
        y_vals.append(1 if lb.label == "threat" else 0)
    return kept_events, kept_labels, np.array(y_vals, dtype=np.int8)


def _stack_v7(events: list[Any]) -> NDArray[np.float32]:
    if not events:
        return np.zeros((0, 7), dtype=np.float32)
    return np.stack([extract_features_v7(e) for e in events]).astype(np.float32)


def _stack_v128(events: list[Any]) -> NDArray[np.float32]:
    if not events:
        return np.zeros((0, 128), dtype=np.float32)
    return np.stack([extract_features_v128(e, HistoryContext(now=e.ts)) for e in events]).astype(
        np.float32
    )


def _epoch_s(e: Any) -> int:
    # Prefer raw epoch seconds if the loader provided it (CSV mode).
    raw = getattr(e, "event_timestamp_epoch_s", None)
    if raw is not None:
        return int(raw)
    # Treat naive datetimes as UTC for stable export.
    return int(e.ts.replace(tzinfo=timezone.utc).timestamp())


def export_frozen_contract_to_dir(
    data_cfg_path: str,
    out_dir: str,
    include_splits: bool = True,
    contract_id: str | None = None,
) -> ContractPaths:
    os.makedirs(out_dir, exist_ok=True)

    loaded = load_data(data_cfg_path)
    splits = time_split(loaded.events, loaded.labels, data_cfg_path)
    train_events, train_labels = splits.train
    val_events, val_labels = splits.val
    test_events, test_labels = splits.test

    train_kept, train_labels_kept, y_train = _filter_labeled(train_events, train_labels)
    val_kept, val_labels_kept, y_val = _filter_labeled(val_events, val_labels)
    test_kept, test_labels_kept, y_test = _filter_labeled(test_events, test_labels)

    X7_train = _stack_v7(train_kept)
    X128_train = _stack_v128(train_kept)
    X7_val = _stack_v7(val_kept)
    X128_val = _stack_v128(val_kept)
    X7_test = _stack_v7(test_kept)
    X128_test = _stack_v128(test_kept)

    ts_train = np.array([_epoch_s(e) for e in train_kept], dtype=np.int64)
    ts_val = np.array([_epoch_s(e) for e in val_kept], dtype=np.int64)
    ts_test = np.array([_epoch_s(e) for e in test_kept], dtype=np.int64)

    def _df(items: list[Any]) -> pd.DataFrame:
        return pd.DataFrame([item.model_dump() for item in items])

    def _write_tabular(df: pd.DataFrame, parquet_path: str) -> str:
        try:
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        except ImportError:
            # Keep contract generation usable without optional parquet engines.
            csv_path = os.path.splitext(parquet_path)[0] + ".csv"
            df.to_csv(csv_path, index=False)
            return csv_path

    events_train_parquet = os.path.join(out_dir, "events_train.parquet")
    labels_train_parquet = os.path.join(out_dir, "labels_train.parquet")
    events_val_parquet = os.path.join(out_dir, "events_val.parquet")
    labels_val_parquet = os.path.join(out_dir, "labels_val.parquet")
    events_test_parquet = os.path.join(out_dir, "events_test.parquet")
    labels_test_parquet = os.path.join(out_dir, "labels_test.parquet")

    if include_splits:
        events_train_parquet = _write_tabular(_df(train_kept), events_train_parquet)
        labels_train_parquet = _write_tabular(_df(train_labels_kept), labels_train_parquet)
        events_val_parquet = _write_tabular(_df(val_kept), events_val_parquet)
        labels_val_parquet = _write_tabular(_df(val_labels_kept), labels_val_parquet)

    events_test_parquet = _write_tabular(_df(test_kept), events_test_parquet)
    labels_test_parquet = _write_tabular(_df(test_labels_kept), labels_test_parquet)

    v7_train_npy = os.path.join(out_dir, "features_v7_train.npy")
    v128_train_npy = os.path.join(out_dir, "features_v128_train.npy")
    y_train_npy = os.path.join(out_dir, "y_train.npy")
    v7_val_npy = os.path.join(out_dir, "features_v7_val.npy")
    v128_val_npy = os.path.join(out_dir, "features_v128_val.npy")
    y_val_npy = os.path.join(out_dir, "y_val.npy")
    v7_test_npy = os.path.join(out_dir, "features_v7_test.npy")
    v128_test_npy = os.path.join(out_dir, "features_v128_test.npy")
    y_true_npy = os.path.join(out_dir, "y_true.npy")
    timestamps_train_npy = os.path.join(out_dir, "timestamps_epoch_s_train.npy")
    timestamps_val_npy = os.path.join(out_dir, "timestamps_epoch_s_val.npy")
    timestamps_test_npy = os.path.join(out_dir, "timestamps_epoch_s_test.npy")

    np.save(v7_train_npy, X7_train)
    np.save(v128_train_npy, X128_train)
    np.save(y_train_npy, y_train)
    np.save(v7_val_npy, X7_val)
    np.save(v128_val_npy, X128_val)
    np.save(y_val_npy, y_val)
    np.save(v7_test_npy, X7_test)
    np.save(v128_test_npy, X128_test)
    np.save(y_true_npy, y_test)
    np.save(timestamps_train_npy, ts_train)
    np.save(timestamps_val_npy, ts_val)
    np.save(timestamps_test_npy, ts_test)

    # Backward-compatible test names
    v7_npy = os.path.join(out_dir, "features_v7.npy")
    v128_npy = os.path.join(out_dir, "features_v128.npy")
    np.save(v7_npy, X7_test)
    np.save(v128_npy, X128_test)

    schema_hash, schema_repr = schema_hash_from_config(data_cfg_path)
    schema_hash_txt = os.path.join(out_dir, "schema_hash.txt")
    with open(schema_hash_txt, "w") as f:
        f.write(schema_hash)

    cfg = _load_yaml(data_cfg_path)
    meta = {
        "contract_id": contract_id or os.path.basename(out_dir),
        "git_commit": _git_commit_hash(),
        "splits": cfg.get("splits", {}),
        "schema_repr": schema_repr,
        "schema_hash": schema_hash,
        "counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "label_pos_rate": {
            "train": float(y_train.mean()) if len(y_train) else 0.0,
            "val": float(y_val.mean()) if len(y_val) else 0.0,
            "test": float(y_test.mean()) if len(y_test) else 0.0,
        },
        "normalization": {
            "iforest_scaler": "standard",
            "notes": "Scaler statistics stored in IF bundle",
        },
    }
    meta_json = os.path.join(out_dir, "meta.json")
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    return ContractPaths(
        root=out_dir,
        events_parquet=events_test_parquet,
        labels_parquet=labels_test_parquet,
        v7_npy=v7_npy,
        v128_npy=v128_npy,
        y_true_npy=y_true_npy,
        meta_json=meta_json,
        schema_hash_txt=schema_hash_txt,
        v7_train_npy=v7_train_npy,
        v128_train_npy=v128_train_npy,
        y_train_npy=y_train_npy,
        v7_val_npy=v7_val_npy,
        v128_val_npy=v128_val_npy,
        y_val_npy=y_val_npy,
        v7_test_npy=v7_test_npy,
        v128_test_npy=v128_test_npy,
        events_train_parquet=events_train_parquet if include_splits else None,
        labels_train_parquet=labels_train_parquet if include_splits else None,
        events_val_parquet=events_val_parquet if include_splits else None,
        labels_val_parquet=labels_val_parquet if include_splits else None,
        events_test_parquet=events_test_parquet,
        labels_test_parquet=labels_test_parquet,
        timestamps_train_npy=timestamps_train_npy,
        timestamps_val_npy=timestamps_val_npy,
        timestamps_test_npy=timestamps_test_npy,
    )


def export_frozen_contract(
    data_cfg_path: str,
    out_root: str = "./reports/contracts",
    include_splits: bool = True,
    contract_id: str | None = None,
) -> ContractPaths:
    os.makedirs(out_root, exist_ok=True)
    cid = contract_id or _contract_id()
    root = os.path.join(out_root, cid)
    return export_frozen_contract_to_dir(
        data_cfg_path=data_cfg_path,
        out_dir=root,
        include_splits=include_splits,
        contract_id=cid,
    )
