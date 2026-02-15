from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
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
    session_id_train_npy: str | None = None
    session_id_val_npy: str | None = None
    session_id_test_npy: str | None = None
    user_train_npy: str | None = None
    user_val_npy: str | None = None
    user_test_npy: str | None = None
    action_train_npy: str | None = None
    action_val_npy: str | None = None
    action_test_npy: str | None = None
    page_train_npy: str | None = None
    page_val_npy: str | None = None
    page_test_npy: str | None = None
    row_key_train_npy: str | None = None
    row_key_val_npy: str | None = None
    row_key_test_npy: str | None = None
    alarm_id_train_npy: str | None = None
    alarm_id_val_npy: str | None = None
    alarm_id_test_npy: str | None = None
    label_row_key_train_npy: str | None = None
    label_row_key_val_npy: str | None = None
    label_row_key_test_npy: str | None = None
    label_alarm_id_train_npy: str | None = None
    label_alarm_id_val_npy: str | None = None
    label_alarm_id_test_npy: str | None = None
    label_session_id_train_npy: str | None = None
    label_session_id_val_npy: str | None = None
    label_session_id_test_npy: str | None = None
    label_user_train_npy: str | None = None
    label_user_val_npy: str | None = None
    label_user_test_npy: str | None = None
    label_created_at_train_npy: str | None = None
    label_created_at_val_npy: str | None = None
    label_created_at_test_npy: str | None = None
    episode_id_train_npy: str | None = None
    episode_id_val_npy: str | None = None
    episode_id_test_npy: str | None = None
    episode_start_ts_train_npy: str | None = None
    episode_start_ts_val_npy: str | None = None
    episode_start_ts_test_npy: str | None = None
    episode_end_ts_train_npy: str | None = None
    episode_end_ts_val_npy: str | None = None
    episode_end_ts_test_npy: str | None = None


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


def export_frozen_contract(
    data_cfg_path: str,
    out_root: str = "./reports/contracts",
) -> ContractPaths:
    os.makedirs(out_root, exist_ok=True)
    cid = _contract_id()
    root = os.path.join(out_root, cid)
    os.makedirs(root, exist_ok=True)

    loaded = load_data(data_cfg_path)
    splits = time_split(loaded.events, loaded.labels, data_cfg_path)
    test_events, test_labels = splits.test
    label_map = {lb.event_id: lb for lb in test_labels}

    kept_events = []
    kept_labels = []
    for e in test_events:
        lb = label_map.get(e.event_id)
        if lb is None or lb.label == "unknown":
            continue
        kept_events.append(e)
        kept_labels.append(lb)

    X7: NDArray[np.float32] = np.stack([extract_features_v7(e) for e in kept_events]).astype(
        np.float32
    )
    X128: NDArray[np.float32] = np.stack(
        [extract_features_v128(e, HistoryContext(now=e.ts)) for e in kept_events]
    ).astype(np.float32)
    y_true: NDArray[np.int8] = np.array(
        [1 if lb.label == "threat" else 0 for lb in kept_labels], dtype=np.int8
    )

    events_df = pd.DataFrame([e.model_dump() for e in kept_events])
    labels_df = pd.DataFrame([lb.model_dump() for lb in kept_labels])

    events_parquet = os.path.join(root, "events.parquet")
    labels_parquet = os.path.join(root, "labels.parquet")
    events_df.to_parquet(events_parquet, index=False)
    labels_df.to_parquet(labels_parquet, index=False)

    v7_npy = os.path.join(root, "features_v7.npy")
    v128_npy = os.path.join(root, "features_v128.npy")
    y_true_npy = os.path.join(root, "y_true.npy")
    np.save(v7_npy, X7)
    np.save(v128_npy, X128)
    np.save(y_true_npy, y_true)

    schema_hash, schema_repr = schema_hash_from_config(data_cfg_path)
    schema_hash_txt = os.path.join(root, "schema_hash.txt")
    with open(schema_hash_txt, "w") as f:
        f.write(schema_hash)

    cfg = _load_yaml(data_cfg_path)
    meta = {
        "contract_id": cid,
        "git_commit": _git_commit_hash(),
        "n_test": int(len(y_true)),
        "label_pos_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "label_distribution": {
            "threat": int(np.sum(y_true == 1)),
            "benign": int(np.sum(y_true == 0)),
        },
        "schema_repr": schema_repr,
        "schema_hash": schema_hash,
        "splits": cfg.get("splits", {}),
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


def _string_array(values: list[str | None]) -> NDArray[np.str_]:
    safe = [v if v is not None else "" for v in values]
    max_len = max((len(v) for v in safe), default=1)
    return np.asarray(safe, dtype=f"<U{max_len}")


def _attr_array(events: list[Any], attr: str) -> NDArray[np.str_]:
    return _string_array([cast(str | None, getattr(e, attr, None)) for e in events])


def _label_attr_array(labels: list[Any], attr: str) -> NDArray[np.str_]:
    return _string_array([cast(str | None, getattr(lb, attr, None)) for lb in labels])


def _label_created_array(labels: list[Any], events: list[Any]) -> NDArray[np.int64]:
    if len(labels) != len(events):
        raise ValueError(
            "Alignment error: labels and events differ in length while exporting label_created_at"
        )
    created: list[int] = []
    for lb, ev in zip(labels, events):
        raw = getattr(lb, "label_created_at_epoch_s", None)
        if raw is None:
            created.append(_epoch_s(ev))
        else:
            created.append(int(raw))
    return np.asarray(created, dtype=np.int64)


def _episode_gap_seconds(cfg: dict[str, Any]) -> int:
    episodes_cfg = cast(dict[str, Any], cfg.get("episodes", {}))
    raw = episodes_cfg.get("inactivity_gap_seconds", 1800)
    gap = int(raw)
    return max(0, gap)


def _compute_episode_arrays(
    session_ids: NDArray[np.str_],
    timestamps: NDArray[np.int64],
    gap_seconds: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    n = int(timestamps.shape[0])
    episode_id = np.zeros((n,), dtype=np.int64)
    episode_start = np.zeros((n,), dtype=np.int64)
    episode_end = np.zeros((n,), dtype=np.int64)
    if n == 0:
        return episode_id, episode_start, episode_end

    current_episode = 0
    current_start_idx = 0
    prev_sid = str(session_ids[0])
    prev_ts = int(timestamps[0])

    for i in range(n):
        sid = str(session_ids[i])
        ts = int(timestamps[i])
        if i > 0:
            is_new = sid != prev_sid or ts < prev_ts or (ts - prev_ts) > gap_seconds
            if is_new:
                start_ts = int(timestamps[current_start_idx])
                end_ts = int(timestamps[i - 1])
                episode_start[current_start_idx:i] = start_ts
                episode_end[current_start_idx:i] = end_ts
                current_episode += 1
                current_start_idx = i
        episode_id[i] = current_episode
        prev_sid = sid
        prev_ts = ts

    start_ts = int(timestamps[current_start_idx])
    end_ts = int(timestamps[n - 1])
    episode_start[current_start_idx:n] = start_ts
    episode_end[current_start_idx:n] = end_ts
    return episode_id, episode_start, episode_end


def compute_episode_segments(
    session_ids: NDArray[np.str_],
    timestamps: NDArray[np.int64],
    gap_seconds: int = 1800,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    return _compute_episode_arrays(session_ids, timestamps, max(0, int(gap_seconds)))


def _assert_aligned(
    split: str,
    y: NDArray[np.int8],
    timestamps: NDArray[np.int64],
    session_ids: NDArray[np.str_],
    **extra_arrays: NDArray[Any],
) -> None:
    n = int(y.shape[0])
    if int(timestamps.shape[0]) != n:
        raise ValueError(
            f"Alignment error in split '{split}': y has {n} rows but timestamps has"
            f" {int(timestamps.shape[0])}"
        )
    if int(session_ids.shape[0]) != n:
        raise ValueError(
            f"Alignment error in split '{split}': y has {n} rows but session_id has"
            f" {int(session_ids.shape[0])}"
        )
    for name, arr in extra_arrays.items():
        if int(arr.shape[0]) != n:
            raise ValueError(
                f"Alignment error in split '{split}': y has {n} rows but {name} has"
                f" {int(arr.shape[0])}"
            )


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
    session_train = _attr_array(train_kept, "session_id")
    session_val = _attr_array(val_kept, "session_id")
    session_test = _attr_array(test_kept, "session_id")
    label_created_train = _label_created_array(train_labels_kept, train_kept)
    label_created_val = _label_created_array(val_labels_kept, val_kept)
    label_created_test = _label_created_array(test_labels_kept, test_kept)

    cfg = _load_yaml(data_cfg_path)
    gap_seconds = _episode_gap_seconds(cfg)
    episode_id_train, episode_start_train, episode_end_train = _compute_episode_arrays(
        session_train, ts_train, gap_seconds
    )
    episode_id_val, episode_start_val, episode_end_val = _compute_episode_arrays(
        session_val, ts_val, gap_seconds
    )
    episode_id_test, episode_start_test, episode_end_test = _compute_episode_arrays(
        session_test, ts_test, gap_seconds
    )

    _assert_aligned(
        "train",
        y_train,
        ts_train,
        session_train,
        label_created_at_epoch_s=label_created_train,
        episode_id=episode_id_train,
        episode_start_ts=episode_start_train,
        episode_end_ts=episode_end_train,
    )
    _assert_aligned(
        "val",
        y_val,
        ts_val,
        session_val,
        label_created_at_epoch_s=label_created_val,
        episode_id=episode_id_val,
        episode_start_ts=episode_start_val,
        episode_end_ts=episode_end_val,
    )
    _assert_aligned(
        "test",
        y_test,
        ts_test,
        session_test,
        label_created_at_epoch_s=label_created_test,
        episode_id=episode_id_test,
        episode_start_ts=episode_start_test,
        episode_end_ts=episode_end_test,
    )

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
    session_id_train_npy = os.path.join(out_dir, "session_id_train.npy")
    session_id_val_npy = os.path.join(out_dir, "session_id_val.npy")
    session_id_test_npy = os.path.join(out_dir, "session_id_test.npy")
    user_train_npy = os.path.join(out_dir, "user_train.npy")
    user_val_npy = os.path.join(out_dir, "user_val.npy")
    user_test_npy = os.path.join(out_dir, "user_test.npy")
    action_train_npy = os.path.join(out_dir, "action_train.npy")
    action_val_npy = os.path.join(out_dir, "action_val.npy")
    action_test_npy = os.path.join(out_dir, "action_test.npy")
    page_train_npy = os.path.join(out_dir, "page_train.npy")
    page_val_npy = os.path.join(out_dir, "page_val.npy")
    page_test_npy = os.path.join(out_dir, "page_test.npy")
    row_key_train_npy = os.path.join(out_dir, "row_key_train.npy")
    row_key_val_npy = os.path.join(out_dir, "row_key_val.npy")
    row_key_test_npy = os.path.join(out_dir, "row_key_test.npy")
    alarm_id_train_npy = os.path.join(out_dir, "alarm_id_train.npy")
    alarm_id_val_npy = os.path.join(out_dir, "alarm_id_val.npy")
    alarm_id_test_npy = os.path.join(out_dir, "alarm_id_test.npy")
    label_row_key_train_npy = os.path.join(out_dir, "label_row_key_train.npy")
    label_row_key_val_npy = os.path.join(out_dir, "label_row_key_val.npy")
    label_row_key_test_npy = os.path.join(out_dir, "label_row_key_test.npy")
    label_alarm_id_train_npy = os.path.join(out_dir, "label_alarm_id_train.npy")
    label_alarm_id_val_npy = os.path.join(out_dir, "label_alarm_id_val.npy")
    label_alarm_id_test_npy = os.path.join(out_dir, "label_alarm_id_test.npy")
    label_session_id_train_npy = os.path.join(out_dir, "label_session_id_train.npy")
    label_session_id_val_npy = os.path.join(out_dir, "label_session_id_val.npy")
    label_session_id_test_npy = os.path.join(out_dir, "label_session_id_test.npy")
    label_user_train_npy = os.path.join(out_dir, "label_user_train.npy")
    label_user_val_npy = os.path.join(out_dir, "label_user_val.npy")
    label_user_test_npy = os.path.join(out_dir, "label_user_test.npy")
    label_created_at_train_npy = os.path.join(out_dir, "label_created_at_epoch_s_train.npy")
    label_created_at_val_npy = os.path.join(out_dir, "label_created_at_epoch_s_val.npy")
    label_created_at_test_npy = os.path.join(out_dir, "label_created_at_epoch_s_test.npy")
    episode_id_train_npy = os.path.join(out_dir, "episode_id_train.npy")
    episode_id_val_npy = os.path.join(out_dir, "episode_id_val.npy")
    episode_id_test_npy = os.path.join(out_dir, "episode_id_test.npy")
    episode_start_ts_train_npy = os.path.join(out_dir, "episode_start_ts_train.npy")
    episode_start_ts_val_npy = os.path.join(out_dir, "episode_start_ts_val.npy")
    episode_start_ts_test_npy = os.path.join(out_dir, "episode_start_ts_test.npy")
    episode_end_ts_train_npy = os.path.join(out_dir, "episode_end_ts_train.npy")
    episode_end_ts_val_npy = os.path.join(out_dir, "episode_end_ts_val.npy")
    episode_end_ts_test_npy = os.path.join(out_dir, "episode_end_ts_test.npy")

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
    np.save(session_id_train_npy, session_train)
    np.save(session_id_val_npy, session_val)
    np.save(session_id_test_npy, session_test)
    np.save(user_train_npy, _attr_array(train_kept, "user"))
    np.save(user_val_npy, _attr_array(val_kept, "user"))
    np.save(user_test_npy, _attr_array(test_kept, "user"))
    np.save(action_train_npy, _attr_array(train_kept, "action"))
    np.save(action_val_npy, _attr_array(val_kept, "action"))
    np.save(action_test_npy, _attr_array(test_kept, "action"))
    np.save(page_train_npy, _attr_array(train_kept, "page"))
    np.save(page_val_npy, _attr_array(val_kept, "page"))
    np.save(page_test_npy, _attr_array(test_kept, "page"))
    np.save(row_key_train_npy, _attr_array(train_kept, "row_key"))
    np.save(row_key_val_npy, _attr_array(val_kept, "row_key"))
    np.save(row_key_test_npy, _attr_array(test_kept, "row_key"))
    np.save(alarm_id_train_npy, _attr_array(train_kept, "alarm_id"))
    np.save(alarm_id_val_npy, _attr_array(val_kept, "alarm_id"))
    np.save(alarm_id_test_npy, _attr_array(test_kept, "alarm_id"))
    np.save(label_row_key_train_npy, _label_attr_array(train_labels_kept, "row_key"))
    np.save(label_row_key_val_npy, _label_attr_array(val_labels_kept, "row_key"))
    np.save(label_row_key_test_npy, _label_attr_array(test_labels_kept, "row_key"))
    np.save(label_alarm_id_train_npy, _label_attr_array(train_labels_kept, "alarm_id"))
    np.save(label_alarm_id_val_npy, _label_attr_array(val_labels_kept, "alarm_id"))
    np.save(label_alarm_id_test_npy, _label_attr_array(test_labels_kept, "alarm_id"))
    np.save(label_session_id_train_npy, _attr_array(train_kept, "session_id"))
    np.save(label_session_id_val_npy, _attr_array(val_kept, "session_id"))
    np.save(label_session_id_test_npy, _attr_array(test_kept, "session_id"))
    np.save(label_user_train_npy, _attr_array(train_kept, "user"))
    np.save(label_user_val_npy, _attr_array(val_kept, "user"))
    np.save(label_user_test_npy, _attr_array(test_kept, "user"))
    np.save(label_created_at_train_npy, label_created_train)
    np.save(label_created_at_val_npy, label_created_val)
    np.save(label_created_at_test_npy, label_created_test)
    np.save(episode_id_train_npy, episode_id_train)
    np.save(episode_id_val_npy, episode_id_val)
    np.save(episode_id_test_npy, episode_id_test)
    np.save(episode_start_ts_train_npy, episode_start_train)
    np.save(episode_start_ts_val_npy, episode_start_val)
    np.save(episode_start_ts_test_npy, episode_start_test)
    np.save(episode_end_ts_train_npy, episode_end_train)
    np.save(episode_end_ts_val_npy, episode_end_val)
    np.save(episode_end_ts_test_npy, episode_end_test)

    # Backward-compatible test names
    v7_npy = os.path.join(out_dir, "features_v7.npy")
    v128_npy = os.path.join(out_dir, "features_v128.npy")
    np.save(v7_npy, X7_test)
    np.save(v128_npy, X128_test)

    schema_hash, schema_repr = schema_hash_from_config(data_cfg_path)
    schema_hash_txt = os.path.join(out_dir, "schema_hash.txt")
    with open(schema_hash_txt, "w") as f:
        f.write(schema_hash)

    dataset_cfg = cast(dict[str, Any], cfg.get("dataset", {}))
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
    meta_json = os.path.join(root, "meta.json")
        "episodes": {
            "source": os.path.basename(str(dataset_cfg.get("path", "unknown"))),
            "keys": ["session_id", "user"],
            "inactivity_gap_seconds": gap_seconds,
            "files": [
                "session_id_train.npy",
                "session_id_val.npy",
                "session_id_test.npy",
                "user_train.npy",
                "user_val.npy",
                "user_test.npy",
                "episode_id_train.npy",
                "episode_id_val.npy",
                "episode_id_test.npy",
                "episode_start_ts_train.npy",
                "episode_start_ts_val.npy",
                "episode_start_ts_test.npy",
                "episode_end_ts_train.npy",
                "episode_end_ts_val.npy",
                "episode_end_ts_test.npy",
            ],
        },
        "label_availability": {
            "timestamp_field": "label_created_at_epoch_s",
            "files": [
                "label_created_at_epoch_s_train.npy",
                "label_created_at_epoch_s_val.npy",
                "label_created_at_epoch_s_test.npy",
            ],
        },
        "join_fields": {
            "event_keys": ["alarm_id", "row_key", "session_id", "user", "timestamps_epoch_s"],
            "label_keys": ["alarm_id", "row_key", "session_id", "user", "label_created_at_epoch_s"],
        },
    }
    meta_json = os.path.join(out_dir, "meta.json")
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    return ContractPaths(
        root=root,
        events_parquet=events_parquet,
        labels_parquet=labels_parquet,
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
        session_id_train_npy=session_id_train_npy,
        session_id_val_npy=session_id_val_npy,
        session_id_test_npy=session_id_test_npy,
        user_train_npy=user_train_npy,
        user_val_npy=user_val_npy,
        user_test_npy=user_test_npy,
        action_train_npy=action_train_npy,
        action_val_npy=action_val_npy,
        action_test_npy=action_test_npy,
        page_train_npy=page_train_npy,
        page_val_npy=page_val_npy,
        page_test_npy=page_test_npy,
        row_key_train_npy=row_key_train_npy,
        row_key_val_npy=row_key_val_npy,
        row_key_test_npy=row_key_test_npy,
        alarm_id_train_npy=alarm_id_train_npy,
        alarm_id_val_npy=alarm_id_val_npy,
        alarm_id_test_npy=alarm_id_test_npy,
        label_row_key_train_npy=label_row_key_train_npy,
        label_row_key_val_npy=label_row_key_val_npy,
        label_row_key_test_npy=label_row_key_test_npy,
        label_alarm_id_train_npy=label_alarm_id_train_npy,
        label_alarm_id_val_npy=label_alarm_id_val_npy,
        label_alarm_id_test_npy=label_alarm_id_test_npy,
        label_session_id_train_npy=label_session_id_train_npy,
        label_session_id_val_npy=label_session_id_val_npy,
        label_session_id_test_npy=label_session_id_test_npy,
        label_user_train_npy=label_user_train_npy,
        label_user_val_npy=label_user_val_npy,
        label_user_test_npy=label_user_test_npy,
        label_created_at_train_npy=label_created_at_train_npy,
        label_created_at_val_npy=label_created_at_val_npy,
        label_created_at_test_npy=label_created_at_test_npy,
        episode_id_train_npy=episode_id_train_npy,
        episode_id_val_npy=episode_id_val_npy,
        episode_id_test_npy=episode_id_test_npy,
        episode_start_ts_train_npy=episode_start_ts_train_npy,
        episode_start_ts_val_npy=episode_start_ts_val_npy,
        episode_start_ts_test_npy=episode_start_ts_test_npy,
        episode_end_ts_train_npy=episode_end_ts_train_npy,
        episode_end_ts_val_npy=episode_end_ts_val_npy,
        episode_end_ts_test_npy=episode_end_ts_test_npy,
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
