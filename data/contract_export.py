from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
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
        "normalization": {
            "iforest_scaler": "standard",
            "notes": "Scaler statistics stored in IF bundle",
        },
    }
    meta_json = os.path.join(root, "meta.json")
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    return ContractPaths(
        root=root,
        events_parquet=events_parquet,
        labels_parquet=labels_parquet,
        v7_npy=v7_npy,
        v128_npy=v128_npy,
        y_true_npy=y_true_npy,
        meta_json=meta_json,
        schema_hash_txt=schema_hash_txt,
    )
