import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from data.contract_export import export_frozen_contract
from data.schema_hash import schema_hash_from_config


def _write_cfg(tmp_path: Path) -> Path:
    cfg = {
        "dataset": {"source": "toy", "n": 200},
        "splits": {
            "mode": "time",
            "train": {"start": "2025-01-01", "end": "2025-08-31"},
            "val": {"start": "2025-09-01", "end": "2025-10-31"},
            "test": {"start": "2025-11-01", "end": "2025-12-31"},
        },
        "features": {
            "v7": {
                "enabled": True,
                "fields": [
                    "duration",
                    "bytes_sent",
                    "bytes_received",
                    "port",
                    "protocol",
                    "hour_of_day",
                    "day_of_week",
                ],
            },
            "v128": {"enabled": True, "history_window_minutes": 60, "dims": 128},
        },
    }
    path = tmp_path / "data.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_contract_export(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    cfg_path = _write_cfg(tmp_path)
    paths = export_frozen_contract(str(cfg_path), out_root=str(tmp_path))

    assert Path(paths.events_parquet).exists()
    assert Path(paths.labels_parquet).exists()
    assert Path(paths.v7_npy).exists()
    assert Path(paths.v128_npy).exists()
    assert Path(paths.y_true_npy).exists()
    assert Path(paths.meta_json).exists()
    assert Path(paths.schema_hash_txt).exists()

    x7 = np.load(paths.v7_npy)
    x128 = np.load(paths.v128_npy)
    y_true = np.load(paths.y_true_npy)

    assert x7.shape[1] == 7
    assert x128.shape[1] == 128
    assert x7.shape[0] == x128.shape[0] == y_true.shape[0]

    schema_hash, _ = schema_hash_from_config(str(cfg_path))
    with open(paths.meta_json, "r") as f:
        meta = json.load(f)
    assert meta["schema_hash"] == schema_hash
