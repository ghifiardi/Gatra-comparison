from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from data.contract_cache import (
    _is_valid_contract_dir,
    compute_cache_key,
    copy_contract,
    get_or_populate_cache,
)


def _create_minimal_contract(out_dir: str, *, n_train: int = 10, n_val: int = 5) -> None:
    """Write the minimal set of files that _is_valid_contract_dir checks."""
    os.makedirs(out_dir, exist_ok=True)
    n_test = 5
    np.save(
        os.path.join(out_dir, "features_v7_train.npy"), np.zeros((n_train, 7), dtype=np.float32)
    )
    np.save(
        os.path.join(out_dir, "features_v128_train.npy"), np.zeros((n_train, 128), dtype=np.float32)
    )
    np.save(os.path.join(out_dir, "y_train.npy"), np.zeros((n_train,), dtype=np.int8))
    np.save(os.path.join(out_dir, "features_v7_val.npy"), np.zeros((n_val, 7), dtype=np.float32))
    np.save(
        os.path.join(out_dir, "features_v128_val.npy"), np.zeros((n_val, 128), dtype=np.float32)
    )
    np.save(os.path.join(out_dir, "y_val.npy"), np.zeros((n_val,), dtype=np.int8))
    np.save(os.path.join(out_dir, "features_v7_test.npy"), np.zeros((n_test, 7), dtype=np.float32))
    np.save(
        os.path.join(out_dir, "features_v128_test.npy"), np.zeros((n_test, 128), dtype=np.float32)
    )
    np.save(os.path.join(out_dir, "y_true.npy"), np.zeros((n_test,), dtype=np.int8))
    with open(os.path.join(out_dir, "schema_hash.txt"), "w") as f:
        f.write("abc123")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"contract_id": "test", "counts": {"train": n_train}}, f)


def _write_data_yaml(path: Path, source: str = "toy") -> None:
    import yaml

    payload = {"dataset": {"source": source, "n": 100}, "features": {"v7": {"dims": 7}}}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_compute_cache_key_deterministic(tmp_path: Path) -> None:
    cfg = tmp_path / "data.yaml"
    _write_data_yaml(cfg)
    k1 = compute_cache_key(str(cfg))
    k2 = compute_cache_key(str(cfg))
    assert k1 == k2
    assert len(k1) == 16


def test_compute_cache_key_changes_with_config(tmp_path: Path) -> None:
    cfg1 = tmp_path / "data1.yaml"
    cfg2 = tmp_path / "data2.yaml"
    _write_data_yaml(cfg1, source="toy")
    _write_data_yaml(cfg2, source="csv")
    assert compute_cache_key(str(cfg1)) != compute_cache_key(str(cfg2))


def test_is_valid_contract_dir(tmp_path: Path) -> None:
    d = str(tmp_path / "contract")
    assert not _is_valid_contract_dir(d)
    _create_minimal_contract(d)
    assert _is_valid_contract_dir(d)


def test_get_or_populate_cache_miss_then_hit(tmp_path: Path) -> None:
    cfg_path = tmp_path / "data.yaml"
    _write_data_yaml(cfg_path)
    cache_root = str(tmp_path / "cache")
    call_count = [0]

    def fake_export(data_cfg_path: str, out_dir: str) -> None:
        call_count[0] += 1
        _create_minimal_contract(out_dir)

    # First call: cache miss
    dir1, hit1 = get_or_populate_cache(str(cfg_path), cache_root, fake_export)
    assert not hit1
    assert call_count[0] == 1
    assert _is_valid_contract_dir(dir1)

    # Second call: cache hit
    dir2, hit2 = get_or_populate_cache(str(cfg_path), cache_root, fake_export)
    assert hit2
    assert call_count[0] == 1  # export NOT called again
    assert dir1 == dir2


def test_copy_contract(tmp_path: Path) -> None:
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    _create_minimal_contract(src)
    copy_contract(src, dst)
    assert _is_valid_contract_dir(dst)
    # Verify content matches
    for name in os.listdir(src):
        assert os.path.exists(os.path.join(dst, name))


def test_cache_dir_under_expected_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "data.yaml"
    _write_data_yaml(cfg_path)
    cache_root = str(tmp_path / "contracts_cache")

    def fake_export(data_cfg_path: str, out_dir: str) -> None:
        _create_minimal_contract(out_dir)

    dir1, _ = get_or_populate_cache(str(cfg_path), cache_root, fake_export)
    key = compute_cache_key(str(cfg_path))
    expected = os.path.join(cache_root, key, "contract")
    assert dir1 == expected
    assert os.path.isdir(expected)
