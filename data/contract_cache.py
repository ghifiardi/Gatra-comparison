from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, cast

import yaml


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return cast(dict[str, Any], data)


def compute_cache_key(data_cfg_path: str) -> str:
    """Deterministic cache key from the data config content."""
    cfg = _load_yaml(data_cfg_path)
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _is_valid_contract_dir(path: str) -> bool:
    """Check that essential contract files exist."""
    required = [
        "features_v7_train.npy",
        "features_v128_train.npy",
        "y_train.npy",
        "features_v7_val.npy",
        "features_v128_val.npy",
        "y_val.npy",
        "features_v7_test.npy",
        "features_v128_test.npy",
        "y_true.npy",
        "schema_hash.txt",
        "meta.json",
    ]
    return all(os.path.exists(os.path.join(path, f)) for f in required)


def get_or_populate_cache(
    data_cfg_path: str,
    cache_root: str,
    export_fn: Any,
) -> tuple[str, bool]:
    """Return (cached_contract_dir, cache_hit).

    If a valid cache entry exists, return it. Otherwise call export_fn to
    populate a fresh cache entry.

    Parameters
    ----------
    data_cfg_path : str
        Path to the snapshotted data config YAML.
    cache_root : str
        Root directory for the contract cache (e.g. ``reports/contracts_cache``).
    export_fn : callable
        ``export_fn(data_cfg_path, out_dir)`` — writes contract artifacts into *out_dir*.

    Returns
    -------
    tuple[str, bool]
        ``(contract_dir, cache_hit)``
    """
    key = compute_cache_key(data_cfg_path)
    cache_dir = os.path.join(cache_root, key, "contract")

    if _is_valid_contract_dir(cache_dir):
        return cache_dir, True

    os.makedirs(cache_dir, exist_ok=True)
    export_fn(data_cfg_path, cache_dir)
    return cache_dir, False


def copy_contract(src: str, dst: str) -> None:
    """Copy all files from *src* contract directory into *dst*."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isfile(s):
            shutil.copy2(s, d)
