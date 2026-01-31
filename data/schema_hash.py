from __future__ import annotations

import json
import hashlib
from typing import Any

import yaml


FEATURE_SCHEMA_VERSION = "v0.2"


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_feature_schema_repr(cfg: dict[str, Any]) -> dict[str, Any]:
    features_cfg = cfg.get("features", {})
    v7_cfg = features_cfg.get("v7", {})
    v128_cfg = features_cfg.get("v128", {})

    return {
        "version": FEATURE_SCHEMA_VERSION,
        "v7_fields": list(v7_cfg.get("fields", [])),
        "v7_dims": int(v7_cfg.get("dims", 7)) if v7_cfg.get("dims") else 7,
        "v128_dims": int(v128_cfg.get("dims", 128)),
        "v128_history_window_minutes": v128_cfg.get("history_window_minutes", 60),
        "dtype": "float32",
        "label_dtype": "int8",
    }


def compute_schema_hash(schema_repr: dict[str, Any]) -> str:
    payload = json.dumps(schema_repr, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def schema_hash_from_config(data_cfg_path: str) -> tuple[str, dict[str, Any]]:
    cfg = _load_yaml(data_cfg_path)
    schema_repr = build_feature_schema_repr(cfg)
    schema_hash = compute_schema_hash(schema_repr)
    return schema_hash, schema_repr
