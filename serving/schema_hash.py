from __future__ import annotations

import json
import os
from typing import Optional

from data.schema_hash import schema_hash_from_config


def load_expected_schema_hash(contract_dir: str) -> Optional[str]:
    meta_path = os.path.join(contract_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("schema_hash")

    txt_path = os.path.join(contract_dir, "schema_hash.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            return f.read().strip()

    return None


def current_schema_hash(data_cfg_path: str) -> str:
    schema_hash, _ = schema_hash_from_config(data_cfg_path)
    return schema_hash
