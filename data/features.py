from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
import numpy as np
from .schemas import RawEvent


def validate_feature_vector(vec: np.ndarray | Sequence[float], expected_dim: int) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.dtype.kind in ("O", "U", "S"):
        raise ValueError(f"Feature vector must be numeric, got dtype={arr.dtype}")
    if arr.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(f"Feature vector must be numeric, got dtype={arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"Expected dim={expected_dim}, got shape={arr.shape}")
    if arr.shape[0] != expected_dim:
        raise ValueError(f"Expected dim={expected_dim}, got dim={arr.shape[0]}")

    arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError("Feature vector contains NaN or Inf values")
    return arr


def _safe_float(x: float | None) -> float:
    return float(x) if x is not None else 0.0


def _safe_int(x: int | None) -> int:
    return int(x) if x is not None else 0


def extract_features_v7(e: RawEvent) -> np.ndarray:
    # duration, bytes_sent, bytes_received, port, protocol_encoded, hour, dow
    protocol = (e.protocol or "").lower()
    protocol_encoded = 1.0 if protocol == "tcp" else 0.0

    hour = float(e.ts.hour)
    dow = float(e.ts.weekday())
    vec = np.array(
        [
            _safe_float(e.duration),
            _safe_float(e.bytes_sent),
            _safe_float(e.bytes_received),
            float(_safe_int(e.port)),
            protocol_encoded,
            hour,
            dow,
        ],
        dtype=np.float32,
    )
    return validate_feature_vector(vec, expected_dim=7)


@dataclass
class HistoryContext:
    # placeholder - extend later for sliding windows, entropy, recency, etc.
    now: datetime


def extract_features_v128(e: RawEvent, ctx: HistoryContext) -> np.ndarray:
    # Minimal deterministic 128D: fill with derived stats + padding.
    v7 = extract_features_v7(e).astype(np.float32)
    out = np.zeros((128,), dtype=np.float32)
    out[:7] = v7
    # cyclic encoding for hour/dow as example
    hour = e.ts.hour / 24.0
    dow = e.ts.weekday() / 7.0
    out[7] = np.sin(2 * np.pi * hour)
    out[8] = np.cos(2 * np.pi * hour)
    out[9] = np.sin(2 * np.pi * dow)
    out[10] = np.cos(2 * np.pi * dow)
    # (rest are zeros in v1; later you'll populate 4x32 blocks)
    return validate_feature_vector(out, expected_dim=128)
