from __future__ import annotations

import json
import os
import re
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

LabelDelayPolicy = Literal["treat_as_unknown", "treat_as_benign"]

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdSMHD])\s*$")


def parse_duration_to_seconds(raw: str) -> int:
    m = _DURATION_RE.match(raw)
    if m is None:
        raise ValueError(f"Invalid duration value: {raw!r}. Use forms like '30m', '12h', '7d'.")
    value = int(m.group(1))
    unit = m.group(2).lower()
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return value * multiplier


def apply_label_availability(
    y_true: NDArray[np.int_],
    event_timestamps_epoch_s: NDArray[np.int64],
    label_created_at_epoch_s: NDArray[np.int64],
    *,
    delay: str | int,
    policy: LabelDelayPolicy,
    eval_time_epoch_s: int | None = None,
) -> tuple[NDArray[np.int8], NDArray[np.bool_], dict[str, int | str]]:
    if y_true.shape[0] != event_timestamps_epoch_s.shape[0]:
        raise ValueError("event_timestamps_epoch_s length mismatch")
    if y_true.shape[0] != label_created_at_epoch_s.shape[0]:
        raise ValueError("label_created_at_epoch_s length mismatch")

    delay_seconds = parse_duration_to_seconds(delay) if isinstance(delay, str) else int(delay)
    if delay_seconds < 0:
        raise ValueError("delay must be non-negative")

    eval_time = (
        int(eval_time_epoch_s)
        if eval_time_epoch_s is not None
        else int(np.max(event_timestamps_epoch_s, initial=0))
    )
    available = (label_created_at_epoch_s + np.int64(delay_seconds)) <= np.int64(eval_time)

    y_out = y_true.astype(np.int8, copy=True)
    unavailable = np.logical_not(available)
    if policy == "treat_as_unknown":
        y_out[unavailable] = np.int8(-1)
    elif policy == "treat_as_benign":
        y_out[unavailable] = np.int8(0)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported label delay policy: {policy}")

    unavailable_pos = int(np.sum((y_true == 1) & unavailable))
    meta: dict[str, int | str] = {
        "eval_time_epoch_s": eval_time,
        "delay_seconds": delay_seconds,
        "policy": policy,
        "unavailable_count": int(np.sum(unavailable)),
        "unavailable_pos": unavailable_pos,
    }
    return y_out, cast(NDArray[np.bool_], available.astype(np.bool_)), meta


def write_label_availability_artifacts(
    out_dir: str,
    split: str,
    available_mask: NDArray[np.bool_],
    y_available: NDArray[np.int8],
    meta: dict[str, int | str],
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    mask_path = os.path.join(out_dir, f"labels_available_mask_{split}.npy")
    y_path = os.path.join(out_dir, f"y_{split}_available.npy")
    meta_path = os.path.join(out_dir, "label_delay_meta.json")
    np.save(mask_path, available_mask.astype(np.bool_))
    np.save(y_path, y_available.astype(np.int8))
    with open(meta_path, "w") as f:
        json.dump({"split": split, **meta}, f, indent=2)
    return {
        "mask_path": mask_path,
        "y_available_path": y_path,
        "meta_path": meta_path,
    }
