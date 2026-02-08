from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import yaml
from numpy.typing import NDArray

JoinMethod = Literal["alarm_id", "row_key", "time_window", "unmatched"]
CollisionRule = Literal["first", "most_recent"]


@dataclass(frozen=True)
class JoinConfig:
    split: str
    priorities: tuple[str, ...]
    time_window_enabled: bool
    time_window_seconds: int
    collision_rule: CollisionRule


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return cast(dict[str, Any], raw)


def _load_str(path: str, fallback_len: int = 0) -> NDArray[np.str_]:
    if not os.path.exists(path):
        return np.asarray([""] * fallback_len, dtype=np.str_)
    return np.load(path, allow_pickle=False).astype(np.str_)


def _load_i64(path: str, fallback_len: int = 0) -> NDArray[np.int64]:
    if not os.path.exists(path):
        return np.zeros((fallback_len,), dtype=np.int64)
    return np.load(path, allow_pickle=False).astype(np.int64)


def _parse_join_config(path: str) -> JoinConfig:
    payload = _load_yaml(path)
    cfg = cast(dict[str, Any], payload.get("join", {}))
    priorities_raw = cfg.get("priority", ["alarm_id", "row_key", "time_window"])
    if not isinstance(priorities_raw, list):
        raise ValueError("join.priority must be a list")
    rule_raw = str(cfg.get("collision_rule", "first"))
    if rule_raw not in {"first", "most_recent"}:
        raise ValueError(f"Unsupported collision_rule: {rule_raw}")
    return JoinConfig(
        split=str(cfg.get("split", "test")),
        priorities=tuple(str(v) for v in priorities_raw),
        time_window_enabled=bool(cfg.get("time_window_fallback", {}).get("enabled", False)),
        time_window_seconds=int(cfg.get("time_window_fallback", {}).get("window_seconds", 3600)),
        collision_rule=cast(CollisionRule, rule_raw),
    )


def _build_index(values: NDArray[np.str_]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, raw in enumerate(values):
        key = str(raw).strip()
        if key:
            out.setdefault(key, []).append(idx)
    return out


def _resolve_label_idx(
    indices: list[int],
    created_at: NDArray[np.int64],
    collision_rule: CollisionRule,
) -> tuple[int, bool]:
    if not indices:
        return -1, False
    if len(indices) == 1:
        return indices[0], False
    if collision_rule == "most_recent":
        best = max(indices, key=lambda i: int(created_at[i]))
        return best, True
    return min(indices), True


def _attempt_time_window_join(
    event_idx: int,
    *,
    event_ts: NDArray[np.int64],
    event_session: NDArray[np.str_],
    event_user: NDArray[np.str_],
    label_created_at: NDArray[np.int64],
    label_session: NDArray[np.str_],
    label_user: NDArray[np.str_],
    window_seconds: int,
    collision_rule: CollisionRule,
) -> tuple[int, bool]:
    sid = str(event_session[event_idx]).strip()
    usr = str(event_user[event_idx]).strip()
    ts = int(event_ts[event_idx])
    cands: list[int] = []
    for i in range(label_created_at.shape[0]):
        if str(label_session[i]).strip() != sid:
            continue
        if str(label_user[i]).strip() != usr:
            continue
        if abs(int(label_created_at[i]) - ts) <= window_seconds:
            cands.append(i)
    if not cands:
        return -1, False
    if collision_rule == "most_recent":
        cands.sort(key=lambda i: int(label_created_at[i]), reverse=True)
    else:
        cands.sort()
    return cands[0], len(cands) > 1


def build_join_map(
    *,
    contract_dir: str,
    join_cfg_path: str,
    out_dir: str,
) -> dict[str, Any]:
    cfg = _parse_join_config(join_cfg_path)
    split = cfg.split

    event_row_key = _load_str(os.path.join(contract_dir, f"row_key_{split}.npy"))
    n_events = int(event_row_key.shape[0])
    event_alarm = _load_str(
        os.path.join(contract_dir, f"alarm_id_{split}.npy"), fallback_len=n_events
    )
    event_session = _load_str(
        os.path.join(contract_dir, f"session_id_{split}.npy"), fallback_len=n_events
    )
    event_user = _load_str(os.path.join(contract_dir, f"user_{split}.npy"), fallback_len=n_events)
    event_ts = _load_i64(
        os.path.join(contract_dir, f"timestamps_epoch_s_{split}.npy"), fallback_len=n_events
    )

    label_row_key = _load_str(
        os.path.join(contract_dir, f"label_row_key_{split}.npy"), fallback_len=n_events
    )
    label_alarm = _load_str(
        os.path.join(contract_dir, f"label_alarm_id_{split}.npy"), fallback_len=n_events
    )
    label_created_at = _load_i64(
        os.path.join(contract_dir, f"label_created_at_epoch_s_{split}.npy"), fallback_len=n_events
    )
    label_session = _load_str(
        os.path.join(contract_dir, f"label_session_id_{split}.npy"), fallback_len=n_events
    )
    label_user = _load_str(
        os.path.join(contract_dir, f"label_user_{split}.npy"), fallback_len=n_events
    )

    alarm_map = _build_index(label_alarm)
    row_map = _build_index(label_row_key)

    event_index = np.arange(n_events, dtype=np.int64)
    label_index = np.full((n_events,), -1, dtype=np.int64)
    method = np.asarray(["unmatched"] * n_events, dtype=np.str_)

    ambiguous = 0
    matched = 0
    method_counts: dict[str, int] = {"alarm_id": 0, "row_key": 0, "time_window": 0, "unmatched": 0}

    for i in range(n_events):
        chosen_idx = -1
        chosen_method: JoinMethod = "unmatched"
        collision = False

        for priority in cfg.priorities:
            if priority == "alarm_id":
                key = str(event_alarm[i]).strip()
                if key:
                    chosen_idx, collision = _resolve_label_idx(
                        alarm_map.get(key, []), label_created_at, cfg.collision_rule
                    )
                    if chosen_idx >= 0:
                        chosen_method = "alarm_id"
                        break
            elif priority == "row_key":
                key = str(event_row_key[i]).strip()
                if key:
                    chosen_idx, collision = _resolve_label_idx(
                        row_map.get(key, []), label_created_at, cfg.collision_rule
                    )
                    if chosen_idx >= 0:
                        chosen_method = "row_key"
                        break
            elif priority == "time_window" and cfg.time_window_enabled:
                chosen_idx, collision = _attempt_time_window_join(
                    i,
                    event_ts=event_ts,
                    event_session=event_session,
                    event_user=event_user,
                    label_created_at=label_created_at,
                    label_session=label_session,
                    label_user=label_user,
                    window_seconds=cfg.time_window_seconds,
                    collision_rule=cfg.collision_rule,
                )
                if chosen_idx >= 0:
                    chosen_method = "time_window"
                    break

        if chosen_idx >= 0:
            matched += 1
            label_index[i] = chosen_idx
            method[i] = chosen_method
        else:
            method[i] = "unmatched"
        if collision:
            ambiguous += 1

    for name in method:
        method_counts[str(name)] = method_counts.get(str(name), 0) + 1

    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, "join_map.npz")
    np.savez(
        map_path,
        event_index=event_index,
        label_index=label_index,
        method=method.astype(np.str_),
        event_row_key=event_row_key,
        label_row_key=label_row_key,
    )

    meta = {
        "split": split,
        "join_priority": list(cfg.priorities),
        "collision_rule": cfg.collision_rule,
        "time_window_enabled": cfg.time_window_enabled,
        "time_window_seconds": cfg.time_window_seconds,
        "candidate_count": n_events,
        "matched_count": matched,
        "matched_rate": float(matched / n_events) if n_events else 0.0,
        "unmatched_count": int(n_events - matched),
        "ambiguity_count": ambiguous,
        "method_counts": method_counts,
    }
    meta_path = os.path.join(out_dir, "join_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {"join_map": map_path, "join_meta": meta_path, "meta": meta}
