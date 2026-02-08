from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .objectives import ObjectiveSpec, compute_reward_matrix

_REALDATA_TYPES = {"time_to_triage_seconds", "detection_coverage"}
_TIME_TO_TRIAGE = "time_to_triage_seconds"
_DETECTION_COVERAGE = "detection_coverage"


@dataclass(frozen=True)
class RealDataRewardResult:
    reward_matrix: NDArray[np.float32]
    objective_signals: NDArray[np.float32] | None
    objective_names: list[str]
    source: str
    stats: dict[str, float]
    artifacts: dict[str, str]


def validate_alignment(expected_len: int, **arrays: NDArray[Any]) -> None:
    for name, arr in arrays.items():
        if int(arr.shape[0]) != expected_len:
            raise ValueError(
                f"Alignment error: expected {expected_len} rows but '{name}' has {int(arr.shape[0])}"
            )


def _normalize(values: NDArray[np.float32], strategy: str) -> NDArray[np.float32]:
    if strategy == "none":
        return values.astype(np.float32)
    if strategy == "zscore":
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std <= 0.0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - mean) / std).astype(np.float32)
    if strategy == "minmax":
        lo = float(np.min(values))
        hi = float(np.max(values))
        den = hi - lo
        if den <= 0.0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - lo) / den).astype(np.float32)
    raise ValueError(f"Unsupported normalization strategy: {strategy}")


def compute_time_to_triage_seconds(
    session_ids: NDArray[np.str_],
    timestamps_epoch_s: NDArray[np.int64],
) -> NDArray[np.float32]:
    validate_alignment(
        expected_len=int(session_ids.shape[0]),
        timestamps_epoch_s=timestamps_epoch_s,
    )
    min_by_session: dict[str, int] = {}
    max_by_session: dict[str, int] = {}
    for i in range(session_ids.shape[0]):
        sid = str(session_ids[i]) or "__missing_session__"
        ts = int(timestamps_epoch_s[i])
        if sid not in min_by_session:
            min_by_session[sid] = ts
            max_by_session[sid] = ts
        else:
            if ts < min_by_session[sid]:
                min_by_session[sid] = ts
            if ts > max_by_session[sid]:
                max_by_session[sid] = ts
    out = np.zeros((session_ids.shape[0],), dtype=np.float32)
    for i in range(session_ids.shape[0]):
        sid = str(session_ids[i]) or "__missing_session__"
        out[i] = float(max_by_session[sid] - min_by_session[sid])
    return out


def compute_coverage_increments(
    session_ids: NDArray[np.str_],
    pages: NDArray[np.str_],
    actions: NDArray[np.str_],
) -> NDArray[np.float32]:
    validate_alignment(
        expected_len=int(session_ids.shape[0]),
        pages=pages,
        actions=actions,
    )
    seen: dict[str, set[tuple[str, str]]] = {}
    out = np.zeros((session_ids.shape[0],), dtype=np.float32)
    for i in range(session_ids.shape[0]):
        sid = str(session_ids[i]) or "__missing_session__"
        pair = (str(pages[i]) or "__missing_page__", str(actions[i]) or "__missing_action__")
        bucket = seen.setdefault(sid, set())
        if pair not in bucket:
            bucket.add(pair)
            out[i] = 1.0
    return out


def _load_str_array(contract_dir: str, stem: str, split: str) -> NDArray[np.str_]:
    path = os.path.join(contract_dir, f"{stem}_{split}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    loaded = np.load(path, allow_pickle=False)
    return cast(NDArray[np.str_], loaded.astype(np.str_))


def _load_i64_array(contract_dir: str, stem: str, split: str) -> NDArray[np.int64]:
    path = os.path.join(contract_dir, f"{stem}_{split}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    loaded = np.load(path, allow_pickle=False)
    return cast(NDArray[np.int64], loaded.astype(np.int64))


def _load_meta_source(contract_dir: str) -> str:
    path = os.path.join(contract_dir, "meta.json")
    if not os.path.exists(path):
        return "unknown"
    with open(path, "r") as f:
        payload = cast(dict[str, Any], json.load(f))
    episodes = cast(dict[str, Any], payload.get("episodes", {}))
    return str(episodes.get("source", "unknown"))


def _upsert_meta(
    contract_dir: str,
    split: str,
    normalization: str,
    objective_names: list[str],
    stats: dict[str, float],
) -> str:
    path = os.path.join(contract_dir, "objectives_meta.json")
    payload: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            payload = cast(dict[str, Any], loaded)

    splits = cast(dict[str, Any], payload.setdefault("splits", {}))
    splits[split] = {
        "file": f"objectives_{split}.npz",
        "objective_names": objective_names,
        "normalization": normalization,
        "stats": stats,
    }
    payload["definitions"] = {
        _TIME_TO_TRIAGE: "Per-session max(ts)-min(ts); reward is negative when alerting.",
        _DETECTION_COVERAGE: "Session-level novelty of (page,action); reward is positive when alerting.",
    }
    payload["source"] = _load_meta_source(contract_dir)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _needs_realdata(objectives: Sequence[ObjectiveSpec]) -> bool:
    return any(spec.type in _REALDATA_TYPES for spec in objectives)


def compute_reward_matrix_realdata_aware(
    y_true: NDArray[np.int_],
    actions: NDArray[np.int_],
    objectives: Sequence[ObjectiveSpec],
    *,
    contract_dir: str | None,
    split: str,
    normalization: str = "minmax",
) -> RealDataRewardResult:
    rewards = compute_reward_matrix(y_true, actions, objectives)
    names = [spec.name for spec in objectives]

    if contract_dir is None or not _needs_realdata(objectives):
        return RealDataRewardResult(
            reward_matrix=rewards,
            objective_signals=None,
            objective_names=names,
            source="fallback_synthetic",
            stats={},
            artifacts={},
        )

    try:
        session_ids = _load_str_array(contract_dir, "session_id", split)
        timestamps = _load_i64_array(contract_dir, "timestamps_epoch_s", split)
        pages = _load_str_array(contract_dir, "page", split)
        actions_raw = _load_str_array(contract_dir, "action", split)
    except FileNotFoundError:
        return RealDataRewardResult(
            reward_matrix=rewards,
            objective_signals=None,
            objective_names=names,
            source="fallback_synthetic",
            stats={},
            artifacts={},
        )

    expected_len = int(y_true.shape[0])
    validate_alignment(
        expected_len=expected_len,
        session_id=session_ids,
        timestamps_epoch_s=timestamps,
        page=pages,
        action=actions_raw,
        actions=actions,
    )

    ttt_raw = compute_time_to_triage_seconds(session_ids=session_ids, timestamps_epoch_s=timestamps)
    cov_raw = compute_coverage_increments(session_ids=session_ids, pages=pages, actions=actions_raw)
    ttt_norm = _normalize(ttt_raw, normalization)
    cov_norm = _normalize(cov_raw, normalization)

    signals = np.zeros((expected_len, len(objectives)), dtype=np.float32)
    actions_f = actions.astype(np.float32)
    for i, spec in enumerate(objectives):
        if spec.type == _TIME_TO_TRIAGE:
            signals[:, i] = ttt_norm
            rewards[:, i] = -ttt_norm * actions_f
        elif spec.type == _DETECTION_COVERAGE:
            signals[:, i] = cov_norm
            rewards[:, i] = cov_norm * actions_f

    npz_path = os.path.join(contract_dir, f"objectives_{split}.npz")
    np.savez(
        npz_path,
        objective_signals=signals,
        objective_names=np.asarray(names, dtype=np.str_),
        time_to_triage_seconds=ttt_raw.astype(np.float32),
        coverage_increment=cov_raw.astype(np.float32),
    )
    stats = {
        "ttt_median_seconds": float(np.median(ttt_raw)) if ttt_raw.size else 0.0,
        "ttt_p90_seconds": float(np.percentile(ttt_raw, 90)) if ttt_raw.size else 0.0,
        "coverage_novelty_rate": float(np.mean(cov_raw)) if cov_raw.size else 0.0,
        "coverage_per_1k_events": float(1000.0 * np.mean(cov_raw)) if cov_raw.size else 0.0,
    }
    meta_path = _upsert_meta(
        contract_dir=contract_dir,
        split=split,
        normalization=normalization,
        objective_names=names,
        stats=stats,
    )
    return RealDataRewardResult(
        reward_matrix=rewards.astype(np.float32),
        objective_signals=signals,
        objective_names=names,
        source="level1_realdata",
        stats=stats,
        artifacts={
            "objectives_npz": npz_path,
            "objectives_meta_json": meta_path,
        },
    )
