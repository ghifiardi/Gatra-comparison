from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .objectives import ObjectiveSpec, compute_reward_matrix
from .objectives_normalize import (
    NormalizationSpec,
    NormalizationStats,
    ObjectiveSpec as NormalizedObjectiveSpec,
    apply_zscore,
    compute_stats,
    rate_per_1k,
)

_REALDATA_TYPES = {"time_to_triage_seconds", "detection_coverage"}
_TIME_TO_TRIAGE = "time_to_triage_seconds"
_DETECTION_COVERAGE = "detection_coverage"


@dataclass(frozen=True)
class RealDataRewardResult:
    reward_matrix: NDArray[np.float32]
    objective_signals: NDArray[np.float32] | None
    objective_signals_raw: NDArray[np.float32] | None
    objective_names: list[str]
    source: str
    stats: dict[str, float]
    normalization: dict[str, Any]
    artifacts: dict[str, str]


def validate_alignment(expected_len: int, **arrays: NDArray[Any]) -> None:
    for name, arr in arrays.items():
        if int(arr.shape[0]) != expected_len:
            raise ValueError(
                f"Alignment error: expected {expected_len} rows but '{name}' has {int(arr.shape[0])}"
            )


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


def _compute_session_sizes(session_ids: NDArray[np.str_]) -> NDArray[np.float32]:
    counts: dict[str, int] = {}
    for sid_raw in session_ids:
        sid = str(sid_raw) or "__missing_session__"
        counts[sid] = counts.get(sid, 0) + 1
    out = np.zeros((session_ids.shape[0],), dtype=np.float32)
    for i in range(session_ids.shape[0]):
        sid = str(session_ids[i]) or "__missing_session__"
        out[i] = float(counts[sid])
    return out


def _load_split_signals(
    contract_dir: str,
    split: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    session_ids = _load_str_array(contract_dir, "session_id", split)
    timestamps = _load_i64_array(contract_dir, "timestamps_epoch_s", split)
    pages = _load_str_array(contract_dir, "page", split)
    actions_raw = _load_str_array(contract_dir, "action", split)
    ttt_raw = compute_time_to_triage_seconds(session_ids=session_ids, timestamps_epoch_s=timestamps)
    cov_raw = compute_coverage_increments(session_ids=session_ids, pages=pages, actions=actions_raw)
    session_sizes = _compute_session_sizes(session_ids)
    return ttt_raw, cov_raw, session_sizes


def _to_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _parse_norm_spec(raw: dict[str, Any] | None) -> NormalizationSpec:
    if raw is None:
        return NormalizationSpec(
            reference_split="val",
            apply_to=("val", "test"),
            eps=1e-8,
            clip_z=5.0,
            persist=True,
        )
    apply_raw = raw.get("apply_to", ["val", "test"])
    if not isinstance(apply_raw, list):
        raise ValueError("morl.normalization.apply_to must be a list")
    return NormalizationSpec(
        reference_split=str(raw.get("reference_split", "val")),
        apply_to=tuple(str(v) for v in apply_raw),
        eps=_to_float(raw.get("eps"), 1e-8),
        clip_z=_to_float(raw.get("clip_z"), 5.0),
        persist=bool(raw.get("persist", True)),
    )


def _resolve_norm_name(spec: ObjectiveSpec, legacy_default_norm: str) -> str:
    if spec.norm and spec.norm != "none":
        return spec.norm
    if legacy_default_norm and legacy_default_norm != "none":
        return legacy_default_norm
    return "none"


def _as_normalized_spec(spec: ObjectiveSpec, legacy_default_norm: str) -> NormalizedObjectiveSpec:
    return NormalizedObjectiveSpec(
        name=spec.name,
        direction=spec.direction,
        norm=_resolve_norm_name(spec, legacy_default_norm),
        cap_pctl=spec.cap_pctl,
    )


def _preprocess_for_norm(
    values: NDArray[np.float32],
    *,
    norm_name: str,
    denom_events: NDArray[np.float32],
) -> NDArray[np.float32]:
    if norm_name in {"none", "zscore"}:
        return values.astype(np.float32)
    if norm_name == "log1p_zscore":
        return np.log1p(np.maximum(values, np.float32(0.0))).astype(np.float32)
    if norm_name == "rate_per_1k_zscore":
        return rate_per_1k(values, denom_events)
    raise ValueError(f"Unsupported norm: {norm_name}")


def _cap_value(values: NDArray[np.float32], p: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, p, method="linear"))


def _apply_cap(values: NDArray[np.float32], cap: float | None) -> NDArray[np.float32]:
    if cap is None:
        return values.astype(np.float32, copy=True)
    return np.minimum(values, np.float32(cap)).astype(np.float32)


def _build_reference_stats(
    contract_dir: str,
    objectives: Sequence[ObjectiveSpec],
    *,
    norm_cfg: NormalizationSpec,
    legacy_default_norm: str,
) -> dict[str, NormalizationStats]:
    ttt_ref, cov_ref, session_sizes_ref = _load_split_signals(contract_dir, norm_cfg.reference_split)
    stats: dict[str, NormalizationStats] = {}
    for spec in objectives:
        if spec.type not in _REALDATA_TYPES:
            continue
        n_spec = _as_normalized_spec(spec, legacy_default_norm)
        if n_spec.norm == "none":
            continue
        raw = ttt_ref if spec.type == _TIME_TO_TRIAGE else cov_ref
        base = _preprocess_for_norm(raw, norm_name=n_spec.norm, denom_events=session_sizes_ref)
        cap_value: float | None = None
        if n_spec.cap_pctl is not None:
            cap_value = _cap_value(base, n_spec.cap_pctl)
            base = _apply_cap(base, cap_value)
        mean, std = compute_stats(base.astype(np.float32), eps=norm_cfg.eps)
        stats[spec.name] = NormalizationStats(
            name=spec.name,
            norm=n_spec.norm,
            direction=n_spec.direction,
            cap_pctl=n_spec.cap_pctl,
            cap_value=cap_value,
            mean=mean,
            std=std,
            clip_z=norm_cfg.clip_z,
            eps=norm_cfg.eps,
        )
    return stats


def _write_norm_json(
    contract_dir: str,
    *,
    norm_cfg: NormalizationSpec,
    stats: dict[str, NormalizationStats],
) -> str:
    path = os.path.join(contract_dir, "objectives_norm.json")
    payload = {
        "version": "v0.6.1",
        "reference_split": norm_cfg.reference_split,
        "apply_to": list(norm_cfg.apply_to),
        "eps": norm_cfg.eps,
        "clip_z": norm_cfg.clip_z,
        "objectives": {name: asdict(stat) for name, stat in stats.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


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
    objective_names: list[str],
    stats: dict[str, float],
    normalization_summary: dict[str, Any],
    normalization_applied: bool,
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
        "stats": stats,
        "normalization_applied": normalization_applied,
    }
    payload["definitions"] = {
        _TIME_TO_TRIAGE: "Per-session max(ts)-min(ts); reward is negative when alerting.",
        _DETECTION_COVERAGE: "Session-level novelty of (page,action); reward is positive when alerting.",
    }
    payload["source"] = _load_meta_source(contract_dir)
    payload["normalization"] = normalization_summary

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
    normalization_cfg: dict[str, Any] | None = None,
    legacy_default_norm: str = "none",
) -> RealDataRewardResult:
    rewards = compute_reward_matrix(y_true, actions, objectives)
    names = [spec.name for spec in objectives]
    norm_cfg = _parse_norm_spec(normalization_cfg)

    if contract_dir is None or not _needs_realdata(objectives):
        return RealDataRewardResult(
            reward_matrix=rewards,
            objective_signals=None,
            objective_signals_raw=None,
            objective_names=names,
            source="fallback_synthetic",
            stats={},
            normalization={
                "applied": False,
                "reference_split": norm_cfg.reference_split,
                "apply_to": list(norm_cfg.apply_to),
            },
            artifacts={},
        )

    try:
        ttt_raw, cov_raw, session_sizes = _load_split_signals(contract_dir, split)
    except FileNotFoundError:
        return RealDataRewardResult(
            reward_matrix=rewards,
            objective_signals=None,
            objective_signals_raw=None,
            objective_names=names,
            source="fallback_synthetic",
            stats={},
            normalization={
                "applied": False,
                "reference_split": norm_cfg.reference_split,
                "apply_to": list(norm_cfg.apply_to),
            },
            artifacts={},
        )

    expected_len = int(y_true.shape[0])
    validate_alignment(
        expected_len=expected_len,
        actions=actions,
        time_to_triage_seconds=ttt_raw,
        coverage_increment=cov_raw,
    )

    ref_stats = _build_reference_stats(
        contract_dir=contract_dir,
        objectives=objectives,
        norm_cfg=norm_cfg,
        legacy_default_norm=legacy_default_norm,
    )
    norm_json_path: str | None = None
    if norm_cfg.persist and ref_stats:
        norm_json_path = _write_norm_json(contract_dir, norm_cfg=norm_cfg, stats=ref_stats)

    signals_raw = np.zeros((expected_len, len(objectives)), dtype=np.float32)
    signals_used = np.zeros((expected_len, len(objectives)), dtype=np.float32)
    actions_f = actions.astype(np.float32)
    normalization_applied = False

    for i, spec in enumerate(objectives):
        if spec.type == _TIME_TO_TRIAGE:
            raw_signal = ttt_raw
        elif spec.type == _DETECTION_COVERAGE:
            raw_signal = cov_raw
        else:
            continue

        signals_raw[:, i] = raw_signal
        n_spec = _as_normalized_spec(spec, legacy_default_norm)
        used_signal = raw_signal
        if (
            n_spec.norm != "none"
            and split in norm_cfg.apply_to
            and spec.name in ref_stats
            and expected_len > 0
        ):
            stat = ref_stats[spec.name]
            base = _preprocess_for_norm(raw_signal, norm_name=n_spec.norm, denom_events=session_sizes)
            base = _apply_cap(base, stat.cap_value)
            used_signal = apply_zscore(base, mean=stat.mean, std=stat.std, clip_z=norm_cfg.clip_z)
            normalization_applied = True

        signals_used[:, i] = used_signal
        if spec.type == _TIME_TO_TRIAGE:
            rewards[:, i] = -used_signal * actions_f
        elif spec.type == _DETECTION_COVERAGE:
            rewards[:, i] = used_signal * actions_f

    coverage_rate_raw = rate_per_1k(cov_raw, session_sizes)
    stats = {
        "ttt_raw_median_seconds": float(np.median(ttt_raw)) if ttt_raw.size else 0.0,
        "ttt_raw_p90_seconds": float(np.percentile(ttt_raw, 90)) if ttt_raw.size else 0.0,
        "coverage_raw_novelty_rate": float(np.mean(cov_raw)) if cov_raw.size else 0.0,
        "coverage_raw_rate_per_1k": float(np.mean(coverage_rate_raw))
        if coverage_rate_raw.size
        else 0.0,
    }
    for i, spec in enumerate(objectives):
        if spec.type == _TIME_TO_TRIAGE:
            stats["ttt_norm_median_z"] = float(np.median(signals_used[:, i])) if expected_len else 0.0
            stats["ttt_norm_p90_z"] = (
                float(np.percentile(signals_used[:, i], 90)) if expected_len else 0.0
            )
        elif spec.type == _DETECTION_COVERAGE:
            stats["coverage_norm_median_z"] = (
                float(np.median(signals_used[:, i])) if expected_len else 0.0
            )
            stats["coverage_norm_p90_z"] = (
                float(np.percentile(signals_used[:, i], 90)) if expected_len else 0.0
            )

    npz_path = os.path.join(contract_dir, f"objectives_{split}.npz")
    np.savez(
        npz_path,
        objective_signals=signals_used,
        objective_signals_raw=signals_raw,
        objective_names=np.asarray(names, dtype=np.str_),
        time_to_triage_seconds=ttt_raw.astype(np.float32),
        coverage_increment=cov_raw.astype(np.float32),
    )

    normalization_summary = {
        "applied": normalization_applied,
        "reference_split": norm_cfg.reference_split,
        "apply_to": list(norm_cfg.apply_to),
        "eps": norm_cfg.eps,
        "clip_z": norm_cfg.clip_z,
        "objectives": {name: asdict(stat) for name, stat in ref_stats.items()},
    }
    meta_path = _upsert_meta(
        contract_dir=contract_dir,
        split=split,
        objective_names=names,
        stats=stats,
        normalization_summary=normalization_summary,
        normalization_applied=normalization_applied,
    )
    artifacts = {
        "objectives_npz": npz_path,
        "objectives_meta_json": meta_path,
    }
    if norm_json_path is not None:
        artifacts["objectives_norm_json"] = norm_json_path

    return RealDataRewardResult(
        reward_matrix=rewards.astype(np.float32),
        objective_signals=signals_used,
        objective_signals_raw=signals_raw,
        objective_names=names,
        source="level1_realdata",
        stats=stats,
        normalization=normalization_summary,
        artifacts=artifacts,
    )
