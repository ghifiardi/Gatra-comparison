from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

import numpy as np
from numpy.typing import NDArray

ObjectiveType = Literal["classification", "fp_penalty", "per_alert_cost"]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    type: ObjectiveType
    tp: float = 0.0
    fn: float = 0.0
    fp: float = 0.0
    tn: float = 0.0
    fp_penalty: float = 0.0
    per_alert_penalty: float = 0.0


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def parse_objectives(raw: Sequence[dict[str, object]]) -> list[ObjectiveSpec]:
    specs: list[ObjectiveSpec] = []
    for obj in raw:
        obj_type = str(obj.get("type", "")).strip()
        if obj_type not in {"classification", "fp_penalty", "per_alert_cost"}:
            raise ValueError(f"Unsupported objective type: {obj_type}")
        objective_type = cast(ObjectiveType, obj_type)
        specs.append(
            ObjectiveSpec(
                name=str(obj.get("name", obj_type)),
                type=objective_type,
                tp=_to_float(obj.get("tp", 0.0)),
                fn=_to_float(obj.get("fn", 0.0)),
                fp=_to_float(obj.get("fp", 0.0)),
                tn=_to_float(obj.get("tn", 0.0)),
                fp_penalty=_to_float(obj.get("fp_penalty", 0.0)),
                per_alert_penalty=_to_float(obj.get("per_alert_penalty", 0.0)),
            )
        )
    return specs


def compute_reward_vector(
    y_true: int,
    action: int,
    objectives: Sequence[ObjectiveSpec],
) -> NDArray[np.float32]:
    if action not in (0, 1):
        raise ValueError(f"Binary action expected (0/1), got: {action}")
    if y_true not in (0, 1):
        raise ValueError(f"Binary label expected (0/1), got: {y_true}")

    rewards: list[float] = []
    is_alert = action == 1
    is_threat = y_true == 1

    for spec in objectives:
        if spec.type == "classification":
            if is_threat and is_alert:
                rewards.append(spec.tp)
            elif is_threat and not is_alert:
                rewards.append(spec.fn)
            elif (not is_threat) and is_alert:
                rewards.append(spec.fp)
            else:
                rewards.append(spec.tn)
        elif spec.type == "fp_penalty":
            rewards.append(spec.fp_penalty if ((not is_threat) and is_alert) else 0.0)
        elif spec.type == "per_alert_cost":
            rewards.append(spec.per_alert_penalty if is_alert else 0.0)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported objective type: {spec.type}")

    return np.asarray(rewards, dtype=np.float32)


def compute_reward_matrix(
    y_true: NDArray[np.int_],
    actions: NDArray[np.int_],
    objectives: Sequence[ObjectiveSpec],
) -> NDArray[np.float32]:
    if y_true.shape[0] != actions.shape[0]:
        raise ValueError("y_true and actions must have identical length")

    out = np.zeros((y_true.shape[0], len(objectives)), dtype=np.float32)
    for i in range(y_true.shape[0]):
        out[i] = compute_reward_vector(int(y_true[i]), int(actions[i]), objectives)
    return out


def scalarize_matrix(
    matrix: NDArray[np.float32],
    weights: NDArray[np.float32],
) -> NDArray[np.float32]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be rank-2")
    if weights.ndim != 1:
        raise ValueError("weights must be rank-1")
    if matrix.shape[1] != weights.shape[0]:
        raise ValueError("matrix objective dim must match weight dim")
    return (matrix @ weights).astype(np.float32)
