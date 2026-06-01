from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Protocol
import logging
import numpy as np
from numpy.typing import NDArray
from architecture_a_rl.env_bandit import ACTIONS
from data.features import validate_feature_vector
from .contract import InferenceRequest, InferenceResponse, InferenceErrorResponse

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float32]


class IForestScorer(Protocol):
    def score(self, features_v7: FloatArray) -> float: ...


class PPOScorer(Protocol):
    def action_probs(self, features_v128: FloatArray) -> list[float]: ...


INVALID_FEATURE_COUNTERS = {
    "dim_mismatch": 0,
    "nan_inf": 0,
    "dtype": 0,
    "schema_mismatch": 0,
    "unknown": 0,
}


def _record_invalid_feature(reason: str, feature_version: str, message: str) -> None:
    if reason not in INVALID_FEATURE_COUNTERS:
        reason = "unknown"
    INVALID_FEATURE_COUNTERS[reason] += 1
    # TODO: Replace with metrics counter, e.g., invalid_feature_total{reason=...}
    logger.warning(
        "invalid_feature_total reason=%s feature_version=%s message=%s",
        reason,
        feature_version,
        message,
    )


def _infer_actual_dim(vec: Any) -> Optional[int]:
    try:
        arr = np.asarray(vec)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return int(arr.shape[0])


def _error_response(
    *,
    error_code: str,
    feature_version: str,
    expected_dim: int,
    actual_dim: Optional[int],
    message: str,
    request_id: Optional[str],
) -> Dict[str, Any]:
    return InferenceErrorResponse(
        error="Invalid features",
        error_code=error_code,
        feature_version=feature_version,
        expected_dim=expected_dim,
        actual_dim=actual_dim,
        message=message,
        request_id=request_id,
    ).model_dump()


def _validate_features(
    features: Any,
    *,
    version: str,
    expected_dim: int,
    request_id: Optional[str],
) -> Tuple[Optional[NDArray[np.float32]], Optional[Dict[str, Any]]]:
    try:
        validated = validate_feature_vector(features, expected_dim=expected_dim)
        return validated, None
    except ValueError as exc:
        message = str(exc)
        actual_dim = _infer_actual_dim(features)
        if "Expected dim=" in message:
            reason = "dim_mismatch"
            error_code = "FEATURE_DIM_MISMATCH"
        elif "NaN or Inf" in message:
            reason = "nan_inf"
            error_code = "FEATURE_NAN_INF"
        elif "numeric" in message:
            reason = "dtype"
            error_code = "FEATURE_DTYPE"
        else:
            reason = "unknown"
            error_code = "FEATURE_DTYPE"
        _record_invalid_feature(reason, version, message)
        return None, _error_response(
            error_code=error_code,
            feature_version=version,
            expected_dim=expected_dim,
            actual_dim=actual_dim,
            message=message,
            request_id=request_id,
        )


def handle_request(
    req: InferenceRequest,
    mode: str,
    active: str,
    iforest: Optional[IForestScorer] = None,
    ppo: Optional[PPOScorer] = None,
    expected_schema_hash: Optional[str] = None,
    current_schema_hash: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    if expected_schema_hash and current_schema_hash and expected_schema_hash != current_schema_hash:
        _record_invalid_feature(
            "schema_mismatch",
            "schema",
            f"Expected schema hash {expected_schema_hash}, got {current_schema_hash}",
        )
        return 400, _error_response(
            error_code="FEATURE_SCHEMA_MISMATCH",
            feature_version="schema",
            expected_dim=0,
            actual_dim=None,
            message="Feature schema hash mismatch",
            request_id=req.request_id,
        )

    v7, err = _validate_features(
        req.features_v7,
        version="v7",
        expected_dim=7,
        request_id=req.request_id,
    )
    if err:
        return 400, err

    v128, err = _validate_features(
        req.features_v128,
        version="v128",
        expected_dim=128,
        request_id=req.request_id,
    )
    if err:
        return 400, err

    if v7 is None or v128 is None:
        return 400, _error_response(
            error_code="FEATURE_DTYPE",
            feature_version="unknown",
            expected_dim=0,
            actual_dim=None,
            message="Feature validation failed",
            request_id=req.request_id,
        )

    response = _predict(req, mode, active, v7, v128, iforest, ppo)
    return 200, response.model_dump()


def _predict(
    req: InferenceRequest,
    mode: str,
    active: str,
    features_v7: FloatArray,
    features_v128: FloatArray,
    iforest: Optional[IForestScorer],
    ppo: Optional[PPOScorer],
) -> InferenceResponse:
    iforest_score = iforest.score(features_v7) if iforest else None
    rl_action_probs = ppo.action_probs(features_v128) if ppo else None

    rl_action = None
    if rl_action_probs:
        rl_action = ACTIONS[int(np.argmax(np.array(rl_action_probs)))]

    routing_decision = "shadow" if mode == "shadow" else active

    return InferenceResponse(
        event_id=req.event_id,
        iforest_score=iforest_score,
        rl_action_probs=rl_action_probs,
        rl_action=rl_action,
        routing_decision=routing_decision,
        meta={"mode": mode, "active": active},
    )


def route_request(
    req: InferenceRequest,
    mode: str,
    active: str,
    iforest: Optional[IForestScorer] = None,
    ppo: Optional[PPOScorer] = None,
    expected_schema_hash: Optional[str] = None,
    current_schema_hash: Optional[str] = None,
) -> InferenceResponse:
    if expected_schema_hash and current_schema_hash and expected_schema_hash != current_schema_hash:
        raise ValueError("Feature schema hash mismatch")

    v7 = validate_feature_vector(req.features_v7, expected_dim=7)
    v128 = validate_feature_vector(req.features_v128, expected_dim=128)
    return _predict(req, mode, active, v7, v128, iforest, ppo)
