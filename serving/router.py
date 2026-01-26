from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import logging
import numpy as np
from architecture_a_rl.env_bandit import ACTIONS
from data.features import validate_feature_vector
from .contract import InferenceRequest, InferenceResponse, InferenceErrorResponse
from .adapters import IForestAdapter, PPOAdapter

logger = logging.getLogger(__name__)

INVALID_FEATURE_COUNTERS = {
    "dim_mismatch": 0,
    "nan_inf": 0,
    "dtype": 0,
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
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
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
    iforest: Optional[IForestAdapter] = None,
    ppo: Optional[PPOAdapter] = None,
) -> Tuple[int, Dict[str, Any]]:
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

    response = _predict(req, mode, active, v7, v128, iforest, ppo)
    return 200, response.model_dump()

def _predict(
    req: InferenceRequest,
    mode: str,
    active: str,
    features_v7: np.ndarray,
    features_v128: np.ndarray,
    iforest: Optional[IForestAdapter],
    ppo: Optional[PPOAdapter],
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
    iforest: Optional[IForestAdapter] = None,
    ppo: Optional[PPOAdapter] = None,
) -> InferenceResponse:
    return _predict(req, mode, active, req.features_v7, req.features_v128, iforest, ppo)
