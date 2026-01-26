from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    event_id: str
    ts: str
    features_v7: List[float | int | str]
    features_v128: List[float | int | str]
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    event_id: str
    iforest_score: Optional[float] = None
    rl_action_probs: Optional[List[float]] = None
    rl_action: Optional[str] = None
    routing_decision: str
    meta: Dict[str, str] = {}

class InferenceErrorResponse(BaseModel):
    error: str
    error_code: str
    feature_version: str
    expected_dim: int
    actual_dim: Optional[int] = None
    message: str
    request_id: Optional[str] = None
