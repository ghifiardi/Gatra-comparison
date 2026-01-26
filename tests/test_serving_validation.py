import numpy as np
from serving.contract import InferenceRequest
from serving.router import handle_request


class DummyIForest:
    def score(self, features_v7):
        return 0.42


class DummyPPO:
    def action_probs(self, features_v128):
        return [0.1, 0.2, 0.3, 0.4]


def test_invalid_v7_dim_rejected() -> None:
    req = InferenceRequest(
        event_id="evt_bad_v7",
        ts="2025-01-01T00:00:00",
        features_v7=[1.0, 2.0],
        features_v128=[0.0] * 128,
        request_id="req_bad_v7",
    )
    status, payload = handle_request(req, mode="shadow", active="iforest")
    assert status == 400
    assert payload["error_code"] == "FEATURE_DIM_MISMATCH"
    assert payload["feature_version"] == "v7"


def test_invalid_v128_dim_rejected() -> None:
    req = InferenceRequest(
        event_id="evt_bad_v128",
        ts="2025-01-01T00:00:00",
        features_v7=[1.0, 2.0, 3.0, 80.0, 1.0, 12.0, 3.0],
        features_v128=[0.0] * 10,
        request_id="req_bad_v128",
    )
    status, payload = handle_request(req, mode="shadow", active="iforest")
    assert status == 400
    assert payload["error_code"] == "FEATURE_DIM_MISMATCH"
    assert payload["feature_version"] == "v128"


def test_nan_inf_rejected() -> None:
    req = InferenceRequest(
        event_id="evt_nan",
        ts="2025-01-01T00:00:00",
        features_v7=[1.0, 2.0, np.nan, 80.0, 1.0, 12.0, 3.0],
        features_v128=[0.0] * 128,
        request_id="req_nan",
    )
    status, payload = handle_request(req, mode="shadow", active="iforest")
    assert status == 400
    assert payload["error_code"] == "FEATURE_NAN_INF"
    assert payload["feature_version"] == "v7"


def test_non_numeric_rejected() -> None:
    req = InferenceRequest(
        event_id="evt_dtype",
        ts="2025-01-01T00:00:00",
        features_v7=["a", "b", "c", "d", "e", "f", "g"],
        features_v128=[0.0] * 128,
        request_id="req_dtype",
    )
    status, payload = handle_request(req, mode="shadow", active="iforest")
    assert status == 400
    assert payload["error_code"] == "FEATURE_DTYPE"
    assert payload["feature_version"] == "v7"


def test_valid_request_success() -> None:
    req = InferenceRequest(
        event_id="evt_ok",
        ts="2025-01-01T00:00:00",
        features_v7=[1.0, 2.0, 3.0, 80.0, 1.0, 12.0, 3.0],
        features_v128=[0.0] * 128,
        request_id="req_ok",
    )
    status, payload = handle_request(
        req,
        mode="shadow",
        active="iforest",
        iforest=DummyIForest(),
        ppo=DummyPPO(),
    )
    assert status == 200
    assert payload["event_id"] == "evt_ok"
    assert payload["iforest_score"] == 0.42
    assert payload["rl_action"] == "dismiss"
