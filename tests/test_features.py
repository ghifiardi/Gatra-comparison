import numpy as np
import pytest
from data.toy import ToyDataset
from data.features import extract_features_v7, extract_features_v128, HistoryContext, validate_feature_vector

def test_feature_shapes() -> None:
    events, labels = ToyDataset(n=10).generate()
    e = events[0]
    v7 = extract_features_v7(e)
    v128 = extract_features_v128(e, HistoryContext(now=e.ts))
    assert v7.shape == (7,)
    assert v128.shape == (128,)

def test_validate_feature_vector_good_v7_v128() -> None:
    v7 = validate_feature_vector([1, 2, 3, 4, 5, 6, 7], expected_dim=7)
    v128 = validate_feature_vector(np.ones(128, dtype=np.float64), expected_dim=128)
    assert v7.dtype == np.float32
    assert v128.dtype == np.float32
    assert v7.shape == (7,)
    assert v128.shape == (128,)

def test_validate_feature_vector_wrong_dims() -> None:
    with pytest.raises(ValueError, match="Expected dim=7"):
        validate_feature_vector([1, 2, 3], expected_dim=7)

def test_validate_feature_vector_nan_inf() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_feature_vector([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0], expected_dim=7)
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_feature_vector([0.0, 1.0, np.inf, 3.0, 4.0, 5.0, 6.0], expected_dim=7)

def test_validate_feature_vector_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="numeric"):
        validate_feature_vector(["a", "b", "c", "d", "e", "f", "g"], expected_dim=7)
