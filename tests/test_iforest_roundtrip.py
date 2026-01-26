import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from architecture_b_iforest.model import IForestModel
from architecture_b_iforest.preprocess import Preprocessor

from pathlib import Path


def test_iforest_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 7)).astype(np.float32)

    prep = Preprocessor().fit(X)
    model = IsolationForest(n_estimators=10, random_state=0)
    model.fit(prep.transform(X))

    bundle = {
        "preprocessor": prep,
        "model": IForestModel(model=model, threshold=0.5),
        "meta": {"model_name": "iforest", "threshold": 0.5},
    }

    path = tmp_path / "iforest_bundle.joblib"
    joblib.dump(bundle, path)

    loaded = joblib.load(path)
    scores = loaded["model"].score(loaded["preprocessor"].transform(X))
    assert scores.shape == (50,)
