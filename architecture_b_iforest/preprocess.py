from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.floating[Any]]


@dataclass
class Preprocessor:
    scaler: StandardScaler | None = None

    def fit(self, X: FloatArray) -> "Preprocessor":
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        if self.scaler is None:
            return X
        result: FloatArray = self.scaler.transform(X)
        return result

    def fit_transform(self, X: FloatArray) -> FloatArray:
        return self.fit(X).transform(X)
