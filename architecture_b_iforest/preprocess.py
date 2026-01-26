from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    scaler: StandardScaler | None = None

    def fit(self, X: np.ndarray) -> "Preprocessor":
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        result: np.ndarray = self.scaler.transform(X)
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
