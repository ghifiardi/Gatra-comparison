from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IForestModel:
    """Isolation Forest model wrapper with fixed calibration.

    The anomaly score calibration uses statistics computed from the training set
    to ensure consistent scoring across different batch sizes. This is critical
    for fair evaluation - scoring a single sample should give the same result
    as scoring it within a batch of 100 samples.

    Attributes:
        model: The fitted sklearn IsolationForest model.
        threshold: Decision threshold for anomaly classification.
        calibration_min: Minimum raw score from training set (for normalization).
        calibration_max: Maximum raw score from training set (for normalization).
    """
    model: IsolationForest
    threshold: float = 0.8
    calibration_min: float | None = field(default=None)
    calibration_max: float | None = field(default=None)

    def fit_calibration(self, X_train: np.ndarray) -> "IForestModel":
        """Compute calibration statistics from training data.

        This should be called after fitting the IsolationForest model
        to establish fixed min/max bounds for score normalization.

        Args:
            X_train: Training features (already preprocessed).

        Returns:
            Self for method chaining.
        """
        raw_scores = self.model.score_samples(X_train)
        self.calibration_min = float(raw_scores.min())
        self.calibration_max = float(raw_scores.max())
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores in [0, 1] range.

        Uses fixed calibration statistics from training set to ensure
        consistent scoring regardless of batch size.

        Args:
            X: Input features (already preprocessed).

        Returns:
            Anomaly scores where higher = more anomalous.
        """
        # sklearn IsolationForest: higher score_samples = more normal
        raw = self.model.score_samples(X)

        # Use fixed calibration from training set if available
        if self.calibration_min is not None and self.calibration_max is not None:
            mn, mx = self.calibration_min, self.calibration_max
        else:
            # Fallback to batch statistics (not recommended for evaluation)
            mn, mx = float(raw.min()), float(raw.max())

        denom = (mx - mn) if mx > mn else 1.0
        # Invert so higher = more anomalous
        s = (mx - raw) / denom
        result: np.ndarray = np.clip(s, 0.0, 1.0)
        return result

    def predict_alert(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to binary predictions using threshold."""
        return (scores >= self.threshold).astype(np.int32)
