from __future__ import annotations
import os
import json
import numpy as np
import yaml
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, f1_score
from data.loaders import load_data
from data.splits import time_split
from data.features import extract_features_v7
from data.schemas import RawEvent, Label
from .preprocess import Preprocessor
from .model import IForestModel


def _get_scores_and_labels(
    events: list[RawEvent],
    labels: list[Label],
    prep: Preprocessor,
    ifm: IForestModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract anomaly scores and ground truth labels for a data split."""
    label_map = {lb.event_id: lb for lb in labels}

    y_true: list[int] = []
    y_scores: list[float] = []

    for e in events:
        lb = label_map.get(e.event_id)
        if lb is None or lb.label == "unknown":
            continue

        y_true.append(1 if lb.label == "threat" else 0)

        x7 = extract_features_v7(e)[None, :]
        x7p = prep.transform(x7)
        score = ifm.score(x7p)[0]
        y_scores.append(float(score))

    return np.array(y_true), np.array(y_scores)


def _tune_threshold_on_validation(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Find optimal threshold that maximizes F1 score on validation set.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted anomaly scores in [0, 1].
        thresholds: List of thresholds to try. Defaults to [0.1, 0.2, ..., 0.9].

    Returns:
        Tuple of (best_threshold, metrics_at_best_threshold).
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics: dict[str, float] = {}

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            p, r, _, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            best_metrics = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "threshold": thresh,
            }

    return best_threshold, best_metrics


def _evaluate_iforest(
    events: list[RawEvent],
    labels: list[Label],
    prep: Preprocessor,
    ifm: IForestModel,
) -> dict[str, float]:
    """Evaluate Isolation Forest on a data split."""
    y_true, y_scores = _get_scores_and_labels(events, labels, prep, ifm)

    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_samples": 0.0}

    y_pred = (y_scores >= ifm.threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_samples": float(len(y_true)),
    }


def train_iforest(iforest_cfg: str, data_cfg: str) -> str:
    with open(iforest_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["io"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    loaded = load_data(data_cfg)
    splits = time_split(loaded.events, loaded.labels, data_cfg)
    train_events, train_labels = splits.train
    val_events, val_labels = splits.val

    X_train = np.stack([extract_features_v7(e) for e in train_events], axis=0)

    prep = Preprocessor()
    Xp = prep.fit_transform(X_train)

    mcfg = cfg["model"]
    model = IsolationForest(
        n_estimators=mcfg["n_estimators"],
        max_samples=mcfg["max_samples"],
        contamination=mcfg["contamination"],
        max_features=mcfg["max_features"],
        bootstrap=mcfg["bootstrap"],
        random_state=mcfg["random_state"],
        n_jobs=-1,
    )
    model.fit(Xp)

    # Create model with temporary threshold for calibration
    ifm = IForestModel(model=model, threshold=0.5)

    # Fit calibration statistics from training data for consistent scoring
    ifm.fit_calibration(Xp)

    # Tune threshold on validation set if enabled
    tune_threshold = cfg.get("scoring", {}).get("tune_on_validation", False)
    if tune_threshold:
        y_true_val, y_scores_val = _get_scores_and_labels(val_events, val_labels, prep, ifm)
        best_threshold, tuning_metrics = _tune_threshold_on_validation(y_true_val, y_scores_val)
        ifm.threshold = best_threshold
    else:
        ifm.threshold = cfg["scoring"]["threshold"]
        tuning_metrics = {}

    # Evaluate on validation set with final threshold
    val_metrics = _evaluate_iforest(val_events, val_labels, prep, ifm)

    bundle = {
        "preprocessor": prep,
        "model": ifm,
        "meta": {
            "model_name": "iforest",
            "threshold": ifm.threshold,
            "threshold_tuned": tune_threshold,
            "n_estimators": mcfg["n_estimators"],
            "calibration_min": ifm.calibration_min,
            "calibration_max": ifm.calibration_max,
        },
    }
    path = os.path.join(out_dir, "iforest_bundle.joblib")
    joblib.dump(bundle, path)

    # Save training log with validation metrics
    train_log: dict[str, float | int | str | dict[str, float]] = {
        "train_samples": len(train_events),
        "val_samples": int(val_metrics["n_samples"]),
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "final_threshold": ifm.threshold,
        "threshold_tuned": tune_threshold,
    }
    if tuning_metrics:
        train_log["tuning_metrics"] = tuning_metrics

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    return path
