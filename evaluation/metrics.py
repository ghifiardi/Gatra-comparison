from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score


def classification_metrics(
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    threshold: float,
) -> dict[str, float]:
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    mask = y_true != -1
    y = y_true[mask]
    score = y_score[mask]
    if y.shape[0] == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
        }

    y_pred = (score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    # guard for constant labels
    try:
        out["roc_auc"] = float(roc_auc_score(y, score))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y, score))
    except Exception:
        out["pr_auc"] = float("nan")
    return out
