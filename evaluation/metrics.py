from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score


def classification_metrics(
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    # guard for constant labels
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = float("nan")
    return out
