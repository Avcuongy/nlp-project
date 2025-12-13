from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import make_scorer


def evaluate_classification(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    average: str = "macro",
    zero_division: int = 0,
) -> Dict[str, float]:
    """
    Compute common classification metrics.

    Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            average: Averaging strategy for multi-class metrics (e.g., 'macro', 'micro', 'weighted').
            zero_division: Value to use when there is a zero division.

    Returns:
            Dictionary containing accuracy, precision, recall, and f1.
    """
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=zero_division
    )
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f1)}


def try_compute_multiclass_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Optional[float]:
    """
    Compute macro ROC-AUC if probabilities are provided.
    Returns None if computation is not applicable.
    """
    try:
        # y_true integers [0..n_classes-1], y_proba shape (n_samples, n_classes)
        if y_proba.ndim == 1:
            # binary probabilities of positive class
            return float(roc_auc_score(y_true, y_proba))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        return None


def get_scorer(scoring: str):
    """
    Return a sklearn-compatible scorer based on a scoring string.
    Falls back to the string for built-in scorer names.
    """
    # For common customizations we could construct scorers explicitly
    mapping = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_micro": make_scorer(f1_score, average="micro"),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
    }
    return mapping.get(scoring, scoring)
