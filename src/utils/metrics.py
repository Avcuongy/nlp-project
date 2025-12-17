from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_1d_labels(arr: Any) -> np.ndarray:
    """Convert labels to a 1D integer/string array.

    Supports:
    - Pandas DataFrame/Series
    - One-hot matrices (argmax)
    - Probability matrices (argmax)
    - Already 1D arrays
    """
    if isinstance(arr, pd.DataFrame):
        # If single column, squeeze; if multi-column, treat as matrix
        if arr.shape[1] == 1:
            return arr.iloc[:, 0].to_numpy()
        arr = arr.to_numpy()
    if isinstance(arr, pd.Series):
        return arr.to_numpy()

    a = np.asarray(arr)
    if a.ndim == 2:
        # Heuristics: one-hot/probabilities -> argmax
        with np.errstate(invalid="ignore"):
            row_sums = a.sum(axis=1)
        if np.all((a >= 0) & (a <= 1)) and (
            np.allclose(row_sums, 1, atol=1e-5) or np.all(np.isin(a, [0, 1]))
        ):
            return a.argmax(axis=1)
    return a


def evaluate(
    y_true: Any,
    y_pred: Any,
    label_names: Optional[list[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate single-label multiclass predictions.

    Args:
        y_true: Ground-truth labels (1D array-like, or 2D one-hot/proba).
        y_pred: Predicted labels (1D array-like, or 2D one-hot/proba).
        label_names: Class names for classification report (ordered to match encodings).
        verbose: If True, print summary metrics and report.

    Returns:
        Dict[str, float]: accuracy and macro-averaged precision/recall/f1.
    """
    y_true_arr = _to_1d_labels(y_true)
    y_pred_arr = _to_1d_labels(y_pred)

    metrics: Dict[str, float] = {
        "accuracy_score": accuracy_score(y_true_arr, y_pred_arr),
        "precision_macro": precision_score(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
    }

    if verbose:
        print("\nEvaluation Results:")
        print(
            pd.DataFrame.from_dict(metrics, orient="index", columns=["score"]).round(4)
        )

        # Print classification report if label names are provided
        try:
            if label_names is not None:
                print("\nClassification Report:")
                print(
                    classification_report(
                        y_true_arr,
                        y_pred_arr,
                        target_names=label_names,
                        zero_division=0,
                    )
                )
        except Exception:
            # Fallback without target_names if a mismatch occurs
            print("\nClassification Report:")
            print(classification_report(y_true_arr, y_pred_arr, zero_division=0))

        # Also display confusion matrix
        try:
            cm = confusion_matrix(y_true_arr, y_pred_arr)
            cm_df = pd.DataFrame(
                cm, index=label_names or None, columns=label_names or None
            )
            print("\nConfusion Matrix:")
            print(cm_df)
        except Exception:
            pass

    return metrics