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
from sklearn.metrics import make_scorer


def get_scorer(scoring: str):
    """
    Return a sklearn-compatible scorer based on a scoring string.
    Falls back to the string for built-in scorer names.

    Args:
        scoring (str): Scoring metric name.

    Returns:
        scorer or str: sklearn scorer object or original string.
    """
    # For common customizations we could construct scorers explicitly
    mapping = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_micro": make_scorer(f1_score, average="micro"),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
    }
    return mapping.get(scoring, scoring)


def evaluate(
    y_true,
    y_pred,
    label_names: Optional[list[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute standard multilabel metrics and optionally print report.

    Args:
        y_true: Ground-truth labels (ndarray or DataFrame of binary matrix).
        y_pred: Predicted labels (ndarray or DataFrame of binary matrix).
        label_names (list[str] | None): Class names for classification report.
        verbose (bool): If True, print summary metrics and report.

    Returns:
        Dict[str, float]: precision/recall/f1 for micro and macro averages.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true.values
    else:
        y_true_arr = y_true

    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred.values
    else:
        y_pred_arr = y_pred

    metrics: Dict[str, float] = {
        "precision_micro": precision_score(
            y_true_arr, y_pred_arr, average="micro", zero_division=0
        ),
        "recall_micro": recall_score(
            y_true_arr, y_pred_arr, average="micro", zero_division=0
        ),
        "f1_micro": f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0),
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
        if label_names is not None:
            print("\nClassification Report:")
            print(
                classification_report(
                    y_true_arr, y_pred_arr, target_names=label_names, zero_division=0
                )
            )

    return metrics
