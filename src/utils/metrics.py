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
