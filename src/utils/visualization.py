from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
	y_true: Iterable[int],
	y_pred: Iterable[int],
	*,
	labels: Optional[Sequence[str]] = None,
	normalize: Optional[str] = None,
	cmap: str = "Blues",
	title: Optional[str] = None,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot confusion matrix using sklearn's display helper."""
	disp = ConfusionMatrixDisplay.from_predictions(
		y_true, y_pred, display_labels=labels, normalize=normalize, cmap=cmap, ax=ax
	)
	if title is None:
		title = "Confusion Matrix"
		if normalize:
			title += f" ({normalize})"
	disp.ax_.set_title(title)
	plt.tight_layout()
	return disp.ax_


def plot_roc_curves(
	y_true: Iterable[int],
	y_score: np.ndarray,
	*,
	n_classes: Optional[int] = None,
	labels: Optional[Sequence[str]] = None,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""
	Plot ROC curves for binary or multi-class classification using OvR.

	Args:
		y_true: Ground-truth integer labels.
		y_score: Score/probability matrix of shape (n_samples, n_classes) or vector for binary.
		n_classes: Optionally provide number of classes.
		labels: Optional class names for legend.
	"""
	y_true = np.asarray(list(y_true))
	if y_score.ndim == 1:
		ax = RocCurveDisplay.from_predictions(y_true, y_score, name=labels[1] if labels else None, ax=ax).ax_
		ax.set_title("ROC Curve")
		plt.tight_layout()
		return ax

	if n_classes is None:
		n_classes = y_score.shape[1]

	y_bin = label_binarize(y_true, classes=list(range(n_classes)))
	if ax is None:
		_, ax = plt.subplots(figsize=(6, 5))
	for i in range(n_classes):
		RocCurveDisplay.from_predictions(
			y_bin[:, i], y_score[:, i], name=(labels[i] if labels else f"class {i}"), ax=ax
		)
	ax.plot([0, 1], [0, 1], "k--", label="chance")
	ax.set_title("ROC Curves (OvR)")
	ax.legend()
	plt.tight_layout()
	return ax


def plot_pr_curves(
	y_true: Iterable[int],
	y_score: np.ndarray,
	*,
	n_classes: Optional[int] = None,
	labels: Optional[Sequence[str]] = None,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot Precision-Recall curves for binary or multi-class classification."""
	y_true = np.asarray(list(y_true))
	if y_score.ndim == 1:
		ax = PrecisionRecallDisplay.from_predictions(
			y_true, y_score, name=labels[1] if labels else None, ax=ax
		).ax_
		ax.set_title("Precision-Recall Curve")
		plt.tight_layout()
		return ax

	if n_classes is None:
		n_classes = y_score.shape[1]

	y_bin = label_binarize(y_true, classes=list(range(n_classes)))
	if ax is None:
		_, ax = plt.subplots(figsize=(6, 5))
	for i in range(n_classes):
		PrecisionRecallDisplay.from_predictions(
			y_bin[:, i], y_score[:, i], name=(labels[i] if labels else f"class {i}"), ax=ax
		)
	ax.set_title("Precision-Recall Curves (OvR)")
	ax.legend()
	plt.tight_layout()
	return ax


def plot_training_curves(
	history: dict,
	*,
	metrics: Sequence[str] = ("loss",),
	title: str = "Training Curves",
) -> plt.Axes:
	"""
	Plot simple training curves given a history dict (e.g., from DL training).

	Expected format: keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy', each a list of values.
	"""
	_, ax = plt.subplots(figsize=(6, 4))
	for m in metrics:
		if m in history:
			ax.plot(history[m], label=m)
		val_key = f"val_{m}"
		if val_key in history:
			ax.plot(history[val_key], label=val_key)
	ax.set_xlabel("Epoch")
	ax.set_title(title)
	ax.legend()
	plt.tight_layout()
	return ax

