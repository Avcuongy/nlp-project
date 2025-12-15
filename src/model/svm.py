from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from src.utils.config_loader import load_config
from src.utils.metrics import get_scorer


class SVMModel:
    """SVM Model with OneVsRestClassifier for multi-label ABSA.

    Attributes:
        config (dict): Configuration dictionary loaded from YAML.
        model (OneVsRestClassifier): The underlying SVM classifier.
        grid_search (GridSearchCV): GridSearchCV instance for hyperparameter tuning.
        best_estimator_ (OneVsRestClassifier): Best model from grid search.
        is_fitted (bool): Whether the model has been fitted.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize SVM model with configuration.

        Args:
            config_path (str | None): Path to config YAML file. If None,
                uses default config/ml/svm.yaml.
        """
        if config_path is None:
            root = Path(__file__).resolve().parents[2]
            config_path = str(root / "config" / "ml" / "svm.yaml")

        self.config = load_config(config_path)
        self.model: Optional[OneVsRestClassifier] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.best_estimator_: Optional[OneVsRestClassifier] = None
        self.is_fitted = False

    def _build_model(self) -> OneVsRestClassifier:
        """Build the OneVsRestClassifier with SVC base estimator.

        Returns:
            OneVsRestClassifier: Initialized model.
        """
        return OneVsRestClassifier(SVC())

    def _build_param_grid(self) -> Dict[str, Any]:
        """Build parameter grid for GridSearchCV from config.

        Returns:
            dict: Parameter grid with proper prefixes.
        """
        raw_grid = self.config.get("grid_search", {}).get("param_grid", {})
        param_grid = {}

        for key, values in raw_grid.items():
            # Convert single values to list for GridSearchCV
            if not isinstance(values, list):
                values = [values]
            param_grid[f"estimator__{key}"] = values

        return param_grid

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame,
        verbose: bool = True,
    ) -> "SVMModel":
        """Train the SVM model with GridSearchCV.

        Args:
            X_train: Training features (vectorized text).
            y_train: Training labels (binary matrix).
            verbose (bool): Whether to print training progress.

        Returns:
            self: Fitted model instance.
        """
        # Convert DataFrames to arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values

        # Build model and param grid
        self.model = self._build_model()
        param_grid = self._build_param_grid()

        # Get GridSearch config
        gs_config = self.config.get("grid_search", {})
        scoring_name = gs_config.get("scoring", "f1_micro")
        cv = gs_config.get("cv", 5)
        n_jobs = gs_config.get("n_jobs", -1)

        # Build scorer
        scorer = make_scorer(f1_score, average="micro", zero_division=0)

        # Initialize GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=n_jobs,
            verbose=2 if verbose else 0,
        )

        if verbose:
            print("Training SVM with GridSearchCV...")
            print(f"Parameter grid: {param_grid}")
            print(f"CV folds: {cv}, Scoring: {scoring_name}")

        # Fit
        self.grid_search.fit(X_train, y_train)
        self.best_estimator_ = self.grid_search.best_estimator_
        self.is_fitted = True

        if verbose:
            print("\nBest parameters:")
            for key, value in self.grid_search.best_params_.items():
                print(f"  {key}: {value}")
            print(f"\nBest CV score (f1_micro): {self.grid_search.best_score_:.4f}")

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict labels for input samples.

        Args:
            X: Input features (vectorized text).

        Returns:
            np.ndarray: Binary label matrix.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted or self.best_estimator_ is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.best_estimator_.predict(X)

    def evaluate(
        self,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.DataFrame,
        label_names: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X_test: Test features.
            y_test: Test labels (binary matrix).
            label_names (list[str] | None): Names of label classes for report.
            verbose (bool): Whether to print evaluation results.

        Returns:
            dict: Dictionary of metrics (precision, recall, f1 for micro/macro).
        """
        # Convert to arrays
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.DataFrame):
            y_test_arr = y_test.values
        else:
            y_test_arr = y_test

        # Predict
        y_pred = self.predict(X_test)

        # Compute metrics
        from sklearn.metrics import precision_score, recall_score

        metrics = {
            "precision_micro": precision_score(
                y_test_arr, y_pred, average="micro", zero_division=0
            ),
            "recall_micro": recall_score(
                y_test_arr, y_pred, average="micro", zero_division=0
            ),
            "f1_micro": f1_score(y_test_arr, y_pred, average="micro", zero_division=0),
            "precision_macro": precision_score(
                y_test_arr, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_test_arr, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(y_test_arr, y_pred, average="macro", zero_division=0),
        }

        if verbose:
            print("\n=== Evaluation Results ===")
            print(
                pd.DataFrame.from_dict(
                    metrics, orient="index", columns=["Score"]
                ).round(4)
            )

            if label_names is not None:
                print("\n=== Classification Report ===")
                print(
                    classification_report(
                        y_test_arr, y_pred, target_names=label_names, zero_division=0
                    )
                )

        return metrics

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path (str): Path to save the model file (.pkl or .joblib).
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        save_obj = {
            "best_estimator": self.best_estimator_,
            "best_params": self.grid_search.best_params_ if self.grid_search else {},
            "best_score": self.grid_search.best_score_ if self.grid_search else None,
            "config": self.config,
        }

        joblib.dump(save_obj, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> "SVMModel":
        """Load a trained model from disk.

        Args:
            path (str): Path to the saved model file.

        Returns:
            self: Loaded model instance.
        """
        save_obj = joblib.load(path)

        self.best_estimator_ = save_obj["best_estimator"]
        self.config = save_obj.get("config", self.config)
        self.is_fitted = True

        print(f"Model loaded from {path}")
        if "best_params" in save_obj:
            print(f"Best params: {save_obj['best_params']}")
        if "best_score" in save_obj and save_obj["best_score"]:
            print(f"Best CV score: {save_obj['best_score']:.4f}")

        return self

    def get_feature_importance(self) -> None:
        """SVM does not provide direct feature importance.

        For linear kernel, coefficients can be extracted, but this is complex
        with OneVsRestClassifier. This method is a placeholder.
        """
        print("Feature importance not directly available for SVM.")
        print("For linear kernel, you can inspect model.estimators_[i].coef_")


if __name__ == "__main__":
    # Example usage
    print("SVM Model module loaded successfully.")
    print("Use: model = SVMModel() to initialize.")
