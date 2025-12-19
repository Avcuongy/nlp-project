from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from utils.config_loader import load_config
from utils.metrics import evaluate


class LogisticModel:
    """Logistic Regression classifier for single-label multiclass classification.

    Attributes:
        config (dict): Configuration dictionary loaded from YAML.
        model (LogisticRegression): The underlying Logistic Regression classifier.
        grid_search (GridSearchCV): GridSearchCV instance for hyperparameter tuning.
        best_estimator_ (LogisticRegression): Best model from grid search.
        is_fitted (bool): Whether the model has been fitted.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize Logistic Regression model with configuration.

        Args:
            config_path (str | None): Path to config YAML file. If None,
                uses default config/ml/logistic.yaml.
        """
        if config_path is None:
            root = Path(__file__).resolve().parents[2]
            config_path = str(root / "config" / "ml" / "logistic.yaml")

        self.config = load_config(config_path)
        self.model: Optional[LogisticRegression] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.best_estimator_: Optional[LogisticRegression] = None
        self.is_fitted = False

    def _build_model(self) -> LogisticRegression:
        """Build the LogisticRegression estimator.

        Returns:
            LogisticRegression: Initialized model with default parameters.
        """
        return LogisticRegression(max_iter=1000, random_state=42)

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

            # Filter incompatible solver-penalty combinations
            if key == "penalty" or key == "solver":
                param_grid[key] = values
            else:
                param_grid[key] = values

        return param_grid

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame | pd.Series,
        verbose: bool = True,
    ) -> "LogisticModel":
        """Train the Logistic Regression model with GridSearchCV.

        Args:
            X_train: Training features (vectorized text).
            y_train: Training labels (1D array-like of class ids or names).
            verbose (bool): Whether to print training progress.

        Returns:
            self: Fitted model instance.
        """
        # Convert DataFrames to arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.values
        # Ensure 1D label vector
        y_train = np.ravel(y_train)

        # Build model and param grid
        self.model = self._build_model()
        param_grid = self._build_param_grid()

        # Get GridSearch config
        gs_config = self.config.get("grid_search", {})
        scoring_name = gs_config.get("scoring", "f1_macro")
        cv = gs_config.get("cv", 5)
        n_jobs = gs_config.get("n_jobs", -1)

        # Make CV deterministic with a seeded splitter if cv is an int
        if isinstance(cv, int):
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_splitter = cv

        # Initialize GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=scoring_name,
            cv=cv_splitter,
            n_jobs=n_jobs,
            verbose=2 if verbose else 0,
            error_score="raise",
        )

        if verbose:
            print("Training Logistic Regression with GridSearchCV...")
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
            print(
                f"\nBest CV score ({scoring_name}): {self.grid_search.best_score_:.4f}"
            )

        return self

    def evaluate(
        self,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.DataFrame | pd.Series,
        label_names: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model on test data using shared metrics utilities.

        Args:
            X_test: Test features (vectorized text).
            y_test: True test labels.
            label_names (list[str] | None): Optional list of label names for reporting.
            verbose (bool): Whether to print evaluation results.

        Returns:
            dict: Dictionary of metrics (accuracy, precision, recall, f1, etc.).
        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        y_pred = self.predict(X_test)
        return evaluate(
            y_true=y_test, y_pred=y_pred, label_names=label_names, verbose=verbose
        )

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict labels for input samples.

        Args:
            X: Input features (vectorized text).

        Returns:
            np.ndarray: 1D array of predicted class ids or names.

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

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for input samples.

        Args:
            X: Input features (vectorized text).

        Returns:
            np.ndarray: Array of shape (n_samples, n_classes) with probabilities.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted or self.best_estimator_ is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.best_estimator_.predict_proba(X)

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path (str): Path to save the model file (.pkl or .joblib).

        Raises:
            ValueError: If model has not been fitted.
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

    def load(self, path: str) -> "LogisticModel":
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


if __name__ == "__main__":
    pass
