from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier

from utils.config_loader import load_config
from utils.metrics import evaluate


class XGBoostModel:
    """XGBoost classifier for single-label multiclass classification with Optuna tuning.

    Attributes:
        config (dict): Configuration dictionary loaded from YAML.
        model (XGBClassifier): The underlying XGBoost classifier.
        study (optuna.Study): Optuna study instance for hyperparameter optimization.
        best_estimator_ (XGBClassifier): Best model from Optuna optimization.
        is_fitted (bool): Whether the model has been fitted.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize XGBoost model with configuration.

        Args:
            config_path (str | None): Path to config YAML file. If None,
                uses default config/ml/xgboost.yaml.
        """
        if config_path is None:
            root = Path(__file__).resolve().parents[2]
            config_path = str(root / "config" / "ml" / "xgboost.yaml")

        self.config = load_config(config_path)
        self.model: Optional[XGBClassifier] = None
        self.study: Optional[optuna.Study] = None
        self.best_estimator_: Optional[XGBClassifier] = None
        self.is_fitted = False

    def _build_model(self, trial: Optional[optuna.Trial] = None) -> XGBClassifier:
        """Build the XGBClassifier estimator with parameters from trial.

        Args:
            trial (optuna.Trial | None): Optuna trial for hyperparameter suggestion.

        Returns:
            XGBClassifier: Initialized model.
        """
        optuna_config = self.config.get("optuna", {})
        booster_config = optuna_config.get("booster", {})
        search_space = optuna_config.get("search_space", {})

        params = {
            "random_state": 42,
            "tree_method": booster_config.get("tree_method", "auto"),
            "objective": booster_config.get("objective", "multi:softprob"),
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
        }

        if trial is not None:
            # Suggest hyperparameters based on search space config
            for param_name, param_config in search_space.items():
                suggest_type = param_config.get("suggest", "float")

                if suggest_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config.get("low"),
                        param_config.get("high"),
                        step=param_config.get("step", 1),
                    )
                elif suggest_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config.get("low"),
                        param_config.get("high"),
                        log=param_config.get("log", False),
                    )
                elif suggest_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config.get("choices", [])
                    )

        return XGBClassifier(**params)

    def _objective(
        self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, cv_splits: list
    ) -> float:
        """Optuna objective function for hyperparameter optimization.

        Args:
            trial (optuna.Trial): Optuna trial instance.
            X: Training features.
            y: Training labels.
            cv_splits: List of (train_idx, val_idx) tuples for cross-validation.

        Returns:
            float: Mean F1-macro score across CV folds.
        """
        from sklearn.metrics import f1_score

        model = self._build_model(trial)
        scores = []

        for train_idx, val_idx in cv_splits:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            y_pred = model.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average="macro")
            scores.append(score)

        return np.mean(scores)

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame | pd.Series,
        verbose: bool = True,
    ) -> "XGBoostModel":
        """Train the XGBoost model with Optuna hyperparameter optimization.

        Args:
            X_train: Training features (vectorized text).
            y_train: Training labels (1D array-like of class ids or names).
            verbose (bool): Whether to print training progress.

        Returns:
            self: Fitted model instance.
        """
        from sklearn.model_selection import StratifiedKFold

        # Convert DataFrames to arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.values
        # Ensure 1D label vector
        y_train = np.ravel(y_train)

        # Get Optuna config
        optuna_config = self.config.get("optuna", {})
        study_name = optuna_config.get("study_name", "xgb_optimization")
        direction = optuna_config.get("direction", "maximize")
        n_trials = optuna_config.get("n_trials", 50)
        timeout = optuna_config.get("timeout", None)

        # Sampler config
        sampler_config = optuna_config.get("sampler", {})
        sampler_name = sampler_config.get("name", "tpe")
        sampler_seed = sampler_config.get("seed", 42)

        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        elif sampler_name == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=sampler_seed)
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=sampler_seed)

        # Pruner config
        pruner_config = optuna_config.get("pruner", {})
        pruner_name = pruner_config.get("name", "median")

        if pruner_name == "median":
            pruner = optuna.pruners.MedianPruner()
        elif pruner_name == "halving":
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        elif pruner_name == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.MedianPruner()

        # Create study
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        # Create CV splits
        cv_folds = 5
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_splits = list(skf.split(X_train, y_train))

        if verbose:
            print(f"Training XGBoost with Optuna optimization...")
            print(f"Study: {study_name}, Direction: {direction}")
            print(f"Trials: {n_trials}, CV folds: {cv_folds}")
            print(f"Sampler: {sampler_name}, Pruner: {pruner_name}")

        # Optimize
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, cv_splits),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
        )

        # Train final model with best parameters
        best_params = self.study.best_params
        self.best_estimator_ = XGBClassifier(
            **best_params,
            random_state=42,
            tree_method=optuna_config.get("booster", {}).get("tree_method", "auto"),
            objective=optuna_config.get("booster", {}).get(
                "objective", "multi:softprob"
            ),
            eval_metric="mlogloss",
            use_label_encoder=False,
        )

        self.best_estimator_.fit(X_train, y_train, verbose=False)
        self.is_fitted = True

        if verbose:
            print("\nBest parameters:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            print(f"\nBest trial value: {self.study.best_value:.4f}")
            print(f"Number of finished trials: {len(self.study.trials)}")

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
            "best_params": self.study.best_params if self.study else {},
            "best_score": self.study.best_value if self.study else None,
            "config": self.config,
        }

        joblib.dump(save_obj, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> "XGBoostModel":
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
            print(f"Best score: {save_obj['best_score']:.4f}")

        return self


if __name__ == "__main__":
    pass
