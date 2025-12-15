from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

from src.preprocessing.vectorize import build_tfidf_vectorizer
from src.utils.common import ensure_dir, get_logger, save_pickle, set_seed
from src.utils.config_loader import load_project_config, load_yaml_file
from src.utils.metrics import get_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML models (class-based). Currently SVM implemented."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("config/config.yaml")),
        help="Path to project config YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["svm", "logistic", "xgboost"],
        help="Override model family (default from config.experiment.model_family)",
    )
    return parser.parse_args()


def _parse_label_cell(cell: str) -> List[str]:
    if not isinstance(cell, str):
        return []
    labels: List[str] = []
    for tok in cell.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("{") and tok.endswith("}"):
            tok = tok[1:-1]
        tok = tok.strip()
        if tok:
            labels.append(tok)
    return labels


def _build_all_label_space(cfg: dict) -> List[str]:
    aspects = cfg.get("dataset", {}).get("aspects", [])
    polarities = cfg.get("dataset", {}).get("polarities", [])
    aspect_keys = [str(a.get("key", "")).upper() for a in aspects]
    pols = [str(p).capitalize() for p in polarities]
    return [f"{a}#{p}" for a in aspect_keys for p in pols]


def _load_csv_split(
    path: str | Path, text_col: str, label_col: str
) -> Tuple[List[str], List[List[str]]]:
    df = pd.read_csv(path)
    texts = df[text_col].astype(str).tolist()
    labels = [
        _parse_label_cell(v) if pd.notna(v) else []
        for v in df[label_col].astype(str).tolist()
    ]
    return texts, labels


class MLTrainer:
    def __init__(self, cfg: dict, model_family: str) -> None:
        self.cfg = cfg
        self.model_family = model_family
        self.logger = get_logger(f"train_{model_family}")
        set_seed(int(cfg.get("project", {}).get("seed", 42)))

        ds = cfg.get("dataset", {})
        if ds.get("file_type", "csv") != "csv":
            raise ValueError("This training script expects CSV files.")

        self.text_col = ds.get("text_column", "comment")
        self.label_col = ds.get("label_column", "label")

        self.train_path = Path(ds.get("train_file", "data/processed/train.csv"))
        self.val_path = Path(ds.get("val_file", "data/processed/val.csv"))

        self.classes = _build_all_label_space(cfg)
        self.mlb = MultiLabelBinarizer(classes=self.classes)
        self.vectorizer = build_tfidf_vectorizer()

        paths = cfg.get("paths", {})
        self.models_root = ensure_dir(Path(paths.get("models", "models")))
        self.models_ml = ensure_dir(Path(paths.get("models_ml", "models/ml")))

    # Data
    def load_data(self) -> None:
        self.logger.info(f"Loading train CSV: {self.train_path}")
        self.X_train_texts, self.y_train_labels = _load_csv_split(
            self.train_path, self.text_col, self.label_col
        )

        self.X_val_texts: List[str] = []
        self.y_val_labels: List[List[str]] = []
        if self.val_path.exists():
            self.logger.info(f"Loading val CSV: {self.val_path}")
            self.X_val_texts, self.y_val_labels = _load_csv_split(
                self.val_path, self.text_col, self.label_col
            )
        else:
            self.logger.warning(
                "Validation file not found; training without val metrics."
            )

    def prepare(self) -> None:
        self.Y_train = self.mlb.fit_transform(self.y_train_labels)
        self.X_train = self.vectorizer.fit_transform(self.X_train_texts)

    # Model specific
    def build_estimator(self):  # pragma: no cover - abstract-like
        raise NotImplementedError

    def build_search(self, estimator):  # pragma: no cover - abstract-like
        raise NotImplementedError

    # Train/Eval
    def train(self) -> None:
        estimator = self.build_estimator()
        search = self.build_search(estimator)
        self.logger.info(f"Fitting {self.model_family} GridSearchCV...")
        search.fit(self.X_train, self.Y_train)
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_

    def evaluate(self) -> dict:
        metrics = {}
        if getattr(self, "X_val_texts", None):
            X_val = self.vectorizer.transform(self.X_val_texts)
            Y_val_true = self.mlb.transform(self.y_val_labels)
            Y_val_pred = self.best_estimator_.predict(X_val)
            metrics = {
                "f1_micro": float(
                    f1_score(Y_val_true, Y_val_pred, average="micro", zero_division=0)
                ),
                "f1_macro": float(
                    f1_score(Y_val_true, Y_val_pred, average="macro", zero_division=0)
                ),
            }
            self.logger.info(
                f"Validation f1_micro={metrics['f1_micro']:.4f} f1_macro={metrics['f1_macro']:.4f}"
            )
        return metrics

    def save(self, metrics: dict | None = None) -> None:
        # Common vectorizer at models/vectorizer.pkl
        save_pickle(self.models_root / "vectorizer.pkl", self.vectorizer)

        # Model family specific filename under models/ml/
        out_path = self.models_ml / f"{self.model_family}.pkl"
        # Save dict to include label binarizer alongside model
        payload = {
            "model": self.best_estimator_,
            "label_binarizer": self.mlb,
            "best_params": getattr(self, "best_params_", {}),
            "classes": list(self.classes),
            "metrics": metrics or {},
        }
        save_pickle(out_path, payload)
        self.logger.info(f"Saved: {out_path} and vectorizer.pkl")


class SVMTrainer(MLTrainer):
    def build_estimator(self):
        # Parallel across classes and CV folds
        gs_cfg = self._grid_cfg()
        n_jobs = int(gs_cfg.get("n_jobs", -1))
        base = SVC()
        return OneVsRestClassifier(base, n_jobs=n_jobs)

    def _grid_cfg(self) -> dict:
        path = self.cfg.get("configs", {}).get("svm")
        return load_yaml_file(path) if path else {"grid_search": {}}

    def build_search(self, estimator):
        cfg = self._grid_cfg().get("grid_search", {})
        scoring = get_scorer(cfg.get("scoring", "f1_micro"))
        cv = int(cfg.get("cv", 5))
        n_jobs = int(cfg.get("n_jobs", -1))
        # Map param grid keys to OneVsRest(estimator__) namespace
        raw_grid = cfg.get(
            "param_grid", {"C": [1.0], "kernel": ["linear"], "gamma": ["scale"]}
        )
        param_grid = {f"estimator__{k}": v for k, v in raw_grid.items()}
        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
        )


class LogisticTrainer(MLTrainer):
    def build_estimator(self):  # pragma: no cover - placeholder
        raise NotImplementedError("Logistic Regression trainer not implemented yet.")

    def build_search(self, estimator):  # pragma: no cover - placeholder
        raise NotImplementedError("Logistic Regression trainer not implemented yet.")


class XGBoostTrainer(MLTrainer):
    def build_estimator(self):  # pragma: no cover - placeholder
        raise NotImplementedError("XGBoost trainer not implemented yet.")

    def build_search(self, estimator):  # pragma: no cover - placeholder
        raise NotImplementedError("XGBoost trainer not implemented yet.")


def main() -> None:
    args = parse_args()
    cfg = load_project_config(args.config)
    model_family = args.model or cfg.get("experiment", {}).get("model_family", "svm")

    if model_family == "svm":
        trainer: MLTrainer = SVMTrainer(cfg, model_family)
    elif model_family == "logistic":
        trainer = LogisticTrainer(cfg, model_family)
    elif model_family == "xgboost":
        trainer = XGBoostTrainer(cfg, model_family)
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")

    trainer.load_data()
    trainer.prepare()
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save(metrics)


if __name__ == "__main__":
    main()
