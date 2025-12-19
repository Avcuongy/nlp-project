import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from scripts.preprocess import preprocessing_df
from utils.metrics import evaluate as eval_metrics
from model.svm import SVMModel


def load_model(model_name: str, model_path: Path):
    if model_name == "svm":
        m = SVMModel()
        m.load(str(model_path))
        return m
    if model_name == "logistic":
        try:
            from model.logistic import LogisticModel  # type: ignore

            m = LogisticModel()
            m.load(str(model_path))
            return m
        except Exception:
            raise ValueError("Logistic model not implemented or load failed")
    if model_name == "xgboost":
        try:
            from model.xgboost import XGBoostModel  # type: ignore

            m = XGBoostModel()
            m.load(str(model_path))
            return m
        except Exception:
            raise ValueError("XGBoost model not implemented or load failed")
    raise ValueError("Unsupported model type")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML model on a dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=["svm", "logistic", "xgboost"],
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model .pkl (defaults to models/ml/<model>.pkl)",
    )
    parser.add_argument(
        "--vectorizer-path",
        type=str,
        default=None,
        help="Path to shared vectorizer .pkl (defaults to models/vectorizer.pkl)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV data for evaluation (defaults to data/processed/val.csv)",
    )
    parser.add_argument(
        "--label-encoder-path",
        type=str,
        default=None,
        help="Path to saved label encoder .pkl (defaults to models/label_encoder.pkl)",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Set if the data file is already preprocessed (skip text cleaning)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    model_path = (
        Path(args.model_path)
        if args.model_path
        else root / "models" / "ml" / f"{args.model}.pkl"
    )
    vectorizer_path = (
        Path(args.vectorizer_path)
        if args.vectorizer_path
        else root / "models" / "vectorizer.pkl"
    )
    data_path = (
        Path(args.data_path)
        if args.data_path
        else root / "data" / "processed" / "val.csv"
    )
    label_encoder_path = (
        Path(args.label_encoder_path)
        if args.label_encoder_path
        else root / "models" / "label_encoder.pkl"
    )

    print("=" * 80)
    print("EVALUATION".center(80))
    print("=" * 80)

    # Load data
    print("\n1. LOADING DATA...")
    df = pd.read_csv(data_path, encoding="utf-8")
    print(f"\tSamples: {len(df)}")

    # Load model so we can use stored classes
    print(f"\n2. LOADING MODEL FROM {model_path}...")
    model = load_model(args.model, model_path)

    # Load label encoder
    print("\n3. LOADING LABEL ENCODER...")
    if label_encoder_path.exists():
        le = joblib.load(label_encoder_path)
        print(f"\tLoaded from {label_encoder_path}")
        print(f"\tClasses: {list(le.classes_)}")
    else:
        raise FileNotFoundError(
            f"Label encoder not found at {label_encoder_path}. "
            "Please run preprocess.py first to generate label_encoder.pkl"
        )

    # Prepare y_true as integer-encoded labels
    print("\n4. PREPARING TRUTH LABELS...")
    y_true = le.transform(df["label"].astype(str))
    print(f"\tEncoded {len(y_true)} labels")

    # Determine text column name
    if "comment" in df.columns:
        text_col = "comment"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise KeyError(
            "Input CSV must contain a text column named 'comment' or 'text'. "
            f"Found columns: {list(df.columns)}"
        )

    # Preprocess text (skip if already preprocessed)
    print("\n5. PREPROCESSING TEXT...")
    if args.preprocessed:
        print("\tSkipping preprocessing (data already preprocessed)")
        X = df[[text_col]].copy()
    else:
        X = preprocessing_df(df[[text_col]], text_col=text_col)

    # Load vectorizer
    print(f"\n6. LOADING VECTORIZER FROM {vectorizer_path}...")
    vec = joblib.load(vectorizer_path)

    # Vectorize
    print("\n7. VECTORIZE...")
    X_vec = vec.transform(X[text_col])
    print(f"\tEval shape: {X_vec.shape}")

    # Predict
    print("\n8. PREDICTING...")
    y_pred = model.predict(X_vec)

    # Evaluate
    print("\n9. EVALUATING...")
    metrics = eval_metrics(
        y_true=y_true, y_pred=y_pred, label_names=list(le.classes_), verbose=True
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
