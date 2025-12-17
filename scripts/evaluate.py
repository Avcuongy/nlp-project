import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder

from preprocessing.clean import vn_text_clean
from preprocessing.tokenize import vn_word_tokenize
from preprocessing.remove_stopwords import remove_stopwords_wrapper
from utils.metrics import evaluate as eval_metrics
from model.svm import SVMModel


def preprocess_df(df: pd.DataFrame, text_col: str = "comment") -> pd.DataFrame:
    out = df.copy()
    out[text_col] = (
        out[text_col]
        .astype(str)
        .map(vn_text_clean)
        .map(lambda t: vn_word_tokenize(t, method="underthesea"))
    )
    out = remove_stopwords_wrapper(out, text_col=text_col)
    return out


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
        "--labels-path",
        type=str,
        default=None,
        help="Path to saved label classes JSON (defaults to models/labels.json)",
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
    labels_path = (
        Path(args.labels_path) if args.labels_path else root / "models" / "labels.json"
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

    # Resolve label classes (prefer labels.json saved during training)
    print("\n3. RESOLVING LABEL CLASSES...")
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            train_classes = json.load(f)
        print(f"\tUsing classes from labels.json: {len(train_classes)}")
    else:
        # Fallback to unique labels in eval data (alphabetical)
        print("\tWARNING: labels.json not found; inferring classes from eval data")
        train_classes = sorted(df["label"].astype(str).unique().tolist())
        # Persist inferred classes for consistency across runs
        try:
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(train_classes, f, ensure_ascii=False, indent=2)
            print(f"\tSaved inferred classes to {labels_path}")
        except Exception as e:
            print(f"\tCould not save inferred classes: {e}")

    # Prepare y_true as integer-encoded labels in the saved class order
    print("\n4. PREPARING GROUND TRUTH LABELS (single-label multiclass)...")
    le = LabelEncoder()
    le.classes_ = np.array(train_classes)
    y_true = le.transform(df["label"].astype(str))

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
        X = df[[text_col]].copy()
    else:
        X = preprocess_df(df[[text_col]], text_col=text_col)

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
        y_true=y_true, y_pred=y_pred, label_names=train_classes, verbose=True
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
