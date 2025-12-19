import argparse
from pathlib import Path
from joblib import load

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from model.svm import SVMModel
from model.logistic import LogisticModel
from model.xgboost import XGBoostModel
from sklearn.preprocessing import LabelEncoder
from utils.common import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=["svm", "logistic", "xgboost"],
        help="Model type to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config file (optional)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save trained model (optional)",
    )

    args = parser.parse_args()

    # Ensure reproducibility across runs
    set_seed(42)

    # Setup paths
    root = Path(__file__).resolve().parents[1]

    # Read processed data
    train_path = root / "data" / "processed" / "train.csv"
    vectorizer_path = root / "models" / "vectorizer.pkl"

    print("=" * 80)
    print(f"TRAINING {args.model.upper()} MODEL".center(80))
    print("=" * 80)

    # Load processed data
    print("\n1. LOADING PROCESSED DATA...")
    df_train = pd.read_csv(train_path, encoding="utf-8")
    print(f"\tTrain size: {len(df_train)}")

    # Load label encoder
    print("\n2. LOADING LABEL ENCODER...")
    label_encoder_path = root / "models" / "label_encoder.pkl"
    le = load(label_encoder_path)
    print(f"\tLoaded from {label_encoder_path}")
    print(f"\tClasses: {list(le.classes_)}")

    # Transform labels
    labels_str = df_train["label"].astype(str)
    y_train = le.transform(labels_str)

    # Load vectorizer
    print("\n3. LOADING VECTORIZER...")
    print(f"\tLoading from {vectorizer_path}...")
    vec = load(vectorizer_path)
    print(f"\tVocabulary size: {len(vec.get_feature_names_out())}")

    # Vectorize text
    print("\n4. VECTORIZING TEXT...")
    X_train_vec = vec.transform(df_train["comment"])
    print(f"\tTrain shape: {X_train_vec.shape}")

    # Initialize and train model
    print(f"\n5. TRAINING {args.model.upper()} MODEL...")

    if args.model == "svm":
        model = SVMModel(config_path=args.config)
    elif args.model == "logistic":
        model = LogisticModel(config_path=args.config)
    elif args.model == "xgboost":
        model = XGBoostModel(config_path=args.config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model.fit(X_train_vec, y_train, verbose=True)

    # Save model
    if args.save_path:
        save_path = args.save_path
    else:
        model_dir = root / "models" / "ml"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(model_dir / f"{args.model}.pkl")

    print(f"\n6. SAVING MODEL TO {save_path}...")
    model.save(save_path)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
