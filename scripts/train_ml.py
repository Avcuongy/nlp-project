import argparse
import json
from pathlib import Path
from joblib import load

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from model.svm import SVMModel
from sklearn.preprocessing import LabelEncoder


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

    # Transform labels
    print("\n2. ENCODING LABELS...")
    labels_str = df_train["label"].astype(str)
    le = LabelEncoder()
    y_train = le.fit_transform(labels_str)
    print(f"\tClasses: {list(le.classes_)}")

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
    # elif args.model == "logistic":
    #     model = LogisticModel(config_path=args.config)
    # elif args.model == "xgboost":
    #     model = XGBoostModel(config_path=args.config)
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

    # Save label classes for consistent evaluation
    shared_models_dir = root / "models"
    labels_json_path = shared_models_dir / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(list(le.classes_), f, ensure_ascii=False, indent=2)
    print(f"\n7. SAVING LABEL CLASSES TO {labels_json_path}...")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
