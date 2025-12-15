import argparse
from pathlib import Path

import pandas as pd

from src.model.svm import SVMModel
from src.preprocessing.vectorize import build_tfidf_vectorizer
from src.utils.other import matrix_labels


def main():
    parser = argparse.ArgumentParser(description="Train ML models for ABSA")
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
    train_path = root / "data" / "processed" / "train.csv"
    val_path = root / "data" / "processed" / "val.csv"

    print("=" * 80)
    print(f"Training {args.model.upper()} Model")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_val = pd.read_csv(val_path, encoding="utf-8")
    print(f"   Train size: {len(df_train)}, Val size: {len(df_val)}")

    # Transform labels
    print("\n2. Transforming labels to binary matrix...")
    matrix_labels_train, mlb_train = matrix_labels(df_train[["label"]])
    matrix_labels_val, mlb_val = matrix_labels(df_val[["label"]])
    print(f"   Number of labels: {len(mlb_train.classes_)}")
    print(f"   Labels: {list(mlb_train.classes_)}")

    # Prepare train/val splits
    X_train = df_train[["comment"]]
    y_train = matrix_labels_train

    X_val = df_val[["comment"]]
    y_val = matrix_labels_val

    # Vectorize text
    print("\n3. Vectorizing text with TF-IDF...")
    vec = build_tfidf_vectorizer()
    X_train_vec = vec.fit_transform(X_train["comment"])
    X_val_vec = vec.transform(X_val["comment"])
    print(f"   Train shape: {X_train_vec.shape}")
    print(f"   Val shape: {X_val_vec.shape}")
    print(f"   Vocabulary size: {len(vec.get_feature_names_out())}")

    # Initialize and train model
    print(f"\n4. Training {args.model.upper()} model...")
    if args.model == "svm":
        model = SVMModel(config_path=args.config)
    # elif args.model == "logistic":
    #     model = LogisticModel(config_path=args.config)
    # elif args.model == "xgboost":
    #     model = XGBoostModel(config_path=args.config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model.fit(X_train_vec, y_train, verbose=True)

    # Evaluate
    print("\n5. Evaluating on validation set...")
    metrics = model.evaluate(
        X_val_vec, y_val, label_names=y_train.columns.tolist(), verbose=True
    )

    # Save model
    if args.save_path:
        save_path = args.save_path
    else:
        model_dir = root / "models" / "ml"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(model_dir / f"{args.model}_model.pkl")

    print(f"\n6. Saving model to {save_path}...")
    model.save(save_path)

    print("\n" + "=" * 80)
    print("Training completed successfull !")
    print("=" * 80)


if __name__ == "__main__":
    main()
