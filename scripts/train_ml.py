import argparse
from pathlib import Path
from joblib import load

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from utils.common import set_seed
from utils.config_loader import load_config
import joblib


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
    val_path = root / "data" / "processed" / "val.csv"
    vectorizer_path = root / "models" / "vectorizer.pkl"

    # Load config
    if args.config is None:
        config_path = root / "config" / "ml" / f"{args.model}.yaml"
    else:
        config_path = Path(args.config)

    config = load_config(str(config_path))
    params = config.get("parameters", {})

    print("=" * 80)
    print(f"TRAINING {args.model.upper()} MODEL".center(80))
    print("=" * 80)
    print(f"\nConfig: {config_path}")

    # Load processed data
    print("\n1. LOADING PROCESSED DATA...")
    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_val = pd.read_csv(val_path, encoding="utf-8")
    print(f"\tTrain size: {len(df_train)}")
    print(f"\tValidation size: {len(df_val)}")

    # Load label encoder
    print("\n2. LOADING LABEL ENCODER...")
    label_encoder_path = root / "models" / "label_encoder.pkl"
    le = load(label_encoder_path)
    print(f"\tLoaded from {label_encoder_path}")
    print(f"\tClasses: {list(le.classes_)}")

    # Transform labels
    labels_str = df_train["label"].astype(str)
    y_train = le.transform(labels_str)
    
    val_labels_str = df_val["label"].astype(str)
    y_val = le.transform(val_labels_str)

    # Load vectorizer
    print("\n3. LOADING VECTORIZER...")
    print(f"\tLoading from {vectorizer_path}...")
    vec = load(vectorizer_path)
    print(f"\tVocabulary size: {len(vec.get_feature_names_out())}")

    # Vectorize text
    print("\n4. VECTORIZING TEXT...")
    X_train_vec = vec.transform(df_train["comment"])
    X_val_vec = vec.transform(df_val["comment"])
    print(f"\tTrain shape: {X_train_vec.shape}")
    print(f"\tValidation shape: {X_val_vec.shape}")

    # Initialize and train model with specified parameters
    print(f"\n5. TRAINING {args.model.upper()} MODEL...")

    if args.model == "svm":
        model = SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            random_state=42,
            verbose=True,
        )
    elif args.model == "logistic":
        model = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            penalty=params.get("penalty", "l2"),
            solver=params.get("solver", "lbfgs"),
            random_state=42,
            verbose=1,
        )
    elif args.model == "xgboost":
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.3),
                n_estimators=params.get("n_estimators", 100),
                subsample=params.get("subsample", 1.0),
                colsample_bytree=params.get("colsample_bytree", 1.0),
                random_state=42,
                verbosity=1,
            )
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model.fit(X_train_vec, y_train)

    # Evaluate on validation set
    print(f"\n6. EVALUATING {args.model.upper()} MODEL...")
    y_val_pred = model.predict(X_val_vec)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"\n\tValidation Accuracy: {val_accuracy:.4f}")
    print("\n\tClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_, digits=4))

    # Save model
    if args.save_path:
        save_path = args.save_path
    else:
        model_dir = root / "models" / "ml"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(model_dir / f"{args.model}.pkl")

    print(f"\n7. SAVING MODEL TO {save_path}...")
    joblib.dump(model, save_path)
    print("\tModel saved successfully!")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
