import argparse
import json
from pathlib import Path
from joblib import dump

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from model.svm import SVMModel
from preprocessing.vectorize import build_tfidf_vectorizer
from preprocessing.clean import vn_text_clean
from preprocessing.tokenize import vn_word_tokenize
from preprocessing.remove_stopwords import remove_stopwords_wrapper
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

    # Read raw data for training to follow the standard pipeline
    train_path = root / "data" / "raw" / "train.csv"

    print("=" * 80)
    print(f"TRAINING {args.model.upper()} MODEL".center(80))
    print("=" * 80)

    # Load data
    print("\n1. LOANDING DATA...")
    df_train = pd.read_csv(train_path, encoding="utf-8")
    print(f"\tTrain size: {len(df_train)}")

    # Transform labels
    print("\n2. ENCODING SINGLE LABEL COLUMN...")
    labels_str = df_train["label"].astype(str)
    le = LabelEncoder()
    y_train = le.fit_transform(labels_str)
    print(f"\tClasses: {list(le.classes_)}")

    # Prepare train/val splits
    X_train = df_train[["comment"]].copy()

    # Preprocess text
    print("\n3 PREPROCESSING TEXT...")

    # Clean and tokenize first, then apply stopword removal via DataFrame util
    X_train["comment"] = (
        X_train["comment"]
        .astype(str)
        .map(vn_text_clean)
        .map(lambda t: vn_word_tokenize(t, method="underthesea"))
    )
    X_train["comment"] = remove_stopwords_wrapper(X_train, text_col="comment")
    # Persist processed training data for downstream evaluation/pipelines
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_train_path = processed_dir / "train.csv"

    # Combine processed text with original string label for downstream use
    processed_df = pd.DataFrame(
        {
            "comment": X_train["comment"].reset_index(drop=True),
            "label": labels_str.reset_index(drop=True),
        }
    )
    processed_df.to_csv(processed_train_path, index=False, encoding="utf-8")

    # Vectorize
    print("\n4. VECTORIZING TEXT WITH TF-IDF...")
    vec = build_tfidf_vectorizer()
    X_train_vec = vec.fit_transform(X_train["comment"])
    print(f"\tTrain shape: {X_train_vec.shape}")
    print(f"\tVocabulary size: {len(vec.get_feature_names_out())}")

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

    # Save a shared vectorizer for ML models
    shared_models_dir = root / "models"
    shared_models_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = shared_models_dir / "vectorizer.pkl"

    print(f"\n7. SAVING VECTORIZER TO {vectorizer_path}...")
    dump(vec, vectorizer_path)

    # Save label classes for consistent evaluation
    labels_json_path = shared_models_dir / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(list(le.classes_), f, ensure_ascii=False, indent=2)
    print(f"\n8. SAVING LABEL CLASSES TO {labels_json_path}...")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
