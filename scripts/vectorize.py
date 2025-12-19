import argparse
from pathlib import Path
from joblib import dump

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from preprocessing.vectorize import build_tfidf_vectorizer


def build_and_save_vectorizer(
    train_path: str, output_path: str, text_col: str = "comment"
):
    """
    Build TF-IDF vectorizer from training data and save it.

    Args:
        train_path: Path to processed training CSV file
        output_path: Path to save vectorizer (pkl file)
        text_col: Name of text column
    """
    print(f"\n{'='*80}")
    print("BUILDING TF-IDF VECTORIZER".center(80))
    print(f"{'='*80}\n")

    # Load processed training data
    print(f"1. Loading processed data from {train_path}...")
    df_train = pd.read_csv(train_path, encoding="utf-8")
    print(f"\tLoaded {len(df_train)} records")

    # Build and fit vectorizer
    print("\n2. Building and fitting TF-IDF vectorizer...")
    vec = build_tfidf_vectorizer()
    X_train_vec = vec.fit_transform(df_train[text_col])
    print(f"\tTrain shape: {X_train_vec.shape}")
    print(f"\tVocabulary size: {len(vec.get_feature_names_out())}")

    # Save vectorizer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n3. Saving vectorizer to {output_path}...")
    dump(vec, output_path)

    print(f"\n{'='*80}")
    print("VECTORIZER SAVED SUCCESSFULLY".center(80))
    print(f"{'='*80}\n")


def main():
    """
    Main function to build and save vectorizer.
    """
    parser = argparse.ArgumentParser(description="Build and save TF-IDF vectorizer")
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to processed training data (default: data/processed/train.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for vectorizer (default: models/vectorizer.pkl)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="comment",
        help="Name of text column (default: comment)",
    )

    args = parser.parse_args()

    # Setup paths
    root = Path(__file__).resolve().parents[1]

    train_path = args.train or str(root / "data" / "processed" / "train.csv")
    output_path = args.output or str(root / "models" / "vectorizer.pkl")

    # Build and save vectorizer
    build_and_save_vectorizer(
        train_path=train_path, output_path=output_path, text_col=args.text_col
    )


if __name__ == "__main__":
    main()
