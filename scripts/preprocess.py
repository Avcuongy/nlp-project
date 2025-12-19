import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder

from preprocessing.clean import vn_text_clean
from preprocessing.tokenize import vn_word_tokenize
from preprocessing.remove_stopwords import remove_stopwords, remove_stopwords_wrapper


def preprocessing_df(df: pd.DataFrame, text_col: str = "comment") -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame.

    Args:
        df: Input DataFrame
        text_col: Name of the text column to preprocess
        method: Tokenization method (default: "underthesea")

    Returns:
        DataFrame with preprocessed text
    """
    df = df.copy()

    # Step 1: Clean text
    print("\tStep 1: Cleaning text...")
    df[text_col] = df[text_col].astype(str).apply(vn_text_clean)

    # Step 2: Tokenize
    print("\tStep 2: Tokenizing text...")
    df[text_col] = df[text_col].apply(vn_word_tokenize, method="underthesea")

    # Step 3: Remove stopwords with fallback
    print("\tStep 3: Removing stopwords...")
    # Create backup for fallback
    df["text_backup"] = df[text_col].copy()

    # Remove stopwords
    post = df[text_col].apply(remove_stopwords)

    # Normalize empties to NaN to enable fallback
    post = post.replace("", np.nan)

    # Fallback to backup text where stopword removal produced NaN
    df[text_col] = post.fillna(df["text_backup"])

    # Remove backup column
    df.drop(columns=["text_backup"], inplace=True)

    return df


def preprocess_and_save(
    input_path: str,
    output_path: str,
    text_col: str = "comment",
):
    """
    Load, preprocess, and save a dataset.
    Uses underthesea for tokenization.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed CSV file
        text_col: Name of the text column to preprocess
    """
    print(f"\n{'='*80}")
    print(f"PREPROCESSING: {input_path}".center(80))
    print(f"{'='*80}\n")

    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"Loaded {len(df)} records")

    # Preprocess
    df = preprocessing_df(df, text_col=text_col)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved processed data to {output_path}")


def main():
    """
    Main function to preprocess train and validation datasets.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess text data")
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to training data (default: data/raw/train.csv)",
    )
    parser.add_argument(
        "--val",
        type=str,
        default=None,
        help="Path to validation data (default: data/raw/val.csv)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to test data (default: data/raw/test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed)",
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

    train_path = args.train or str(root / "data" / "raw" / "train.csv")
    val_path = args.val or str(root / "data" / "raw" / "val.csv")
    test_path = args.test or str(root / "data" / "raw" / "test.csv")
    output_dir = args.output_dir or str(root / "data" / "processed")

    # Preprocess train
    preprocess_and_save(
        train_path,
        str(Path(output_dir) / "train.csv"),
        text_col=args.text_col,
    )

    # Preprocess validation
    preprocess_and_save(
        val_path,
        str(Path(output_dir) / "val.csv"),
        text_col=args.text_col,
    )

    # Preprocess test
    preprocess_and_save(
        test_path,
        str(Path(output_dir) / "test.csv"),
        text_col=args.text_col,
    )

    # Fit and save label encoder
    print("\n" + "=" * 80)
    print("Fitting and saving label encoder".center(80))
    print("=" * 80)

    # Load train data to fit label encoder
    train_processed = pd.read_csv(Path(output_dir) / "train.csv")
    le = LabelEncoder()
    le.fit(train_processed["label"].astype(str))

    # Save label encoder
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    label_encoder_path = models_dir / "label_encoder.pkl"
    dump(le, label_encoder_path)

    print(
        f"\nFitted label encoder with {len(le.classes_)} classes: {list(le.classes_)}"
    )
    print(f"Saved label encoder to {label_encoder_path}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
