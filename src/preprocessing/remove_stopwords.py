from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd


@lru_cache(maxsize=1)
def _load_stopwords(path: str | None = None) -> tuple[set[str], set[str]]:
    """Load Vietnamese stopwords from text file.

    Args:
        path (str | None, optional): Path to stopwords file. Defaults to None.

    Returns:
        tuple[set[str], set[str]]: (singles, phrases) sets of stopwords.
    """
    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = str(root / "data" / "external" / "vietnamese-stopwords.txt")

    singles: set[str] = set()
    phrases: set[str] = set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().lower()
                if not s:
                    continue
                normalized = "_".join(s.split())
                if "_" in normalized:
                    phrases.add(normalized)
                else:
                    singles.add(normalized)
    except FileNotFoundError:
        pass

    return singles, phrases


def remove_stopwords(text: str) -> str:
    """Remove Vietnamese stopwords from space-tokenized text and rejoin.

    Args:
        text (str): Input text with tokens separated by spaces.

    Returns:
        str: Text with stopwords removed.
    """
    singles, phrases = _load_stopwords()

    # Split exactly by a single space per requirement
    tokens = text.split(" ")

    # Build removal set (underscore-normalized entries)
    remove_set = singles | phrases

    # Filter tokens; compare in lowercase
    kept = [t for t in tokens if t and t.lower() not in remove_set]

    return " ".join(kept).strip()


def remove_stopwords_df(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """Apply stopword removal on a DataFrame column and recover empties.

    Strategy: create an internal backup Series of the original text, apply
    stopword removal, and if a row becomes empty (after strip), restore the
    original text from the backup by index. No additional columns are created.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Name of the text column to process. Defaults to "text".

    Returns:
        pd.DataFrame: Single-column DataFrame containing only the updated `text_col`.
    """
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in DataFrame")

    out = df.copy()

    # Backup original text column
    backup = out[text_col].copy()

    # Apply removal safely (handle NaN)
    def _apply(x: str | float) -> str:
        if pd.isna(x):
            return ""
        return remove_stopwords(str(x))

    out[text_col] = out[text_col].apply(_apply)

    # Identify rows that became empty ("" or whitespace only)
    empties = out[text_col].astype(str) == ""

    # Restore from internal backup for those rows
    if empties.any():
        out.loc[empties, text_col] = backup.loc[empties]

    return out[[text_col]]


if __name__ == "__main__":
    sample = "Vì_vậy tôi muốn hỏi bạn về vấn_đề này và ngay_bây_giờ."
    print(remove_stopwords(sample))
    print(remove_stopwords(" ".join(sample.split())))
    print(
        remove_stopwords_df(
            pd.DataFrame(
                {
                    "text": [
                        "Vì_vậy tôi muốn hỏi bạn về vấn_đề này và ngay_bây_giờ.",
                        "Cảm ơn bạn rất nhiều!",
                        "Một câu chứa toàn stopwords vì vậy và về này.",
                        "rất",
                        "laptop",
                    ],
                }
            )
        )
    )
