from __future__ import annotations

from functools import lru_cache
from pathlib import Path


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


if __name__ == "__main__":
    sample = "Vì_vậy tôi muốn hỏi bạn về vấn_đề này và ngay_bây_giờ."
    print(remove_stopwords(sample))
