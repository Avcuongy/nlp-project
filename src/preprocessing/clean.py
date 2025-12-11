import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from utils.common import load_json


@lru_cache(maxsize=1)
def load_abbrev(path: str | None = None) -> dict[str, str]:
    """Load and cache abbreviation mapping from JSON.

    If no path is given, resolve from project root: data/external/abbreviation.json.
    Returns an empty dict if file is missing or invalid.
    """
    try:
        if path is None:
            root = Path(__file__).resolve().parents[2]
            path = str(root / "data" / "external" / "abbreviation.json")
        data = load_json(path)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def apply_abbrev(text: str, abbrev: dict[str, str]) -> str:
    """Apply abbreviation replacements to text using word-boundary safe regex."""
    for k, v in abbrev.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text


def vn_text_clean(text: str) -> str:
    # Normalize Vietnamese text
    text = unicodedata.normalize("NFC", text)
    
    # Lowercase
    text = text.lower()

    # Remove emails (including cases like @gmail.com)
    text = re.sub(r"\b(?:[A-Za-z0-9._%+-]+)?@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)

    # Remove URLs
    text = re.sub(r"\b(?:https?://|www\.)\S+\b", " ", text)

    # Normalization of repeated punctuation and characters
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[-~_*]{2,}", " ", text)  # Remove repeated special characters
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # Remove repeated characters

    # Normalize slang/abbreviations
    abbrev = load_abbrev()
    text = apply_abbrev(text, abbrev)

    # Remove numbers
    text = re.sub(r"\b\d+[\d.,]*\b", " ", text)

    # Remove unnecessary characters
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    sample = "Máy dùng okkkkk!!! pin .trâuuuu http://example.com hgg@gmail.com wf mạnh lắm~~~ không như mấy con đt khác..... con cá."
    print(vn_text_clean(sample))
