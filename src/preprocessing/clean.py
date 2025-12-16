import re
import unicodedata
from functools import lru_cache
from pathlib import Path
import re
import unicodedata


@lru_cache(maxsize=1)
def load_abbrev(path: str | None = None) -> dict[str, str]:
    """Load and cache abbreviation mapping from a .txt file.

    Only supports text format where each line is "key: value" (split by the
    first colon). Leading/trailing spaces are trimmed. Lines without a colon
    will be split by the first whitespace. Blank or comment lines are ignored.

    Args:
        path (str | None, optional): Path to abbreviation .txt file. Defaults to
            `data/external/abbreviation.txt`.

    Returns:
        dict[str, str]: Mapping of abbreviations to their expansions.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        if path is None:
            path = str(root / "data" / "external" / "abbreviation.txt")

        p = Path(path)
        if p.suffix.lower() != ".txt" or not p.exists():
            return {}

        mapping: dict[str, str] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if ":" in raw:
                    k, v = raw.split(":", 1)
                else:
                    parts = raw.split(None, 1)
                    if len(parts) != 2:
                        continue
                    k, v = parts
                k = k.strip()
                v = v.strip()
                if k and v:
                    mapping[str(k)] = str(v)
        return mapping
    except Exception:
        return {}


def apply_abbrev(text: str, abbrev: dict[str, str]) -> str:
    """Apply abbreviation replacements to text using word-boundary safe regex.

    Args:
        text (str): Input text.
        abbrev (dict[str, str]): Mapping of abbreviations to expansions.

    Returns:
        str: Text with abbreviations replaced.
    """
    for k, v in abbrev.items():
        # Skip empty keys defensively
        if not k:
            continue
        pattern = rf"\b{re.escape(k)}\b"
        try:
            text = re.sub(pattern, v, text)
        except re.error:
            # If a malformed key causes regex compilation to fail, skip it
            continue
    return text


def vn_text_clean(text: str) -> str:
    """Clean Vietnamese text.

    Args:
        text (str): Input Vietnamese text.

    Returns:
        str: Cleaned text.
    """
    # Normalize Vietnamese text
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Normalize line breaks to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # If a word was split across lines with a hyphen, replace with a single space
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", " ", text)

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

    # Collapse all whitespace (including newlines) to single spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    sample = """Mình đang dùng xe Xforce Ultimate. Lấy xe từ 20/11/24. Sau thời gian sử dụng mình thấy có ưu và nhược như này: 
            Nhược điểm: 
            - Camera lùi không được nét. auto@gmail.com
            - Nội thất chỉ có màu đen. Nhìn sẽ hơi tối. Bẩn nhanh lộ. 
            - Sau khoảng 5000km đi thì bị kêu khá to khi rà phanh đi chậm. Đã liên hệ hãng. 
            Hãng đang báo sẽ thay đĩa phanh. 
            Mình đang đợi phanh về để thay. Chưa thay nhưng đợt này đi lại không kêu nữa.
            lol cái ni trông xịn vl
            Liên hệ web: https://example.com/test?query=abc
            Giá xe hơi cao so với mặt bằng chung!!!            Liên hệ: 0123-456-789"""
    print(vn_text_clean(sample))
