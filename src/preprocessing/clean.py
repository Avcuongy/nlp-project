import re
import unicodedata
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

from utils.common import load_json


@lru_cache(maxsize=1)
def load_abbrev(path: str | None = None) -> dict[str, str]:
    """Load and cache abbreviation mapping from a JSON-formatted file.

    Defaults to `data/external/abbreviation.txt` (now stored as JSON),
    and falls back to `data/external/abbreviation.json` if the TXT file
    is not present. Returns an empty dict if file is missing or invalid.
    """
    try:
        root = Path(__file__).resolve().parents[2]
        if path is None:
            txt_path = root / "data" / "external" / "abbreviation.txt"
            json_path = root / "data" / "external" / "abbreviation.json"
            path = str(txt_path if txt_path.exists() else json_path)
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
            1. Nhược điểm: 
            - Camera lùi không được nét. 
            - Nội thất chỉ có màu đen. Nhìn sẽ hơi tối. Bẩn nhanh lộ. 
            - Sau khoảng 5000km đi thì bị kêu khá to khi rà phanh đi chậm. Đã liên hệ hãng. Hãng đang báo sẽ thay đĩa phanh. Mình đang đợi phanh về để thay. Chưa thay nhưng đợt này đi lại không kêu nữa. 
            2. Ưu điểm: 
            - Đẹp, rộng rãi. Nhìn ngoài rất bệ vệ. Chất liệu nội thất ok. Màn hình to. 
            - Quá đủ các tính năng an toàn. Phanh tay điện tử, auto hold, đèn tự động, gạt mưa tự động, cánh báo giảm thiểu va chạm, cảm biến áp suất lốp theo xe…nói chung thoải mái dùng. 
            - Cách âm tốt. Loa nghe rất hay. 
            - Lái rất nhẹ nhàng. Quan sát tốt. 
            - Rất tiết kiệm nhiên liệu. Lái đường cao tốc chưa đến 5l/100km."""
    print(vn_text_clean(sample))
