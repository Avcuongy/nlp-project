import nltk
from underthesea import word_tokenize
from pyvi import ViTokenizer

# Download the 'punkt' resource for NLTK sentence tokenization
nltk.download("punkt")


def vn_word_tokenize(text: str, method: str = "underthesea") -> str:
    """
    Tokenize Vietnamese text into a single string where tokens are separated by
    spaces and multi-word phrases are joined with an underscore ('_').

    Supported methods:
    - "underthesea": underthesea.word_tokenize with text output (phrases as điện_thoại)
    - "pyvi": ViTokenizer.tokenize (returns string with underscores for phrases)

    Args:
        text: Input text to tokenize.
        method: One of {'underthesea', 'pyvi', 'nltk'}.

    Returns:
        str: Tokenized text (e.g., "điện_thoại pin_tốt").

    Raises:
        ValueError: If method is not supported.
    """
    method = method.lower()

    if method == "underthesea":
        # Prefer string output with underscores for phrases if supported
        try:
            return word_tokenize(text, format="text")
        except TypeError:
            # Older versions: join list of tokens
            toks = word_tokenize(text)
            return " ".join(toks)

    if method == "pyvi":
        return ViTokenizer.tokenize(text)

    raise ValueError("method must be one of: 'underthesea', 'pyvi', 'nltk'")


if __name__ == "__main__":
    sample = "ô tô điện này có pin tốt wifi mạnh giá rẻ với nền tảng tính năng đa dạng, ổn định."
    print("Underthesea:", vn_word_tokenize(sample, method="underthesea"))
    print("Pyvi:", vn_word_tokenize(sample, method="pyvi"))
