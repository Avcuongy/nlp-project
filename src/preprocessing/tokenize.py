import nltk
from underthesea import word_tokenize as uts_word_tokenize
from underthesea import sent_tokenize as uts_sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize

nltk.download('punkt')

def vn_word_tokenize(text: str, method: str = "underthesea"):
    """
    Tách từ trong văn bản tiếng Việt.

    Hàm này hỗ trợ tách từ sử dụng thư viện 'underthesea' (tối ưu cho tiếng Việt)
    hoặc 'nltk' (tách từ cơ bản).

    Args:
        text (str): Văn bản đầu vào cần tách từ.
        method (str, optional): Phương pháp tách từ. Mặc định là "underthesea".
            Các giá trị hợp lệ:
            - "underthesea": Sử dụng tokenizer của thư viện underthesea (khuyên dùng).
            - "nltk": Sử dụng tokenizer của thư viện NLTK.

    Returns:
        list: Danh sách các từ (tokens) sau khi tách.

    Raises:
        ValueError: Nếu `method` không phải là 'underthesea' hoặc 'nltk'.
    """
    method = method.lower()

    if method == "underthesea":
        return uts_word_tokenize(text)

    elif method == "nltk":
        return nltk_word_tokenize(text)

    else:
        raise ValueError("method phải là 'underthesea' hoặc 'nltk'.")


def vn_sentence_tokenize(text: str, method: str = "underthesea"):
    """
    Tách câu trong văn bản tiếng Việt.

    Hàm này hỗ trợ tách câu sử dụng thư viện 'underthesea' hoặc 'nltk'.

    Args:
        text (str): Văn bản đầu vào cần tách câu.
        method (str, optional): Phương pháp tách câu. Mặc định là "underthesea".
            Các giá trị hợp lệ:
            - "underthesea": Sử dụng tokenizer của thư viện underthesea (tối ưu cho tiếng Việt).
            - "nltk": Sử dụng tokenizer của thư viện NLTK (dựa trên dấu câu).

    Returns:
        list: Danh sách các câu sau khi tách.

    Raises:
        ValueError: Nếu `method` không phải là 'underthesea' hoặc 'nltk'.
    """
    method = method.lower()

    if method == "underthesea":
        return uts_sent_tokenize(text)

    elif method == "nltk":
        return nltk_sent_tokenize(text)

    else:
        raise ValueError("method phải là 'underthesea' hoặc 'nltk'.")
