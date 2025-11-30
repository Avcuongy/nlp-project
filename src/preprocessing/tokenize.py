import nltk
from underthesea import word_tokenize as uts_word_tokenize
from underthesea import sent_tokenize as uts_sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize

# Download the 'punkt' resource for NLTK sentence tokenization
nltk.download("punkt")


def vn_word_tokenize(text: str, method: str = "underthesea"):
    """
    Tokenizes Vietnamese text into words.

    This function supports word tokenization using either the 'underthesea'
    library (optimized for Vietnamese) or 'nltk' (basic tokenization).

    Args:
        text (str): The input text to be tokenized.
        method (str, optional): The tokenization method. Defaults to "underthesea".
            Valid values:
            - "underthesea": Uses the underthesea library's tokenizer (recommended).
            - "nltk": Uses the NLTK library's tokenizer.

    Returns:
        list: A list of words (tokens) after tokenization.

    Raises:
        ValueError: If 'method' is neither 'underthesea' nor 'nltk'.
    """
    method = method.lower()

    if method == "underthesea":
        return uts_word_tokenize(text)

    elif method == "nltk":
        return nltk_word_tokenize(text)

    else:
        raise ValueError("The 'method' must be either 'underthesea' or 'nltk'.")


def vn_sentence_tokenize(text: str, method: str = "underthesea"):
    """
    Tokenizes Vietnamese text into sentences.

    This function supports sentence tokenization using either the 'underthesea'
    library or 'nltk'.

    Args:
        text (str): The input text to be tokenized into sentences.
        method (str, optional): The sentence tokenization method. Defaults to "underthesea".
            Valid values:
            - "underthesea": Uses the underthesea library's tokenizer (optimized for Vietnamese).
            - "nltk": Uses the NLTK library's tokenizer (typically based on punctuation).

    Returns:
        list: A list of sentences after tokenization.

    Raises:
        ValueError: If 'method' is neither 'underthesea' nor 'nltk'.
    """
    method = method.lower()

    if method == "underthesea":
        return uts_sent_tokenize(text)

    elif method == "nltk":
        return nltk_sent_tokenize(text)

    else:
        raise ValueError("The 'method' must be either 'underthesea' or 'nltk'.")
