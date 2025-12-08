import re
import unicodedata

def vn_text_clean(text: str) -> str:
    """
    Clean and normalize Vietnamese text.

    The processing steps include:
        1. Normalize Unicode to the NFC form.
        2. Convert the entire text to lowercase.
        3. Remove URLs and email addresses.
        4. Remove punctuation and special characters (only keeping letters, numbers, and spaces).
        5. Remove digits/numbers.
        6. Remove excessive whitespace at the beginning, end, and between words.
    
    Args:
        text (str): The input text string to be cleaned.
    
    Returns:
        str: The text string after being cleaned and normalized.
    """
    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)

    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and email addresses
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)  # Keep only letters, numbers, and spaces

    # Remove digits/numbers
    text = re.sub(r"\d+", " ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

if __name__ == "__main__":
    sample_text = "Xin chào! Đây là một ví dụ về văn bản Tiếng Việt. Liên hệ: 0932173 uiui@gmail.com hoặc truy cập http://example.com. Giá là 1000 đồng."
    cleaned_text = vn_text_clean(sample_text)
    print(cleaned_text)