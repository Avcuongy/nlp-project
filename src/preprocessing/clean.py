import re
import unicodedata

def vn_text_clean(text: str) -> str:
    """
    Làm sạch và chuẩn hóa văn bản tiếng Việt.
    
    **Quy trình xử lý bao gồm**:
        1. Chuẩn hóa Unicode về dạng NFC.
        2. Chuyển đổi toàn bộ văn bản sang chữ thường.
        3. Loại bỏ các đường dẫn (URL) và địa chỉ email.
        4. Loại bỏ dấu câu và các ký tự đặc biệt (chỉ giữ lại chữ cái, số và khoảng trắng).
        5. Loại bỏ các chữ số.
        6. Loại bỏ khoảng trắng thừa ở đầu, cuối và giữa các từ.

    Args:
        text (str): Chuỗi văn bản đầu vào cần làm sạch.

    Returns:
        str: Chuỗi văn bản sau khi đã được làm sạch và chuẩn hóa.
    """
    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFC", text)
    
    # Chuyển về chữ thường
    text = text.lower()
    
    # Loại bỏ URL, email
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    
    # Loại bỏ dấu câu và ký tự đặc biệt
    text = re.sub(r"[^\w\s]", " ", text)  # giữ chữ + số + khoảng trắng
    
    # Loại bỏ số (tuỳ chọn)
    text = re.sub(r"\d+", " ", text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    
    return text