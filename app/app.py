import streamlit as st

# Cấu hình trang
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# CSS để căn giữa tiêu đề
st.markdown(
    """
    <h1 style='text-align: center;'>Sentiment Analysis</h1>
""",
    unsafe_allow_html=True,
)

# Dữ liệu giả lập các chỉ số đánh giá cho mỗi model
model_metrics = {
    "SVM": {"accuracy": 0.87, "precision": 0.85, "recall": 0.88, "f1_score": 0.86},
    "Logistic": {"accuracy": 0.84, "precision": 0.82, "recall": 0.85, "f1_score": 0.83},
    "XGBoost": {"accuracy": 0.89, "precision": 0.88, "recall": 0.90, "f1_score": 0.89},
    "FNN": {"accuracy": 0.91, "precision": 0.90, "recall": 0.92, "f1_score": 0.91},
}


# Dự đoán giả lập dựa trên từ khóa
def predict_sentiment(text, model):
    positive_words = [
        "tốt",
        "hay",
        "xuất sắc",
        "tuyệt vời",
        "good",
        "great",
        "excellent",
        "amazing",
    ]
    negative_words = ["tệ", "xấu", "kém", "bad", "terrible", "poor", "awful"]

    text_lower = text.lower()

    # Đếm từ tích cực và tiêu cực
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        return "Tích cực (Positive)"
    elif neg_count > pos_count:
        return "Tiêu cực (Negative)"
    else:
        return "Trung tính (Neutral)"


# Khởi tạo session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Tạo 4 nút chọn model
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("SVM", use_container_width=True):
        st.session_state.selected_model = "SVM"

with col2:
    if st.button("Logistic", use_container_width=True):
        st.session_state.selected_model = "Logistic"

with col3:
    if st.button("XGBoost", use_container_width=True):
        st.session_state.selected_model = "XGBoost"

with col4:
    if st.button("FNN", use_container_width=True):
        st.session_state.selected_model = "FNN"

# Hiển thị chỉ số khi đã chọn model
if st.session_state.selected_model:
    metrics = model_metrics[st.session_state.selected_model]
    st.markdown(
        f"""
        <p style='text-align: center; font-size: 18px; margin-top: 20px;'>
            <b>Accuracy:</b> {metrics['accuracy']:.2f} &nbsp;&nbsp;
            <b>Precision:</b> {metrics['precision']:.2f} &nbsp;&nbsp;
            <b>Recall:</b> {metrics['recall']:.2f} &nbsp;&nbsp;
            <b>F1-Score:</b> {metrics['f1_score']:.2f}
        </p>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Ô nhập văn bản
text_input = st.text_area(
    "Nhập văn bản để phân tích:",
    height=150,
    placeholder="Ví dụ: Sản phẩm này thật tuyệt vời...",
)

# Nút predict và hiển thị kết quả
if st.button("Predict", use_container_width=True):
    if not st.session_state.selected_model:
        st.warning("⚠️ Vui lòng chọn model trước khi dự đoán!")
    elif not text_input.strip():
        st.warning("⚠️ Vui lòng nhập văn bản!")
    else:
        prediction = predict_sentiment(text_input, st.session_state.selected_model)
        st.markdown(
            f"""
            <div style='margin-top: 20px; padding: 15px; background-color: #f0f2f6; border-radius: 5px;'>
                <p style='font-size: 18px; margin: 0;'>
                    <b>Text:</b> {text_input[:100]}{'...' if len(text_input) > 100 else ''}
                </p>
                <p style='font-size: 20px; margin-top: 10px; color: #1f77b4;'>
                    <b>Label:</b> {prediction}
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )
