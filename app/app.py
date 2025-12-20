import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow import keras

from preprocessing.clean import vn_text_clean
from preprocessing.tokenize import vn_word_tokenize
from preprocessing.remove_stopwords import remove_stopwords


# Load vectorizer và label encoder
@st.cache_resource
def load_vectorizer_and_encoder():
    with open("../models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("../models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return vectorizer, label_encoder


# Load models
@st.cache_resource
def load_models():
    models = {}

    # Load ML models
    ml_models = ["svm", "logistic", "xgboost"]
    for model_name in ml_models:
        model_path = f"../models/ml/{model_name}.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                models[model_name] = pickle.load(f)

    # Load DL model
    dl_model_path = "../models/dl/fnn.keras"
    if os.path.exists(dl_model_path):
        models["fnn"] = keras.models.load_model(dl_model_path)

    return models


# Preprocessing function theo notebook 2_preprocessing.ipynb
def preprocess_text(text):
    # Clean
    cleaned = vn_text_clean(text)

    # Tokenize
    tokenized = vn_word_tokenize(cleaned, method="underthesea")

    # Remove stopwords với fallback
    text_backup = tokenized
    result = remove_stopwords(tokenized)

    # Fallback nếu kết quả rỗng
    if not result or result.strip() == "":
        result = text_backup

    return result


# Dự đoán với model thực tế
def predict_sentiment_real(text, model_name, models, vectorizer, label_encoder):
    # Preprocessing
    processed_text = preprocess_text(text)

    # Vectorize
    X = vectorizer.transform([processed_text])

    # Lấy model
    model = models.get(model_name.lower())
    if model is None:
        return "Model không tìm thấy"

    # Dự đoán
    if model_name.lower() == "fnn":
        # FNN model cần dense array
        X_dense = X.toarray()
        prediction = model.predict(X_dense, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
    else:
        # ML models
        predicted_class = model.predict(X)[0]

    # Decode label
    label = label_encoder.inverse_transform([predicted_class])[0]

    return label


# Load resources
try:
    vectorizer, label_encoder = load_vectorizer_and_encoder()
    models = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Lỗi khi load models: {e}")
    models_loaded = False


# Cấu hình trang
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# CSS để căn giữa tiêu đề
st.markdown(
    """
    <h1 style='text-align: center;'>Sentiment Analysis</h1>
""",
    unsafe_allow_html=True,
)

# Thông số
model_metrics = {
    "SVM": {"accuracy": 0.87, "precision": 0.85, "recall": 0.88, "f1_score": 0.86},
    "Logistic": {"accuracy": 0.84, "precision": 0.82, "recall": 0.85, "f1_score": 0.83},
    "XGBoost": {"accuracy": 0.89, "precision": 0.88, "recall": 0.90, "f1_score": 0.89},
    "FNN": {"accuracy": 0.91, "precision": 0.90, "recall": 0.92, "f1_score": 0.91},
}


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
        st.warning("Vui lòng chọn model trước khi dự đoán")
    elif not text_input.strip():
        st.warning("Vui lòng nhập văn bản!")
    elif not models_loaded:
        st.error("Models chưa được load thành công")
    else:
        with st.spinner("Chờ tí..."):
            try:
                prediction = predict_sentiment_real(
                    text_input,
                    st.session_state.selected_model,
                    models,
                    vectorizer,
                    label_encoder,
                )
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
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
