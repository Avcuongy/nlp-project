import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import os
import sys
from pathlib import Path
from tensorflow import keras

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.clean import vn_text_clean
from preprocessing.tokenize import vn_word_tokenize
from preprocessing.remove_stopwords import remove_stopwords


# Load vectorizer và label encoder
@st.cache_resource
def load_vectorizer_and_encoder():
    vectorizer_path = PROJECT_ROOT / "models" / "vectorizer.pkl"
    label_encoder_path = PROJECT_ROOT / "models" / "label_encoder.pkl"

    vectorizer = load(vectorizer_path)
    label_encoder = load(label_encoder_path)
    return vectorizer, label_encoder


# Load models
@st.cache_resource
def load_models():
    models = {}

    # Load ML models
    ml_models = ["svm", "logistic", "xgboost"]
    for model_name in ml_models:
        model_path = PROJECT_ROOT / "models" / "ml" / f"{model_name}.pkl"
        if model_path.exists():
            models[model_name] = load(model_path)

    # Load DL model
    dl_model_path = PROJECT_ROOT / "models" / "dl" / "fnn.keras"
    if dl_model_path.exists():
        models["fnn"] = keras.models.load_model(dl_model_path)

    return models


# Preprocessing function
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


# Predict function
def predict_sentiment_real(text, model_name, models, vectorizer, label_encoder):
    # Preprocessing
    processed_text = preprocess_text(text)

    # Vectorize
    X = vectorizer.transform([processed_text])

    # Model
    model = models.get(model_name.lower())
    if model is None:
        return "Model không tìm thấy"

    # Predict
    if model_name.lower() == "fnn":
        # FNN model need dense array
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


# Streamlit app
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# CSS
st.markdown(
    """
    <style>
    /* Style cho button được chọn */
    div[data-testid="column"] button[kind="secondary"] {
        transition: all 0.3s ease;
    }
    
    div[data-testid="column"] button[kind="secondary"]:hover {
        transform: scale(1.05);
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Style cho tất cả các buttons khi hover */
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {
        font-weight: bold !important;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    </style>
    <h1 style='text-align: center;'>Sentiment Analysis</h1>
""",
    unsafe_allow_html=True,
)

# Model metrics
model_metrics = {
    "SVM": {"accuracy": 0.66, "precision": 0.54, "recall": 0.51, "f1_score": 0.52},
    "Logistic": {"accuracy": 0.68, "precision": 0.56, "recall": 0.52, "f1_score": 0.51},
    "XGBoost": {"accuracy": 0.66, "precision": 0.53, "recall": 0.50, "f1_score": 0.50},
    "FNN": {"accuracy": 0.66, "precision": 0.49, "recall": 0.50, "f1_score": 0.49},
}


# Initialize session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Create 4 model selection buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button(
        "SVM",
        use_container_width=True,
        type="primary" if st.session_state.selected_model == "SVM" else "secondary",
    ):
        st.session_state.selected_model = "SVM"
        st.rerun()

with col2:
    if st.button(
        "Logistic",
        use_container_width=True,
        type=(
            "primary" if st.session_state.selected_model == "Logistic" else "secondary"
        ),
    ):
        st.session_state.selected_model = "Logistic"
        st.rerun()

with col3:
    if st.button(
        "XGBoost",
        use_container_width=True,
        type="primary" if st.session_state.selected_model == "XGBoost" else "secondary",
    ):
        st.session_state.selected_model = "XGBoost"
        st.rerun()

with col4:
    if st.button(
        "FNN",
        use_container_width=True,
        type="primary" if st.session_state.selected_model == "FNN" else "secondary",
    ):
        st.session_state.selected_model = "FNN"
        st.rerun()

# Show selected model metrics
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

# Input text area
text_input = st.text_area(
    "Nhập văn bản:",
    height=150,
    placeholder="Ví dụ: Xe vinfast thật tuyệt vời...",
)

# Button predict
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
                        <p style='font-size: 20px; margin-top: 10px;'>
                            <b>Label:</b> {prediction}
                        </p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
