# Cấu Trúc Dự Án NLP - Phân Loại Cảm Xúc Tiếng Việt

## Đánh Giá Cấu Trúc

### Ưu điểm:
1. **Phân tách rõ ràng** giữa code thử nghiệm (notebooks) và production code (src)
2. **Quản lý config tốt** với file riêng cho từng thuật toán ML/DL
3. **Data pipeline chuẩn**: raw → processed → external
4. **Đánh số notebooks** theo workflow nghiên cứu
5. **Lưu trữ artifacts** phân biệt ML (Pickle) và DL (PyTorch)

### Cải tiến đề xuất:
- Tách `app/` ra ngoài cùng cấp với `src/` (không nằm trong src)
- Thêm `scripts/` cho các file chạy độc lập (train_all.py, evaluate_all.py)
- Gộp `features/` vào `preprocessing/` (vectorizer là bước tiền xử lý)
- Thêm `Dockerfile`, `.dockerignore` cho deployment
- Thêm `.env.example` cho biến môi trường

---

## Cấu Trúc Được Tối Ưu Hóa

```
nlp-prj-group-08/
│
├── .gitignore                  # Loại bỏ: data/, models/, .env, __pycache__
├── .dockerignore               # Loại bỏ: notebooks/, .git, *.ipynb
├── .env.example                # Template cho biến môi trường (API keys, paths)
├── Dockerfile                  # Container hóa ứng dụng
├── docker-compose.yml          # (Tùy chọn) Chạy multi-service (app + database)
│
├── README.md                   # Hướng dẫn cài đặt & reproduce kết quả
├── requirements.txt            # Dependencies: scikit-learn, torch, transformers, underthesea
├── setup.py                    # Cài đặt package: pip install -e .
│
├── config/                     # Quản lý cấu hình tập trung
│   ├── config.yaml             # File config chung (paths, random_seed, train/test split)
│   ├── ml/
│   │   ├── a.yaml              # Naive Bayes: alpha, fit_prior
│   │   ├── b.yaml              # Logistic Regression: C, penalty, solver
│   │   └── c.yaml              # SVM: C, kernel, gamma
│   └── dl/
│       └── UNGTHU.yaml         # PhoBERT: lr, batch_size, epochs, max_len, warmup_steps
│
├── data/                       # Quản lý dữ liệu (KHÔNG COMMIT LÊN GIT)
│   ├── raw/                    # Dữ liệu gốc (READ-ONLY)
│   │   ├── VLSP.xml
│   │   └── Foody.csv
│   ├── processed/              # Dữ liệu sau tiền xử lý
│   │   ├── train.csv           # Gồm: text, label
│   │   ├── test.csv
│   │   └── val.csv             # (Tùy chọn) Validation set
│   └── external/               # Tài nguyên bên ngoài
│       ├── vietnamese-stopwords.txt
│       ├── teencode_dict.json  # Từ điển chuyển teencode → tiếng Việt chuẩn
│       └── emojis_sentiment.json
│
├── models/                     # Model artifacts (KHÔNG COMMIT)
│   ├── ml/
│   │   ├── naive_bayes.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── svm.pkl
│   │   └── tfidf_vectorizer.pkl  # Phải lưu vectorizer để inference
│   └── dl/
│       ├── phobert_best.bin      # Checkpoint tốt nhất
│       ├── training_args.json    # Log hyperparameters đã dùng
│       └── tokenizer/            # Custom tokenizer (nếu thêm từ vựng)
│
├── notebooks/                  # Jupyter Notebooks (Research & EDA)
│   ├── analysis/               # Phân tích dữ liệu
│   │   ├── 1_clean.ipynb       # EDA: phân bố nhãn, độ dài text, missing values
│   │   ├── 2_tokenize.ipynb    # Thử nghiệm làm sạch: regex, teencode, emoji
│   │   └── 3_vectorize.ipynb   # So sánh TF-IDF vs Word2Vec vs FastText
│   └── model/                  # Thử nghiệm mô hình
│       ├── a.ipynb             # Chạy & tune 3 mô hình ML
│       ├── b.ipynb             # Chạy & tune 3 mô hình ML
│       ├── c.ipynb             # Chạy & tune 3 mô hình ML
│       └── UNGTHU.ipynb        # Fine-tune PhoBERT trên GPU
│
├── src/                        # Production Code (Clean & Modular)
│   ├── __init__.py
│   │
│   ├── preprocessing/          # Tiền xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── clean.py            # Làm sạch: lowercase, remove URL/emoji, normalize unicode
│   │   ├── tokenize.py         # Tách từ: underthesea.word_tokenize, NLTK
│   │   ├── vectorizer.py       # TF-IDF, CountVectorizer, N-grams
│   │   └── augmentation.py     # (Nâng cao) Back-translation, synonym replacement
│   │
│   ├── models/                 # Định nghĩa & lưu mô hình
│   │   ├── __init__.py
│   │   ├── ml_models.py        # Class wrapper cho NB, LR, SVM
│   │   ├── phobert_model.py    # Class PhoBERTClassifier (PyTorch)
│   │   └── dataset.py          # PyTorch Dataset cho PhoBERT
│   │
│   └── utils/                  # Hàm tiện ích dùng chung
│       ├── __init__.py
│       ├── metrics.py          # Accuracy, Precision, Recall, F1, Confusion Matrix
│       ├── visualization.py    # Vẽ confusion matrix, ROC curve, loss/accuracy plots
│       ├── config_loader.py    # Load YAML config
│       └── common.py           # set_seed(), save_model(), load_model()
│
├── scripts/                    # Scripts chạy độc lập (CLI)
│   ├── train_ml.py             # Train 3 mô hình ML: python scripts/train_ml.py --model svm
│   ├── train_dl.py             # Train PhoBERT: python scripts/train_dl.py --epochs 5
│   ├── evaluate.py             # Đánh giá tất cả mô hình trên test set
│   └── predict.py              # Dự đoán: python scripts/predict.py --text "Sản phẩm rất tốt"
│
├── app/                        # Streamlit Web Application
│   ├── app.py                  # Main Streamlit app (entry point)
│   ├── pages/                  # Multi-page Streamlit app
│   │   ├── 1_Analyze.py     # Trang phân tích văn bản đơn lẻ
│   │   ├── 2_Batch.py       # Upload file CSV để phân tích hàng loạt
│   │   └── 3_Dashboard.py   # Dashboard thống kê & visualization
│   ├── components/             # Reusable UI components
│   │   ├── model_selector.py  # Component chọn model (NB/LR/SVM/PhoBERT)
│   │   ├── text_input.py      # Component nhập text với preprocessing preview
│   │   └── result_display.py  # Component hiển thị kết quả (sentiment + confidence)
│   └── utils/
│       ├── inference.py        # Load models & predict
│       ├── preprocessing.py    # Wrapper cho src.preprocessing
│       └── visualization.py    # Vẽ charts cho Streamlit
│
└── tests/                      # Unit Tests (pytest)
    ├── __init__.py
    ├── test_preprocessing.py   # Test clean_text(), tokenize()
    ├── test_models.py          # Test model training/prediction
    └── test_api.py             # Test API endpoints
```

---

## Workflow Thực Thi

### 1️⃣ Tiền xử lý dữ liệu
```bash
python scripts/preprocess_data.py --input data/raw/VLSP.xml --output data/processed/
```

### 2️⃣ Huấn luyện mô hình
```bash
# ML Models
python scripts/train_ml.py --model naive_bayes --config config/ml/a.yaml

# Deep Learning
python scripts/train_dl.py --config config/dl/UNGTHU.yaml --gpu 0
```

### 3️⃣ Đánh giá
```bash
python scripts/evaluate.py --test-data data/processed/test.csv
```

### 4️⃣ Dự đoán
```bash
python scripts/predict.py --text "Món ăn rất ngon, tôi sẽ quay lại"
```

### 5️⃣ Chạy Streamlit App

```bash
streamlit run app/app.py
# Mở browser tại: http://localhost:8501
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t sentiment-analysis:latest .

# Chạy container
docker run -p 8501:8501 sentiment-analysis
```

---

## 📝 Notes Quan Trọng

1. **KHÔNG commit** thư mục `data/`, `models/` lên Git → Dùng DVC hoặc Google Drive
2. **Lưu vectorizer** cùng với ML models (TF-IDF phải được fit trên tập train)
3. **Seed cố định** trong `config.yaml` để reproducible
4. **Requirements.txt** nên pin version: `torch==2.0.1` thay vì `torch`
5. **Logging** kết quả huấn luyện vào file hoặc MLflow/WandB
