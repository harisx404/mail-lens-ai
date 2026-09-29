# 🛡️ PhishGuard AI

### NLP-Based Phishing & Malicious Email Detection and Risk Analysis System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

---

## 📋 Overview

**PhishGuard AI** is an NLP and machine-learning-based system that analyzes email content and classifies it into three categories:

| Category | Description |
|---|---|
| 🟢 **LEGITIMATE** | Normal, non-malicious email |
| 🟠 **PHISHING** | Social engineering / deception attempt |
| 🔴 **MALICIOUS** | Harmful payload / malware delivery |

The system provides:
- ✅ Email classification with confidence score
- 📊 Risk level assessment (LOW / MEDIUM / HIGH / CRITICAL)
- 🔍 Detected security indicators (urgency, credentials, URLs, etc.)
- 📝 Human-readable explanation
- 💡 Security recommendations

---

## 🎯 Problem Statement

Phishing and malicious emails remain one of the most prevalent cybersecurity threats. Traditional rule-based filters struggle with sophisticated social engineering attacks. This project applies NLP and supervised machine learning to detect threatening emails based on their textual content and security-specific features.

---

## 🏗️ Architecture

```
Email Input
    ↓
Text Preprocessing (HTML removal, normalization, tokenization)
    ↓
Feature Extraction
    ├── NLP Features (TF-IDF, n-grams)
    └── Cybersecurity Features (URL analysis, keyword indicators)
    ↓
ML Classification (Logistic Regression / Naive Bayes / SVM)
    ↓
Risk Assessment + Explainability
    ↓
Web Interface (Streamlit)
```

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd phishguard-ai

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1: Train the Model

```bash
python scripts/train.py
```

This will:
1. Download datasets automatically
2. Preprocess and combine data
3. Train multiple models (LR, NB, SVM)
4. Evaluate and select the best model
5. Save the trained model and artifacts

### Step 2: Run the Web Application

```bash
streamlit run app/main.py
```

The application will open at `http://localhost:8501`

### Step 3: Analyze Emails

1. Navigate to **🔍 Analyze Email**
2. Enter the email subject and body
3. Click **Analyze Email**
4. Review the classification, risk level, and security indicators

---

## 📊 Dataset & Unified Taxonomy

| Dataset | Source | License | Cleaned Samples | Role in Project |
|---|---|---|---|---|
| **Phishing Email Detection** | [HuggingFace](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) | LGPL-3.0 | 17,522 | High-volume baseline (Legitimate & Phishing) |
| **Multiclass NLP Threats** | [Zenodo](https://zenodo.org/records/15235123) | CC BY 4.0 | 438 | Targeted social engineering & Malware samples |
| **Unified Corpus** | **Combined & Deduplicated** | Academic | **17,960** | **3-Class Taxonomy (Stratified 70/10/20)** |

### Class Distribution
- **LEGITIMATE**: 10,978 samples (61.1%)
- **PHISHING**: 6,805 samples (37.9%)
- **MALICIOUS**: 177 samples (1.0%)

---

## 📈 Model Benchmark & Experimental Results

All experiments evaluated on a stratified validation set of **1,796 samples** with fixed seed (42):

| Exp | Architecture | Features | Accuracy | Macro F1 | Weighted F1 | Selection Status |
|---|---|---|---|---|---|---|
| **E1** | Logistic Regression | TF-IDF (10k n-grams) | 97.33% | 0.9717 | 0.9733 | Baseline |
| **E2** | Multinomial Naive Bayes | TF-IDF (10k n-grams) | 96.05% | 0.8922 | 0.9611 | Probabilistic baseline |
| **E3** | **Linear SVM (Calibrated)** | **TF-IDF (10k n-grams)** | **97.94%** | **0.9760** | **0.9794** | 🏆 **Best Model** |
| **E4** | Logistic Regression | TF-IDF + 17 Security Features | 97.38% | 0.9721 | 0.9739 | Feature fusion |
| **E5** | Linear SVM (Calibrated) | TF-IDF + 17 Security Features | 97.77% | 0.9748 | 0.9777 | Feature fusion |

### 🏆 Final Evaluation on Held-Out Test Set (3,592 Samples)

The winning model (**Linear SVM**) was evaluated on the unseen test split:

* **Overall Test Accuracy:** **98.64%**
* **Macro F1-Score:** **0.9761**
* **Weighted F1-Score:** **0.9864**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **LEGITIMATE** | 0.9909 | 0.9877 | 0.9893 | 2,196 |
| **PHISHING** | 0.9795 | 0.9853 | 0.9824 | 1,361 |
| **MALICIOUS** | 0.9706 | 0.9429 | 0.9565 | 35 |

#### Test Set Confusion Matrix

```
                      Predicted Legitimate   Predicted Phishing   Predicted Malicious
Actual Legitimate :          2,169                   26                    1
Actual Phishing   :             20                1,341                    0
Actual Malicious  :              0                    2                   33
```
* **Malicious Detection Recall:** **94.3%** (33/35 hostile emails caught) with only **1 false positive** across 2,196 legitimate emails.


---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## 📁 Project Structure

```
phishguard-ai/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
│
├── docs/                    # Documentation
├── data/
│   ├── raw/                 # Raw datasets (not committed)
│   └── processed/           # Processed splits (not committed)
│
├── src/
│   ├── data/                # Data loading & pipeline
│   ├── preprocessing/       # NLP text preprocessing
│   ├── features/            # Feature engineering (TF-IDF + security)
│   ├── models/              # Model training
│   ├── evaluation/          # Model evaluation
│   ├── inference/           # Production inference pipeline
│   └── utils/               # Configuration & utilities
│
├── app/                     # Streamlit web application
│   └── main.py              # Main application entry point
│
├── models/saved/            # Trained model artifacts (not committed)
├── tests/                   # Test suite
├── scripts/                 # Training & utility scripts
├── notebooks/               # Jupyter notebooks for EDA
└── presentation/            # Capstone presentation
```

---

## ⚠️ Limitations

- This is a **research/educational capstone prototype**, not a production security product
- The MALICIOUS class has limited training data (~200 samples) — detection may be less reliable
- Training data is English-only
- No real-time email interception or mailbox integration
- No dynamic analysis (no URL visiting, no file execution)
- Trained on specific datasets — domain shift to real-world emails is possible
- False positives and false negatives are expected and documented
- Cannot replace professional email security products (e.g., SEGs, sandboxes)

---

## 🔮 Future Improvements

- Larger, more balanced dataset for MALICIOUS class
- Transformer-based models (DistilBERT) for improved accuracy
- Multi-language support
- Email header analysis (SPF, DKIM, DMARC)
- Real-time API integration
- Active learning from user feedback
- SHAP-based model explainability

---

## 🔒 Security & Privacy

- No email content is sent to external APIs
- All analysis is performed locally
- No URLs are visited during analysis
- No files are executed
- Sanitized input handling
- No sensitive data logging
- Configuration via environment variables

---

## 📄 Credits & References

- **Datasets:** HuggingFace (zefang-liu), Zenodo (Engineering Ingegneria Informatica Spa)
- **Libraries:** scikit-learn, NLTK, Streamlit, pandas, numpy
- **Methodology:** TF-IDF vectorization, supervised classification, cybersecurity feature engineering

---

## 👤 Author

**Muhammad Haris**
AI/ML Training Program — KPITB

---

*Built with ❤️ and 🔬 as an AI/ML capstone project.*
