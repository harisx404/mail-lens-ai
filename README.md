# Mail-Lens AI

**NLP-based email threat classification and risk analysis**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-71%20passing-brightgreen)](tests/)

Mail-Lens AI analyzes email content and security-related signals to classify messages as **legitimate**, **phishing**, or **malicious**, and explains the factors behind each decision.

---

## What it does

- Classifies emails into three categories: `LEGITIMATE`, `PHISHING`, or `MALICIOUS`
- Assigns a risk score (0–1) and a risk level (`LOW` / `MEDIUM` / `HIGH` / `CRITICAL`)
- Detects security indicators: urgency tactics, credential harvesting language, lookalike domains, executable references
- Highlights suspicious keywords directly in the input text
- Provides plain-English explanations and actionable recommendations

## How it works

```
Email Input (sender, subject, body)
  ↓
Text Preprocessing  →  HTML removal, URL/email normalization, lemmatization
  ↓
Feature Extraction  →  TF-IDF (10,000 n-grams) + 20 cybersecurity heuristics
  ↓
ML Classification   →  Calibrated Linear SVM (Experiment E3)
  ↓
Risk Assessment     →  Combined probability + heuristic score (0.0–1.0)
  ↓
Explainability      →  Token attribution + plain-English reason bullets
  ↓
Web Interface       →  Streamlit dashboard
```

## Technology stack

| Component | Technology |
|---|---|
| Web application | Streamlit |
| ML classifier | scikit-learn (Calibrated LinearSVC) |
| NLP | TF-IDF vectorization, NLTK lemmatization |
| Data handling | pandas, numpy, scipy |
| Visualization | Plotly |
| Testing | pytest, pytest-cov |

## Dataset

| Source | Samples | Role |
|---|---|---|
| HuggingFace — `zefang-liu/phishing-email-dataset` | 17,522 cleaned | Legitimate + Phishing baseline |
| Zenodo — CC BY 4.0 multiclass NLP dataset | 438 samples | Targeted social engineering + Malware |
| **Combined (deduplicated, stratified)** | **17,960** | **Final training corpus** |

**Class distribution:** Legitimate 61.1% · Phishing 37.9% · Malicious 1.0%

## Model results

Five architectures were benchmarked on the same stratified splits (seed 42):

| Exp | Model | Features | Val Acc | Val Macro F1 |
|---|---|---|---|---|
| E1 | Logistic Regression | TF-IDF 10k | 97.33% | 0.9717 |
| E2 | Multinomial Naive Bayes | TF-IDF 10k | 96.05% | 0.8922 |
| **E3** | **Calibrated LinearSVC** | **TF-IDF 10k** | **97.94%** | **0.9760** |
| E4 | Logistic Regression | TF-IDF + 20 security features | 97.44% | 0.9725 |
| E5 | Calibrated LinearSVC | TF-IDF + 20 security features | 97.88% | 0.9657 |

**E3 (Calibrated LinearSVC)** was selected as the production model.

### Held-out test set results (3,592 samples)

| Metric | Value |
|---|---|
| Accuracy | **98.64%** |
| Macro F1 | **0.9761** |
| Weighted F1 | **0.9864** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LEGITIMATE | 0.9909 | 0.9877 | 0.9893 | 2,196 |
| PHISHING | 0.9795 | 0.9853 | 0.9824 | 1,361 |
| MALICIOUS | 0.9706 | 0.9429 | 0.9565 | 35 |

> The MALICIOUS class has limited training support (177 samples total). Detection is functional but less robust than the other two classes.

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/harisx404/mail-lens-ai.git
cd mail-lens-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Running the application

The trained model is included in the repository (`models/saved/`). No training step is required to run the app.

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

## Training from scratch

If you want to retrain the model (downloads ~18k email datasets):

```bash
python scripts/train.py
```

This runs the full pipeline: data loading → preprocessing → feature engineering → 5-model benchmark → artifact saving.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

**71/71 tests passing** · **76.77% statement coverage** across `src/`

Test breakdown:
- `tests/test_core.py` (36): Preprocessing, features, data pipeline, configuration
- `tests/test_extended.py` (12): Feature engineering, model factory, evaluation metrics
- `tests/test_integration.py` (10): End-to-end inference, typosquatting, cloaking regression
- `tests/test_qa_regression.py` (13): XSS escaping, boundary inputs, injection handling, latency

## Project structure

```
mail-lens-ai/
├── app/
│   └── main.py              # Streamlit web application
├── src/
│   ├── data/                # Dataset loading and processing pipeline
│   ├── preprocessing/       # Text cleaning and normalization
│   ├── features/            # TF-IDF feature engineering + 20 security features
│   ├── models/              # Model training (LR, NB, SVM)
│   ├── evaluation/          # Evaluation metrics and experiment comparison
│   ├── inference/           # Production inference engine
│   └── utils/               # Configuration and constants
├── models/saved/            # Trained model artifacts
├── data/
│   ├── raw/                 # Raw datasets (downloaded by train.py)
│   └── processed/           # Processed train/val/test splits
├── tests/                   # Test suite (71 tests)
├── scripts/
│   └── train.py             # Training script
├── notebooks/               # Jupyter EDA notebook
├── docs/                    # Technical documentation
├── samples/                 # Synthetic email examples for testing
├── README.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Security and privacy

- All analysis is performed locally — no email content is sent to external services
- No URLs are visited during analysis (pure static inspection)
- No files or scripts referenced in emails are executed
- Analyzed emails are processed in-memory and not persisted to disk
- See [docs/SECURITY.md](docs/SECURITY.md) for the full security posture

## Limitations

- Research prototype, not a production email security gateway
- MALICIOUS class has limited training data (~177 samples)
- English-only training data
- No email header analysis (SPF, DKIM, DMARC)
- No dynamic URL or attachment detonation
- False positives and false negatives are expected and documented in [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## Documentation

| Document | Description |
|---|---|
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | What the project is and why it exists |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and data flow |
| [docs/MACHINE_LEARNING.md](docs/MACHINE_LEARNING.md) | ML pipeline, training, and evaluation |
| [docs/SECURITY.md](docs/SECURITY.md) | Security posture and privacy |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Installation and local development |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment options and considerations |
| [docs/API.md](docs/API.md) | Inference engine API reference |
| [docs/TESTING.md](docs/TESTING.md) | Test strategy and commands |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Known limitations and caveats |
| [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) | Comprehensive technical documentation |

## Author

**Muhammad Haris** — S.No: 70  
KPITB AI/ML Training Program — Final Capstone Project

## License

MIT — see [LICENSE](LICENSE).
