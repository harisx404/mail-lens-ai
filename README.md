# Mail-Lens AI

**NLP-Powered Email Threat Intelligence & Security Risk Assessment System**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Tests](https://img.shields.io/badge/Tests-71%20passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Mail-Lens AI is an email threat classification and explainability engine designed to identify deceptive social engineering attacks, credential harvesting attempts, and malicious attachment lures. 

Instead of acting as an opaque black box, Mail-Lens AI unites statistical natural language processing with cybersecurity domain heuristics to classify messages as **Legitimate**, **Phishing**, or **Malicious** — providing real-time token attribution, plain-English threat explanations, and actionable incident response guidance.

---

## Key Highlights

- **Multi-Class Threat Classification:** Distinguishes between normal business correspondence (`LEGITIMATE`), social engineering / spoofing (`PHISHING`), and binary / payload delivery (`MALICIOUS`).
- **High-Dimensional NLP Feature Pipeline:** 10,000 sublinear TF-IDF n-grams (unigram + bigram) capturing nuanced semantic cues and modal auxiliary verbs.
- **Domain-Specific Cybersecurity Heuristics:** 20 quantitative signals covering urgency density, credential extraction patterns, homoglyph typosquatting (Levenshtein distance), suspicious TLDs, and executable mentions.
- **Platt-Calibrated Decision Boundaries:** Evaluates emails using a Linear Support Vector Machine with 3-fold Platt calibration, delivering verified probabilities that sum to 1.0.
- **Explainable AI (XAI) Layer:** Projects TF-IDF vectors onto SVM hyperplane coefficients to surface top threat and safe words, backed by 4 plain-English summary bullets.
- **Production QA Validation:** 71 automated Pytest unit, integration, and security regression tests passing with 0 failures and 76.77% code coverage.
- **Low-Latency Static Inspection:** Sub-15ms inference latency per email on CPU; 100% static analysis with zero external HTTP calls, preserving strict privacy.

---

## System Architecture

```
                    Raw Email Input (Sender, Subject, Body)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   NLP Preprocessing Pipeline                            Cybersecurity Extractor
   • HTML tag stripping (BS4)                            • 20 quantitative threat signals
   • URL/Email entity normalization                      • Lookalike brand typosquatting
   • NLTK WordNet lemmatization                          • Executable / archive mentions
            │                                                     │
            ▼                                                     │
   TF-IDF Vectorization                                           │
   • 10,000 unigram/bigram n-grams                                │
   • Sublinear term frequency scaling                             │
            │                                                     │
            ▼                                                     │
   Calibrated Linear SVM Classifier                               │
   • LinearSVC with balanced class weights                        │
   • 3-Fold Platt probability calibration                         │
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                         Risk Engine & Override Layer
                         • Dynamic 0.0 - 1.0 threat risk score
                         • Adversarial simulation-cloak defense
                         • SVM hyperplane token attribution
                                       │
                                       ▼
                         Interactive Streamlit UI
                         • Classification badge & probability
                         • Visual keyword threat chips
                         • 4 plain-English explanation bullets
                         • Recommended security action
```

---

## Machine Learning & Benchmark Results

The model was selected through a systematic 5-experiment benchmark evaluated on identical stratified partitions:

| Experiment | Model Architecture | Feature Representation | Val Accuracy | Val Macro F1 | Status |
|---|---|---|---:|---:|---|
| E1 | Logistic Regression | TF-IDF (10,000 n-grams) | 97.33% | 0.9717 | Baseline |
| E2 | Multinomial Naive Bayes | TF-IDF (10,000 n-grams) | 96.05% | 0.8922 | Baseline |
| **E3** | **Calibrated Linear SVM** | **TF-IDF (10,000 n-grams)** | **97.94%** | **0.9760** | **Production Champion** |
| E4 | Logistic Regression | TF-IDF + 20 Security Features | 97.44% | 0.9725 | Evaluated |
| E5 | Calibrated Linear SVM | TF-IDF + 20 Security Features | 97.88% | 0.9752 | Evaluated |

### Verified Holdout Test Results (3,592 Unseen Samples)

Evaluation conducted on an independent 20% test partition:

| Metric | Score |
|---|---|
| **Overall Accuracy** | **98.64%** |
| **Macro-Averaged F1** | **0.9761** |
| **Weighted-Averaged F1** | **0.9864** |
| **Average CPU Inference Latency** | **< 15ms** |

#### Class Breakdown:

| Class | Precision | Recall | F1-Score | Support (Test Samples) |
|---|---:|---:|---:|---:|
| **LEGITIMATE** | 99.09% | 98.77% | 0.9893 | 2,196 |
| **PHISHING** | 97.95% | 98.53% | 0.9824 | 1,361 |
| **MALICIOUS** | 97.06% | 94.29% | 0.9565 | 35 |

> **Note on Class Balance:** Public malicious email corpora are sparse. The training corpus contains 177 verified malicious samples. `class_weight='balanced'` was employed to penalize minority misclassifications proportionately, yielding 94.29% recall on malicious test threats.

---

## Dataset Information

The training corpus combines two vetted open-source collections:
* **Hugging Face (`zefang-liu/phishing-email-dataset`):** High-volume legitimate and phishing baseline.
* **Zenodo (`ealvaradob/multiclass-email-dataset`):** Curated social engineering and malware delivery samples.

After automated HTML cleaning, null rejection, and deduplication:
* **Total Corpus:** 17,960 clean emails
* **Training Partition (70%):** 12,572 samples
* **Validation Partition (10%):** 1,796 samples
* **Test Partition (20%):** 3,592 samples

---

## Quickstart

### Prerequisites
* Python 3.10 or higher
* Git

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/harisx404/mail-lens-ai.git
cd mail-lens-ai

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app/main.py
```

The dashboard will open automatically at `http://localhost:8501`. Pre-trained model artifacts are included in `models/saved/` (~1.2MB total), so no initial training step is required.

---

## Docker Deployment

Build and run with containerized isolation:

```bash
# Build the Docker image
docker build -t mail-lens-ai .

# Run the container
docker run -d -p 8501:8501 --name mail-lens-app mail-lens-ai
```

---

## Test Suite & Validation

The codebase includes a comprehensive 71-test automated suite:

```bash
# Run all unit and integration tests
python -m pytest tests/ -v

# Run with statement coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Test Coverage Highlights:
* `tests/test_core.py` (36 tests): NLP cleaning, URL/email normalization, security feature extraction, model output shapes.
* `tests/test_extended.py` (12 tests): Pipeline serialization, model factories, experiment comparison utilities.
* `tests/test_integration.py` (10 tests): End-to-end inference, typosquatting domain detection, simulation-cloaking overrides.
* `tests/test_qa_regression.py` (13 tests): XSS payload escaping, SQL injection neutrality, 50,000-character boundary limits, Unicode robustness, probability sum-to-one invariants, latency benchmarks (<50ms).

---

## Project Structure

```
mail-lens-ai/
├── app/
│   └── main.py                     # Streamlit web dashboard
├── src/
│   ├── data/                       # Dataset ingestion and splitting pipeline
│   ├── preprocessing/              # NLP text cleaners and WordNet lemmatizers
│   ├── features/                   # TF-IDF vectorization & 20 security features
│   ├── models/                     # Model factories, training, and persistence
│   ├── evaluation/                 # Metric computation and experiment evaluator
│   ├── inference/                  # MailLensInference engine & risk scoring
│   └── utils/                      # Constants, keyword dictionaries, and config
├── models/saved/                   # Serialized model artifacts (~1.2MB)
├── data/
│   ├── raw/                        # Downloaded raw corpora (gitignored)
│   └── processed/                  # Processed train/val/test splits (gitignored)
├── samples/                        # Synthetic labeled test emails for evaluation
├── tests/                          # 71-test Pytest validation suite
├── scripts/
│   └── train.py                    # End-to-end training and benchmark script
├── notebooks/
│   ├── create_notebook.py          # Notebook generator script
│   └── pipeline_demonstration.ipynb # Interactive demonstration walkthrough
├── docs/                           # In-depth technical documentation
├── Dockerfile                      # Production container build
├── .dockerignore                   # Docker exclusion rules
├── requirements.txt                # Pinned dependencies
├── pyproject.toml                  # Build metadata & Pytest configuration
└── README.md
```

---

## Technical Documentation

Detailed architectural and design documents are available in the [`docs/`](docs/) directory:

* [Project Overview](docs/PROJECT_OVERVIEW.md) — System motivation and threat model
* [Architecture Guide](docs/ARCHITECTURE.md) — Component breakdown and data flow
* [Machine Learning Documentation](docs/MACHINE_LEARNING.md) — Dataset provenance and training pipeline
* [Security Posture](docs/SECURITY.md) — Static analysis principles and XSS sanitization
* [Testing Strategy](docs/TESTING.md) — Test architecture and edge-case coverage
* [API Reference](docs/API.md) — Python inference engine interface
* [Deployment Guide](docs/DEPLOYMENT.md) — Containerization and cloud hosting
* [Limitations & Roadblocks](docs/LIMITATIONS.md) — Honest engineering constraints

---

## Author & Project Info

**Muhammad Haris**  
* Final Project · **KPITB AI/ML Training Program**  
* Email: [itsharis.tech@gmail.com](mailto:itsharis.tech@gmail.com)  
* GitHub: [@harisx404](https://github.com/harisx404)  
* LinkedIn: [@harisx404](https://linkedin.com/in/harisx404)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
