# Mail-Lens AI — Quickstart & Execution Guide

This guide outlines the verified system status and instructions for running, testing, and demonstrating the **Mail-Lens AI** email threat classification system.

---

## System Status & Component Verification

| Component | Status | Details |
|---|---|---|
| **Python Environment** | Configured | Python 3.10+ virtual environment (`venv/`) with scikit-learn, streamlit, nltk, pandas, scipy |
| **Dataset Ingestion** | Verified | Curated corpus of 17,960 emails (HuggingFace + Zenodo), deduplicated and cleaned |
| **Data Partitioning** | Verified | Stratified 70/10/20 train/val/test splits (12,572 train / 1,796 val / 3,592 test) |
| **Feature Extraction** | Verified | 10,000 TF-IDF n-grams (unigram + bigram) + 20 cybersecurity domain heuristics |
| **Model Benchmark** | Verified | 5-model evaluation; production model: Calibrated Linear SVM (Val Macro F1 = 0.9760) |
| **Test Performance** | Verified | 98.64% Test Accuracy, 0.9761 Macro F1 on 3,592 holdout test samples |
| **Model Artifacts** | Verified | Serialized in `models/saved/` (`best_model.joblib`, `tfidf_vectorizer.joblib`, metadata) |
| **Inference Engine** | Verified | Sub-15ms CPU inference with Platt-scaled probabilities and plain-English reasons |
| **Test Suite** | Verified | 71/71 tests passing (100% pass rate, 76.77% code coverage) in Pytest |
| **Web Dashboard** | Verified | Interactive Streamlit web UI with keyword threat highlighting and clear reset |
| **Containerization** | Verified | Production `Dockerfile` and `.dockerignore` for containerized deployments |

---

## Quickstart Instructions

### 1. Launch the Web Application

The application can be run directly using Streamlit:

```powershell
# Windows
.\venv\Scripts\streamlit run app/main.py

# macOS / Linux
source venv/bin/activate
streamlit run app/main.py
```

Then open your browser at **`http://localhost:8501`**.

### 2. Run the Full Test Suite

To run all 71 automated tests with code coverage analysis:

```powershell
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### 3. Open the Interactive Demonstration Notebook

To inspect the pipeline step-by-step:

```powershell
jupyter notebook notebooks/pipeline_demonstration.ipynb
```

### 4. Container Deployment (Docker)

To build and run the Docker container locally:

```bash
docker build -t mail-lens-ai .
docker run -d -p 8501:8501 mail-lens-ai
```

---

## Project Repository

* **GitHub Repository:** [https://github.com/harisx404/mail-lens-ai](https://github.com/harisx404/mail-lens-ai)
* **Author:** Muhammad Haris ([itsharis.tech@gmail.com](mailto:itsharis.tech@gmail.com))
* **Affiliation:** Final Project · KPITB AI/ML Training Program
