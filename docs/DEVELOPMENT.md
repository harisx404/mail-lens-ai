# Development Guide

## Prerequisites

- Python 3.10 or higher
- pip
- Git
- ~2GB free disk space (for datasets if retraining)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/harisx404/mail-lens-ai.git
cd mail-lens-ai

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Environment configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env  # macOS/Linux
copy .env.example .env  # Windows
```

The defaults work out-of-the-box. Only change them if you move the model or data directories.

| Variable | Default | Description |
|---|---|---|
| `DATA_RAW_DIR` | `data/raw` | Raw dataset location |
| `DATA_PROCESSED_DIR` | `data/processed` | Processed splits location |
| `MODELS_DIR` | `models/saved` | Model artifacts location |
| `RANDOM_SEED` | `42` | Global reproducibility seed |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Running the application

The trained model is already included in `models/saved/`. You can run the app immediately:

```bash
streamlit run app/main.py
```

Opens at `http://localhost:8501`.

## Retraining the model

If you want to regenerate the model from scratch:

```bash
python scripts/train.py
```

This will:
1. Download Dataset A from HuggingFace (`zefang-liu/phishing-email-dataset`)
2. Download Dataset B from Zenodo (XLSX file)
3. Process and merge both datasets into train/val/test splits
4. Train 5 model architectures (LR, NB, SVM, LR+Sec, SVM+Sec)
5. Evaluate all models on the validation set
6. Save the best model (E3 — Calibrated LinearSVC) and all artifacts

Training takes approximately 3–5 minutes on a standard laptop CPU.

> **Note**: First run downloads ~100MB of datasets. Subsequent runs use cached files.

## Testing

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_integration.py -v

# Run a single test
python -m pytest tests/test_core.py::TestTextPreprocessor::test_html_removal -v
```

### NLTK data

The preprocessor downloads NLTK resources (`punkt`, `wordnet`, `stopwords`) automatically on first run. If you're offline, download them manually:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

## Code structure

```
src/
├── data/
│   ├── loader.py          # Dataset download and raw loading
│   └── pipeline.py        # Data processing pipeline
├── preprocessing/
│   └── text_preprocessor.py   # TextPreprocessor class
├── features/
│   ├── feature_engineer.py    # TF-IDF wrapper
│   └── security_features.py   # SecurityFeatureExtractor (20 features)
├── models/
│   └── trainer.py             # Model creation and training
├── evaluation/
│   └── evaluator.py           # Metrics, reports, experiment comparison
├── inference/
│   └── engine.py              # MailLensInference (production pipeline)
└── utils/
    └── config.py              # Project configuration and constants
```

## Logging

The application uses Python's standard `logging` module. Set `LOG_LEVEL=DEBUG` in `.env` for verbose output during development.

Training logs are written to `training.log` in the project root (excluded from git).

## Common issues

**Model not loading:**
Ensure `models/saved/best_model.joblib` exists. If not, run `python scripts/train.py`.

**NLTK resources missing:**
Run the download snippet above or set `NLTK_DATA` env variable to a directory with pre-downloaded resources.

**Streamlit port already in use:**
```bash
streamlit run app/main.py --server.port 8502
```

**Datasets fail to download:**
HuggingFace and Zenodo require an internet connection on first run. Check connectivity and proxy settings.
