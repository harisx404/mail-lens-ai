# Resume / Portfolio Summary

## Project: Mail-Lens AI

### Short description (for resume bullet points)

Built an NLP-based email threat classifier that categorizes emails as legitimate, phishing, or malicious with 98.64% accuracy and provides token-level explainability.

### Technical description (for portfolio / project sections)

**Mail-Lens AI** is an end-to-end NLP and machine learning system for email threat classification. The pipeline includes text preprocessing (HTML stripping, URL normalization, lemmatization), TF-IDF feature extraction (10,000 n-grams, unigrams + bigrams), and a Calibrated LinearSVC classifier. A secondary security heuristic engine (20 domain-specific features) feeds a post-classification risk scoring layer. The system produces risk scores, token attributions, and plain-English explanations delivered through a Streamlit web dashboard.

### Key technologies

- **Language:** Python 3.10+
- **ML/NLP:** scikit-learn (LinearSVC, CalibratedClassifierCV, TfidfVectorizer), NLTK (lemmatization)
- **Web interface:** Streamlit with custom CSS
- **Data processing:** pandas, numpy, scipy (sparse matrices)
- **Visualization:** Plotly
- **Testing:** pytest, pytest-cov (71 tests, 76.77% coverage)
- **Datasets:** HuggingFace, Zenodo (17,960 labeled emails)

### Key contributions

- Designed and implemented a 5-experiment model benchmark comparing Logistic Regression, Naive Bayes, and Linear SVM architectures with and without security feature fusion
- Implemented Platt-scaled probability calibration (`CalibratedClassifierCV`) to convert SVM margin distances into interpretable probabilities
- Built a 20-feature cybersecurity heuristic extractor covering URL analysis, brand typosquatting detection (Levenshtein distance), urgency language detection, and credential-seeking pattern recognition
- Developed an XSS-hardened token highlighting system that escapes all user input before HTML injection
- Wrote 71 automated tests including regression tests for security hardening, boundary conditions, and inference latency benchmarks

### Verified metrics

All metrics are from evaluation on the held-out test set (3,592 samples, never seen during training):

| Metric | Value |
|---|---|
| Test Accuracy | 98.64% |
| Test Macro F1 | 0.9761 |
| Test Weighted F1 | 0.9864 |
| MALICIOUS Recall | 94.3% (33/35 detected) |
| Mean Inference Latency | ~12ms per email on CPU |

### GitHub repository description

> NLP-based email threat classifier — identifies phishing and malicious emails using TF-IDF + Calibrated Linear SVM with risk scoring and token-level explainability. 98.64% test accuracy on 17,960 email corpus.

### Portfolio description

Mail-Lens AI is a practical NLP security project that classifies emails as legitimate, phishing, or malicious. It combines a trained Calibrated LinearSVC model (10,000 TF-IDF features) with a 20-feature security heuristic engine to produce threat classifications, risk scores, and plain-language explanations. The project includes an interactive Streamlit dashboard, 71 automated tests, and full technical documentation.

### What to highlight in interviews

- Multi-class imbalanced text classification (MALICIOUS class at 0.99% of training data)
- Rationale for keeping stopwords (function words carry phishing signal)
- Experiment design: why E3 (TF-IDF only + LinearSVC) outperformed E5 (TF-IDF + security features) despite having fewer features
- Calibration: difference between raw SVM decision margins and Platt-calibrated probabilities
- Security hardening: XSS prevention in HTML rendering, exception narrowing, whitespace edge cases

---

## Author & Contact

**Muhammad Haris**  
- Final Project · KPITB AI/ML Training Program  
- Email: [itsharis.tech@gmail.com](mailto:itsharis.tech@gmail.com)  
- GitHub: [@harisx404](https://github.com/harisx404)  
- LinkedIn: [@harisx404](https://linkedin.com/in/harisx404)
