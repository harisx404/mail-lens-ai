# Mail-Lens AI — Technical Documentation

## Background

Email-based threats remain among the most common entry points in security incidents. Most phishing and malware delivery campaigns rely on social engineering — manipulating recipients through urgency, authority impersonation, or deception — rather than technical exploits. These behaviors leave statistical fingerprints in the text content of emails.

Supervised text classification can learn these fingerprints from labeled historical data and apply them to new, unseen emails. This project implements that approach as a working prototype with a web interface.

## Objectives

1. Build a functional three-class email classifier (Legitimate / Phishing / Malicious)
2. Produce calibrated probability estimates, not just binary flags
3. Explain model decisions at the token level and in plain language
4. Provide a composited risk score that combines ML output with security heuristics
5. Package everything in a usable web application

## System design

### Data layer

Two external datasets were combined:
- **HuggingFace** (`zefang-liu/phishing-email-dataset`): ~18k rows, originally a two-class dataset (ham/spam). Ham maps to LEGITIMATE, spam to PHISHING.
- **Zenodo** (CC BY 4.0): 624 rows covering Phishing Email, Safe Email, and Malicious categories. Malicious samples were the primary motivation for adding this dataset.

After cleaning, deduplication, and label unification, the corpus is 17,960 emails across three classes.

### NLP pipeline

Text preprocessing uses a configurable pipeline in `TextPreprocessor`:
1. HTML tag and script removal via BeautifulSoup
2. URL and email address normalization to placeholder tokens
3. Unicode NFKD normalization
4. Lowercasing
5. Non-alphanumeric character removal
6. Whitespace normalization
7. WordNet lemmatization (reduces morphological variants)

**Decision: stopwords are kept.** Testing with stopword removal showed a measurable drop in Macro F1. Function words like "your", "please", "you" carry genuine phishing signal in context (e.g., "verify your account", "click here now").

### Feature engineering

**TF-IDF vectorization** (`TfidfVectorizer`, scikit-learn):
- 10,000 most frequent terms
- Word unigrams and bigrams `(1, 2)`
- Sublinear TF scaling (dampens term frequency dominance)
- `min_df=2`, `max_df=0.95` (filters rare and ubiquitous terms)

Bigrams are important for capturing multi-word phishing phrases: "click here", "verify account", "dear customer", "act now", "urgent action".

**Security heuristic features** (`SecurityFeatureExtractor`):
20 domain-specific features computed independently from raw text. These are not fed into the ML classifier (which only uses TF-IDF). They are used for:
- Post-classification risk score calculation
- Safety override rules
- Human-readable security indicator strings in the UI

### Classification

Five model architectures were trained and evaluated on the same 70/10/20 stratified split:

| Model | Val Macro F1 | Notes |
|---|---|---|
| Logistic Regression (TF-IDF) | 0.9717 | Strong baseline |
| Multinomial Naive Bayes | 0.8922 | Struggles with MALICIOUS class |
| **Calibrated LinearSVC (TF-IDF)** | **0.9760** | **Selected** |
| Logistic Regression (TF-IDF + Sec) | 0.9725 | Marginal improvement over LR baseline |
| Calibrated LinearSVC (TF-IDF + Sec) | 0.9657 | Scale mismatch reduces F1 |

**Why LinearSVC?** High-dimensional sparse text data is a natural fit for linear SVMs. The maximum-margin objective tends to generalize well. Combining it with Platt scaling (`CalibratedClassifierCV`) gives calibrated probabilities.

**Why TF-IDF only (E3 vs E5)?** Concatenating raw security feature counts with sublinear TF-IDF weights introduces a scale mismatch. The security features are on different scales and units, and without explicit scaling or weighting, they can hurt rather than help. Decoupling them (E3 + post-hoc heuristics) performed better.

### Risk scoring

Risk score (0.0–1.0) is computed from:
- Classification probabilities for threat classes
- Security indicator count density
- URL count modifier
- Executable extension modifier

Risk level thresholds:
| Level | Score range | Color |
|---|---|---|
| LOW | 0.00–0.30 | Green |
| MEDIUM | 0.30–0.60 | Amber |
| HIGH | 0.60–0.85 | Orange |
| CRITICAL | 0.85–1.00 | Red |

Override rules escalate risk independent of the ML prediction when hard security signals fire (e.g., executable file reference with any threat probability, or brand typosquatting detected).

### Explainability

**Token attribution:** The LinearSVC produces a coefficient matrix `(n_classes, n_features)`. For an input email, the non-zero TF-IDF features are projected against the coefficients for the predicted threat class and the LEGITIMATE class. The top tokens sorted by `tfidf × coefficient` are displayed as threat tokens and safe tokens.

This is a first-order approximation, not a rigorous SHAP-style attribution. It correlates with importance but does not account for feature interactions.

**Plain-English reasons:** Generated by `generate_plain_english_reasons()` in `app/main.py`. This function uses pattern matching on the input text and detected indicators to produce human-readable bullet points. It deliberately avoids ML jargon.

**Visual highlighting:** `highlight_email_content()` scans raw input text for known threat and safe keywords using precompiled regex patterns and wraps matches in styled HTML spans.

### Web interface

Built with Streamlit. Key design decisions:
- Model is loaded once via `@st.cache_resource` and reused across sessions
- Input state is managed via `st.session_state` to support clearing and preset loading
- All user-controlled strings rendered inside `unsafe_allow_html=True` blocks are escaped with `html.escape()`
- Layout is fixed-width (max 1680px) and tested at 1920×1080

## Error handling

| Scenario | Behavior |
|---|---|
| Empty input | Returns `UNKNOWN` with guidance message |
| Whitespace-only input | Treated same as empty after strip() |
| Model not found | `RuntimeError` with clear message |
| NLTK resources missing | Auto-downloaded on first use |
| Malformed URL in input | Caught by `(ValueError, AttributeError)`, logged at DEBUG |
| Very long input | Processed but may be slow; no hard size limit in the engine |

## Testing

71 automated tests across 4 test files:
- `test_core.py`: Unit tests for preprocessing, features, data, configuration
- `test_extended.py`: Feature engineering, models, evaluation
- `test_integration.py`: End-to-end inference scenarios
- `test_qa_regression.py`: Security hardening, boundary conditions, performance

Statement coverage: 76.77% across `src/` (core modules >90%, data loading and training scripts lower as expected).

## Security measures implemented

- `html.escape()` on all user-controlled strings in HTML output
- Whitespace stripping to prevent whitespace-only false positives
- Narrow exception handling in URL parsing (not `except Exception`)
- Streamlit CORS and XSRF protection enabled
- No external network calls during inference

## Known limitations

See [LIMITATIONS.md](LIMITATIONS.md) for a full discussion.

## Future directions

- Larger, better-balanced MALICIOUS class dataset
- Transformer-based encoder (DistilBERT) for contextual language understanding
- Email header analysis (SPF, DKIM, DMARC alignment)
- Active learning loop with analyst feedback
- FastAPI backend for programmatic API access
- Real-time VirusTotal URL lookup (opt-in, privacy-conscious)
