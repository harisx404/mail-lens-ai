# Architecture

## System overview

Mail-Lens AI is a single-tier Streamlit application. The ML inference pipeline runs in the same Python process as the web UI. There is no separate backend API or database.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Application                    │
│                       app/main.py                               │
│                                                                 │
│  ┌────────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │  Input Console │ → │  Analysis Engine │ → │  Result UI   │  │
│  │  (sender,      │   │  (inference +    │   │  (verdict,   │  │
│  │   subject,     │   │   risk scoring + │   │   risk,      │  │
│  │   body)        │   │   explanation)   │   │   reasons,   │  │
│  └────────────────┘   └──────────────────┘   │   tokens)    │  │
│                                               └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   src/  (pipeline)  │
                    └─────────────────────┘
```

## Component breakdown

### `app/main.py`

The Streamlit entry point. Handles:
- UI rendering (CSS, layout, cards)
- Session state for input fields and results
- Model loading via `@st.cache_resource` (loaded once, reused across requests)
- Calling `engine.analyze()` and rendering the result
- Visual threat keyword highlighting
- Plain-English reason generation
- Preset scenario management

### `src/inference/engine.py` — `MailLensInference`

The production inference class. Methods:
- `load_model(model_dir)`: Loads `best_model.joblib`, `tfidf_vectorizer.joblib`, and config from disk
- `analyze(text, subject, sender)`: Full pipeline execution, returns `AnalysisResult`
- `_compute_risk_score()`: Combines prediction probabilities with security indicator count
- `_apply_risk_override()`: Escalates risk if hard security signals fire (executables, typosquats)
- `_get_risk_level()`: Maps 0.0–1.0 score to LOW/MEDIUM/HIGH/CRITICAL
- `_extract_nlp_insights()`: Projects TF-IDF input through LinearSVC coefficients to produce token attribution

### `src/preprocessing/text_preprocessor.py` — `TextPreprocessor`

Configurable preprocessing pipeline:
1. HTML tag and script block removal (BeautifulSoup)
2. URL normalization → `urlplaceholder`
3. Email address normalization → `emailplaceholder`
4. Unicode NFKD normalization + combining character removal
5. Lowercasing
6. Special character removal (keep alphanumeric + spaces)
7. Whitespace normalization
8. Optional stopword removal (disabled by default — function words carry phishing signal)
9. Optional lemmatization (enabled by default — WordNetLemmatizer)
10. Minimum token length filtering (≥2 chars)

### `src/features/feature_engineer.py` — `FeatureEngineer`

Wraps `TfidfVectorizer` with a consistent fit/transform interface:
- `fit(train_df)`: Fits vectorizer on training text
- `transform(df)`: Returns sparse TF-IDF matrix
- `transform_single(text)`: Transforms a single string for inference
- `save(dir)` / `load(dir)`: Persists vectorizer and config to disk

The vectorizer config: `max_features=10000`, `ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`, `max_df=0.95`

### `src/features/security_features.py` — `SecurityFeatureExtractor`

Extracts 20 domain-specific security heuristics from raw email text:
URL features, brand typosquatting detection, keyword category counts (urgency, credential, threat, reward, impersonation), simulation cloaking detection, sender domain analysis, and attachment extension detection.

These features are **not** fed into the E3 LinearSVC classifier. They are computed separately and used for:
- Risk score adjustment
- Safety override rules (e.g., force HIGH risk if executable mentioned)
- Populating the "detected indicators" list in the UI

### `src/models/trainer.py`

Creates and trains classifiers (`LogisticRegression`, `MultinomialNB`, `LinearSVC` wrapped in `CalibratedClassifierCV`).

### `src/evaluation/evaluator.py`

Computes classification reports, macro/weighted F1, confusion matrices, and cross-experiment comparison tables.

### `src/data/`

- `loader.py`: Downloads and caches raw datasets from HuggingFace and Zenodo
- `pipeline.py`: Full data processing pipeline — label unification, cleaning, deduplication, stratified splitting

## Data flow

```
User Input
  │
  ├── security_extractor.extract(full_text)
  │     → SecurityFeatures (20 heuristic values)
  │
  ├── feature_engineer.transform_single(full_text)
  │     → preprocessor.preprocess(text)
  │           → cleaned text
  │     → tfidf_vectorizer.transform([cleaned_text])
  │           → sparse CSR matrix (1, 10000)
  │
  ├── model.predict(X)       → prediction index (0, 1, 2)
  ├── model.predict_proba(X) → [p_legit, p_phish, p_malic]
  │
  ├── _compute_risk_score(prediction, confidence, indicators, sec_features)
  │     → risk_score (float)
  ├── _apply_risk_override(...)
  │     → final risk_score, risk_level
  │
  ├── _generate_explanation(...)
  ├── _generate_recommendation(...)
  └── _extract_nlp_insights(full_text, prediction)
        → top_threat_tokens, top_safe_tokens, pipeline_trace

→ AnalysisResult(prediction, confidence, probabilities, risk_score,
                 risk_level, risk_color, detected_indicators,
                 explanation, recommendation, top_threat_tokens,
                 top_safe_tokens, pipeline_trace)
```

## Model artifacts

Stored in `models/saved/`:
| File | Contents |
|---|---|
| `best_model.joblib` | `CalibratedClassifierCV` (LinearSVC base, Platt scaling, 3-fold CV) |
| `tfidf_vectorizer.joblib` | Fitted `TfidfVectorizer` (10,000 features, vocab + IDF weights) |
| `feature_config.joblib` | Dict: `{use_security_features: False, max_features: 10000, ...}` |
| `feature_names.joblib` | Array of 10,000 n-gram strings |
| `model_metadata.json` | Experiment record, metrics, classes, random seed |
| `experiment_results.json` | Benchmark table for all 5 experiments |

## Threading and caching

Streamlit runs each user session in a thread. `@st.cache_resource` ensures the inference engine (model + vectorizer) is loaded exactly once at startup and shared across all sessions. The engine's `analyze()` method is stateless and thread-safe.

## Deployment architecture

The application is a single-process Python program with no external database or queue. It can be deployed as-is on any machine with Python ≥3.10 and the model artifacts present.

Streamlit Cloud deployment is possible if model file sizes stay within the platform's limits (currently ≤1GB total). See [docs/DEPLOYMENT.md](DEPLOYMENT.md) for details.
