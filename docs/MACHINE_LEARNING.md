# Machine Learning

## Dataset

### Sources

| Dataset | URL | License | Purpose |
|---|---|---|---|
| Phishing Email Detection (HuggingFace) | `zefang-liu/phishing-email-dataset` | LGPL-3.0 | High-volume legitimate + phishing samples |
| Multiclass NLP Threats (Zenodo) | zenodo.org/records/15235123 | CC BY 4.0 | Malicious and targeted social engineering samples |

### Processing pipeline (`src/data/`)

1. **Loading** (`loader.py`): Downloads Dataset A from HuggingFace datasets library; Dataset B from Zenodo via HTTP. Both are cached locally.
2. **Label unification** (`pipeline.py`): Maps source-specific labels to the unified schema:
   - `0 = LEGITIMATE` (ham / safe email)
   - `1 = PHISHING` (spam / phishing / social engineering)
   - `2 = MALICIOUS` (malware delivery, dangerous payloads)
3. **Cleaning**: Removes empty rows, normalizes whitespace, drops duplicate text entries
4. **Deduplication**: Hash-based exact match across combined dataset
5. **Stratified splitting**: Train 70% / Val 10% / Test 20% with `random_state=42`

### Final corpus

| Split | Legitimate | Phishing | Malicious | Total |
|---|---|---|---|---|
| Train | 7,688 | 4,760 | 124 | 12,572 |
| Val | 1,098 | 680 | 18 | 1,796 |
| Test | 2,196 | 1,361 | 35 | 3,592 |
| **Total** | **10,982** | **6,801** | **177** | **17,960** |

Class imbalance is significant for the MALICIOUS class (0.99%). This is an inherent limitation of the available data, not a design choice.

### Data leakage verification

- Vectorizer is fit exclusively on training data
- No target labels or risk scores are included in the feature matrix
- Zero overlapping samples between train, val, and test splits (verified by text hash comparison)

## Preprocessing

See `src/preprocessing/text_preprocessor.py`. Key decisions:

**Stopword removal is disabled** by default. Initial experiments showed that words like "your", "please", "you", "click" carry meaningful phishing signal and removing them reduced Macro F1 by ~0.8%.

**Lemmatization is enabled** using NLTK's `WordNetLemmatizer`. This reduces morphological variants (e.g., "verifying" → "verify") without discarding the base token's meaning.

**URL and email normalization** replaces actual URLs and addresses with placeholder tokens (`urlplaceholder`, `emailplaceholder`). Their _presence_ is a useful signal; their specific content is handled separately by the security feature extractor.

## Feature engineering

### TF-IDF vectorization (`src/features/feature_engineer.py`)

- **Algorithm**: `TfidfVectorizer` from scikit-learn
- **Vocabulary**: Top 10,000 terms by frequency, with sublinear TF scaling (`sublinear_tf=True`)
- **N-grams**: Word unigrams + bigrams `(1, 2)` — captures two-word phrases like "click here", "verify account", "dear customer"
- **Frequency bounds**: `min_df=2` (ignores hapax legomena), `max_df=0.95` (ignores ubiquitous terms)
- **Output**: Sparse CSR matrix of shape `(n_samples, 10000)`

### Security heuristic features (`src/features/security_features.py`)

20 domain-specific features computed from raw text, used for post-classification risk adjustment:

| Feature | Type | Description |
|---|---|---|
| `num_urls` | int | Count of URLs |
| `has_ip_url` | bool | URL contains IP address |
| `has_shortened_url` | bool | Known URL shortener detected |
| `has_suspicious_tld` | bool | Suspicious top-level domain |
| `has_brand_typosquat` | bool | Lookalike brand domain (Levenshtein) |
| `num_emails` | int | Email address count |
| `num_dollar_signs` | int | Financial symbol count |
| `urgency_keyword_count` | int | Urgency language hits |
| `threat_keyword_count` | int | Threat language hits |
| `credential_keyword_count` | int | Credential-seeking language hits |
| `reward_keyword_count` | int | Reward/bait language hits |
| `impersonation_score` | float | Authority impersonation ratio |
| `has_executable_mention` | bool | Executable file extension reference |
| `has_archive_mention` | bool | Archive file extension reference |
| `sender_domain_suspicious` | bool | Free webmail / mismatched domain |
| `subject_has_re_fwd` | bool | Deceptive RE:/FWD: prefix |
| `simulation_cloaking_detected` | bool | Phishing simulation disclaimer spoofing |
| `subject_caps_ratio` | float | Uppercase ratio in subject |
| `body_caps_ratio` | float | Uppercase ratio in body |
| `sentiment_polarity` | float | Text sentiment score |

**Important**: These 20 features are **not** part of the E3 (winning) classifier's input vector. The LinearSVC is trained and infers on 10,000 TF-IDF features only. The security features feed the risk scoring and override logic.

Experiment E4 and E5 tested concatenating security features into the classification feature matrix. The hybrid approach produced slightly lower macro F1, likely due to feature scale mismatch between sublinear TF-IDF weights and raw count-based security features.

## Model training

### Architectures benchmarked

```python
E1: LogisticRegression(C=1.0, max_iter=1000, random_state=42)
E2: MultinomialNB(alpha=0.1)
E3: CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3, method='sigmoid')
E4: LogisticRegression(C=1.0, ...)  # with security features
E5: CalibratedClassifierCV(LinearSVC(...))  # with security features
```

All models trained on the same `train.csv` split with fixed `random_state=42`.

### Model selection

E3 (Calibrated LinearSVC on TF-IDF only) was selected based on highest validation Macro F1 (`0.9760`). LinearSVC is well-suited for high-dimensional sparse text data. Platt scaling (sigmoid calibration via `CalibratedClassifierCV`) converts the SVM's decision margins into calibrated probabilities, which the risk scorer requires.

### Training script

```bash
python scripts/train.py
```

Outputs:
- `models/saved/best_model.joblib`
- `models/saved/tfidf_vectorizer.joblib`
- `models/saved/feature_config.joblib`
- `models/saved/feature_names.joblib`
- `models/saved/model_metadata.json`
- `models/saved/experiment_results.json`
- `training.log`

## Evaluation

Evaluation was run on the held-out `test.csv` (3,592 samples, unseen during training and validation).

### Metrics (E3 — selected model)

| Metric | Value |
|---|---|
| Accuracy | 98.64% |
| Macro F1 | 0.9761 |
| Weighted F1 | 0.9864 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LEGITIMATE | 0.9909 | 0.9877 | 0.9893 | 2,196 |
| PHISHING | 0.9795 | 0.9853 | 0.9824 | 1,361 |
| MALICIOUS | 0.9706 | 0.9429 | 0.9565 | 35 |

### Confusion matrix

```
                     Predicted L   Predicted P   Predicted M
Actual Legitimate       2,169          26              1
Actual Phishing            20       1,341              0
Actual Malicious            0           2             33
```

2 malicious emails were misclassified as phishing. 0 were misclassified as legitimate.

### Inference performance

Mean CPU inference latency: ~12ms per email (measured over 100 iterations in `test_qa_regression.py`). Well within a 50ms interactive response threshold.

## Reproducibility

- Random seed fixed at 42 across all data splits, vectorizer fitting, and model training
- All artifacts are serialized with `joblib` and deserialize consistently
- `scripts/train.py` is deterministic on the same data and Python version

## Known ML limitations

- The MALICIOUS class has only 177 total samples and 35 in the test set. The 94.3% recall is encouraging but should not be treated as reliable performance on a diverse malware corpus.
- The model was trained on English-language emails. Performance on multilingual email is untested and likely poor.
- Domain shift is possible: emails from environments significantly different from the training corpus may produce worse performance.
- The model cannot explain _why_ a token contributed in the way a human can. The token attribution is coefficient-based and represents statistical correlation, not causal security reasoning.
