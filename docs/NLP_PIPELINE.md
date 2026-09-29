# PhishGuard AI — NLP Pipeline Documentation

## Overview

The NLP pipeline transforms raw email text into a numerical representation suitable for machine learning classification.

## Pipeline Architecture

```
Raw Email Text
    ↓
1. HTML Tag Removal
    ↓
2. URL Extraction → Security Features (separate path)
   URL Replacement → "urlplaceholder"
    ↓
3. Email Address Normalization → "emailplaceholder"
    ↓
4. Unicode Normalization (NFKD)
    ↓
5. Lowercasing
    ↓
6. Special Character Removal
    ↓
7. Whitespace Normalization
    ↓
8. Lemmatization (WordNet)
    ↓
9. Token Length Filtering (≥2 chars)
    ↓
Clean Text
    ↓
10. TF-IDF Vectorization
    ↓
Feature Vector (up to 10,000 dimensions)
```

## Design Decisions

### Stopword Removal: OFF by default
Phishing-relevant words like "your", "you", "please", "immediately" are standard English stopwords but carry important signal for phishing detection. Removing them would lose information.

### Lemmatization: ON by default
Lemmatization normalizes word forms ("running" → "run", "accounts" → "account") which reduces vocabulary size and helps the model generalize. We use WordNet lemmatizer via NLTK.

### URL/Email Placeholders
URLs and email addresses are replaced with placeholders rather than removed. This preserves the information that "this text contained URLs" while preventing the model from memorizing specific domains.

### TF-IDF Configuration

| Parameter | Value | Justification |
|---|---|---|
| max_features | 10,000 | Balance between information and dimensionality |
| ngram_range | (1, 2) | Captures single words and two-word phrases |
| min_df | 2 | Ignore terms appearing only once (likely noise) |
| max_df | 0.95 | Ignore terms in 95%+ of documents (too common) |
| sublinear_tf | True | Log normalization reduces impact of very frequent terms |

## Implementation

- **Module:** `src/preprocessing/text_preprocessor.py`
- **Class:** `TextPreprocessor`
- **Dependencies:** BeautifulSoup, NLTK (punkt, stopwords, wordnet)

## Leakage Prevention

The TF-IDF vectorizer is **fit only on training data**. Validation and test data are **transformed only** — this prevents vocabulary information from the test set from influencing the model.
