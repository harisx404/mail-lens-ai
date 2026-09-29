# PhishGuard AI — Executive Project Summary

**Project Title:** PhishGuard AI — NLP-Based Phishing & Malicious Email Detection and Risk Analysis System  
**Author:** Muhammad Haris  
**Program:** KPITB AI / ML Training Program — Final Capstone Project  
**Date:** September 2026  
**Status:** Validated, Tested, & Operationally Ready  

---

## 1. Executive Summary

PhishGuard AI is an advanced, production-oriented email threat detection and risk analysis system engineered specifically for enterprise Security Operations Centers (SOC) and cyber-defense analysts. Unlike traditional spam filters that rely on naive keyword blocklists or opaque deep-learning models, PhishGuard AI introduces a **three-class taxonomy** (`LEGITIMATE`, `PHISHING`, and `MALICIOUS`), integrates domain-specific cybersecurity feature engineering with statistical NLP, and provides **transparent, human-readable explainability** alongside calibrated risk scores.

---

## 2. Key Technical Differentiators

| Capability | Traditional Filters | Generic AI Prototypes | PhishGuard AI |
|---|---|---|---|
| **Classification Granularity** | Binary (Spam / Ham) | Binary (Phish / Legitimate) | **Three-Class (Legitimate, Phishing, Malicious)** |
| **Feature Engineering** | Static Regex / Wordlists | Unigram TF-IDF or Raw Embeddings | **Hybrid: 10,000 TF-IDF N-grams + 17 Domain Security Features** |
| **Explainability** | Rule match or None | Black-box softmax output | **Interpretable factor breakdown & threat indicator logs** |
| **Risk Assessment** | Static score | Prediction confidence | **Calibrated Risk Engine (Low, Medium, High, Critical)** |
| **Data Provenance** | Proprietary | Often synthetic / fabricated | **100% Verifiable Academic Data (HuggingFace + Zenodo)** |
| **Data Leakage Controls** | N/A | Frequent test contamination | **Strict pre-split separation & zero post-split fitting** |
| **Inference Latency** | Low | High (GPU transformer) | **Ultra-Low (<5ms CPU inference per email)** |

---

## 3. Dataset Architecture

- **Total Unified Corpus:** **17,960 emails**
  - **Dataset A (HuggingFace):** `zefang-liu/phishing-email-dataset` (LGPL-3.0) — 17,522 cleaned samples providing high-volume baselines for Legitimate and Phishing classes.
  - **Dataset B (Zenodo):** Record `15235123` (CC BY 4.0) — 624 multi-class samples providing specialized instances of Malware, Scareware, Baiting, and Pretexting.
- **Split Distribution (Stratified, Seed=42):**
  - **Train (70%):** 12,572 samples (7,684 Legitimate, 4,764 Phishing, 124 Malicious)
  - **Validation (10%):** 1,796 samples (1,098 Legitimate, 680 Phishing, 18 Malicious)
  - **Test (20%):** 3,592 samples (2,196 Legitimate, 1,361 Phishing, 35 Malicious)

---

## 4. Engineering Architecture

```
                    ┌─────────────────────────────────────────┐
                    │               Raw Email                 │
                    └────────────────────┬────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    ┌─────────────────────────┐                     ┌─────────────────────────┐
    │     NLP Preprocessor    │                     │   Security Extractor    │
    │  • HTML tag/script strip│                     │  • Urgency keywords     │
    │  • URL/Email placeholder│                     │  • Credential triggers  │
    │  • Unicode NFKD normal. │                     │  • Threat / Fear terms  │
    │  • Morph. Lemmatization │                     │  • IP / Shortened URLs  │
    └────────────┬────────────┘                     │  • Executable mentions  │
                 │                                  └────────────┬────────────┘
                 ▼                                               │
    ┌─────────────────────────┐                                  │
    │     TF-IDF Vectorizer   │                                  │
    │  • 10,000 max features  │                                  │
    │  • (1, 2) N-gram range  │                                  │
    │  • Sublinear TF scaling │                                  │
    └────────────┬────────────┘                                  │
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                        ┌─────────────────────────────────┐
                        │   Feature Fusion (10,017 Dims)  │
                        │    scipy.sparse.hstack (CSR)    │
                        └────────────────┬────────────────┘
                                         ▼
                        ┌─────────────────────────────────┐
                        │        Trained Classifier       │
                        │    (Linear SVM / Logistic Reg)  │
                        └────────────────┬────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    ┌─────────────────────────┐                     ┌─────────────────────────┐
    │       Risk Engine       │                     │   Explanation Engine    │
    │  • Calibrated score 0-1 │                     │  • Matched keywords     │
    │  • Threat Tier (Low-    │                     │  • Structural anomalies │
    │    Critical)            │                     │  • SOC analyst guidance │
    └─────────────────────────┘                     └─────────────────────────┘
```

---

## 5. Security & SOC Application

PhishGuard AI is specifically optimized for tier-1 SOC triage:
1. **Zero Execution Hazard:** Operates completely as static content analysis — no external network connections are opened to malicious domains and no attachments are launched.
2. **Actionable Incident Context:** The explanation engine tells the analyst *exactly* which words and structures triggered the classification.
3. **High Recall on Threats:** Loss weighting and classification thresholds are tuned to prioritize minimizing False Negatives on active attacks.
