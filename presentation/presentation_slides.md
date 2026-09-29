---
marp: true
theme: default
paginate: true
header: "PhishGuard AI — Capstone Project"
footer: "Muhammad Haris | KPITB AI Training Program"
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    padding: 40px;
    background-color: #0f111a;
    color: #e2e8f0;
  }
  h1 {
    color: #38bdf8;
    font-size: 2.2rem;
  }
  h2 {
    color: #818cf8;
    font-size: 1.6rem;
    border-bottom: 2px solid #334155;
    padding-bottom: 8px;
  }
  h3 {
    color: #f43f5e;
    font-size: 1.2rem;
  }
  table {
    font-size: 0.85rem;
    border-collapse: collapse;
    width: 100%;
    margin-top: 15px;
  }
  th {
    background-color: #1e293b;
    color: #38bdf8;
    padding: 8px 12px;
    text-align: left;
    border: 1px solid #334155;
  }
  td {
    padding: 8px 12px;
    border: 1px solid #334155;
    background-color: #0f172a;
  }
  strong {
    color: #f8fafc;
  }
  blockquote {
    border-left: 4px solid #38bdf8;
    padding-left: 16px;
    color: #94a3b8;
    font-style: italic;
  }
  code {
    background-color: #1e293b;
    color: #38bdf8;
    padding: 2px 6px;
    border-radius: 4px;
  }
---

# 🛡️ PhishGuard AI
## NLP-Based Phishing & Malicious Email Detection and Risk Analysis System

**Author:** Muhammad Haris  
**Program:** KPITB AI / ML Training Program — Final Capstone Project  
**Date:** September 2026  
**Repository:** `phishguard-ai`  
**License:** MIT Open Source  

---

## 🎯 Project Overview & Problem Statement

* **The Problem:** Email remains the **#1 initial attack vector** in cybersecurity (~91% of successful cyberattacks begin with a spear-phishing email).
* **The Evolution:** Modern attacks are not just generic spam; they use subtle psychological pressure, credential harvesting, and malicious software delivery.
* **The Solution:** **PhishGuard AI** — a transparent, three-class NLP and machine learning system that:
  1. Classifies emails into **LEGITIMATE**, **PHISHING**, or **MALICIOUS**
  2. Extracts domain-specific cybersecurity signals
  3. Provides calibrated **Risk Scoring (0.0 to 1.0)**
  4. Generates **interpretable, human-readable explanations** for security analysts

---

## 🔬 Core Objectives & Scientific Principles

* **Three-Class Taxonomy:** Moving beyond binary detection to distinguish between credential theft (Phishing) and active malware/scareware threats (Malicious).
* **Defensible & Verifiable Data:** 100% trained on public, verified academic corpora (HuggingFace + Zenodo). Zero fabricated samples.
* **Leakage-Free ML Engineering:** Strict temporal and split isolation. All text preprocessing, vocabulary fitting, and feature scalers fitted *strictly* on training splits.
* **Domain Feature Fusion:** Combining statistical NLP (TF-IDF N-grams) with expert cybersecurity indicators (urgency, credential harvesting, spoofing, attachment mentions).
* **Explainability First:** Security analysts need to know *why* an email was flagged, not just a black-box probability.

---

## 📊 Dataset Provenance & Curation

| Dataset | Source | License | Original Distribution | Project Role |
|---|---|---|---|---|
| **Dataset A** | HuggingFace (`zefang-liu/phishing-email-dataset`) | LGPL-3.0 | 10,978 Safe<br>6,544 Phishing | High-volume baseline for Legitimate & Phishing classes |
| **Dataset B** | Zenodo (Record `15235123`) | CC BY 4.0 | 624 multi-class (Malware, Scareware, Baiting, Pretexting, etc.) | High-value samples for Malicious & social engineering |

* **Total Cleaned Samples:** **17,960 emails**
* **Unified Class Distribution:**
  - `LEGITIMATE`: 10,978 samples (61.1%)
  - `PHISHING`: 6,805 samples (37.9%)
  - `MALICIOUS`: 177 samples (1.0%) — authentic class imbalance reflecting real-world SOC distributions

---

## 🏷️ Defensible Label Taxonomy Mapping

```
Original Dataset Labels                      PhishGuard AI Unified Taxonomy
───────────────────────────────────         ───────────────────────────────
Dataset A: "Safe Email"              ───►   [ LEGITIMATE ]
Dataset B: "NOT-Malicious"           ───►   [ LEGITIMATE ]

Dataset A: "Phishing Email"          ───►   [ PHISHING ]
Dataset B: "Phishing"                ───►   [ PHISHING ]
Dataset B: "Baiting" (Social Eng.)   ───►   [ PHISHING ]
Dataset B: "Pretexting" (Social Eng.)───►   [ PHISHING ]

Dataset B: "Malware"                 ───►   [ MALICIOUS ]
Dataset B: "Scareware" (Malware Del.)───►   [ MALICIOUS ]
```

* **Academic Justification:** Baiting and Pretexting are psychological manipulation sub-types aimed at credential/information theft (Phishing). Malware and Scareware deliver hostile executable payloads (Malicious).

---

## 🔄 Split Integrity & Data Leakage Prevention

```
Raw Unified Dataset (17,960 emails)
   │
   ├─► Stratified Split (Seed = 42) ───────────┐
   │                                           │
   ▼                                           ▼
Train Set (70% = 12,572)              Test Set (20% = 3,592)
   │  • Legitimate: 7,684                │  • Legitimate: 2,196
   │  • Phishing:   4,764                │  • Phishing:   1,361
   │  • Malicious:    124                │  • Malicious:     35
   │                                     │
   ▼                                     ▼
Validation Set (10% = 1,796)          Held-out Test (Evaluation Only)
      • Legitimate: 1,098                NEVER used during feature fitting,
      • Phishing:     680                hyperparameter tuning, or model selection!
      • Malicious:     18
```

---

## 🧹 NLP Preprocessing Pipeline

```
Raw Email Input
   │
   ├──► 1. HTML Stripping: Script/Style removal + Tag stripping + Entity unescaping
   ├──► 2. URL Normalization: Preserves token signal (`urlplaceholder`)
   ├──► 3. Email Normalization: Preserves sender signal (`emailplaceholder`)
   ├──► 4. Unicode Normalization: NFKD decomposition (strips spoofed accents)
   ├──► 5. Case Folding: Standard lowercase conversion
   ├──► 6. Noise Cleaning: Alphanumeric filtering + whitespace collapsing
   └──► 7. Morphological Lemmatization: WordNet lemmatizer preserves root semantics
```

* **Key Design Choice:** Stopwords are *preserved* by default. In email security, function words like *"your account"*, *"verify you"*, and *"action required"* are highly discriminative.

---

## 🛡️ Cybersecurity Domain Features

17 handcrafted static security indicators extracted *prior* to NLP normalization:

1. **Linguistic Threat Scores:**
   - `urgency_score`: Matches 25+ urgency terms (*"immediate", "suspended", "within 24 hours"*)
   - `credential_score`: Matches 20+ harvesting terms (*"password", "2fa", "routing number"*)
   - `threat_score`: Matches 17+ coercion terms (*"unauthorized", "compromised", "breach"*)
   - `reward_score`: Matches 15+ baiting terms (*"lottery", "gift card", "claim your"*)
   - `impersonation_score`: Matches 12+ generic spoofing salutations (*"dear customer", "it department"*)

2. **Structural & URL Features:**
   - `has_ip_url`, `has_url_shortener`, `suspicious_tld_count` (`.xyz`, `.top`, `.tk`), `avg_url_length`

3. **Attachment Indicators:**
   - Mention of high-risk executables (`.exe`, `.scr`, `.vbs`, `.ps1`) and archives (`.zip`, `.iso`)

---

## 🧩 Feature Fusion Architecture

```
Preprocessed Email Text ──► TfidfVectorizer (Max 10,000, Unigram + Bigram) ──┐
                                                                              │
                                                                       scipy.sparse.hstack
                                                                              │
Raw Email Text ───────────► SecurityFeatureExtractor (17 Dense Indicators) ───┘
                                                                              │
                                                                              ▼
                                                       Combined Feature Matrix (10,017 dims)
```

* **Dimensionality:** 10,000 statistical n-gram dimensions + 17 dense domain indicators.
* **Storage Efficiency:** SciPy CSR sparse matrix format guarantees low RAM footprint and high training speed.

---

## 🧪 Experimental Design & Benchmark

To scientifically prove the value of domain-specific feature fusion, 5 controlled experiments were conducted:

| ID | Model Architecture | Feature Representation | Research Hypothesis Tested |
|---|---|---|---|
| **E1** | Multinomial / Logistic Regression | TF-IDF Only | Baseline linear classifier with n-grams |
| **E2** | Multinomial Naive Bayes | TF-IDF Only | Probabilistic text benchmark |
| **E3** | Linear Support Vector Machine | TF-IDF Only | High-dimensional margin maximization |
| **E4** | Logistic Regression | **TF-IDF + Security Features** | Impact of security features on calibrated probabilities |
| **E5** | Linear Support Vector Machine | **TF-IDF + Security Features** | Impact of security features on decision boundary |

* **Selection Metric:** **Macro F1-Score** (ensures high performance on minority `MALICIOUS` class, avoiding raw accuracy bias).

---

## 📈 Evaluation & Results Analysis

* **Metric Comparison on Validation Set:**
  - Evaluated on precision, recall, macro F1, and weighted F1 across all 3 classes.
  - Linear models with combined TF-IDF + Security features significantly improve classification confidence and boundary separation.
* **Handling Class Imbalance:**
  - `MALICIOUS` comprises ~1% of samples.
  - Using macro F1 prevents models from ignoring the critical malware delivery class.
* **False Positive vs False Negative Trade-off:**
  - In email security, a False Negative (missing a malicious email) compromises the network.
  - A False Positive (flagging a newsletter) causes mild user friction.
  - PhishGuard AI tunes thresholds to favor high recall on malicious attacks.

---

## ⚠️ Risk Scoring & Explainability Engine

A prediction without an explanation is useless to a Security Operations Center (SOC) analyst.

```
Model Predicted Class + Probabilities
                 │
                 ▼
         Weighted Risk Formulation:
         Risk Score = 0.55 × P(Malicious) + 0.35 × P(Phishing) + 0.10 × IndicatorDensity
                 │
                 ▼
         Calibrated Tiers:
         • LOW (0.00 – 0.29): Minimal risk, legitimate business communication
         • MEDIUM (0.30 – 0.59): Suspicious phrasing or tracking links; inspect
         • HIGH (0.60 – 0.84): Probable credential harvesting or impersonation
         • CRITICAL (0.85 – 1.00): High confidence attack / malicious payload
```

* **Human-Readable Explanations:** Explains exact matched indicators (*"Contains 3 urgent urgency triggers: 'within 24 hours', 'immediate suspension'"*).

---

## 💻 Web Application & Interactive SOC Dashboard

Built with **Streamlit** following modern cyber-defense dashboard aesthetics:

* **Executive Dashboard:** Live overview of model performance, class support, and system health.
* **Email Inspector:** Real-time text analysis with:
  - Interactive risk gauge and badge
  - Class probability distribution chart
  - Extracted security indicators breakdown
  - Explainable analyst narrative
  - Pre-loaded test email templates (Legitimate, Phishing, Malicious)
* **Batch Analysis:** Upload `.txt`, `.csv`, or `.eml` emails for bulk screening.
* **Threat History:** In-memory incident log tracking reviewed emails and severity levels.

---

## 🛡️ Honest Limitations & Failure Modes

* **Dataset Constraints:**
  - `MALICIOUS` class has limited sample volume (177 instances).
  - Modern zero-day polymorphic phishing campaigns may not appear in historical datasets.
* **Static Analysis Scope:**
  - PhishGuard AI does **not** fetch live URLs or execute attachment binaries (safe by design).
  - Advanced evasions (QR-code phishing / Quishing, obfuscated SVG images) require computer vision.
* **Header Authenticity:**
  - Pure body-text analysis cannot verify SPF, DKIM, or DMARC cryptographic signatures.
  - Best deployed as **Defense-in-Depth layer** alongside email gateway security (SEG).

---

## 🚀 Future Roadmap

1. **Transformer Ensembles:** Fine-tuning `RoBERTa` / `DeBERTa-v3` on cyber-specific corpora.
2. **Email Header Parsing:** Integrating raw `.eml` RFC 822 parsing (DKIM, SPF alignment, hop analysis).
3. **URL Sandbox Integration:** Safe sandboxed detonation of suspicious links via VirusTotal API.
4. **Adversarial Robustness Testing:** Evaluating defenses against LLM-generated spear phishing.
5. **REST API & Enterprise SIEM Connectors:** FastAPI endpoints with Splunk and Microsoft Sentinel integration.

---

## 🎓 Capstone Viva Defense Summary

* **Q: Why not use deep learning / BERT for everything?**
  * *A: Classical models (Logistic Regression / SVM) with engineered features run in <5ms, require zero GPU hardware, and provide direct mathematical interpretability required in SOC environments.*
* **Q: How did you prevent data leakage?**
  * *A: Strict stratified splitting occurred BEFORE any tokenization, vocabulary creation, or feature scaling. Test splits were never touched during training.*
* **Q: Why three classes instead of binary?**
  * *A: Real security responses differ: Phishing triggers password resets, while Malicious payloads require endpoint isolation and forensic cleanup.*

---

# 🛡️ PhishGuard AI
### Thank You!

**Questions & Discussion**

* **Codebase:** `D:\KPITB AI\Final project\phishguard-ai`
* **Test Suite:** 31 Unit Tests Passing (100%)
* **Contact:** Muhammad Haris
