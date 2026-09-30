# Mail-Lens AI — Project Blueprint

## Vision

Build an NLP and machine-learning-based system that analyzes email content to detect phishing and malicious emails, providing classification, confidence, risk assessment, and human-readable explanations — suitable as an AI/ML capstone project.

---

## Problem Statement

Phishing emails remain a primary vector for cyberattacks, causing billions of dollars in losses annually. Traditional email filters use rule-based detection that can be evaded by sophisticated social engineering. This project applies Natural Language Processing to learn patterns from email text and classify emails based on their linguistic and structural characteristics.

---

## Scope

### In Scope
- Email text classification (subject + body)
- Three-class taxonomy: LEGITIMATE, PHISHING, MALICIOUS
- NLP preprocessing pipeline (HTML removal, tokenization, TF-IDF)
- Cybersecurity feature engineering (URL analysis, keyword indicators)
- Supervised ML models (LR, NB, SVM)
- Model evaluation with full metrics
- Risk scoring and explainability
- Web-based analysis interface (Streamlit)
- Professional documentation and presentation

### Non-Goals (Explicitly Out of Scope)
- Real-time email interception / mailbox integration
- Dynamic malware analysis / sandboxing
- Live URL visiting or DNS resolution
- Email header authentication (SPF/DKIM/DMARC)
- Production deployment with authentication
- Multi-language support
- LLM-based analysis
- SIEM integration
- Microservices architecture

---

## Requirements

### Functional
1. Classify emails into LEGITIMATE, PHISHING, or MALICIOUS
2. Provide confidence/probability for each class
3. Assess risk level (LOW, MEDIUM, HIGH, CRITICAL)
4. Detect security indicators (urgency, credentials, URLs, etc.)
5. Generate human-readable explanations
6. Provide security recommendations
7. Maintain analysis history in the web interface
8. Support free-text email input via web form

### Non-Functional
1. Reproducible training pipeline (fixed random seed)
2. No data leakage (fit on train only)
3. Honest evaluation (no fabricated metrics)
4. Configurable via environment variables
5. Tested with pytest
6. Documented with honest limitations

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   MAIL-LENS AI                   │
│                                                  │
│  Input Layer                                     │
│  ┌──────────────────────────────────────────┐    │
│  │ Email (subject + body + optional sender) │    │
│  └──────────────┬───────────────────────────┘    │
│                 ↓                                │
│  Processing Layer                                │
│  ┌──────────────┴───────────────────────────┐    │
│  │ Text Preprocessor                        │    │
│  │ (HTML, URL, Unicode, lowercase, lemma)   │    │
│  └──────────────┬───────────────────────────┘    │
│                 ↓                                │
│  Feature Layer                                   │
│  ┌──────────────┴──────────────┐                 │
│  │ NLP Features    │ Security  │                 │
│  │ (TF-IDF)       │ Features  │                 │
│  └──────────────┬──┴──────────┘                 │
│                 ↓                                │
│  Model Layer                                     │
│  ┌──────────────┴───────────────────────────┐    │
│  │ ML Classifier (best of LR/NB/SVM)       │    │
│  └──────────────┬───────────────────────────┘    │
│                 ↓                                │
│  Output Layer                                    │
│  ┌──────────────┴───────────────────────────┐    │
│  │ Prediction + Risk + Explanation          │    │
│  └──────────────┬───────────────────────────┘    │
│                 ↓                                │
│  Presentation Layer                              │
│  ┌──────────────┴───────────────────────────┐    │
│  │ Streamlit Web Application                │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

---

## Dataset Strategy

**Combined dataset from two verified public sources:**

| Source | Samples | Labels | License |
|---|---|---|---|
| HuggingFace (zefang-liu) | ~18,650 | Safe/Phishing | LGPL-3.0 |
| Zenodo (Eng. Informatica) | 624 | 6-class | CC BY 4.0 |

**Known limitation:** MALICIOUS class severely underrepresented. Mitigated with class weighting and honest documentation.

---

## NLP Strategy

- TF-IDF vectorization with unigrams + bigrams
- HTML removal via BeautifulSoup
- URL/email normalization (replaced with placeholders)
- Unicode normalization (NFKD)
- Optional stopword removal (tested both ways)
- Lemmatization via NLTK WordNetLemmatizer

---

## ML Strategy

1. Train three baseline models: LR, NB, SVM
2. Compare with TF-IDF only vs TF-IDF + Security features
3. Select best by Macro F1 (balanced per-class performance)
4. Evaluate on held-out test set
5. Class weighting for imbalance handling

---

## Cybersecurity Strategy

- Keyword-based indicator detection (urgency, credentials, threats, rewards)
- Static URL analysis (count, HTTP/HTTPS, IP-based, shorteners, suspicious TLDs)
- Attachment keyword detection (executables, archives)
- All analysis is STATIC — no URLs visited, no files executed

---

## UI Strategy

- Streamlit for single-deployment simplicity
- Four pages: Dashboard, Analyze, History, About
- Professional dark theme with gradient headers
- Session state for analysis history
- Risk badges with color coding
- Clear distinction between model prediction and heuristic indicators

---

## Testing Strategy

- pytest for all tests
- Test categories: preprocessing, security features, data pipeline, configuration, edge cases
- Target: 60%+ code coverage
- Manual testing for web UI

---

## Documentation Strategy

- README.md: Overview, installation, usage
- PROJECT_BLUEPRINT.md: Architecture and decisions (this file)
- PROJECT_MASTER_GUIDE.md: Comprehensive beginner-friendly explanation
- DATASET.md: Data sources, labels, processing
- VIVA_QA.md: 40+ Q&A for viva preparation
- WORKDONE.md: Phase tracking log
- MANUAL_TASKS.md: Tasks requiring user action
- LIMITATIONS.md: Honest limitations
- FUTURE_WORK.md: Improvement roadmap

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| MALICIOUS class too small | HIGH | Class weighting, binary fallback, honest docs |
| Dataset domain shift | HIGH | Document as limitation, no production claims |
| Overfitting | MEDIUM | Cross-validation, stratified split |
| pip/dependency issues | LOW | Fixed versions, fallback instructions |
| Transformer too slow | LOW | Optional Phase 7, baselines are priority |

---

## Milestones

1. ✅ Phase 0: Research & Feasibility
2. ✅ Phase 1: Environment & Repository
3. ✅ Phase 2-4: Data + NLP + Features (code complete)
4. ✅ Phase 5: Model Training (code complete)
5. ✅ Phase 8: Inference Pipeline (code complete)
6. ✅ Phase 9: Web Application (code complete)
7. ✅ Phase 10: Test Suite (code complete)
8. ⬜ Run actual training pipeline
9. ⬜ Verify actual metrics
10. ⬜ Complete all documentation with real results
11. ⬜ Create presentation with actual screenshots
12. ⬜ Final audit

---

## Deliverables

- Working training pipeline
- Working inference pipeline
- Working web application
- Trained model artifacts
- Test suite
- Professional documentation
- Capstone presentation
- Viva Q&A preparation
