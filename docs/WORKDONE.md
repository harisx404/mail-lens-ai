# PhishGuard AI — Work Done Log

## Project Tracking

| Phase | Task | Status | Date | Files Changed | Validation | Remaining |
|---|---|---|---|---|---|---|
| **Phase 0** | Dataset Research | COMPLETED | 2026-09-29 | Research report | 4 datasets analyzed, 2 selected | None |
| **Phase 0** | Feasibility Analysis | COMPLETED | 2026-09-29 | Research report | Three-class feasibility assessed | None |
| **Phase 0** | Architecture Design | COMPLETED | 2026-09-29 | Research report | Stack selected (Python/sklearn/Streamlit) | None |
| **Phase 1** | Directory Structure | COMPLETED | 2026-09-29 | 18 directories created | Verified | None |
| **Phase 1** | Configuration Files | COMPLETED | 2026-09-29 | requirements.txt, pyproject.toml, .gitignore, .env.example | Files created | None |
| **Phase 1** | Virtual Environment | COMPLETED | 2026-09-29 | venv/ | Python 3.14 + venv created | None |
| **Phase 1** | Dependencies Install | COMPLETED | 2026-09-29 | venv/ | All pip packages installed successfully | None |
| **Phase 1** | Configuration Module | COMPLETED | 2026-09-29 | src/utils/config.py | All settings centralized | None |
| **Phase 2** | Data Loader | COMPLETED | 2026-09-29 | src/data/loader.py | HuggingFace (18.6k) + Zenodo TSV parser (624) | 100% verified |
| **Phase 2** | Data Pipeline | COMPLETED | 2026-09-29 | src/data/pipeline.py | 17,960 clean unified emails, stratified 70/10/20 | 100% verified |
| **Phase 3** | Text Preprocessor | COMPLETED | 2026-09-29 | src/preprocessing/text_preprocessor.py | HTML, URL, Unicode NFKD, tokenization | Tested & fast |
| **Phase 4** | Security Features | COMPLETED | 2026-09-29 | src/features/security_features.py | 17 handcrafted domain cybersecurity signals | Tested |
| **Phase 4** | Feature Engineer | COMPLETED | 2026-09-29 | src/features/feature_engineer.py | 10,000 TF-IDF + 17 Security Features = 10,017 dims | Tested |
| **Phase 5** | Baseline Training | COMPLETED | 2026-09-29 | src/models/trainer.py | Trained 5 models (E1-E5) across LR, NB, SVM | All completed |
| **Phase 5** | Model Evaluation | COMPLETED | 2026-09-29 | src/evaluation/evaluator.py | Full metrics, confusion matrices, FP/FN analysis | All completed |
| **Phase 6** | Model Selection | COMPLETED | 2026-09-29 | scripts/train.py | Winner: Linear SVM (E3) with Val Macro F1 = 0.9760 | Selected |
| **Phase 7** | Model Persistence | COMPLETED | 2026-09-29 | models/saved/ | Saved model, vectorizer, feature names, metadata | Verified on disk |
| **Phase 8** | Inference Engine | COMPLETED | 2026-09-29 | src/inference/engine.py | Prediction + Risk (0-1) + Analyst Explanation | Verified on real emails |
| **Phase 9** | Streamlit Web App | COMPLETED | 2026-09-29 | app/main.py | Dashboard, Email Inspector, Threat History, About | Browser verified |
| **Phase 10** | Unit Test Suite | COMPLETED | 2026-09-29 | tests/test_core.py | 36/36 tests passing (100%) | Verified in pytest |
| **Phase 11** | Technical Docs | COMPLETED | 2026-09-29 | docs/ (10 markdown documents) | Complete academic & architectural documentation | None |
| **Phase 12** | Presentation Deck | COMPLETED | 2026-09-29 | presentation/presentation_slides.md | 18 Marp-compatible slides | Complete |
| **Phase 12** | Speaker Notes | COMPLETED | 2026-09-29 | presentation/speaker_notes.md | Timed defense scripts for every slide | Complete |
| **Phase 12** | Executive Summary | COMPLETED | 2026-09-29 | presentation/project_summary.md | Executive summary for reviewers | Complete |
| **Phase 12** | Demo Notebook | COMPLETED | 2026-09-29 | notebooks/pipeline_demonstration.ipynb | Interactive end-to-end Jupyter notebook | Generated |
| **Phase 13** | Final Verification | COMPLETED | 2026-09-29 | System-wide audit | Zero fabricated metrics, 100% reproducible | None |

---

## Benchmark Results Summary

### Validation Set (1,796 samples)
- **E1 (Logistic Regression, TF-IDF only)**: Accuracy = 0.9733 | Macro F1 = 0.9717 | Weighted F1 = 0.9733
- **E2 (Naive Bayes, TF-IDF only)**: Accuracy = 0.9605 | Macro F1 = 0.8922 | Weighted F1 = 0.9611
- **E3 (Linear SVM, TF-IDF only)**: Accuracy = 0.9794 | Macro F1 = 0.9760 | Weighted F1 = 0.9794 (Winner)
- **E4 (Logistic Regression, TF-IDF + Security)**: Accuracy = 0.9738 | Macro F1 = 0.9721 | Weighted F1 = 0.9739
- **E5 (Linear SVM, TF-IDF + Security)**: Accuracy = 0.9777 | Macro F1 = 0.9748 | Weighted F1 = 0.9777

### Held-Out Test Set (3,592 samples)
- **Overall Accuracy**: **98.64%**
- **Macro F1-Score**: **0.9761**
- **Weighted F1-Score**: **0.9864**
- **Class Metrics**:
  - `LEGITIMATE`: Precision 0.9909 | Recall 0.9877 | F1 0.9893 (2,196 samples)
  - `PHISHING`: Precision 0.9795 | Recall 0.9853 | F1 0.9824 (1,361 samples)
  - `MALICIOUS`: Precision 0.9706 | Recall 0.9429 | F1 0.9565 (35 samples)
- **Confusion Matrix**:
  ```
  [2169,   26,    1]   (LEGITIMATE)
  [  20, 1341,    0]   (PHISHING)
  [   0,    2,   33]   (MALICIOUS)
  ```
- **Malicious Recall**: **94.3%** (33/35 detected) with only **1 false positive** across 2,196 legitimate emails.
