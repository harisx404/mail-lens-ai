# Mail-Lens AI — Future Work

## Potential Improvements

### 1. Larger & More Balanced Dataset
- **Problem:** MALICIOUS class has ~100-200 samples
- **Solution:** Integrate additional malware email corpora (CEAS, Nazario, custom collection)
- **Impact:** Significantly improved MALICIOUS class detection

### 2. Transformer-Based Models
- **What:** Fine-tune DistilBERT or BERT for email classification
- **Why:** Contextual embeddings capture semantic meaning better than TF-IDF
- **Consideration:** Requires GPU for training; increases inference time

### 3. Multi-Language Support
- **What:** Support non-English emails
- **How:** Multilingual models (mBERT, XLM-R) or language detection + per-language models
- **Impact:** Global applicability

### 4. Email Header Analysis
- **What:** Analyze SPF, DKIM, DMARC records, sender authentication
- **Impact:** Detects spoofed senders, a key phishing indicator

### 5. SHAP-Based Explainability
- **What:** Use SHAP (SHapley Additive exPlanations) for per-prediction feature attribution
- **Impact:** Shows exactly which words/features drove the model's decision

### 6. Active Learning
- **What:** Allow users to provide feedback on predictions to improve the model
- **How:** Store user corrections, periodically retrain with feedback data
- **Impact:** Continuous model improvement

### 7. Real-Time API
- **What:** FastAPI endpoint for programmatic email analysis
- **Impact:** Integration with email gateways and security tools

### 8. Persistent Storage
- **What:** SQLite/PostgreSQL database for analysis history
- **Impact:** Persistent history, trend analysis, analytics dashboard

### 9. Adversarial Robustness
- **What:** Test and improve resilience against adversarial email crafting
- **How:** Adversarial training, text augmentation
- **Impact:** Harder for attackers to evade detection

### 10. Ensemble Models
- **What:** Combine multiple classifiers (voting, stacking)
- **Impact:** Potentially higher accuracy and robustness
