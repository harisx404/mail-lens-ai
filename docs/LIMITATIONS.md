# PhishGuard AI — Limitations

## Honest Assessment of Project Limitations

This document transparently describes the known limitations of PhishGuard AI. This system is a **research/educational capstone prototype** and should NOT be treated as a production email security product.

---

## Dataset Limitations

### 1. Class Imbalance
The MALICIOUS class has significantly fewer training samples (~100-200) compared to LEGITIMATE and PHISHING (~9,000+ each). This means:
- The model may be less reliable at detecting malicious emails
- Per-class metrics for MALICIOUS may be lower
- Class weighting is used to mitigate but cannot fully compensate

### 2. Dataset Representativeness
- Training data comes from specific public datasets, which may not represent the full diversity of real-world email threats
- Modern phishing techniques (AI-generated phishing, highly targeted spear-phishing) may not be well-represented
- The dataset is English-only — non-English emails are not supported

### 3. Dataset B Message Quality
- Dataset B (Zenodo) contains "email or SMS-like" messages (624 total), not necessarily full email formats
- These shorter messages may not represent typical email patterns

### 4. Domain Shift
- Models trained on historical data may not generalize to new, unseen attack patterns
- Phishing techniques evolve constantly — the model cannot detect previously unseen attack types

---

## Model Limitations

### 5. No Guaranteed Detection
- No ML model achieves 100% accuracy
- False positives (legitimate email flagged as threat) and false negatives (threat missed) are expected
- The model cannot detect zero-day attacks or novel social engineering techniques

### 6. Text-Only Analysis
- The model analyzes text content only
- No analysis of: email headers (SPF, DKIM, DMARC), sender reputation, attachment binaries, network indicators
- Real email security requires multiple layers of defense

### 7. Static Analysis Only
- No URLs are visited or resolved
- No files are executed or sandboxed
- No DNS lookups performed
- This limits detection of sophisticated attacks that require dynamic analysis

### 8. Baseline Models
- The project uses traditional ML models (LR, NB, SVM) with TF-IDF features
- More sophisticated approaches (transformer models, graph neural networks) may perform better
- The baseline approach is chosen for interpretability and computational efficiency

---

## Application Limitations

### 9. Risk Score
- The risk assessment is an **application-level heuristic**, not an objectively validated security rating
- It combines model confidence with keyword-based indicators
- It should not be used as a sole basis for security decisions

### 10. Security Indicators
- The detected security indicators (urgency, credentials, etc.) are based on keyword matching
- They are NOT the exact reasons the ML model made its prediction
- They are additional contextual signals, not causal explanations

### 11. No Real-Time Processing
- The application processes individual emails on demand
- No real-time email stream processing
- No integration with email servers (IMAP, Exchange, etc.)

### 12. Session-Only History
- Analysis history is stored in Streamlit session state
- History is lost when the application restarts
- No persistent database storage

---

## Security Limitations

### 13. Not a Replacement
This system **cannot replace** professional email security solutions such as:
- Secure Email Gateways (SEGs)
- Email sandboxing solutions
- Threat intelligence platforms
- SIEM systems
- Anti-malware products

### 14. Adversarial Vulnerability
- An attacker aware of the model's features could craft emails to evade detection
- Text obfuscation, homoglyph attacks, and image-based phishing are not handled
- The system does not have adversarial robustness training

---

## What This System IS

✅ A research/educational prototype demonstrating NLP + ML for email security
✅ A capstone project showing supervised learning, feature engineering, and evaluation
✅ An explainable analysis tool with transparent limitations
✅ A well-documented, reproducible ML pipeline

## What This System IS NOT

❌ A production email security product
❌ A guaranteed phishing detector
❌ A replacement for security training and awareness
❌ A real-time email filtering system
