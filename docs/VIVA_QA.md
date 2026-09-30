# Mail-Lens AI — Viva Questions & Answers

> **40+ realistic questions covering NLP, ML, Cybersecurity, Evaluation, and the Project.**
> All answers match the actual implementation.

---

## NLP Questions

### Q1: What is NLP?
**A:** Natural Language Processing (NLP) is a branch of AI that enables computers to understand, interpret, and process human language. In Mail-Lens AI, NLP is used to analyze the textual content of emails to identify patterns associated with phishing and malicious emails.

### Q2: Why is NLP useful for phishing detection?
**A:** Phishing emails rely on language patterns — urgency, deception, impersonation. NLP can learn these linguistic patterns from training data and generalize to detect similar patterns in unseen emails. Simple keyword matching fails because attackers vary their wording; NLP captures statistical patterns across thousands of examples.

### Q3: What is tokenization?
**A:** Tokenization is the process of splitting text into individual units (tokens), typically words or subwords. In our project, we tokenize email text after preprocessing to convert it into a format suitable for TF-IDF feature extraction. We use whitespace-based tokenization after cleaning.

### Q4: What is TF-IDF?
**A:** TF-IDF stands for Term Frequency–Inverse Document Frequency. It measures how important a word is to a document relative to the entire corpus.
- **TF (Term Frequency):** How often a word appears in this document
- **IDF (Inverse Document Frequency):** How rare the word is across all documents
- **TF-IDF = TF × IDF:** High for words that are frequent in this document but rare overall

In Mail-Lens AI, we use TF-IDF to convert email text into numerical feature vectors with up to 10,000 features (unigrams + bigrams).

### Q5: What is an n-gram?
**A:** An n-gram is a contiguous sequence of n items from text. A unigram is a single word ("urgent"), a bigram is two consecutive words ("verify account"), a trigram is three ("click here now"). We use (1,2)-grams (unigrams + bigrams) to capture both individual words and common phrases.

### Q6: Why not only use keyword matching for phishing detection?
**A:** Keyword matching is brittle — attackers can paraphrase. "Verify your credentials" vs "confirm your login details" mean the same thing but use different keywords. TF-IDF + ML learns statistical patterns across many examples, capturing nuanced relationships that keyword lists miss. Our system uses BOTH: ML for the primary classification and keyword matching for supplementary security indicators.

### Q7: What preprocessing steps do you perform?
**A:**
1. HTML tag removal (BeautifulSoup)
2. URL extraction and replacement with `urlplaceholder`
3. Email address normalization to `emailplaceholder`
4. Unicode normalization (NFKD)
5. Lowercasing
6. Special character removal
7. Whitespace normalization
8. Lemmatization (WordNet)
9. Minimum token length filtering (2+ chars)

### Q8: Why do you replace URLs with placeholders instead of removing them?
**A:** The presence of URLs is itself an important signal. By replacing them with a placeholder, TF-IDF can still learn that "emails containing URLs" have different classification patterns, while the actual URL content is analyzed separately by the cybersecurity feature extractor.

---

## Machine Learning Questions

### Q9: What is supervised classification?
**A:** Supervised classification is an ML approach where the model learns from labeled examples (emails with known categories) and then predicts categories for unseen emails. The "supervision" comes from the labeled training data.

### Q10: Why did you choose Logistic Regression?
**A:** Logistic Regression is an excellent baseline for text classification because:
- It's interpretable (coefficients show feature importance)
- It works well with high-dimensional sparse features (TF-IDF)
- It naturally outputs probabilities
- It handles multi-class problems (multinomial)
- It supports class weighting for imbalance

### Q11: Why Naive Bayes?
**A:** Multinomial Naive Bayes is specifically designed for text classification with word counts/frequencies. It's fast, works well with small training sets, and has strong theoretical foundations for bag-of-words models. We used sample weighting to handle class imbalance since NB doesn't support class_weight directly.

### Q12: Why SVM (Support Vector Machine)?
**A:** Linear SVM finds the maximum-margin hyperplane separating classes in feature space. It excels with high-dimensional text features and is known for strong performance on text classification tasks. We wrapped it with CalibratedClassifierCV to get probability outputs.

### Q13: What is overfitting?
**A:** Overfitting occurs when a model memorizes training data patterns (including noise) rather than learning generalizable patterns. Signs include very high training accuracy but low test accuracy. We mitigate overfitting through train/test splitting, cross-validation (in SVM calibration), and using regularization (C parameter).

### Q14: What is underfitting?
**A:** Underfitting occurs when a model is too simple to capture the underlying patterns. Signs include low accuracy on both training and test data. If our models showed very poor performance, we would need more complex features or models.

### Q15: What is train/test data leakage?
**A:** Data leakage occurs when information from the test set influences the training process, leading to artificially inflated metrics. In Mail-Lens AI, we prevent leakage by:
- Splitting data BEFORE any preprocessing or feature extraction
- Fitting the TF-IDF vectorizer ONLY on training data
- Transforming (not fitting) validation and test data

### Q16: Why use stratified splitting?
**A:** Stratification ensures each split (train/val/test) preserves the original class distribution. This is critical when classes are imbalanced (our MALICIOUS class is small) — without stratification, a split might have zero MALICIOUS samples in the test set.

### Q17: How do you handle class imbalance?
**A:** Our MALICIOUS class has ~100-200 samples vs ~9,000+ for others. We handle this with:
- **Class weighting** (`class_weight='balanced'` in LR and SVM)
- **Sample weighting** (for Naive Bayes)
- **Stratified splitting** (preserve ratios in all splits)
- **Macro F1 for model selection** (treats all classes equally)
- **Honest documentation** of the limitation

---

## Evaluation Questions

### Q18: What is accuracy?
**A:** Accuracy = (correct predictions) / (total predictions). It's intuitive but misleading for imbalanced datasets. If 95% of emails are legitimate, a model predicting "LEGITIMATE" always achieves 95% accuracy while detecting zero threats.

### Q19: What is precision?
**A:** Precision = TP / (TP + FP). For the PHISHING class: "Of all emails the model flagged as phishing, what fraction actually were phishing?" High precision means few false alarms.

### Q20: What is recall?
**A:** Recall = TP / (TP + FN). For the PHISHING class: "Of all actual phishing emails, what fraction did the model catch?" High recall means few threats are missed.

### Q21: What is F1-score?
**A:** F1 = 2 × (Precision × Recall) / (Precision + Recall). It's the harmonic mean of precision and recall, useful when both matter. We prioritize F1 because both missing threats (low recall) and false alarms (low precision) are problematic.

### Q22: What is Macro F1?
**A:** Macro F1 computes F1 for each class independently, then averages. It treats all classes equally regardless of size. We use Macro F1 for model selection because it ensures the model performs well on ALL classes, including the small MALICIOUS class.

### Q23: What is a confusion matrix?
**A:** A confusion matrix shows predicted vs actual labels in a grid. For 3 classes, it's a 3×3 matrix where:
- Diagonal = correct predictions
- Off-diagonal = errors (which class was confused with which)
It reveals specific error patterns — e.g., if MALICIOUS emails are often confused with PHISHING.

### Q24: What is a false positive?
**A:** A false positive occurs when the model incorrectly classifies a legitimate email as phishing or malicious. In email security, this means a safe email gets flagged — annoying for users but not dangerous.

### Q25: What is a false negative?
**A:** A false negative occurs when the model fails to detect a phishing or malicious email, classifying it as legitimate. In email security, this is MORE DANGEROUS than a false positive because a threat reaches the user.

### Q26: Why is Macro F1 better than accuracy for your model selection?
**A:** Because our dataset is imbalanced. Accuracy would favor the majority classes. A model ignoring the MALICIOUS class entirely might still achieve high accuracy. Macro F1 ensures the model performs well on ALL classes, including the rare MALICIOUS class.

---

## Cybersecurity Questions

### Q27: What is phishing?
**A:** Phishing is a social engineering attack where an attacker sends deceptive messages (usually emails) designed to trick the recipient into revealing sensitive information (passwords, credit cards), clicking malicious links, or downloading malware. It exploits human psychology rather than technical vulnerabilities.

### Q28: What is social engineering?
**A:** Social engineering is the manipulation of people into performing actions or divulging confidential information. It exploits trust, urgency, authority, and fear. Phishing is the most common form of social engineering.

### Q29: What is a malicious email?
**A:** A malicious email is one that carries harmful payloads — malware attachments, links to exploit kits, ransomware delivery, or trojan downloaders. Unlike phishing (which primarily seeks information), malicious emails aim to infect the recipient's system.

### Q30: How is phishing different from malware delivery?
**A:** Phishing primarily uses deception to steal information (credentials, financial data). Malware delivery aims to install harmful software on the victim's device. In practice, they overlap — a phishing email might also deliver malware. Our model treats them as separate classes but acknowledges the overlap in documentation.

### Q31: What security indicators does your system detect?
**A:** Five categories:
1. **Urgency language** — "immediately", "act now", "expires"
2. **Credential requests** — "verify your password", "confirm your account"
3. **Threat language** — "unauthorized access", "compromised", "suspended"
4. **Reward language** — "congratulations", "won a prize", "free gift"
5. **Impersonation** — "dear customer", "support team", "security team"

Plus URL features (HTTP, IP-based, shorteners, suspicious TLDs) and attachment mentions (executable extensions).

### Q32: Why are false negatives important in email security?
**A:** A false negative means a threatening email reaches the user undetected. This can lead to credential theft, malware infection, financial loss, or data breaches. False negatives are generally more dangerous than false positives in security contexts, because one missed threat can cause significant damage.

### Q33: How could an attacker evade your system?
**A:** Several ways:
- Using images instead of text (our model only analyzes text)
- Using character obfuscation (homoglyphs, Unicode tricks)
- Novel phrasing not seen in training data
- Embedding phishing content in attachments
- Using trusted domains as intermediaries
- Zero-day social engineering techniques
This is a fundamental limitation of any ML-based detection system.

---

## Project-Specific Questions

### Q34: Why did you choose this project?
**A:** This project sits at the intersection of NLP and cybersecurity — two critical areas in AI. Phishing detection is a real-world problem that directly demonstrates how NLP can be applied to security. It covers supervised learning, feature engineering, text processing, and web application development — comprehensive skills for an AI/ML capstone.

### Q35: What is novel about your implementation?
**A:** The combination of NLP features (TF-IDF) with cybersecurity-specific features (URL analysis, urgency detection, credential request indicators). Most academic implementations use only text features. Our system also provides explainable results with risk scoring and security recommendations — not just a binary prediction.

### Q36: What datasets did you use and why?
**A:** Two verified public datasets:
1. **HuggingFace phishing-email-dataset** (18,650 samples, Safe/Phishing) — large, well-established, provides our binary classification backbone
2. **Zenodo multiclass NLP dataset** (624 samples, 6 classes including Malware/Scareware) — provides the MALICIOUS class samples

Both have permissive licenses (LGPL-3.0 and CC BY 4.0).

### Q37: What were your biggest limitations?
**A:** The MALICIOUS class having only ~100-200 training samples is the biggest limitation. The model may struggle to reliably detect malicious emails compared to phishing emails. I documented this honestly and used class weighting to mitigate.

### Q38: What would you improve with more time?
**A:**
1. Larger, balanced dataset for the MALICIOUS class
2. Fine-tuned DistilBERT for potentially higher accuracy
3. Email header analysis (SPF/DKIM/DMARC validation)
4. Multi-language support
5. SHAP-based model explainability
6. Persistent database for analysis history
7. Active learning from user feedback

### Q39: How would you deploy this system in production?
**A:** For production:
1. Containerize with Docker
2. Deploy behind an API (FastAPI) with authentication
3. Add persistent database (PostgreSQL)
4. Integrate with email gateway via IMAP/API
5. Add monitoring and alerting
6. Implement model versioning and A/B testing
7. Regular retraining with new threat data
8. Security audit and penetration testing

### Q40: How does the risk scoring work?
**A:** The risk score (0-1) combines:
- **Model confidence** (60% weight) — probability from the classifier
- **Security indicator density** (25% weight) — count of detected indicators
- **URL/attachment modifiers** (15% weight) — presence of suspicious URLs or executables

It maps to four levels: LOW (0-0.3), MEDIUM (0.3-0.6), HIGH (0.6-0.85), CRITICAL (0.85-1.0). This is clearly labeled as an **application-level heuristic**, not an objectively validated security rating.

### Q41: What happens from clicking "Analyze" until the result appears?
**A:**
1. User submits email text via Streamlit form
2. Subject and body are combined into full text
3. Security Feature Extractor runs on RAW text (URL detection, keywords)
4. Text Preprocessor cleans text (HTML, lowercase, lemmatize)
5. TF-IDF vectorizer transforms cleaned text to feature vector
6. Security features are appended to feature vector
7. ML model predicts class and probabilities
8. Risk score is computed from confidence + indicators
9. Human-readable explanation is generated
10. Result is displayed with risk badge, indicators, and recommendation

### Q42: How do you prevent data leakage?
**A:** Three mechanisms:
1. Data is split (train/val/test) BEFORE any preprocessing
2. TF-IDF vectorizer is FIT only on training data (`fit_transform` on train, `transform` on val/test)
3. Random seed is fixed for reproducibility

### Q43: Why Streamlit instead of a full web framework?
**A:** Streamlit is ideal for an ML capstone because:
- Single-file deployment (no separate frontend/backend)
- Native Python — all ML code stays in Python
- Built-in support for charts, forms, session state
- Fast to develop and demonstrate
- Professional enough for a capstone presentation
- Evaluators can see results immediately

### Q44: What is the difference between the model explanation and security indicators?
**A:** This is an important distinction:
- **Model prediction:** Based on learned statistical patterns from TF-IDF + features — this is WHAT the ML model decided
- **Security indicators:** Heuristic keyword/pattern matching — these are CONTEXTUAL signals detected separately

The indicators are NOT necessarily the reasons the model made its decision. They provide additional human-interpretable context. The documentation clearly distinguishes between the two.
