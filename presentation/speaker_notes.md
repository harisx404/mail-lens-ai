# PhishGuard AI — Capstone Presentation Speaker Notes

> **Comprehensive presentation script and defense guide.**  
> Designed for Muhammad Haris for the KPITB AI Training Program Capstone Defense.

---

## Slide 1: Title & Introduction
* **Estimated Time:** 45 seconds
* **Script:**  
  "Respected instructors, evaluation committee members, and fellow peers, good morning/afternoon. My name is Muhammad Haris, and today I am excited to present my AI/ML capstone project: **PhishGuard AI — An NLP-Based Phishing & Malicious Email Detection and Risk Analysis System**."  
* **Key Point to Emphasize:**  
  "This project bridges the gap between modern Natural Language Processing and practical Cybersecurity operations, designed specifically to detect social engineering attacks and explain its decisions to human analysts."

---

## Slide 2: Project Overview & Problem Statement
* **Estimated Time:** 1 minute 15 seconds
* **Script:**  
  "To understand why this system is essential, consider this reality: according to modern cybersecurity threat reports, over 90% of all organizational cyber breaches start with an email. Whether it is spear phishing targeting executives, credential harvesting portals, or malicious attachments delivering ransomware, email remains the softest entry point into any perimeter."  
  "However, traditional email filters frequently rely on static rule sets or basic keyword blacklists that attackers trivially bypass using obfuscated text, homoglyphs, or urgent social engineering phrasing. On the other hand, complex deep learning models are often deployed as opaque black boxes that offer no explanation. PhishGuard AI solves both challenges by combining fine-grained three-class classification with domain-specific security feature engineering and interpretable risk explanations."

---

## Slide 3: Core Objectives & Scientific Principles
* **Estimated Time:** 1 minute
* **Script:**  
  "When designing PhishGuard AI, our engineering was guided by five core scientific principles:  
  First, **Three-Class Taxonomy**: Rather than treating email security as a naive binary 'spam vs ham' problem, we distinguish between legitimate email, credential phishing, and hostile malicious software delivery.  
  Second, **Scientific Reproducibility**: Every number, model weight, and metric in this project comes from real, verifiable academic corpora. There are zero fabricated metrics or artificial datasets.  
  Third, **Strict Leakage Prevention**: We enforce zero data contamination across all processing boundaries.  
  Fourth, **Domain Feature Fusion**: We merge statistical NLP N-grams with expert cybersecurity signals.  
  And fifth, **Explainability First**: A security tool is only as trustworthy as the reasoning behind its alerts."

---

## Slide 4: Dataset Provenance & Curation
* **Estimated Time:** 1 minute 30 seconds
* **Script:**  
  "Let's discuss the data foundations. Machine learning systems are only as good as their data. For PhishGuard AI, we curated a unified dataset of **17,960 emails** by integrating two verified public datasets:  
  1. **Dataset A** from HuggingFace, providing over 17,000 real-world emails labeled as Safe or Phishing under the LGPL-3.0 license. This provides our high-volume baseline for normal correspondence and common phishing campaigns.  
  2. **Dataset B** from Zenodo, published under CC BY 4.0, which provides granular multi-class instances including malware, scareware, baiting, and pretexting.  
  Notice that our final cleaned dataset reflects real-world SOC class imbalance: Legitimate emails represent 61.1%, Phishing represents 37.9%, and active Malicious emails represent 1.0%. This realistic distribution ensures our models are tested under operational conditions rather than artificial 50/50 balance."

---

## Slide 5: Defensible Label Taxonomy Mapping
* **Estimated Time:** 1 minute
* **Script:**  
  "A common question in machine learning is: *How did you justify merging disparate dataset classes?*  
  We established a clear, defensible taxonomy mapping grounded in cybersecurity literature:  
  - 'Safe Email' and 'NOT-Malicious' map directly to **LEGITIMATE**.  
  - 'Phishing', 'Baiting', and 'Pretexting' map to **PHISHING**. In cybersecurity, pretexting and baiting are social engineering vectors whose primary objective is psychological deception to extract confidential credentials or personal data.  
  - 'Malware' and 'Scareware' map to **MALICIOUS**, because they deliver hostile executable payloads or coerce users into downloading trojans and ransomware. This mapping is academically sound and operationally meaningful."

---

## Slide 6: Split Integrity & Data Leakage Prevention
* **Estimated Time:** 1 minute 15 seconds
* **Script:**  
  "One of the most frequent flaws in student AI projects is data leakage — where information from the test set subtly contaminates the training phase.  
  In PhishGuard AI, we implemented strict pipeline boundaries:  
  - The dataset was partitioned into 70% Train (12,572 emails), 10% Validation (1,796 emails), and 20% Test (3,592 emails) using **stratified sampling** with a fixed random seed of 42.  
  - This split was executed **before** any NLP preprocessing, before fitting the TF-IDF vocabulary, and before extracting feature distributions.  
  - The test split was completely held out until the final model was selected on the validation set, ensuring that our reported test metrics reflect true out-of-sample generalization."

---

## Slide 7: NLP Preprocessing Pipeline
* **Estimated Time:** 1 minute 15 seconds
* **Script:**  
  "Our natural language processing pipeline converts noisy, unstructured email text into clean linguistic tokens through seven distinct stages:  
  First, HTML tag and script removal using optimized regular expressions.  
  Second, URL normalization, where raw URLs are extracted for security analysis, and then replaced in the text by a specialized `urlplaceholder` token. This preserves the semantic signal that a link was present without causing vocabulary explosion from unique domains.  
  Third, email address normalization replacing addresses with `emailplaceholder`.  
  Fourth, Unicode normalization using NFKD decomposition to neutralize homoglyph and accent spoofing.  
  Fifth, lowercasing and whitespace collapse.  
  And finally, WordNet morphological lemmatization.  
  Crucially, we opted **not** to remove stopwords by default. In email phishing, short grammatical phrases like 'your account', 'action required', or 'contact us' are critical behavioral indicators that generic stopword removal would destroy."

---

## Slide 8: Domain-Specific Cybersecurity Features
* **Estimated Time:** 1 minute 30 seconds
* **Script:**  
  "Pure bag-of-words NLP often struggles with domain context. For instance, an attacker might write a grammatically clean email that subtly conveys severe urgency.  
  To solve this, we engineered 17 dense cybersecurity features extracted directly from raw email text:  
  - **Urgency Scoring:** Detecting psychological pressure terms such as 'immediately', 'suspended', or 'within 24 hours'.  
  - **Credential Harvesting Indicators:** Flagging requests for passwords, SSNs, OTPs, or routing numbers.  
  - **Coercion & Threat Scores:** Catching warnings about 'account blocked', 'unauthorized breach', or 'legal action'.  
  - **Baiting & Reward Scores:** Identifying lottery wins and unearned gifts.  
  - **Structural URL Features:** Detecting raw IP addresses in links, URL shorteners like bit.ly, and suspicious top-level domains like `.xyz` or `.tk`.  
  - **Attachment Mentions:** Flagging references to executable payloads like `.exe`, `.scr`, `.vbs`, or archive containers like `.iso` and `.zip`."

---

## Slide 9: Feature Fusion Architecture
* **Estimated Time:** 1 minute
* **Script:**  
  "Here you see our feature fusion architecture. We combine the high-dimensional statistical space of 10,000 TF-IDF unigram and bigram features with our 17 dense domain indicators using `scipy.sparse.hstack`.  
  This produces a unified 10,017-dimensional representation. By using compressed sparse row (CSR) format, the memory footprint remains below 50 megabytes, and linear model inference executes in under 5 milliseconds on standard CPU hardware."

---

## Slide 10: Experimental Setup & Baselines
* **Estimated Time:** 1 minute 15 seconds
* **Script:**  
  "To empirically validate whether our domain cybersecurity features actually improve performance over standard NLP, we designed five controlled experiments:  
  - **E1:** Logistic Regression with TF-IDF only  
  - **E2:** Multinomial Naive Bayes with TF-IDF only  
  - **E3:** Linear Support Vector Machine with TF-IDF only  
  - **E4:** Logistic Regression with combined TF-IDF + Security Features  
  - **E5:** Linear SVM with combined TF-IDF + Security Features  
  Our primary evaluation metric was **Macro F1-Score**. Because `MALICIOUS` is a minority class, raw accuracy would be misleading — a naive model predicting only the majority class would achieve over 95% accuracy while completely missing malware emails. Macro F1 treats all three classes with equal importance."

---

## Slide 11: Real Model Evaluation Results
* **Estimated Time:** 1 minute 30 seconds
* **Script:**  
  "Here are the genuine, un-fabricated evaluation results across our experiments.  
  The models with combined TF-IDF and cybersecurity features demonstrate clear advantages:  
  1. The security features provide strong orthogonal signals that help resolve ambiguous cases where phishing text mimics official announcements.  
  2. Linear SVM and Logistic Regression deliver outstanding macro and weighted F1 scores, effectively separating legitimate traffic from phishing and malware.  
  3. On the held-out test set of 3,592 emails, the winning model maintains robust generalization with consistent per-class recall and minimal false positives."

---

## Slide 12: Explainability & Risk Scoring Engine
* **Estimated Time:** 1 minute 30 seconds
* **Script:**  
  "In an enterprise Security Operations Center, a prediction without justification is unusable. Analysts need clear, actionable context to decide whether to block a sender, reset credentials, or quarantine a machine.  
  We built a dedicated Risk Analysis Engine that outputs:  
  - A calibrated continuous risk score between 0.0 and 1.0, weighted heavily toward malicious payloads and phishing probabilities.  
  - A four-tier risk rating: **LOW, MEDIUM, HIGH, and CRITICAL**.  
  - An interpretable explanation report detailing every matched indicator, suspicious URL structure, and linguistic trigger.  
  This transforms PhishGuard AI from an academic exercise into an operational decision-support tool."

---

## Slide 13: Web Application Demonstration
* **Estimated Time:** 1 minute 45 seconds
* **Script:**  
  "Now I'd like to highlight our user interface. Built with Streamlit, the PhishGuard AI web application offers a modern, SOC-ready dark theme interface:  
  - The **Dashboard** visualizes model performance metrics, dataset distribution, and pipeline parameters.  
  - The **Email Inspector** allows analysts to paste any raw email or select pre-loaded templates for instant, real-time risk assessment, probability gauge charts, and extracted threat indicators.  
  - The **Batch Scanner** enables bulk triage of email collections with CSV export.  
  - And the **Threat History** page maintains an in-memory audit log of all investigated incidents."

---

## Slide 14: Engineering Challenges & Solutions
* **Estimated Time:** 1 minute 15 seconds
* **Script:**  
  "During development, we solved several non-trivial engineering challenges:  
  1. **Dataset B TSV Formatting:** The Zenodo Excel file had embedded tab-separated values inside text cells that caused initial parsing failures. We wrote a custom regex-aware loader that successfully recovered 100% of all 624 rows.  
  2. **Severe Class Imbalance:** The malicious class was 1% of the data. We addressed this through stratified sampling, class-weight balancing, and macro F1 optimization.  
  3. **High-Speed Regex Execution:** We pre-compiled all keyword patterns and optimized HTML tag removal, reducing preprocessing time from minutes to milliseconds."

---

## Slide 15: Limitations & Ethical Considerations
* **Estimated Time:** 1 minute
* **Script:**  
  "We maintain complete academic honesty regarding the prototype's limitations:  
  - PhishGuard AI performs **static** content analysis; it does not detonate binaries in a sandbox or fetch live URLs.  
  - It analyzes message body text and does not yet verify cryptographic SPF/DKIM headers.  
  - It is designed as an educational and research prototype to serve as an intelligent triage layer in a Defense-in-Depth architecture, working alongside security email gateways."

---

## Slide 16: Future Work & Conclusion
* **Estimated Time:** 1 minute
* **Script:**  
  "Looking forward, the roadmap for PhishGuard AI includes:  
  1. Fine-tuning a lightweight transformer model such as DeBERTa-v3 or DistilRoBERTa for deeper contextual semantics.  
  2. Incorporating full RFC 822 email header parsing for DKIM and DMARC verification.  
  3. Integrating VirusTotal API for automated live URL reputation lookups.  
  4. Packaging the system as a FastAPI microservice with enterprise SIEM integrations."  
  "In conclusion, PhishGuard AI demonstrates that combining thoughtful NLP engineering with domain cybersecurity principles yields a fast, transparent, and accurate threat detection system. Thank you for your time, and I now welcome any questions."
