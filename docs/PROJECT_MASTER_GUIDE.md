# PhishGuard AI — Project Master Guide

## Complete Beginner-Friendly Explanation

This guide explains the entire PhishGuard AI project in plain language, suitable for capstone presentation, viva defense, and future reference.

---

## Part 1: What Does This Project Do?

PhishGuard AI analyzes email content and answers one question: **"Is this email safe, a phishing attempt, or carrying malicious content?"**

Think of it as a smart email security assistant that reads an email and tells you:
1. **What it is** — Legitimate, Phishing, or Malicious
2. **How confident** — e.g., "92% sure this is phishing"
3. **How risky** — LOW to CRITICAL
4. **Why it thinks so** — "Urgency language detected, suspicious URL found"
5. **What to do** — "Do not click links, report to IT"

---

## Part 2: How Does It Work?

### Step 1: Email Input
The user pastes an email (subject + body) into the web interface.

### Step 2: Text Preprocessing (NLP)
The raw email text is cleaned:
- HTML tags are removed (`<b>Bold</b>` → `Bold`)
- URLs are detected and replaced with a placeholder
- Text is lowercased
- Special characters are removed
- Words are lemmatized ("running" → "run")

**Why?** ML models need clean, consistent text. "CLICK HERE!!!" and "click here" should be treated the same way.

### Step 3: Feature Extraction
Two types of features are extracted:

**A. NLP Features (TF-IDF)**
The cleaned text is converted into a vector of numbers using TF-IDF:
- Each word/phrase gets a score based on how important it is
- Words that are common in phishing but rare in legitimate emails get high scores
- This creates a feature vector of up to 10,000 dimensions

**B. Cybersecurity Features**
The RAW text (before cleaning) is analyzed for security-specific signals:
- How many urgency words? ("immediately", "act now")
- Any credential requests? ("verify your password")
- How many URLs? Are they HTTP (unsafe) or HTTPS?
- Any IP-based URLs? URL shorteners?
- Any executable file mentions? (.exe, .bat)

### Step 4: ML Classification
The combined feature vector (TF-IDF + security features) is fed into a trained machine learning model. The model outputs:
- A predicted class (LEGITIMATE, PHISHING, or MALICIOUS)
- A probability for each class

### Step 5: Risk Assessment
The model's confidence is combined with the security indicators to compute a risk score (0-1), mapped to a risk level (LOW/MEDIUM/HIGH/CRITICAL).

### Step 6: Explanation
A human-readable explanation is generated combining the model prediction with the detected security indicators and a security recommendation.

---

## Part 3: How Was the Model Trained?

### Data Collection
We used two public datasets:
1. **18,650 emails** from HuggingFace (Safe vs Phishing)
2. **624 messages** from Zenodo (6 classes including Malware/Scareware)

### Label Mapping
We mapped the original labels to our three-class system:
- Safe Email / NOT-Malicious → **LEGITIMATE**
- Phishing / Baiting / Pretexting → **PHISHING**
- Malware / Scareware → **MALICIOUS**

### Data Splitting
The combined dataset was split into three parts:
- **Training set (~70%)**: The model learns from this
- **Validation set (~10%)**: Used to compare different models
- **Test set (~20%)**: Used ONCE for final evaluation (never seen during training)

### Model Training
We trained 5 experiments:

| Experiment | Model | Features |
|---|---|---|
| E1 | Logistic Regression | TF-IDF only |
| E2 | Naive Bayes | TF-IDF only |
| E3 | Linear SVM | TF-IDF only |
| E4 | Logistic Regression | TF-IDF + Security |
| E5 | Linear SVM | TF-IDF + Security |

### Model Selection
The best model was selected based on **Macro F1** score — this metric treats all classes equally, which is important because our MALICIOUS class has fewer samples.

---

## Part 4: Understanding the Results

### Classification
- **LEGITIMATE**: The email appears normal and non-threatening
- **PHISHING**: The email uses social engineering to deceive the recipient
- **MALICIOUS**: The email is associated with malware or harmful content

### Risk Levels
- **🟢 LOW (0-0.3)**: Minimal concern
- **🟡 MEDIUM (0.3-0.6)**: Review recommended
- **🟠 HIGH (0.6-0.85)**: Significant risk indicators found
- **🔴 CRITICAL (0.85-1.0)**: Strong threat indicators detected

### Security Indicators
The system detects specific patterns:
- Urgency language ("act now", "immediately")
- Credential requests ("verify your password")
- Threat language ("account suspended", "unauthorized")
- Reward language ("you have won", "congratulations")
- Suspicious URLs (HTTP, IP-based, shorteners)
- Executable file mentions

---

## Part 5: Code Architecture

```
phishguard-ai/
├── src/
│   ├── data/
│   │   ├── loader.py          # Downloads and loads raw datasets
│   │   └── pipeline.py        # Label mapping, cleaning, splitting
│   ├── preprocessing/
│   │   └── text_preprocessor.py  # NLP text cleaning
│   ├── features/
│   │   ├── security_features.py  # Cybersecurity feature extraction
│   │   └── feature_engineer.py   # TF-IDF + Security feature combination
│   ├── models/
│   │   └── trainer.py         # Model creation, training, saving
│   ├── evaluation/
│   │   └── evaluator.py       # Metrics computation and comparison
│   ├── inference/
│   │   └── engine.py          # Production inference pipeline
│   └── utils/
│       └── config.py          # Central configuration
├── app/
│   └── main.py                # Streamlit web application
├── scripts/
│   └── train.py               # Master training script
└── tests/
    └── test_core.py           # Test suite
```

### Key Design Decisions
1. **Separation of concerns**: Each module has a single responsibility
2. **Configuration centralized**: All settings in `config.py`
3. **No data leakage**: TF-IDF fit on train only
4. **Security by design**: No URL visiting, no file execution
5. **Honest metrics**: All results from actual runs, nothing fabricated
6. **Documented limitations**: Every known weakness is documented

---

## Part 6: The Math Behind TF-IDF

**TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)**

Where:
- **t** = a term (word or phrase)
- **d** = a document (email)
- **D** = the corpus (all emails)

**TF(t, d)** = (number of times t appears in d) / (total terms in d)
**IDF(t, D)** = log(total documents / documents containing t)

**Example:**
- "password" appears 3 times in a phishing email of 50 words: TF = 3/50 = 0.06
- "password" appears in 200 of 18,000 emails: IDF = log(18000/200) = 4.5
- TF-IDF = 0.06 × 4.5 = 0.27

A word that's frequent in this email but rare across the corpus gets a HIGH score. Common words like "the" get very low scores.

We use **sublinear TF** (log normalization) to prevent very frequent words from dominating.

---

## Part 7: What Makes This Project Capstone-Worthy?

1. **Real-world problem**: Phishing detection is genuinely important
2. **Complete ML pipeline**: Data → Features → Training → Evaluation → Deployment
3. **NLP + Cybersecurity**: Combines two important domains
4. **Multiple models compared**: Not just one model, but systematic experimentation
5. **Professional engineering**: Config management, testing, documentation
6. **Honest evaluation**: Limitations are documented, not hidden
7. **Working demo**: Functional web application for live demonstration
8. **Reproducible**: Fixed seeds, documented data sources, clear instructions
9. **Well-documented**: README, blueprint, master guide, viva Q&A, limitations

---

## Part 8: Running the Project

### Quick Start
```bash
# Setup
cd "D:\KPITB AI\Final project\phishguard-ai"
venv\Scripts\activate

# Train
python scripts/train.py

# Test
python -m pytest tests/ -v

# Run
streamlit run app/main.py
```

### Demo Flow for Presentation
1. Open the web app → show Dashboard
2. Go to Analyze Email → paste a sample phishing email
3. Show the classification result, risk level, indicators
4. Analyze a legitimate email for comparison
5. Show the History page
6. Explain the architecture and methodology
