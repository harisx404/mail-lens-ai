# PhishGuard AI — Dataset Documentation

## Overview

PhishGuard AI uses a combined dataset from two verified public sources to train a three-class email classifier.

---

## Dataset A — Phishing Email Detection

| Field | Value |
|---|---|
| **Name** | Phishing Email Detection |
| **Source** | HuggingFace (mirror of Kaggle subhajournal/phishingemails) |
| **URL** | https://huggingface.co/datasets/zefang-liu/phishing-email-dataset |
| **License** | LGPL-3.0 (GNU Lesser General Public License v3.0) |
| **Samples** | ~18,650 |
| **Labels** | Binary: `Safe Email`, `Phishing Email` |
| **Text Fields** | `Email Text` (body content) |
| **Format** | CSV / Parquet (via HuggingFace datasets library) |
| **Language** | English |
| **Citation** | subhajournal on Kaggle |

### Known Limitations
- Binary only (no malware class)
- No email metadata (subject, sender, headers)
- Some entries may have empty/missing body text
- No information about temporal distribution

---

## Dataset B — Multiclass NLP Dataset for Phishing and Social Engineering Threat Detection

| Field | Value |
|---|---|
| **Name** | Multiclass NLP Dataset for Phishing and Social Engineering Threat Detection |
| **Source** | Zenodo (Engineering Ingegneria Informatica Spa) |
| **URL** | https://zenodo.org/records/15235123 |
| **DOI** | 10.5281/zenodo.15235123 |
| **License** | CC BY 4.0 (Creative Commons Attribution 4.0 International) |
| **Samples** | 624 |
| **Labels** | 6 classes: Phishing, Malware, Scareware, Baiting, Pretexting, NOT-Malicious |
| **Text Fields** | `Corpus` (message content — email or SMS-like) |
| **Format** | .xlsx (Excel) |
| **Language** | English |

### Known Limitations
- Very small (624 samples)
- Messages are email/SMS-like, not necessarily full email format
- No metadata fields
- Limited diversity within each class

---

## Label Mapping to Project Taxonomy

The project uses a three-class taxonomy: **LEGITIMATE**, **PHISHING**, **MALICIOUS**

### Dataset A Mapping

| Original Label | Project Label | Justification |
|---|---|---|
| `Safe Email` | **LEGITIMATE** | Direct semantic mapping — email with no malicious intent |
| `Phishing Email` | **PHISHING** | Direct semantic mapping — email using deception to steal information |

### Dataset B Mapping

| Original Label | Project Label | Justification |
|---|---|---|
| `NOT-Malicious` | **LEGITIMATE** | Direct semantic mapping — safe, benign message |
| `Phishing` | **PHISHING** | Direct semantic mapping — deceptive message seeking sensitive data |
| `Baiting` | **PHISHING** | Social engineering subcategory — uses lures/incentives to extract info or action |
| `Pretexting` | **PHISHING** | Social engineering subcategory — uses fabricated scenarios to establish trust |
| `Malware` | **MALICIOUS** | Direct semantic mapping — message delivering harmful software |
| `Scareware` | **MALICIOUS** | Malware variant — uses fear tactics to trick users into installing malware |

### Justification for Combining Baiting/Pretexting → PHISHING
Baiting and pretexting are social engineering techniques fundamentally similar to phishing — they use deception to manipulate the recipient. From an NLP perspective, they share similar linguistic patterns (urgency, deception, impersonation). Grouping them under PHISHING is academically defensible and supported by cybersecurity taxonomy standards.

### Justification for Combining Scareware → MALICIOUS
Scareware delivers malicious software by exploiting fear. While it uses social engineering, the end goal is malware installation, making MALICIOUS the appropriate category.

---

## Merged Dataset Characteristics

### Expected Distribution (Before Cleaning/Dedup)

| Class | Approx. Samples | Sources |
|---|---|---|
| LEGITIMATE | ~9,000+ | Dataset A (Safe Email) |
| PHISHING | ~9,500+ | Dataset A (Phishing Email) + Dataset B (Phishing, Baiting, Pretexting) |
| MALICIOUS | ~100-200 | Dataset B (Malware, Scareware) |

> **⚠️ WARNING:** The MALICIOUS class is severely underrepresented. This is the project's most significant data limitation.

### Splitting Strategy

| Split | Fraction | Purpose |
|---|---|---|
| Training | ~70% | Model training (TF-IDF fitted here) |
| Validation | ~10% | Hyperparameter selection, model comparison |
| Test | ~20% | Final evaluation (used once) |

All splits are **stratified** to preserve class ratios.

---

## Data Processing Pipeline

1. **Download** — Automatic via HuggingFace `datasets` library (A) and HTTP download (B)
2. **Load** — Standardize column names to `text`, `label_original`, `source`
3. **Validate** — Check for missing values, empty texts, duplicates
4. **Map Labels** — Apply the mapping table above
5. **Exclude Dataset B Legitimate** — Already covered by Dataset A
6. **Clean** — Drop missing labels, empty texts, NaN values
7. **Deduplicate** — Remove exact text duplicates (keep first occurrence)
8. **Split** — Stratified train/val/test split
9. **Save** — CSV files in `data/processed/`

---

## Ethical & Legal Compliance

- Both datasets have permissive licenses allowing research and educational use
- CC BY 4.0 requires attribution (provided in this document and README)
- LGPL-3.0 allows use in non-LGPL projects
- No personally identifiable information (PII) concerns in the email text
- No proprietary or confidential data used

---

## Actual Statistics

> **Note:** Actual statistics (exact row counts, class distributions, text length statistics) will be populated after running the training pipeline. See `data/processed/` for the actual split files and `training.log` for validation reports.
