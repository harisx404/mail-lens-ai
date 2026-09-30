# Mail-Lens AI — Data Directory

## Structure

```
data/
├── raw/           # Raw downloaded datasets (NOT committed to git)
│   ├── dataset_a_cache.csv          # HuggingFace dataset cache
│   └── phishing_nlp_dataset.xlsx    # Zenodo dataset
├── processed/     # Processed and split data (NOT committed to git)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── README.md      # This file
```

## Dataset Sources

### Dataset A — Phishing Email Detection
- **Source:** HuggingFace (mirror of Kaggle subhajournal/phishingemails)
- **URL:** https://huggingface.co/datasets/zefang-liu/phishing-email-dataset
- **License:** LGPL-3.0
- **Samples:** ~18,650
- **Labels:** "Safe Email" (→ LEGITIMATE), "Phishing Email" (→ PHISHING)
- **Download:** Automatic via `datasets` library

### Dataset B — Multiclass NLP Dataset for Phishing and Social Engineering
- **Source:** Zenodo (Engineering Ingegneria Informatica Spa)
- **URL:** https://zenodo.org/records/15235123
- **DOI:** 10.5281/zenodo.15235123
- **License:** CC BY 4.0
- **Samples:** 624
- **Labels:** Phishing, Malware, Scareware, Baiting, Pretexting, NOT-Malicious
- **Download:** Automatic via download script

## Label Mapping

| Original Label | Source | Project Label | Justification |
|---|---|---|---|
| Safe Email | Dataset A | LEGITIMATE | Direct mapping |
| Phishing Email | Dataset A | PHISHING | Direct mapping |
| NOT-Malicious | Dataset B | LEGITIMATE | Direct mapping |
| Phishing | Dataset B | PHISHING | Direct mapping |
| Baiting | Dataset B | PHISHING | Social engineering subcategory |
| Pretexting | Dataset B | PHISHING | Social engineering subcategory |
| Malware | Dataset B | MALICIOUS | Direct mapping |
| Scareware | Dataset B | MALICIOUS | Malware delivery via fear tactics |

## How to Prepare Data

```bash
python scripts/train.py
```

The training script handles:
1. Downloading both datasets
2. Validating the data
3. Mapping labels
4. Cleaning and deduplication
5. Stratified train/val/test split
6. Saving processed files

## Important Notes

- Raw data files are NOT committed to git (see .gitignore)
- Data is downloaded automatically during training
- Internet connection required for first run
- Subsequent runs use cached data
