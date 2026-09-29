"""
PhishGuard AI — Dataset Loader

Downloads and loads the raw datasets:
  - Dataset A: HuggingFace phishing-email-dataset (binary: Safe/Phishing)
  - Dataset B: Zenodo multiclass NLP dataset (.xlsx, 6-class)

All dataset sources are documented and verifiable.
"""

import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_dataset_b(output_path: Path) -> Path:
    """
    Download Dataset B (Zenodo multiclass) to the specified path.

    Source: https://zenodo.org/records/15235123
    License: CC BY 4.0
    """
    url = "https://zenodo.org/records/15235123/files/phishing_nlp_dataset.xlsx?download=1"
    output_file = output_path / "phishing_nlp_dataset.xlsx"

    if output_file.exists():
        logger.info(f"Dataset B already exists at {output_file}")
        return output_file

    logger.info(f"Downloading Dataset B from Zenodo...")
    output_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(output_file, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Dataset B") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Dataset B saved to {output_file}")
    return output_file


def load_dataset_a(raw_dir: Path) -> pd.DataFrame:
    """
    Load Dataset A from HuggingFace.

    Source: https://huggingface.co/datasets/zefang-liu/phishing-email-dataset
    License: LGPL-3.0
    Original: Kaggle (subhajournal/phishingemails)

    Returns DataFrame with columns: ['text', 'label_original', 'source']
    """
    cache_file = raw_dir / "dataset_a_cache.csv"

    if cache_file.exists():
        logger.info(f"Loading Dataset A from cache: {cache_file}")
        return pd.read_csv(cache_file)

    logger.info("Loading Dataset A from HuggingFace...")

    try:
        from datasets import load_dataset

        ds = load_dataset("zefang-liu/phishing-email-dataset", split="train")
        df = ds.to_pandas()
    except Exception as e:
        logger.error(f"Failed to load from HuggingFace: {e}")
        raise RuntimeError(
            "Could not load Dataset A. Ensure 'datasets' is installed "
            "and you have internet access. Run: pip install datasets"
        ) from e

    # Standardize column names
    # The dataset has columns: 'Email Text' and 'Email Type'
    col_map = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        if "text" in col_lower or "body" in col_lower or "email text" in col_lower:
            col_map[col] = "text"
        elif "type" in col_lower or "label" in col_lower:
            col_map[col] = "label_original"

    if "text" not in col_map.values() or "label_original" not in col_map.values():
        # Fallback: use positional columns
        cols = list(df.columns)
        logger.warning(f"Could not map columns by name. Columns found: {cols}")
        col_map = {cols[0]: "text", cols[1]: "label_original"}

    df = df.rename(columns=col_map)
    df = df[["text", "label_original"]].copy()
    df["source"] = "dataset_a"

    # Cache locally
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    logger.info(f"Dataset A cached to {cache_file} ({len(df)} rows)")

    return df


def load_dataset_b(raw_dir: Path) -> pd.DataFrame:
    """
    Load Dataset B from Zenodo .xlsx file.

    Source: https://zenodo.org/records/15235123
    License: CC BY 4.0

    Returns DataFrame with columns: ['text', 'label_original', 'source']
    """
    xlsx_file = raw_dir / "phishing_nlp_dataset.xlsx"

    if not xlsx_file.exists():
        logger.info("Dataset B not found locally. Downloading...")
        download_dataset_b(raw_dir)

    logger.info(f"Loading Dataset B from {xlsx_file}")
    df = pd.read_excel(xlsx_file, engine="openpyxl")

    parsed_texts = []
    parsed_labels = []

    # Handle the Zenodo format where 'Corpus' and 'Labels' contain tab-separated text and label
    for _, row in df.iterrows():
        c = str(row.iloc[0]) if pd.notnull(row.iloc[0]) else ""
        l = str(row.iloc[1]) if len(row) > 1 and pd.notnull(row.iloc[1]) else ""
        full = (c + " " + l).strip()
        if "\t" in full:
            parts = full.rsplit("\t", 1)
            parsed_texts.append(parts[0].strip())
            parsed_labels.append(parts[1].strip())
        elif len(row) > 1 and str(row.iloc[1]).strip():
            parsed_texts.append(c.strip())
            parsed_labels.append(str(row.iloc[1]).strip())
        else:
            parsed_texts.append(c.strip())
            parsed_labels.append("")

    df_clean = pd.DataFrame(
        {
            "text": parsed_texts,
            "label_original": parsed_labels,
            "source": "dataset_b",
        }
    )

    logger.info(f"Dataset B loaded: {len(df_clean)} rows")
    return df_clean



def validate_dataframe(df: pd.DataFrame, name: str) -> dict:
    """
    Validate a loaded DataFrame and return a validation report.

    Checks:
    - Total rows
    - Missing values per column
    - Duplicate texts
    - Empty texts
    - Label distribution
    - Text length statistics
    """
    report = {
        "name": name,
        "total_rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "empty_texts": int((df["text"].astype(str).str.strip() == "").sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "label_distribution": df["label_original"].value_counts().to_dict(),
        "text_length_stats": {
            "min": int(df["text"].astype(str).str.len().min()),
            "max": int(df["text"].astype(str).str.len().max()),
            "mean": round(df["text"].astype(str).str.len().mean(), 1),
            "median": round(df["text"].astype(str).str.len().median(), 1),
        },
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET VALIDATION: {name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total rows: {report['total_rows']}")
    logger.info(f"Missing values: {report['missing_values']}")
    logger.info(f"Empty texts: {report['empty_texts']}")
    logger.info(f"Duplicate texts: {report['duplicate_texts']}")
    logger.info(f"Label distribution: {report['label_distribution']}")
    logger.info(f"Text length (min/max/mean): "
                f"{report['text_length_stats']['min']}/"
                f"{report['text_length_stats']['max']}/"
                f"{report['text_length_stats']['mean']}")
    logger.info(f"{'='*60}")

    return report
