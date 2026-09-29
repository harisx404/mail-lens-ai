"""
PhishGuard AI — Data Pipeline

Handles:
1. Loading raw datasets (A + B)
2. Label mapping to project taxonomy
3. Cleaning (missing values, empty texts)
4. Deduplication
5. Train/validation/test splitting (stratified)
6. Exporting processed data

Label Mapping Document:
-----------------------
Dataset A (HuggingFace):
    "Safe Email"     → LEGITIMATE
    "Phishing Email" → PHISHING

Dataset B (Zenodo):
    "NOT-Malicious"  → LEGITIMATE
    "Phishing"       → PHISHING
    "Baiting"        → PHISHING  (social engineering subcategory)
    "Pretexting"     → PHISHING  (social engineering subcategory)
    "Malware"        → MALICIOUS
    "Scareware"      → MALICIOUS (malware delivery via fear tactics)

Justification:
- Baiting and Pretexting are social engineering techniques fundamentally
  similar to phishing — they deceive the recipient to extract information
  or take an action. Mapping them to PHISHING is defensible.
- Scareware is a form of malicious software delivery that uses fear to
  trick users into installing malware. Mapping to MALICIOUS is defensible.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset_a, load_dataset_b, validate_dataframe

logger = logging.getLogger(__name__)

# ============================================
# Label Mapping Dictionaries
# ============================================

DATASET_A_LABEL_MAP = {
    "Safe Email": "LEGITIMATE",
    "safe email": "LEGITIMATE",
    "Phishing Email": "PHISHING",
    "phishing email": "PHISHING",
    # Numeric variants (some versions use 0/1)
    0: "LEGITIMATE",
    1: "PHISHING",
}

DATASET_B_LABEL_MAP = {
    "NOT-Malicious General Class": "LEGITIMATE",
    "NOT-Malicious": "LEGITIMATE",
    "Not-Malicious": "LEGITIMATE",
    "not-malicious": "LEGITIMATE",
    "Phishing": "PHISHING",
    "phishing": "PHISHING",
    "Baiting": "PHISHING",
    "baiting": "PHISHING",
    "Pretexting": "PHISHING",
    "pretexting": "PHISHING",
    "Malware": "MALICIOUS",
    "malware": "MALICIOUS",
    "Scareware": "MALICIOUS",
    "scareware": "MALICIOUS",
}


def map_labels(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Map original labels to project taxonomy (LEGITIMATE / PHISHING / MALICIOUS).

    Args:
        df: DataFrame with 'label_original' column
        source: 'dataset_a' or 'dataset_b'

    Returns:
        DataFrame with new 'label' column
    """
    label_map = DATASET_A_LABEL_MAP if source == "dataset_a" else DATASET_B_LABEL_MAP

    df = df.copy()
    df["label"] = df["label_original"].map(label_map)

    unmapped = df[df["label"].isna()]
    if len(unmapped) > 0:
        unmapped_labels = unmapped["label_original"].unique()
        logger.warning(
            f"[{source}] {len(unmapped)} rows with unmapped labels: {unmapped_labels}"
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    1. Drop rows with missing text or label
    2. Drop rows with empty text (after stripping)
    3. Strip whitespace from text
    4. Convert text to string type
    """
    initial_count = len(df)

    # Convert text to string, handle NaN
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()

    # Drop missing labels
    df = df.dropna(subset=["label"])

    # Drop empty text
    df = df[df["text"].str.len() > 0]
    df = df[df["text"] != "nan"]

    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Cleaning removed {removed} rows ({initial_count} -> {len(df)})")

    return df.reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate texts, keeping the first occurrence.
    Deduplication is based on exact text match (case-sensitive).
    """
    initial_count = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Deduplication removed {removed} rows ({initial_count} -> {len(df)})")

    return df


def build_unified_dataset(
    raw_dir: Path,
    include_dataset_b_legitimate: bool = False,
) -> pd.DataFrame:
    """
    Build the unified dataset from Dataset A + Dataset B.

    Args:
        raw_dir: Directory containing raw data files
        include_dataset_b_legitimate: If True, include NOT-Malicious from Dataset B.
            Default False since Dataset A already provides ample legitimate samples.

    Returns:
        Unified DataFrame with columns: ['text', 'label_original', 'source', 'label']
    """
    logger.info("=" * 60)
    logger.info("BUILDING UNIFIED DATASET")
    logger.info("=" * 60)

    # Load datasets
    df_a = load_dataset_a(raw_dir)
    df_b = load_dataset_b(raw_dir)

    # Validate raw data
    report_a = validate_dataframe(df_a, "Dataset A (HuggingFace)")
    report_b = validate_dataframe(df_b, "Dataset B (Zenodo)")

    # Map labels
    df_a = map_labels(df_a, "dataset_a")
    df_b = map_labels(df_b, "dataset_b")

    # Optionally exclude Dataset B legitimate samples to avoid duplication
    if not include_dataset_b_legitimate:
        before = len(df_b)
        df_b = df_b[df_b["label"] != "LEGITIMATE"]
        logger.info(
            f"Excluded {before - len(df_b)} LEGITIMATE rows from Dataset B "
            f"(already covered by Dataset A)"
        )

    # Combine
    df = pd.concat([df_a, df_b], ignore_index=True)
    logger.info(f"Combined dataset: {len(df)} rows")

    # Clean
    df = clean_data(df)

    # Deduplicate
    df = deduplicate(df)

    # Final validation
    validate_dataframe(df, "Unified Dataset (Final)")

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test with stratification.

    Data Leakage Prevention:
    - Split is done BEFORE any preprocessing/feature extraction
    - Stratified split ensures class distribution is preserved
    - Random seed is fixed for reproducibility

    Args:
        df: Unified dataset
        test_size: Fraction for test set (default 0.2)
        val_size: Fraction for validation set (default 0.1, taken from train)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting dataset: test={test_size}, val={val_size}, seed={random_seed}")

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["label"],
    )

    # Second split: train vs val
    # Adjust val_size relative to train+val
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=train_val_df["label"],
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"Train distribution:\n{train_df['label'].value_counts().to_string()}")
    logger.info(f"Val distribution:\n{val_df['label'].value_counts().to_string()}")
    logger.info(f"Test distribution:\n{test_df['label'].value_counts().to_string()}")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save processed splits to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Processed data saved to {output_dir}/")
    logger.info(f"  train.csv: {len(train_df)} rows")
    logger.info(f"  val.csv:   {len(val_df)} rows")
    logger.info(f"  test.csv:  {len(test_df)} rows")


def run_data_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete data pipeline.

    Steps:
    1. Load and validate raw datasets
    2. Map labels to project taxonomy
    3. Combine datasets
    4. Clean and deduplicate
    5. Split into train/val/test (stratified)
    6. Save processed files

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load existing processed splits if available
    train_file = processed_dir / "train.csv"
    val_file = processed_dir / "val.csv"
    test_file = processed_dir / "test.csv"
    if train_file.exists() and val_file.exists() and test_file.exists():
        logger.info(f"Loading existing processed splits from {processed_dir}")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        return train_df, val_df, test_df

    # Build unified dataset
    df = build_unified_dataset(raw_dir)


    # Split
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
    )

    # Save
    save_processed_data(train_df, val_df, test_df, processed_dir)

    return train_df, val_df, test_df
