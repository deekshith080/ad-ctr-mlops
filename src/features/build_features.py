"""
build_features.py
-----------------
Loads raw Criteo CTR data, engineers features, and saves
reference and current datasets for training and drift detection.

Reference data  → first 800K rows  (used for training + PSI baseline)
Current data    → last  200K rows  (simulates new incoming data with drift)
"""

import pandas as pd
import numpy as np
from pathlib import Path

#Paths
RAW_DATA_PATH       = Path("data/raw/train.csv")
PROCESSED_DIR       = Path("data/processed")
REFERENCE_DATA_PATH = PROCESSED_DIR / "reference.csv"
CURRENT_DATA_PATH   = PROCESSED_DIR / "current.csv"

# Column definitions
# Criteo dataset columns
NUMERIC_COLS = [f"intCol_{i}" for i in range(13)]
CATEGORICAL_COLS = [f"catCol_{i}" for i in range(26)]
TARGET_COL = "target"


def load_raw_data(path: Path, nrows: int = 1_000_000) -> pd.DataFrame:
    """Load raw Criteo CSV — use first 1M rows for speed."""
    print(f"Loading data from {path} ...")
    col_names = [TARGET_COL] + NUMERIC_COLS + CATEGORICAL_COLS
    df = pd.read_csv(path, nrows=nrows)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values."""
    print("Cleaning data ...")

    # Numeric: fill with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical: fill with 'missing'
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("missing")

    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features on top of raw columns.
    These are simple but realistic for CTR models.
    """
    print("Engineering features ...")

    # Log-transform skewed numeric features (common in CTR data)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # Frequency encode categorical columns
    # Replace each category with how often it appears (normalised)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq)

    print(f"Features after engineering: {df.shape[1]} columns")
    return df


def split_and_save(df: pd.DataFrame) -> None:
    """
    Split into reference (first 800K) and current (last 200K).
    Save both to data/processed/.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    reference = df.iloc[:800_000].copy()
    current   = df.iloc[800_000:].copy()

    print(f"Reference data: {len(reference):,} rows")
    print(f"Current data:   {len(current):,} rows")

    reference.to_csv(REFERENCE_DATA_PATH, index=False)
    current.to_csv(CURRENT_DATA_PATH, index=False)

    print(f"Saved reference → {REFERENCE_DATA_PATH}")
    print(f"Saved current   → {CURRENT_DATA_PATH}")


def main():
    print("=" * 55)
    print("  CTR Feature Engineering Pipeline")
    print("=" * 55)

    df = load_raw_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    split_and_save(df)

    print("\n Feature engineering complete.")
    print(f"   Reference: {REFERENCE_DATA_PATH}")
    print(f"   Current:   {CURRENT_DATA_PATH}")


if __name__ == "__main__":
    main()