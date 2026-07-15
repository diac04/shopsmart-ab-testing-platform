# phase5/data_loader.py
"""
Loads Experiment 3 banner data and validates it for Phase 5 analysis.
Returns a clean DataFrame with boolean/int 'clicked' column.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    DATA_PATH, TREATMENT_COL, OUTCOME_COL,
    DEVICE_COL, USER_ID_COL,
    CONTROL_LABEL, TREATMENT_LABEL, DEVICE_SEGMENTS
)


def load_data(verbose: bool = True) -> pd.DataFrame:
    """
    Load and validate Experiment 3 data.

    Returns
    -------
    pd.DataFrame  with columns guaranteed:
        user_id, variant, clicked (int 0/1), device_type
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # ── Basic column checks ────────────────────────────────────────────────
    required = [USER_ID_COL, TREATMENT_COL, OUTCOME_COL, DEVICE_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}\n"
                         f"Available: {list(df.columns)}")

    # ── Clean outcome column ───────────────────────────────────────────────
    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors="coerce").fillna(0).astype(int)

    # ── Filter to known variants and devices ──────────────────────────────
    df = df[df[TREATMENT_COL].isin([CONTROL_LABEL, TREATMENT_LABEL])].copy()
    df = df[df[DEVICE_COL].isin(DEVICE_SEGMENTS)].copy()
    df = df.drop_duplicates(subset=[USER_ID_COL]).copy()

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*55}")
        print(f"  Experiment 3 Data Loaded")
        print(f"{'='*55}")
        print(f"  Total rows      : {len(df):,}")
        print(f"  Variants        : {df[TREATMENT_COL].value_counts().to_dict()}")
        print(f"  Devices         : {df[DEVICE_COL].value_counts().to_dict()}")
        print(f"  Overall CTR     : {df[OUTCOME_COL].mean():.4f}")
        print(f"{'='*55}\n")

    return df


def get_segment(df: pd.DataFrame, device: str) -> pd.DataFrame:
    """Return rows for one device segment."""
    return df[df[DEVICE_COL] == device].copy()


def split_variants(df: pd.DataFrame):
    """Return (control_df, treatment_df)."""
    ctrl = df[df[TREATMENT_COL] == CONTROL_LABEL]
    trt  = df[df[TREATMENT_COL] == TREATMENT_LABEL]
    return ctrl, trt


if __name__ == "__main__":
    data = load_data()
    print(data.head())