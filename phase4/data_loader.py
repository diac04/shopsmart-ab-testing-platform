# phase4/data_loader.py
# ============================================================
# Loads Experiment 2 data, applies the novelty-period filter
# flagged in Phase 2, and returns clean analysis-ready frames.
# ============================================================

import pandas as pd
import numpy as np
from phase4.config import (
    DATA_PATH, GROUP_COL, REVENUE_COL, PRE_EXP_COL,
    DAY_COL, NOVELTY_CUTOFF_DAY,
    CONTROL_LABEL, TREATMENT_LABEL
)


def load_raw() -> pd.DataFrame:
    """Load the full experiment 2 dataset from Phase 2 output."""
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} rows from {DATA_PATH}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Groups  : {df[GROUP_COL].value_counts().to_dict()}")
    return df


def apply_novelty_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows from day >= NOVELTY_CUTOFF_DAY.
    Phase 2 found early/late lift ratio = 1.61 (threshold 1.15).
    The late-period lift (+10.08%) is the trusted estimate.
    """
    if DAY_COL not in df.columns:
        print(f"  WARNING: '{DAY_COL}' column not found. "
              f"Skipping novelty filter — using ALL days.")
        return df

    before = len(df)
    df_late = df[df[DAY_COL] >= NOVELTY_CUTOFF_DAY].copy()
    after   = len(df_late)
    print(f"  Novelty filter: kept days {NOVELTY_CUTOFF_DAY}–14 "
          f"({after:,} of {before:,} rows, "
          f"{100*after/before:.1f}%)")
    return df_late


def split_groups(df: pd.DataFrame):
    """Return (control_series, treatment_series) of revenue values."""
    ctrl  = df.loc[df[GROUP_COL] == CONTROL_LABEL,   REVENUE_COL].values
    treat = df.loc[df[GROUP_COL] == TREATMENT_LABEL, REVENUE_COL].values
    return ctrl, treat


def get_covariate(df: pd.DataFrame, group: str) -> np.ndarray:
    """Return pre-experiment revenue covariate for a given group."""
    return df.loc[df[GROUP_COL] == group, PRE_EXP_COL].values


def load_experiment2():
    """
    Master loader used by all Phase 4 modules.
    Returns:
        df_full   — full dataset (all days)
        df_late   — novelty-filtered dataset (days 7–14)  ← PRIMARY
        ctrl_full, treat_full   — revenue arrays, all days
        ctrl_late, treat_late   — revenue arrays, late period only
        pre_ctrl, pre_treat     — pre-experiment covariate arrays
    """
    print("\n" + "="*60)
    print("  PHASE 4 — DATA LOADER")
    print("="*60)

    df_full = load_raw()
    df_late = apply_novelty_filter(df_full)

    ctrl_full,  treat_full  = split_groups(df_full)
    ctrl_late,  treat_late  = split_groups(df_late)

    # Covariates come from the full dataset (pre-experiment, same for all days)
    pre_ctrl  = get_covariate(df_full, CONTROL_LABEL)
    pre_treat = get_covariate(df_full, TREATMENT_LABEL)

    print(f"\n  Full dataset   — control: {len(ctrl_full):,} "
          f"| treatment: {len(treat_full):,}")
    print(f"  Late period    — control: {len(ctrl_late):,} "
          f"| treatment: {len(treat_late):,}")
    print(f"  Pre-exp covar  — control: {len(pre_ctrl):,} "
          f"| treatment: {len(pre_treat):,}")

    return (df_full, df_late,
            ctrl_full, treat_full,
            ctrl_late, treat_late,
            pre_ctrl, pre_treat)