# phase6/data_loader.py
"""
Load and validate Experiment 1 data for Phase 6 analysis.
Reads from phase2/experiment1/exp1_checkout_data.csv
"""

import pandas as pd
import numpy as np
import os
from phase6.config import PHASE2_EXP1, PRE_REG


def load_experiment1() -> pd.DataFrame:
    """
    Load Experiment 1 checkout data.
    Expected columns: user_id, group, converted, date (or day index).
    Returns cleaned DataFrame.
    """
    if not os.path.exists(PHASE2_EXP1):
        raise FileNotFoundError(
            f"Experiment 1 data not found at:\n  {PHASE2_EXP1}\n"
            "Run Phase 2 first."
        )

    df = pd.read_csv(PHASE2_EXP1)
    print(f"[data_loader] Loaded {len(df):,} rows from experiment 1.")

    # ── Normalise column names ────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Validate required columns ─────────────────────────────────────────
    required = {"group", "converted"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}\n"
                         f"Available: {list(df.columns)}")

    # ── Standardise group labels ──────────────────────────────────────────
    df["group"] = df["group"].str.strip().str.lower()
    valid_groups = {"control", "treatment"}
    found_groups = set(df["group"].unique())
    if not found_groups.issubset(valid_groups):
        # Try to remap common alternatives
        remap = {}
        for g in found_groups:
            if g in ("ctrl", "a", "0"):
                remap[g] = "control"
            elif g in ("treat", "b", "1"):
                remap[g] = "treatment"
        if remap:
            df["group"] = df["group"].replace(remap)
            print(f"[data_loader] Remapped groups: {remap}")

    # ── Ensure converted is binary int ───────────────────────────────────
    df["converted"] = df["converted"].astype(int)

    # ── Add sequential row index as proxy day if no date column ──────────
    if "date" not in df.columns and "day" not in df.columns:
        # Assign day based on row order partitioned by group
        df = df.sort_index()
        df["day"] = (df.groupby("group").cumcount() //
                     (PRE_REG["n_per_group"] // PRE_REG["min_run_days"])) + 1
        df["day"] = df["day"].clip(upper=PRE_REG["min_run_days"])
        print("[data_loader] No date column found — synthetic day index added.")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["day"]  = (df["date"] - df["date"].min()).dt.days + 1

    # ── Summary ───────────────────────────────────────────────────────────
    for grp in ["control", "treatment"]:
        sub = df[df["group"] == grp]
        rate = sub["converted"].mean()
        print(f"  {grp:>10}: n={len(sub):,}  conversions={sub['converted'].sum():,}"
              f"  rate={rate:.4f}")

    return df


def get_group_arrays(df: pd.DataFrame):
    """
    Returns (control_conversions, control_n, treatment_conversions, treatment_n)
    as plain integers — ready for Beta-Binomial likelihood.
    """
    ctrl  = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]

    ctrl_conv  = int(ctrl["converted"].sum())
    ctrl_n     = int(len(ctrl))
    treat_conv = int(treat["converted"].sum())
    treat_n    = int(len(treat))

    return ctrl_conv, ctrl_n, treat_conv, treat_n


def get_daily_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns daily conversion rates per group — used for novelty effect (Part C).
    """
    daily = (df.groupby(["day", "group"])["converted"]
               .agg(["sum", "count"])
               .rename(columns={"sum": "conversions", "count": "n"})
               .reset_index())
    daily["rate"] = daily["conversions"] / daily["n"]
    return daily