# phase3/data_loader.py
"""
Loads and prepares exp1 data for Phase 3.
Applies deduplication (Phase 2 mandate) and optional bot removal.
"""

import pandas as pd
from phase3.config import (DATA_FILE, BOT_COLUMN, GROUP_COLUMN,
                            OUTCOME_COLUMN, DAY_COLUMN, NOVELTY_CUTOFF)


def load_exp1(remove_bots: bool = True,
              late_period_only: bool = False) -> pd.DataFrame:
    """
    Returns a clean DataFrame for Experiment 1.

    Parameters
    ----------
    remove_bots      : drop rows where is_bot == 1
    late_period_only : keep only experiment_day >= NOVELTY_CUTOFF

    Returns
    -------
    pd.DataFrame with columns: group, converted, experiment_day, [is_bot, ...]
    """
    df = pd.read_csv(DATA_FILE)

    # ── 1. Deduplication (Phase 2 flag: keep first occurrence) ────────────
    user_col = _detect_user_col(df)
    before   = len(df)
    if user_col:
        df = df.drop_duplicates(subset=[user_col], keep="first")
    after = len(df)
    print(f"[DataLoader] Deduplication: {before:,} → {after:,} rows "
          f"(removed {before - after:,} duplicates)")

    # ── 2. Bot removal ─────────────────────────────────────────────────────
    if remove_bots and BOT_COLUMN in df.columns:
        before_bot = len(df)
        df = df[df[BOT_COLUMN] == 0].copy()
        print(f"[DataLoader] Bot removal: {before_bot:,} → {len(df):,} rows "
              f"(removed {before_bot - len(df):,} bots)")
    elif remove_bots:
        print(f"[DataLoader] WARNING: '{BOT_COLUMN}' column not found — "
              f"no bot removal applied.")

    # ── 3. Late-period filter (novelty effect check) ───────────────────────
    if late_period_only and DAY_COLUMN in df.columns:
        before_late = len(df)
        df = df[df[DAY_COLUMN] >= NOVELTY_CUTOFF].copy()
        print(f"[DataLoader] Late-period filter (day >= {NOVELTY_CUTOFF}): "
              f"{before_late:,} → {len(df):,} rows")

    # ── 4. Validate required columns ───────────────────────────────────────
    for col in [GROUP_COLUMN, OUTCOME_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")

    # ── 5. Basic type enforcement ─────────────────────────────────────────
    df[OUTCOME_COLUMN] = df[OUTCOME_COLUMN].astype(int)

    return df


def split_groups(df: pd.DataFrame):
    """Return (control_df, treatment_df)."""
    ctrl = df[df[GROUP_COLUMN] == "control"].copy()
    trt  = df[df[GROUP_COLUMN] == "treatment"].copy()
    print(f"[DataLoader] Split → Control: {len(ctrl):,} | "
          f"Treatment: {len(trt):,}")
    return ctrl, treatment_df_alias(trt)


def treatment_df_alias(trt):
    return trt


def get_conversions(df: pd.DataFrame):
    """Return (n_total, n_converted, conversion_rate) for a group."""
    n     = len(df)
    conv  = df[OUTCOME_COLUMN].sum()
    rate  = conv / n if n > 0 else 0.0
    return n, int(conv), rate


def _detect_user_col(df: pd.DataFrame) -> str | None:
    """Auto-detect the user-ID column name."""
    candidates = ["user_id", "userid", "id", "visitor_id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None