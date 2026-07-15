# ============================================================
# ShopSmart India — Phase 2 — Data Loader
# ============================================================
# Loads the real Kaggle AB dataset for Experiment 1.
# Injects a realistic 52/48 Sample Ratio Mismatch (SRM)
# by selectively dropping control users to simulate what
# happens when a tracking/assignment bug favours one group.
# ============================================================

import pandas as pd
import numpy as np
import os
import sys

# ── Make sure config is importable ───────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import KAGGLE_RAW, EXP1_DIR, EXP1, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# ============================================================
# SECTION 1 — Load Raw Kaggle Data
# ============================================================

def load_raw_kaggle(filepath: str) -> pd.DataFrame:
    """
    Load the raw Kaggle AB_Test_Results CSV.

    Expected columns:
        user_id   : int
        timestamp : str  (datetime)
        group     : str  ('control' or 'treatment')
        landing_page : str ('old_page' or 'new_page')
        converted : int  (0 or 1)

    Returns a cleaned DataFrame.
    """
    print("\n" + "="*60)
    print("LOADING RAW KAGGLE DATASET — EXPERIMENT 1")
    print("="*60)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n❌ File not found: {filepath}"
            f"\n   Please download ab_data.csv from Kaggle and place it in:"
            f"\n   {os.path.dirname(filepath)}"
        )

    df = pd.read_csv(filepath)

    print(f"  Raw shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns            : {list(df.columns)}")
    print(f"  Memory usage       : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # ── Basic dtype fixes ────────────────────────────────────
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['converted'] = df['converted'].astype(int)
    df['user_id']   = df['user_id'].astype(int)
    df['group']     = df['group'].str.strip().str.lower()
    df['landing_page'] = df['landing_page'].str.strip().str.lower()

    print(f"\n  Group counts (raw):")
    grp = df['group'].value_counts()
    for g, n in grp.items():
        print(f"    {g:>10}: {n:>10,}  ({n/len(df)*100:.2f}%)")

    print(f"\n  Date range         : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Converted (raw)    : {df['converted'].sum():,} / {len(df):,} "
          f"({df['converted'].mean()*100:.2f}%)")

    return df


# ============================================================
# SECTION 2 — Clean Mismatched Page Assignments
# ============================================================

def clean_page_group_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Kaggle dataset has a known issue: some users in the
    'control' group were shown the 'new_page' and vice versa.
    These are MISASSIGNED users — they must be removed before
    any analysis, as they contaminate both groups.

    Valid assignments:
        control   → old_page
        treatment → new_page
    """
    print("\n" + "="*60)
    print("CLEANING PAGE/GROUP MISMATCHES")
    print("="*60)

    before = len(df)

    valid_mask = (
        ((df['group'] == 'control')   & (df['landing_page'] == 'old_page')) |
        ((df['group'] == 'treatment') & (df['landing_page'] == 'new_page'))
    )

    mismatched = df[~valid_mask]
    df_clean   = df[valid_mask].copy()

    after = len(df_clean)
    removed = before - after

    print(f"  Rows before cleaning : {before:>10,}")
    print(f"  Mismatched rows      : {removed:>10,}  ({removed/before*100:.2f}%)")
    print(f"  Rows after cleaning  : {after:>10,}")

    if len(mismatched) > 0:
        print(f"\n  Mismatch breakdown:")
        cross = pd.crosstab(mismatched['group'], mismatched['landing_page'])
        print(cross.to_string(index=True))

    print(f"\n  Group counts after cleaning:")
    grp = df_clean['group'].value_counts()
    for g, n in grp.items():
        print(f"    {g:>10}: {n:>10,}  ({n/len(df_clean)*100:.2f}%)")

    return df_clean


# ============================================================
# SECTION 3 — Inject Sample Ratio Mismatch (SRM)
# ============================================================

def inject_srm(df: pd.DataFrame,
               target_treatment_share: float = 0.52,
               seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Inject a realistic 52/48 SRM into Experiment 1.

    HOW IT WORKS:
    -------------
    Real SRMs happen when:
      - A tracking pixel fires only on fast-loading pages
        (treatment page loads faster → more treatment events logged)
      - A cookie assignment bug creates a race condition that
        favours one group
      - An SDK version mismatch silently drops some control events

    We simulate this by RANDOMLY DROPPING a fraction of control
    users, leaving treatment users intact. This mirrors what
    happens when control-side logging fails intermittently.

    The result: treatment gets ~52% of users, control ~48%.
    This is enough to trigger a statistically significant SRM
    at chi-square test while being subtle enough that a naive
    analyst might miss it visually.

    Parameters
    ----------
    df                    : cleaned DataFrame (post mismatch removal)
    target_treatment_share: desired treatment fraction (0.52)
    seed                  : numpy random seed
    """
    print("\n" + "="*60)
    print("INJECTING SAMPLE RATIO MISMATCH (SRM) — EXPERIMENT 1")
    print("="*60)
    print(f"  Target split       : {target_treatment_share:.0%} treatment "
          f"/ {1-target_treatment_share:.0%} control")
    print(f"  Injection method   : Randomly drop control users")
    print(f"  Simulates          : Intermittent control-side logging failure")

    rng = np.random.default_rng(seed)

    control_df   = df[df['group'] == 'control'].copy()
    treatment_df = df[df['group'] == 'treatment'].copy()

    n_treatment = len(treatment_df)

    # If treatment is fraction t of total, then
    # total = n_treatment / t → n_control = total * (1-t)
    n_control_target = int(n_treatment * (1 - target_treatment_share)
                           / target_treatment_share)

    n_control_current = len(control_df)
    n_to_drop = max(0, n_control_current - n_control_target)

    print(f"\n  Control users (before) : {n_control_current:>10,}")
    print(f"  Control users (target) : {n_control_target:>10,}")
    print(f"  Users to drop          : {n_to_drop:>10,}")

    # Randomly choose which control users to drop
    drop_idx = rng.choice(control_df.index,
                           size=n_to_drop,
                           replace=False)
    control_df_srm = control_df.drop(index=drop_idx)

    df_srm = pd.concat([control_df_srm, treatment_df], ignore_index=True)
    df_srm = df_srm.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Report final split ────────────────────────────────────
    total = len(df_srm)
    grp   = df_srm['group'].value_counts()

    print(f"\n  Final group counts (after SRM injection):")
    for g, n in grp.items():
        print(f"    {g:>10}: {n:>10,}  ({n/total*100:.2f}%)")

    actual_treatment_share = grp.get('treatment', 0) / total
    print(f"\n  Actual treatment share : {actual_treatment_share:.4f} "
          f"(target was {target_treatment_share:.4f})")
    print(f"\n  ⚠️  SRM INJECTED — This will be caught by the chi-square "
          f"goodness-of-fit check in sanity_checks.py")

    return df_srm


# ============================================================
# SECTION 4 — Add Experiment Metadata
# ============================================================

def add_experiment_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful derived columns for downstream analysis:
      - experiment_day  : integer day of experiment (1-indexed)
      - hour_of_day     : for bot and time-anomaly checks
      - day_of_week     : 0=Monday … 6=Sunday
      - week_of_exp     : which week of the experiment (1-indexed)
    """
    print("\n" + "="*60)
    print("ADDING EXPERIMENT METADATA COLUMNS")
    print("="*60)

    df = df.copy()
    start_date = df['timestamp'].min()

    df['experiment_day'] = (
        (df['timestamp'] - start_date).dt.days + 1
    ).astype(int)

    df['hour_of_day']  = df['timestamp'].dt.hour
    df['day_of_week']  = df['timestamp'].dt.dayofweek
    df['week_of_exp']  = ((df['experiment_day'] - 1) // 7 + 1).astype(int)

    print(f"  experiment_day range : {df['experiment_day'].min()} "
          f"→ {df['experiment_day'].max()}")
    print(f"  hour_of_day range    : {df['hour_of_day'].min()} "
          f"→ {df['hour_of_day'].max()}")
    print(f"  Weeks in experiment  : {df['week_of_exp'].max()}")
    print(f"  Columns now          : {list(df.columns)}")

    return df


# ============================================================
# SECTION 5 — Save Dataset
# ============================================================

def save_experiment1(df: pd.DataFrame, output_dir: str) -> str:
    """Save the final Experiment 1 dataset to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "exp1_checkout_data.csv")
    df.to_csv(filepath, index=False)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"\n  ✅ Saved: {filepath}")
    print(f"     Rows : {len(df):,}")
    print(f"     Size : {size_mb:.2f} MB")
    return filepath


# ============================================================
# SECTION 6 — Master Function
# ============================================================

def load_experiment1_data(verbose: bool = True) -> pd.DataFrame:
    """
    Master function — runs the full Experiment 1 data pipeline:
      1. Load raw Kaggle CSV
      2. Clean page/group mismatches
      3. Inject SRM (52/48 split)
      4. Add metadata columns
      5. Save to experiment1/

    Returns the final DataFrame.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1 — FULL DATA LOADING PIPELINE")
    print("="*60)

    # Step 1: Load
    df_raw = load_raw_kaggle(KAGGLE_RAW)

    # Step 2: Clean mismatches
    df_clean = clean_page_group_mismatch(df_raw)

    # Step 3: Inject SRM
    df_srm = inject_srm(
        df_clean,
        target_treatment_share=EXP1['srm_split'][0]  # 0.52
    )

    # Step 4: Add metadata
    df_final = add_experiment_metadata(df_srm)

    # Step 5: Save
    save_experiment1(df_final, EXP1_DIR)

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 1 — FINAL DATASET SUMMARY")
    print("="*60)
    total = len(df_final)
    for g, gdf in df_final.groupby('group'):
        cr = gdf['converted'].mean() * 100
        print(f"  {g:>10} : {len(gdf):>8,} users | "
              f"Conversion rate: {cr:.4f}%")
    print(f"  {'TOTAL':>10} : {total:>8,} users")
    print(f"\n  Columns : {list(df_final.columns)}")

    return df_final


# ============================================================
# SECTION 7 — Quick standalone test
# ============================================================

if __name__ == "__main__":
    df = load_experiment1_data()

    print("\n" + "="*60)
    print("SAMPLE ROWS")
    print("="*60)
    print(df.head(10).to_string(index=False))

    print("\n" + "="*60)
    print("DTYPES")
    print("="*60)
    print(df.dtypes)

    print("\n✅ data_loader.py complete.")