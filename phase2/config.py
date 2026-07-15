# ============================================================
# ShopSmart India — Phase 2 Configuration
# ============================================================
# All parameters flow from Phase 1 outputs.
# Change values here only if Phase 1 parameters change.
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
PHASE2_DIR = os.path.join(BASE_DIR, "phase2")

EXP1_DIR   = os.path.join(PHASE2_DIR, "experiment1")
EXP2_DIR   = os.path.join(PHASE2_DIR, "experiment2")
EXP3_DIR   = os.path.join(PHASE2_DIR, "experiment3")

KAGGLE_RAW = os.path.join(DATA_DIR, "ab_data.csv")

# ── Random seed (reproducibility) ────────────────────────────
RANDOM_SEED = 42

# ── Phase 1 Parameters ───────────────────────────────────────

# Experiment 1 — Checkout Redesign
EXP1 = {
    "baseline_conversion" : 0.0380,
    "treatment_conversion": 0.0420,
    "n_per_group"         : 37_671,
    "total_n"             : 75_342,
    "daily_traffic"       : 40_000,
    "days_required"       : 2,
    "min_run_days"        : 14,
    "alpha"               : 0.05,
    "power"               : 0.80,
    # Injected problem: SRM — 52/48 split instead of 50/50
    "srm_split"           : (0.52, 0.48),
}

# Experiment 2 — Personalized Recommendations
EXP2 = {
    "baseline_mean"   : 850.0,
    "treatment_mean"  : 920.0,
    "baseline_std"    : 420.0,
    "mde_absolute"    : 70.0,
    "n_per_group"     : 566,
    "total_n"         : 1_132,
    "daily_traffic"   : 160_000,
    "days_required"   : 1,
    "min_run_days"    : 14,
    "alpha"           : 0.05,
    "power"           : 0.80,
    # Novelty effect: early lift larger, decays over ~10 days
    "novelty_days"    : 10,
    "novelty_boost"   : 120.0,   # extra Rs on top of treatment mean, day 1
    "novelty_decay"   : 0.75,    # multiplier per day (geometric decay)
    # Pre-experiment covariate correlation with during-experiment revenue
    "pre_exp_corr"    : 0.65,
}

# Experiment 3 — Discount Banner Placement
EXP3 = {
    "baseline_ctr_overall"  : 0.0320,
    "treatment_ctr_overall" : 0.0380,
    "mobile_share"          : 0.65,
    "desktop_share"         : 0.35,
    "baseline_ctr_mobile"   : 0.0350,
    "treatment_ctr_mobile"  : 0.0430,
    "baseline_ctr_desktop"  : 0.0280,
    "treatment_ctr_desktop" : 0.0320,
    "n_per_group"           : 98_772,
    "total_n"               : 197_544,
    "daily_traffic"         : 120_000,
    "days_required"         : 2,
    "min_run_days"          : 14,
    "alpha"                 : 0.05,
    "power"                 : 0.80,
}

# ── Sanity Check Thresholds ───────────────────────────────────
CHECKS = {
    "srm_alpha"              : 0.01,   # SRM uses stricter alpha
    "bot_session_threshold"  : 5,      # actions within N seconds = bot
    "bot_max_daily_sessions" : 50,     # >50 sessions/day = bot candidate
    "novelty_ratio_threshold": 1.15,   # early/late > 1.15 = novelty concern
    "leakage_threshold"      : 0,      # any crossover = flag
    "duplicate_threshold"    : 0,      # any duplicate = flag
    "time_anomaly_hour_start": 2,      # 2am–4am = low-traffic window
    "time_anomaly_hour_end"  : 4,
}

print("✅ Config loaded.")
print(f"   BASE_DIR  : {BASE_DIR}")
print(f"   DATA_DIR  : {DATA_DIR}")
print(f"   PHASE2_DIR: {PHASE2_DIR}")