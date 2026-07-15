# phase3/config.py
"""
Phase 3 Configuration — ShopSmart India A/B Testing Platform
Experiment 1: Checkout Redesign
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE   = os.path.join(BASE_DIR, "phase2", "experiment1",
                           "exp1_checkout_data.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "phase3")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "phase3_results.json")

# ── Pre-registered values (from Phase 1) ──────────────────────────────────
PRE_REGISTERED = {
    "baseline_rate"   : 0.038,
    "treatment_rate"  : 0.042,
    "mde_absolute"    : 0.004,
    "mde_relative_pct": 10.53,
    "alpha"           : 0.05,
    "power"           : 0.80,
    "cohens_h_prereg" : 0.0204,
    "n_per_group"     : 37_671,
}

# ── Statistical test settings ──────────────────────────────────────────────
ALPHA           = 0.05          # two-sided
CI_LEVEL        = 0.95
RANDOM_SEED     = 42

# ── Bot-filtering column (created by Phase 2 simulator) ───────────────────
BOT_COLUMN      = "is_bot"      # bool column; 0/1 or True/False
GROUP_COLUMN    = "group"       # values: "control" | "treatment"
OUTCOME_COLUMN  = "converted"   # binary 0/1
DAY_COLUMN      = "experiment_day"
NOVELTY_CUTOFF  = 7             # days 1-6 = early, 7+ = late

# ── Business impact assumptions ───────────────────────────────────────────
BUSINESS = {
    "daily_checkout_visitors" : 40_000,   # from Phase 1 traffic assumptions
    "aov_inr"                 : 1_200,    # Average Order Value ₹1,200
                                          # (Statista India e-comm 2023)
    "annual_days"             : 365,
    "traffic_split"           : 0.50,     # 50% in treatment at full rollout
    "currency"                : "INR",
}

# ── Analysis variants to run ───────────────────────────────────────────────
ANALYSIS_VARIANTS = {
    "primary"  : "full_deduplicated_no_bots",   # headline number
    "secondary": "full_deduplicated_with_bots",  # sensitivity
    "tertiary" : "late_period_no_bots",          # novelty check (days 7-14)
}