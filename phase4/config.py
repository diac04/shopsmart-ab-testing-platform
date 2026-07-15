# phase4/config.py
# ============================================================
# Phase 4 Configuration — ShopSmart India A/B Testing Platform
# Experiment 2: Personalized Recommendations
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "phase2", "experiment2",
                          "exp2_recommendations_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "phase4", "experiment2")

# ── Experiment identity ──────────────────────────────────────
EXPERIMENT_NAME = "Experiment 2 — Personalized Recommendations"
CONTROL_LABEL   = "control"
TREATMENT_LABEL = "treatment"

# ── Column names ─────────────────────────────────────────────
REVENUE_COL     = "revenue"
PRE_EXP_COL     = "pre_exp_revenue"
GROUP_COL       = "group"
DAY_COL         = "experiment_day"

# ── Phase 1 pre-registered values ────────────────────────────
BASELINE_MEAN   = 850.0       # Rs. — from Phase 1
EXPECTED_TREAT  = 920.0       # Rs.
BASELINE_STD    = 420.0       # Rs.
MDE_ABSOLUTE    = 70.0        # Rs.
ALPHA           = 0.05
POWER           = 0.80
N_PER_GROUP_P1  = 566         # Phase 1 required n per group

# ── Novelty effect: use late-period only (Phase 2 finding) ───
# Days 1–6 contaminated by novelty boost
# Primary analysis window = days 7–14
NOVELTY_CUTOFF_DAY = 7        # use day >= 7 as primary

# ── Bootstrap settings ───────────────────────────────────────
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED      = 42
BOOTSTRAP_CI        = 0.95

# ── Outlier / winsorization settings ─────────────────────────
WINSOR_LOWER = 0.01           # 1st percentile
WINSOR_UPPER = 0.99           # 99th percentile
TRIM_FRAC    = 0.05           # 5% each tail for trimmed mean

# ── Delta method flag ────────────────────────────────────────
# Revenue-per-user here is a SIMPLE AVERAGE (revenue / n_users)
# NOT a ratio metric (it does not divide by visits per user)
# Delta method therefore NOT required — explained in written decision
IS_RATIO_METRIC = False

# ── Business scale ───────────────────────────────────────────
MONTHLY_ACTIVE_USERS = 5_000_000   # ShopSmart India MAU assumption