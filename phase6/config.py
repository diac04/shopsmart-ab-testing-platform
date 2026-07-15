# phase6/config.py
"""
Phase 6 Configuration
ShopSmart India A/B Testing Platform
Experiment 1 (Checkout Flow) — Advanced Statistical Layer
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2_EXP1 = os.path.join(BASE_DIR, "phase2", "experiment1",
                            "exp1_checkout_data.csv")
PHASE6_OUT  = os.path.join(BASE_DIR, "phase6", "experiment1")

# ── Phase 1 Pre-registered values (Experiment 1) ─────────────────────────────
PRE_REG = {
    "baseline_rate"    : 0.038,
    "treatment_rate"   : 0.042,
    "mde_absolute"     : 0.004,
    "mde_relative_pct" : 10.53,
    "alpha"            : 0.05,
    "power"            : 0.80,
    "cohens_h"         : 0.0204,
    "n_per_group"      : 37_671,
    "total_n"          : 75_342,
    "days_required"    : 2,
    "min_run_days"     : 14,
}

# ── Phase 3 Observed values (Experiment 1) ────────────────────────────────────
OBSERVED = {
    "cohens_h"        : 0.005,
    "lift_abs"        : -0.001617,
    "lift_rel_pct"    : -1.343,
    "p_value"         : 0.188081,
    "significant"     : False,
}

# ── Traffic (Phase 1 assumptions) ────────────────────────────────────────────
DAILY_CHECKOUT_VISITORS = 40_000

# ── Bayesian priors to test (Part A) ─────────────────────────────────────────
PRIORS = {
    "uniform"  : {"alpha": 1,    "beta": 1,
                  "label": "Beta(1,1) — Uniform (primary)"},
    "weakly"   : {"alpha": 2,    "beta": 2,
                  "label": "Beta(2,2) — Weakly informative (sensitivity)"},
    "informed" : {"alpha": 3.8,  "beta": 96.2,
                  "label": "Beta(3.8,96.2) — Centred on baseline 3.8%"},
}

# ── Expected loss threshold for shipping decision ─────────────────────────────
LOSS_THRESHOLD_ABS = 0.001   # 0.1 pp — below this we tolerate shipping

# ── Sequential testing (Part B) ──────────────────────────────────────────────
SEQUENTIAL = {
    "alpha"              : 0.05,
    "beta"               : 0.20,          # 1 - power
    "n_simulations"      : 10_000,        # for peeking simulation
    "peek_every"         : 500,           # peek frequency           # peek frequency
    "h0_rate"            : 0.038,         # null: no difference
    "h1_lift"            : 0.004,         # alternative: +0.4pp
    "ob_flemming_looks"  : 5,             # number of interim looks
}

# ── Novelty effect (Part C) ──────────────────────────────────────────────────
NOVELTY = {
    "warmup_days"        : 3,   # days to flag as "initial" period
    "rolling_window"     : 3,   # rolling average window in days
}

# ── Business impact (used in cost-of-delay) ──────────────────────────────────
BUSINESS = {
    # From Phase 3 — if treatment had lifted by MDE (+0.4pp on 3.8% base)
    # 40 000 daily visitors × 0.4% lift × ₹850 AOV = ₹136 000/day
    "daily_revenue_impact_inr" : 136_000,
    "aov_inr"                  : 850,
    "daily_visitors"           : 40_000,
}

# ── Plot style ────────────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
BRAND_COLORS = {
    "control"   : "#2E86AB",
    "treatment" : "#E84855",
    "neutral"   : "#F4A261",
    "success"   : "#2DC653",
}