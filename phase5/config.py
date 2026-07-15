# phase5/config.py
"""
Phase 5 Configuration — Experiment 3: Discount Banner Placement
Metric: Click-Through Rate (CTR) by device (mobile / desktop)
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "phase2", "experiment3", "exp3_banner_data.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "phase5", "experiment3")

# ── Experiment identity ─────────────────────────────────────────────────────
EXPERIMENT_NAME   = "Experiment 3 – Discount Banner Placement"
TREATMENT_COL     = "group"            # actual column name in data
OUTCOME_COL       = "clicked"          # binary 0/1
DEVICE_COL        = "device"           # actual column name in data
USER_ID_COL       = "user_id"

CONTROL_LABEL     = "control"
TREATMENT_LABEL   = "treatment"

DEVICE_SEGMENTS   = ["mobile", "desktop"]   # tablet does not exist in this data

# ── Statistical settings ────────────────────────────────────────────────────
ALPHA             = 0.05
TWO_SIDED         = True

# ── Business parameters (₹) ────────────────────────────────────────────────
MONTHLY_ACTIVE_USERS        = 2_000_000
DEVICE_SHARE = {
    "mobile":  0.62,
    "desktop": 0.38,           # redistributed tablet share to desktop
}

REVENUE_PER_USER_PER_MONTH  = 180.0        # ₹ ARPU
REVENUE_LOSS_FRACTION       = 0.04         # 4% ARPU loss per affected user

# ── Phase 1 expectation (for narrative) ────────────────────────────────────
PHASE1_HYPOTHESIS = (
    "Phase 1 predicted mobile users would show the strongest positive "
    "treatment effect because the top-of-screen banner is more prominent "
    "on small viewports."
)