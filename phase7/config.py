# phase7/config.py
"""
All file paths and business constants for Phase 7 synthesis.
Phase 7 reads upstream results — it does NOT re-run statistics.
"""

import os

# ── Root of the whole project ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Upstream result files ───────────────────────────────────────────────────
PATHS = {
    # Experiment 1 — UPI Checkout
    "exp1_phase3": os.path.join(PROJECT_ROOT, "phase3", "phase3_results.json"),
    "exp1_phase6_bayesian": os.path.join(
        PROJECT_ROOT, "phase6", "experiment1", "bayesian_results.json"
    ),
    "exp1_phase6_sequential": os.path.join(
        PROJECT_ROOT, "phase6", "experiment1", "sequential_results.json"
    ),
    "exp1_phase6_novelty": os.path.join(
        PROJECT_ROOT, "phase6", "experiment1", "novelty_results.json"
    ),
    "exp1_phase6_power": os.path.join(
        PROJECT_ROOT, "phase6", "experiment1", "power_retrospective.json"
    ),
    # Experiment 2 — Personalized Recommendations
    "exp2_phase4": os.path.join(
        PROJECT_ROOT, "phase4", "experiment2", "phase4_exp2_results.json"
    ),
    # Experiment 3 — Discount Banner Placement
    "exp3_phase5": os.path.join(
        PROJECT_ROOT, "phase5", "experiment3", "phase5_exp3_results.json"
    ),
    "exp3_segmented_corrected": os.path.join(
        PROJECT_ROOT, "phase5", "experiment3", "segmented_tests_corrected.csv"
    ),
}

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "phase7", "experiment_outputs")

# ── Business constants (shared across all experiments) ─────────────────────
BUSINESS = {
    # General site traffic
    "daily_visitors": 40_000,
    "split_fraction": 0.50,          # 50/50 A/B split
    "days_per_month": 30,
    "months_per_year": 12,

    # Experiment 1 — UPI Checkout (conversion experiment)
    "exp1_aov_inr": 1_200,           # Average Order Value ₹1,200
    "exp1_baseline_cvr": 0.1204,     # 12.04% control conversion rate

    # Experiment 2 — Recommendations (revenue-per-user experiment)
    # Trusted CUPED lift: ₹18.90/user/month  (from Phase 4)
    # Conservative annual: ₹5.35 crore       (from Phase 4)
    "exp2_cuped_lift_monthly": 18.90,         # ₹/user/month
    "exp2_annual_conservative_inr": 53_52_000_00,  # ₹535,200,000
    "exp2_annual_point_inr":       113_37_25_185,  # ₹1,133,725,185

    # Experiment 3 — Banner Placement (CTR → revenue conversion)
    "exp3_baseline_ctr_mobile":  0.0360,
    "exp3_treatment_ctr_mobile": 0.0441,
    "exp3_baseline_ctr_desktop": 0.0283,
    "exp3_treatment_ctr_desktop":0.0325,
    "exp3_mobile_share": 0.649,      # 128171/197544
    "exp3_desktop_share": 0.351,
    # Assume CTR lift → conversion at 15% of clicks, AOV ₹1,200
    "exp3_click_to_conversion": 0.15,
    "exp3_aov_inr": 1_200,

    # Dev cost assumptions (stated explicitly — adjust as needed)
    "exp1_dev_cost_inr": 5_00_000,    # ₹5 lakh (DO NOT SHIP — counterfactual)
    "exp2_dev_cost_inr": 20_00_000,   # ₹20 lakh (ML infra + A/B harness)
    "exp3_dev_cost_inr": 3_00_000,    # ₹3 lakh (CSS/layout change)

    # Cost-of-delay from Phase 6 (sequential_results.json)
    # ₹2,52,083 = revenue foregone per month if we had waited
    # Protected revenue = ₹17 lakh/month (avoided bad ship)
    "exp1_cost_of_delay_monthly_inr": 2_52_083,
    "exp1_protected_revenue_monthly_inr": 17_00_000,
}