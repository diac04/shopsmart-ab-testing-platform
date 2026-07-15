"""
dashboard_data.py
-----------------
Central data layer for the ShopSmart A/B Testing dashboard.
Reads existing phase JSON/CSV outputs and exposes canonical figures.
"""

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_json(rel_path: str) -> dict:
    full = ROOT / rel_path
    if not full.exists():
        return {}
    with open(full, "r") as f:
        return json.load(f)


def _load_csv(rel_path: str) -> pd.DataFrame:
    full = ROOT / rel_path
    if not full.exists():
        return pd.DataFrame()
    return pd.read_csv(full)


# ── Experiment 1 loaders ──────────────────────────────────────────────────
def load_exp1_phase3():
    return _load_json("phase3/phase3_results.json")

def load_exp1_bayesian():
    return _load_json("phase6/experiment1/bayesian_results.json")

def load_exp1_sequential():
    return _load_json("phase6/experiment1/sequential_results.json")

def load_exp1_novelty():
    return _load_json("phase6/experiment1/novelty_results.json")

def load_exp1_power_retro():
    return _load_json("phase6/experiment1/power_retrospective.json")

def load_exp1_daily_rates():
    return _load_csv("phase6/experiment1/daily_conversion_rates.csv")

def load_exp1_sprt_boundaries():
    return _load_csv("phase6/experiment1/obrien_fleming_boundaries.csv")


# ── Experiment 2 loaders ──────────────────────────────────────────────────
def load_exp2_phase4():
    return _load_json("phase4/experiment2/phase4_exp2_results.json")

def load_exp2_method_comparison():
    return _load_csv("phase4/experiment2/method_comparison.csv")

def load_exp2_outlier_sensitivity():
    return _load_csv("phase4/experiment2/outlier_sensitivity.csv")

def load_exp2_cuped_results():
    return _load_csv("phase4/experiment2/cuped_results.csv")


# ── Experiment 3 loaders ──────────────────────────────────────────────────
def load_exp3_phase5():
    return _load_json("phase5/experiment3/phase5_exp3_results.json")

def load_exp3_overall_test():
    return _load_json("phase5/experiment3/overall_test.json")

def load_exp3_segmented_corrected():
    return _load_csv("phase5/experiment3/segmented_tests_corrected.csv")

def load_exp3_interaction_coefs():
    return _load_csv("phase5/experiment3/interaction_model_coefs.csv")

def load_exp3_decision_table():
    return _load_csv("phase5/experiment3/decision_table.csv")


# ── Phase 7 loaders ───────────────────────────────────────────────────────
def load_business_summary():
    return _load_json("phase7/experiment_outputs/business_summary.json")

def load_decision_table():
    return _load_csv("phase7/experiment_outputs/final_decision_table.csv")

def load_power_analysis():
    return _load_json("phase1/phase1_power_analysis.json")


# ── Canonical experiment summary (used everywhere) ────────────────────────
EXPERIMENT_SUMMARY = {
    "exp1": {
        "name": "UPI Checkout Redesign",
        "phase": "Phase 3 and 6",
        "status": "DO NOT SHIP",
        "significant": False,
        "direction": "negative",
        "n_control": 129129,
        "n_treatment": 150315,
        "duration_days": 14,
        "metric": "Conversion Rate",
        "control_rate": 0.03326,
        "treatment_rate": 0.03164,
        "lift_pp": -0.162,
        "p_value": 0.188,
        "ci_lower_pp": -0.40,
        "ci_upper_pp": 0.08,
        "srm_flag": True,
        "srm_chi2": 447.13,
        "srm_ratio": 1.083,
        "p_treatment_better": 0.094,
        "e_loss_ship_pp": 0.1670,
        "e_loss_hold_pp": 0.0055,
        "avoided_loss_crore_50pct": 1.40,
        "avoided_loss_crore_100pct": 2.79,
        "protected_revenue_lakh_month": 17.00,
        "headline": (
            "The redesigned UPI checkout performed worse than the current design. "
            "Rollout is blocked."
        ),
        "impact_plain": (
            "Blocking this rollout avoids an estimated loss of up to "
            "1.40 Crore per year. There is only a 9 percent chance the new "
            "design is actually better than the current one."
        ),
        "caveat": (
            "A traffic split imbalance (SRM) was detected, which weakens confidence "
            "in these numbers. The experiment should be re-run after the traffic "
            "allocation mechanism is fixed."
        ),
    },
    "exp2": {
        "name": "Personalized Recommendations",
        "phase": "Phase 4",
        "status": "SHIP",
        "significant": True,
        "direction": "positive",
        "n_control": 6000,
        "n_treatment": 6000,
        "duration_days": 14,
        "metric": "Revenue per User per Month",
        "cuped_lift": 18.90,
        "ci_lower": 8.92,
        "ci_upper": 28.91,
        "raw_lift_rejected": 56.35,
        "bias_removed": 37.46,
        "annual_conservative_crore": 53.52,
        "annual_point_crore": 113.37,
        "dev_cost_lakh": 20,
        "payback_days": 1.3,
        "cohen_d": 0.0647,
        "cuped_rho": 0.8473,
        "headline": (
            "The personalized recommendation engine drives a measurable increase "
            "in revenue per user. Full rollout is approved."
        ),
        "impact_plain": (
            "We are 95 percent confident this adds between 8.92 and 28.91 rupees "
            "per user per month. At the conservative end, that translates to "
            "53.52 Crore per year. The 20 lakh build cost is recovered in 1.3 days."
        ),
        "caveat": (
            "The raw uncorrected number was 3x higher due to a group composition "
            "imbalance at the start of the experiment. The conservative estimate "
            "of 53 Crore is the figure to communicate to leadership."
        ),
    },
    "exp3": {
        "name": "Discount Banner Placement",
        "phase": "Phase 5",
        "status": "SHIP",
        "significant": True,
        "direction": "positive",
        "n_control": 98772,
        "n_treatment": 98772,
        "duration_days": 14,
        "metric": "Click Through Rate",
        "overall_lift_pp": 0.68,
        "overall_lift_pct": 20.4,
        "z_stat": 8.02,
        "p_value": 0.0,
        "mobile_lift_pp": 0.81,
        "desktop_lift_pp": 0.42,
        "interaction_p": 0.18,
        "add_conversions_month": 1212,
        "add_revenue_lakh_month": 14.54,
        "annual_revenue_crore": 1.74,
        "dev_cost_lakh": 3,
        "payback_months": 0.2,
        "headline": (
            "Repositioning the discount banner increases click through rate by "
            "20 percent across both mobile and desktop. Full rollout is approved."
        ),
        "impact_plain": (
            "This change is projected to add approximately 1,212 additional purchases "
            "per month and 14.54 Lakh in monthly revenue, or 1.74 Crore per year. "
            "The 3 lakh build cost is recovered in about 6 days."
        ),
        "caveat": (
            "The click to purchase conversion rate of 15 percent is an assumption "
            "based on historical benchmarks and should be validated in GA4 after "
            "launch. Tablet users were not measured and should be monitored separately."
        ),
    },
}

COMBINED_IMPACT = {
    "shipped_annual_conservative_crore": 55.26,
    "avoided_loss_crore": 1.40,
    "total_protection_crore": 56.66,
}