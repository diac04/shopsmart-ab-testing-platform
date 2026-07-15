# phase4/business_impact.py
# ============================================================
# Part E — Business Impact
# Cohen's d + Rs. business impact using the trusted estimate
# The trusted estimate is the CUPED-adjusted mean difference
# because:
#   1. It corrects for pre-experiment imbalance (Rs.37 of the
#      raw Rs.56 lift was pre-existing, not caused by recs)
#   2. It is robust to outliers (outlier removal only moved
#      raw lift by ~5%, so outliers are not the issue)
#   3. It uses rho=0.85 covariate — the strongest possible
#      variance reduction we can achieve with this data
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
import os

from phase4.config import (
    OUTPUT_DIR, ALPHA, BOOTSTRAP_CI,
    MONTHLY_ACTIVE_USERS, N_PER_GROUP_P1,
    BASELINE_MEAN, BASELINE_STD, MDE_ABSOLUTE
)


# ── Cohen's d ─────────────────────────────────────────────────────────────────

def cohens_d(ctrl: np.ndarray,
             treat: np.ndarray,
             label: str = "") -> dict:
    """
    Cohen's d = (mean_treat - mean_ctrl) / pooled_std
    Uses pooled standard deviation (equal to the two-sample version).
    """
    n_c, n_t   = len(ctrl), len(treat)
    mean_c     = np.mean(ctrl)
    mean_t     = np.mean(treat)
    var_c      = np.var(ctrl,  ddof=1)
    var_t      = np.var(treat, ddof=1)

    pooled_std = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) /
                          (n_c + n_t - 2))
    d          = (mean_t - mean_c) / pooled_std

    if abs(d) >= 0.8:
        magnitude = "Large"
    elif abs(d) >= 0.5:
        magnitude = "Medium"
    elif abs(d) >= 0.2:
        magnitude = "Small"
    else:
        magnitude = "Negligible"

    return {
        "label"      : label,
        "n_ctrl"     : n_c,
        "n_treat"    : n_t,
        "mean_ctrl"  : round(float(mean_c),     2),
        "mean_treat" : round(float(mean_t),      2),
        "pooled_std" : round(float(pooled_std),  2),
        "cohens_d"   : round(float(d),           4),
        "magnitude"  : magnitude,
    }


# ── Business impact calculator ────────────────────────────────────────────────

def compute_business_impact(cuped_diff: float,
                             cuped_ci_lo: float,
                             cuped_ci_hi: float,
                             mau: int = MONTHLY_ACTIVE_USERS) -> dict:
    """
    Compute monthly and annual revenue impact of the recommendation
    feature, using the CUPED-adjusted lift as the point estimate.

    Conservative estimate uses the lower bound of the 95% CI.
    Point estimate uses the CUPED mean diff.
    Optimistic estimate uses the upper bound of the 95% CI.

    All estimates are on a per-user-per-month basis.
    We assume the experiment is representative of the full MAU base.
    """
    monthly_point        = cuped_diff   * mau
    monthly_conservative = cuped_ci_lo  * mau
    monthly_optimistic   = cuped_ci_hi  * mau

    annual_point         = monthly_point        * 12
    annual_conservative  = monthly_conservative * 12
    annual_optimistic    = monthly_optimistic   * 12

    relative_lift_pct = 100 * cuped_diff / BASELINE_MEAN

    return {
        "cuped_lift_rs"              : round(cuped_diff,            2),
        "cuped_ci_lo"                : round(cuped_ci_lo,           2),
        "cuped_ci_hi"                : round(cuped_ci_hi,           2),
        "relative_lift_pct"          : round(relative_lift_pct,     2),
        "mau"                        : mau,
        "monthly_point_rs"           : round(monthly_point,         0),
        "monthly_conservative_rs"    : round(monthly_conservative,  0),
        "monthly_optimistic_rs"      : round(monthly_optimistic,    0),
        "annual_point_rs"            : round(annual_point,          0),
        "annual_conservative_rs"     : round(annual_conservative,   0),
        "annual_optimistic_rs"       : round(annual_optimistic,     0),
        "caveat"                     : (
            "Conservative estimate uses the 95% CI lower bound. "
            "CUPED adjusts for pre-experiment imbalance (rho=0.85). "
            "Novelty effect removed (days 7-14 only). "
            "Annual figure assumes the late-period lift is stable — "
            "long-run decay should be monitored post-launch."
        )
    }


# ── Master runner ─────────────────────────────────────────────────────────────

def run_business_impact(ctrl_late: np.ndarray,
                         treat_late: np.ndarray,
                         ctrl_cuped: np.ndarray,
                         treat_cuped: np.ndarray,
                         cuped_ci_lo: float,
                         cuped_ci_hi: float) -> dict:
    """
    Full Part E pipeline.
    """

    print("\n" + "="*60)
    print("  PART E — BUSINESS IMPACT")
    print("="*60)

    # ── Cohen's d: raw vs CUPED ───────────────────────────────
    d_raw   = cohens_d(ctrl_late,  treat_late,  "Raw (unadjusted)")
    d_cuped = cohens_d(ctrl_cuped, treat_cuped, "CUPED-adjusted")

    print("\n  COHEN'S d — EFFECT SIZE")
    print(f"  {'Metric':<30} {'Raw':>10} {'CUPED':>10}")
    print("  " + "-"*52)
    for key in ["mean_ctrl", "mean_treat", "pooled_std",
                "cohens_d", "magnitude"]:
        print(f"  {key:<30} {str(d_raw[key]):>10} {str(d_cuped[key]):>10}")

    phase1_d = round(MDE_ABSOLUTE / BASELINE_STD, 4)
    print(f"\n  Interpretation:")
    print(f"    Raw Cohen's d        = {d_raw['cohens_d']}  "
          f"-> {d_raw['magnitude']}")
    print(f"    CUPED Cohen's d      = {d_cuped['cohens_d']}  "
          f"-> {d_cuped['magnitude']}")
    print(f"    Phase 1 pre-reg d    = {phase1_d}  "
          f"(MDE={MDE_ABSOLUTE} / std={BASELINE_STD})")

    # ── Business impact ───────────────────────────────────────
    cuped_diff = float(np.mean(treat_cuped) - np.mean(ctrl_cuped))
    impact     = compute_business_impact(cuped_diff, cuped_ci_lo, cuped_ci_hi)

    # Pre-format all currency strings to avoid backslash-in-fstring
    lift_str   = f"Rs.{impact['cuped_lift_rs']}"
    ci_lo_str  = f"Rs.{impact['cuped_ci_lo']}"
    ci_hi_str  = f"Rs.{impact['cuped_ci_hi']}"
    rel_str    = f"{impact['relative_lift_pct']}%"
    mau_str    = f"{impact['mau']:,}"

    m_point    = f"Rs.{impact['monthly_point_rs']:,.0f}"
    m_cons     = f"Rs.{impact['monthly_conservative_rs']:,.0f}"
    m_opt      = f"Rs.{impact['monthly_optimistic_rs']:,.0f}"
    a_point    = f"Rs.{impact['annual_point_rs']:,.0f}"
    a_cons     = f"Rs.{impact['annual_conservative_rs']:,.0f}"
    a_opt      = f"Rs.{impact['annual_optimistic_rs']:,.0f}"

    print("\n  BUSINESS IMPACT (CUPED-adjusted estimate)")
    print(f"  {'Metric':<40} {'Value':>20}")
    print("  " + "-"*62)
    print(f"  {'CUPED lift (point estimate)':<40} {lift_str:>20}")
    print(f"  {'95% CI lower bound':<40} {ci_lo_str:>20}")
    print(f"  {'95% CI upper bound':<40} {ci_hi_str:>20}")
    print(f"  {'Relative lift over baseline':<40} {rel_str:>20}")
    print(f"  {'Monthly Active Users (MAU)':<40} {mau_str:>20}")
    print()
    print(f"  {'MONTHLY IMPACT (point estimate)':<40} {m_point:>20}")
    print(f"  {'MONTHLY IMPACT (conservative)':<40} {m_cons:>20}")
    print(f"  {'MONTHLY IMPACT (optimistic)':<40} {m_opt:>20}")
    print()
    print(f"  {'ANNUAL IMPACT (point estimate)':<40} {a_point:>20}")
    print(f"  {'ANNUAL IMPACT (conservative)':<40} {a_cons:>20}")
    print(f"  {'ANNUAL IMPACT (optimistic)':<40} {a_opt:>20}")
    print(f"\n  Caveat: {impact['caveat']}")

    # ── Comparison: raw vs CUPED business impact ──────────────
    raw_diff    = float(np.mean(treat_late) - np.mean(ctrl_late))
    raw_monthly = raw_diff * MONTHLY_ACTIVE_USERS
    raw_annual  = raw_monthly * 12

    overest_monthly = raw_monthly - impact["monthly_point_rs"]
    overest_annual  = raw_annual  - impact["annual_point_rs"]

    # Pre-format all currency strings
    raw_lift_str  = f"Rs.{raw_diff:.2f}"
    raw_m_str     = f"Rs.{raw_monthly:,.0f}"
    raw_a_str     = f"Rs.{raw_annual:,.0f}"
    cuped_m_str   = f"Rs.{impact['monthly_point_rs']:,.0f}"
    cuped_a_str   = f"Rs.{impact['annual_point_rs']:,.0f}"
    over_m_str    = f"Rs.{overest_monthly:,.0f}"
    over_a_str    = f"Rs.{overest_annual:,.0f}"

    overest_pct   = round((raw_diff - cuped_diff) / cuped_diff * 100, 0)
    multiplier    = round(raw_diff / cuped_diff, 1)

    print("\n  OVERESTIMATION IF RAW LIFT USED (no CUPED)")
    print(f"    Raw lift              : {raw_lift_str}")
    print(f"    Raw monthly impact    : {raw_m_str}")
    print(f"    Raw annual impact     : {raw_a_str}")
    print(f"    CUPED monthly         : {cuped_m_str}")
    print(f"    CUPED annual          : {cuped_a_str}")
    print(f"    Overestimation/month  : {over_m_str}")
    print(f"    Overestimation/year   : {over_a_str}")
    print(f"    Raw lift is {multiplier}x the true CUPED effect.")
    print(f"    CUPED prevented a {overest_pct:.0f}% overstatement "
          f"of business impact.")

    # ── Save CSV ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = [
        {"metric": "CUPED lift (Rs.)",
         "value": impact["cuped_lift_rs"]},
        {"metric": "95% CI lower (Rs.)",
         "value": impact["cuped_ci_lo"]},
        {"metric": "95% CI upper (Rs.)",
         "value": impact["cuped_ci_hi"]},
        {"metric": "Relative lift (%)",
         "value": impact["relative_lift_pct"]},
        {"metric": "MAU",
         "value": impact["mau"]},
        {"metric": "Monthly impact point (Rs.)",
         "value": impact["monthly_point_rs"]},
        {"metric": "Monthly impact conservative (Rs.)",
         "value": impact["monthly_conservative_rs"]},
        {"metric": "Monthly impact optimistic (Rs.)",
         "value": impact["monthly_optimistic_rs"]},
        {"metric": "Annual impact point (Rs.)",
         "value": impact["annual_point_rs"]},
        {"metric": "Annual impact conservative (Rs.)",
         "value": impact["annual_conservative_rs"]},
        {"metric": "Annual impact optimistic (Rs.)",
         "value": impact["annual_optimistic_rs"]},
        {"metric": "Raw lift (Rs.)",
         "value": round(raw_diff, 2)},
        {"metric": "Raw annual impact (Rs.)",
         "value": round(raw_annual, 0)},
        {"metric": "Overestimation avoided per year (Rs.)",
         "value": round(overest_annual, 0)},
        {"metric": "Cohen's d (raw)",
         "value": d_raw["cohens_d"]},
        {"metric": "Cohen's d (CUPED)",
         "value": d_cuped["cohens_d"]},
    ]

    df_out   = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "business_impact.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  ✅ business_impact.csv saved -> {csv_path}")

    return {
        "cohens_d_raw"   : d_raw,
        "cohens_d_cuped" : d_cuped,
        "impact"         : impact,
        "raw_diff"       : round(raw_diff,       2),
        "raw_annual"     : round(raw_annual,      0),
        "overest_annual" : round(overest_annual,  0),
    }