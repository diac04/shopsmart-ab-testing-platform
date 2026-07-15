# phase3/business_impact.py
"""
Translates statistical lift into ₹ business impact.
Assumptions are stated explicitly — never hidden.
"""

from phase3.config import BUSINESS


def compute_impact(ctrl_rate: float,
                   trt_rate:  float,
                   variant_label: str) -> dict:
    """
    Impact model
    ─────────────
    Additional daily conversions = daily_visitors × 0.50 (treatment share)
                                   × (trt_rate - ctrl_rate)
    Annual revenue uplift         = additional daily conversions
                                   × AOV × 365

    ALL assumptions are returned in the dict for full transparency.
    """
    daily_visitors  = BUSINESS["daily_checkout_visitors"]   # 40,000
    aov             = BUSINESS["aov_inr"]                   # ₹1,200
    annual_days     = BUSINESS["annual_days"]               # 365
    traffic_split   = BUSINESS["traffic_split"]             # 0.50

    lift_abs        = trt_rate - ctrl_rate

    # Users in treatment arm at full 50/50 rollout
    daily_treatment_users       = daily_visitors * traffic_split   # 20,000
    additional_daily_conversions = daily_treatment_users * lift_abs

    # ── Scenarios ─────────────────────────────────────────────────────────
    # If lift is negative → revenue loss
    daily_revenue_impact        = additional_daily_conversions * aov
    annual_revenue_impact       = daily_revenue_impact * annual_days

    # Upside scenario: if lift is positive, what if we roll out to 100%?
    full_rollout_daily          = daily_visitors * lift_abs
    full_rollout_annual         = full_rollout_daily * aov * annual_days

    # Crore formatting (1 crore = 10,000,000)
    def to_crore(x):
        return round(x / 1e7, 3)

    def to_lakh(x):
        return round(x / 1e5, 2)

    return {
        "variant"                     : variant_label,
        "assumptions": {
            "daily_checkout_visitors" : daily_visitors,
            "aov_inr"                 : aov,
            "traffic_split_pct"       : traffic_split * 100,
            "annual_days"             : annual_days,
            "source_daily_visitors"   : "Phase 1 — IAMAI 2023 benchmarks",
            "source_aov"              : "Statista India e-commerce 2023 avg",
        },
        "observed_lift": {
            "control_rate"            : round(ctrl_rate, 6),
            "treatment_rate"          : round(trt_rate,  6),
            "absolute_lift"           : round(lift_abs,  6),
            "relative_lift_pct"       : round(lift_abs / ctrl_rate * 100, 3)
                                        if ctrl_rate > 0 else 0,
        },
        "50pct_rollout": {
            "daily_treatment_users"         : int(daily_treatment_users),
            "additional_daily_conversions"  : round(additional_daily_conversions, 2),
            "daily_revenue_impact_inr"      : round(daily_revenue_impact, 2),
            "annual_revenue_impact_inr"     : round(annual_revenue_impact, 2),
            "annual_revenue_impact_crore"   : to_crore(annual_revenue_impact),
            "annual_revenue_impact_lakh"    : to_lakh(annual_revenue_impact),
        },
        "100pct_rollout": {
            "daily_treatment_users"         : int(daily_visitors),
            "additional_daily_conversions"  : round(full_rollout_daily, 2),
            "annual_revenue_impact_inr"     : round(full_rollout_annual, 2),
            "annual_revenue_impact_crore"   : to_crore(full_rollout_annual),
        },
        "risk_note": ("SRM was detected in this experiment. "
                      "All impact figures carry high uncertainty "
                      "until a clean re-run is completed."),
    }