# phase7/business_summary.py
"""
Translates statistical results into business language:
  - Additional conversions / month
  - Additional revenue / month  (₹)
  - Annual revenue impact       (₹ crore)
  - Dev cost vs revenue gain
  - Payback period              (months)

All assumptions are printed explicitly.
"""

from phase7.config import BUSINESS


# ── helpers ────────────────────────────────────────────────────────────────
def _crore(inr: float) -> str:
    return f"₹{inr / 1e7:.2f} Crore"


def _lakh(inr: float) -> str:
    return f"₹{inr / 1e5:.2f} Lakh"


# ── Experiment 1 — UPI Checkout (DO NOT SHIP) ──────────────────────────────
def summarise_exp1(data: dict) -> dict:
    """
    Even though Exp 1 is a NO-SHIP, we still quantify:
      (a) the cost of the bad decision had we shipped
      (b) the cost-of-delay from Phase 6
      (c) the protected revenue from NOT shipping
    """
    b = BUSINESS
    daily_treatment_users = b["daily_visitors"] * b["split_fraction"]
    monthly_treatment_users = daily_treatment_users * b["days_per_month"]

    # Observed lift is NEGATIVE: -0.162 pp
    observed_lift_abs = -0.001617          # from Phase 3 / Phase 6
    aov = b["exp1_aov_inr"]

    # Revenue delta if shipped (monthly, 50% rollout)
    monthly_rev_delta_50pct = (
        monthly_treatment_users * observed_lift_abs * aov
    )
    annual_rev_delta_50pct  = monthly_rev_delta_50pct * b["months_per_year"]
    annual_rev_delta_100pct = annual_rev_delta_50pct * 2

    # Bayesian expected loss if shipped
    # E[loss | ship] = 0.1670 pp  (from Phase 6 bayesian_results.json)
    bayes_expected_loss_pp = 0.001670
    monthly_bayes_loss = (
        monthly_treatment_users * bayes_expected_loss_pp * aov
    )

    result = {
        "experiment": "Experiment 1 — UPI Checkout Redesign",
        "decision": "DO NOT SHIP",
        "srm_flag": True,
        "direction": "NEGATIVE — treatment worse than control",

        # What we avoided
        "avoided_annual_loss_50pct_inr": abs(annual_rev_delta_50pct),
        "avoided_annual_loss_100pct_inr": abs(annual_rev_delta_100pct),
        "avoided_annual_loss_50pct_str": _crore(abs(annual_rev_delta_50pct)),
        "avoided_annual_loss_100pct_str": _crore(abs(annual_rev_delta_100pct)),

        # Bayesian expected loss
        "bayesian_expected_loss_if_shipped_monthly_inr": monthly_bayes_loss,
        "bayesian_expected_loss_if_shipped_monthly_str": _lakh(monthly_bayes_loss),

        # Protected revenue (from Phase 6 sequential)
        "protected_revenue_monthly_inr": b["exp1_protected_revenue_monthly_inr"],
        "protected_revenue_monthly_str": _lakh(b["exp1_protected_revenue_monthly_inr"]),
        "protected_revenue_annual_inr":  b["exp1_protected_revenue_monthly_inr"] * 12,
        "protected_revenue_annual_str":  _crore(
            b["exp1_protected_revenue_monthly_inr"] * 12
        ),

        # Cost-of-delay (counterfactual: if MDE had been real)
        "cost_of_delay_monthly_inr": b["exp1_cost_of_delay_monthly_inr"],
        "cost_of_delay_monthly_str": _lakh(b["exp1_cost_of_delay_monthly_inr"]),
        "cost_of_delay_narrative": (
            "Had the true effect been +0.40 pp (the pre-registered MDE), "
            "each month of delay would have cost ₹2,52,083 in foregone revenue. "
            "Because the observed effect is NEGATIVE, faster shipping would have "
            "destroyed value — the SPRT correctly stopped at n=600, protecting "
            "₹17 lakh/month that would have been lost to a premature rollout."
        ),

        # Payback period — not applicable (no-ship)
        "dev_cost_inr": b["exp1_dev_cost_inr"],
        "payback_months": "N/A — DO NOT SHIP",

        "assumptions": [
            "Daily visitors = 40,000 | 50/50 split → 20,000 treatment/day",
            "AOV = ₹1,200",
            "Observed lift = -0.162 pp (from Phase 3, SRM-flagged, HIGH UNCERTAINTY)",
            "Bayesian E[loss|ship] = 0.1670 pp (Phase 6, Beta(1,1) prior)",
            "Protected revenue = ₹17 lakh/month (Phase 6 sequential)",
            "Cost-of-delay = ₹2,52,083/month (Phase 6 counterfactual at MDE)",
            "Dev cost = ₹5 lakh (assumption — confirm with engineering)",
            "SRM present: all figures HIGH UNCERTAINTY",
        ],
    }
    return result


# ── Experiment 2 — Personalized Recommendations (SHIP) ─────────────────────
def summarise_exp2(data: dict) -> dict:
    b = BUSINESS
    daily_visitors   = b["daily_visitors"]
    split            = b["split_fraction"]
    days_per_month   = b["days_per_month"]

    # CUPED trusted lift: ₹18.90/user/month
    # This is already per-user-per-month from Phase 4
    lift_per_user_month = b["exp2_cuped_lift_monthly"]

    # Users who will see the feature (treatment arm, monthly)
    # After ship: 100% rollout assumed for revenue projection
    monthly_users_full_rollout = daily_visitors * days_per_month

    monthly_rev_gain   = monthly_users_full_rollout * lift_per_user_month
    annual_rev_gain    = monthly_rev_gain * b["months_per_year"]

    # Conservative (Phase 4 bootstrap lower CI bound)
    annual_conservative = b["exp2_annual_conservative_inr"]
    annual_point        = b["exp2_annual_point_inr"]

    dev_cost            = b["exp2_dev_cost_inr"]
    # Payback using conservative annual
    payback_months      = dev_cost / (annual_conservative / 12)

    result = {
        "experiment": "Experiment 2 — Personalized Recommendations",
        "decision": "SHIP",
        "direction": "POSITIVE — treatment better than control",
        "srm_flag": False,

        # Lift headline
        "cuped_lift_per_user_per_month_inr": lift_per_user_month,
        "cuped_lift_ci_lower_inr": 8.92,
        "cuped_lift_ci_upper_inr": 28.91,

        # Monthly revenue at 100% rollout
        "monthly_rev_gain_full_rollout_inr": monthly_rev_gain,
        "monthly_rev_gain_full_rollout_str": _lakh(monthly_rev_gain),

        # Annual revenue
        "annual_rev_conservative_inr": annual_conservative,
        "annual_rev_conservative_str": _crore(annual_conservative),
        "annual_rev_point_inr":        annual_point,
        "annual_rev_point_str":        _crore(annual_point),

        # Overstatement avoided (outlier-robustness caveat)
        "raw_lift_rejected_inr_per_user": 56.35,
        "overstatement_avoided_annual_inr": 22_50_000_00,   # ₹225 crore
        "overstatement_avoided_annual_str": _crore(22_50_000_00),
        "outlier_caveat": (
            "Raw lift of ₹56.35/user was rejected due to pre-experiment group "
            "composition bias (not outliers). CUPED removed ₹37.46 of bias. "
            "Outlier robustness confirmed: trimming at 1%, 5%, 10% all yielded "
            "stable raw lifts (₹53–₹56), so the issue was group imbalance, "
            "not extreme users. Leadership should use ₹53.52 Crore/year "
            "(conservative) — quoting ₹113 Crore risks credibility if actuals "
            "come in lower."
        ),

        # Payback
        "dev_cost_inr": dev_cost,
        "dev_cost_str": _lakh(dev_cost),
        "payback_months": round(payback_months, 4),
"payback_days":   round(payback_months * 30, 1),
"payback_str": (
    f"{round(payback_months * 30, 1)} days "
    f"({round(payback_months, 4)} months) — "
    f"dev cost recovered in under 2 days on conservative estimate"
),
        "assumptions": [
            "Daily visitors = 40,000 | after ship: 100% rollout (all users)",
            "CUPED lift = ₹18.90/user/month (Phase 4, bootstrap 95% CI [₹8.92, ₹28.91])",
            "Conservative annual = ₹53.52 Crore (Phase 4 bootstrap lower CI)",
            "Point estimate annual = ₹113.37 Crore (Phase 4 bootstrap mean)",
            "Raw lift ₹56.35/user REJECTED — pre-experiment bias confirmed by CUPED",
            "Outlier trimming 1–10%: lift stable ₹53–₹56 → root cause is group bias",
            "Dev cost = ₹20 lakh (ML infra + serving layer — confirm with engineering)",
            "Payback = dev cost ÷ monthly conservative revenue",
            "Analysis window: Days 7–14 only (novelty filter from Phase 2)",
        ],
    }
    return result


# ── Experiment 3 — Discount Banner Placement (SHIP both segments) ──────────
def summarise_exp3(data: dict) -> dict:
    b  = BUSINESS
    dv = b["daily_visitors"]
    dm = b["days_per_month"]
    my = b["months_per_year"]
    c2c = b["exp3_click_to_conversion"]
    aov = b["exp3_aov_inr"]

    mob_share  = b["exp3_mobile_share"]
    dsk_share  = b["exp3_desktop_share"]

    daily_mobile  = dv * mob_share
    daily_desktop = dv * dsk_share

    # Lift in CTR
    mob_ctr_lift = b["exp3_treatment_ctr_mobile"]  - b["exp3_baseline_ctr_mobile"]
    dsk_ctr_lift = b["exp3_treatment_ctr_desktop"] - b["exp3_baseline_ctr_desktop"]

    # Additional clicks/day
    add_clicks_mob = daily_mobile  * mob_ctr_lift
    add_clicks_dsk = daily_desktop * dsk_ctr_lift
    add_clicks_total = add_clicks_mob + add_clicks_dsk

    # Additional conversions/month (clicks × click-to-conversion rate)
    add_conv_mob_month = add_clicks_mob * c2c * dm
    add_conv_dsk_month = add_clicks_dsk * c2c * dm
    add_conv_total_month = add_conv_mob_month + add_conv_dsk_month

    # Additional revenue/month
    add_rev_mob_month   = add_conv_mob_month   * aov
    add_rev_dsk_month   = add_conv_dsk_month   * aov
    add_rev_total_month = add_conv_total_month * aov

    # Annual
    add_rev_annual = add_rev_total_month * my

    dev_cost       = b["exp3_dev_cost_inr"]
    payback_months = dev_cost / add_rev_total_month

    result = {
        "experiment": "Experiment 3 — Discount Banner Placement",
        "decision": "SHIP (mobile + desktop, device-level feature flags)",
        "direction": "POSITIVE — treatment better on both segments",
        "srm_flag": False,

        # CTR lifts
        "mobile_ctr_lift_pp": round(mob_ctr_lift * 100, 2),
        "desktop_ctr_lift_pp": round(dsk_ctr_lift * 100, 2),

        # Additional conversions per month
        "additional_conversions_mobile_month":  round(add_conv_mob_month),
        "additional_conversions_desktop_month": round(add_conv_dsk_month),
        "additional_conversions_total_month":   round(add_conv_total_month),

        # Revenue
        "additional_revenue_mobile_month_inr":  round(add_rev_mob_month),
        "additional_revenue_desktop_month_inr": round(add_rev_dsk_month),
        "additional_revenue_total_month_inr":   round(add_rev_total_month),
        "additional_revenue_total_month_str":   _lakh(add_rev_total_month),
        "additional_revenue_annual_inr":        round(add_rev_annual),
        "additional_revenue_annual_str":        _crore(add_rev_annual),

        # Multiple-testing risk note (from Phase 5)
        "multiple_testing_note": (
            "Two segments (mobile + desktop) were tested. Bonferroni correction "
            "applied: mobile Bonferroni p ≈ 0 (SHIP), desktop Bonferroni p = 0.0027 "
            "(SHIP). No decision changed post-correction → false positive cost = ₹0. "
            "Interaction term (treat×mobile OR=1.073, p=0.18) is NOT significant, "
            "so we cannot formally claim mobile benefits more than desktop. "
            "Device-level feature flags are recommended for monitoring, not for "
            "restricting rollout to mobile-only."
        ),

        # Segment-specific recommendation
        "segment_recommendation": {
            "mobile":  "SHIP. Effect size +0.81pp (+22.6%), Bonferroni p ≈ 0. "
                       "Largest absolute gain. Deploy first.",
            "desktop": "SHIP. Effect size +0.42pp (+14.7%), Bonferroni p = 0.0027. "
                       "Smaller but statistically robust after correction. "
                       "Do NOT restrict to mobile-only (interaction not significant).",
            "tablet":  "NO DATA — tablet segment absent from experiment. "
                       "Recommend running a follow-up or monitoring post-ship.",
        },

        # Payback
        "dev_cost_inr": dev_cost,
        "dev_cost_str": _lakh(dev_cost),
        "payback_months": round(payback_months, 1),
        "payback_str": f"{round(payback_months, 1)} months",

        "assumptions": [
            "Daily visitors = 40,000 | Mobile share = 64.9% | Desktop = 35.1%",
            "CTR lift mobile  = +0.81 pp (Phase 5 corrected)",
            "CTR lift desktop = +0.42 pp (Phase 5 corrected)",
            "Click-to-conversion rate = 15% (ASSUMPTION — validate with GA4)",
            "AOV = ₹1,200",
            "100% rollout assumed post-ship",
            "Dev cost = ₹3 lakh (CSS/layout change — confirm with engineering)",
            "No tablet data — flag for post-ship monitoring",
            "Bonferroni correction applied (confirmatory framing, Phase 5)",
        ],
    }
    return result


# ── Master summary ──────────────────────────────────────────────────────────
def build_all_summaries(loaded_data: dict) -> dict:
    print("[Phase 7] Building business summaries...")
    summaries = {
        "exp1": summarise_exp1(loaded_data["exp1"]),
        "exp2": summarise_exp2(loaded_data["exp2"]),
        "exp3": summarise_exp3(loaded_data["exp3"]),
    }
    print("  ✓ Experiment 1 summary complete")
    print("  ✓ Experiment 2 summary complete")
    print("  ✓ Experiment 3 summary complete")
    return summaries