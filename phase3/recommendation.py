# phase3/recommendation.py
"""
Generates a structured ship/no-ship recommendation for the PM.
Decision logic accounts for SRM contamination, significance,
effect size, and novelty effect.
"""

from phase3.config import PRE_REGISTERED, ALPHA


def generate_recommendation(primary_results: dict,
                             impact: dict,
                             srm_detected: bool = True) -> dict:
    """
    Decision framework
    ──────────────────
    1. SRM → automatic NO-SHIP until clean re-run
    2. p >= alpha → NO-SHIP (insufficient evidence)
    3. Lift < 0  → NO-SHIP (harmful)
    4. Lift > 0 and significant and h >= prereg h → SHIP
    5. Lift > 0 and significant but h < prereg h → SHIP WITH CAUTION
    """

    z       = primary_results["ztest"]
    lift    = primary_results["lift"]
    ci      = primary_results["confidence_interval"]
    h_obs   = primary_results["cohens_h"]["cohens_h"]
    h_pre   = PRE_REGISTERED["cohens_h_prereg"]
    impact_crore = impact["50pct_rollout"]["annual_revenue_impact_crore"]

    significant = z["significant"]
    lift_abs    = lift["absolute_lift"]
    p_value     = z["p_value"]

    # ── Decision logic ─────────────────────────────────────────────────────
    if srm_detected:
        decision = "NO-SHIP — RESTART EXPERIMENT"
        confidence = "NONE (SRM contamination)"
        reason = (
            "A Sample Ratio Mismatch (SRM) was detected in Exp 1 "
            f"(T/C ratio = 1.0833, chi2 = 447.13, p ≈ 0). "
            "This means the randomization was compromised — treatment and "
            "control groups are not comparable. Any observed lift "
            "cannot be attributed to the checkout redesign alone. "
            "The experiment MUST be restarted with a fixed randomization "
            "pipeline before any shipping decision can be made."
        )
        what_changes = (
            "A clean re-run (no SRM, SRM chi2 p > 0.01) showing "
            f"p < {ALPHA} and absolute lift ≥ "
            f"{PRE_REGISTERED['mde_absolute']:.3f} "
            f"({PRE_REGISTERED['mde_relative_pct']}% relative) "
            "would change this recommendation to SHIP."
        )

    elif lift_abs < 0:
        decision = "NO-SHIP — TREATMENT IS HARMFUL"
        confidence = "HIGH (statistically significant harm)" \
                     if significant else "MODERATE"
        reason = (
            f"The treatment group shows a lower conversion rate than control "
            f"(absolute lift = {lift_abs:+.4f}). Shipping would destroy value."
        )
        what_changes = (
            "A redesigned treatment addressing the UX failure points, "
            "followed by a clean re-run showing positive lift, would "
            "change this recommendation."
        )

    elif not significant:
        decision = "NO-SHIP — INSUFFICIENT EVIDENCE"
        confidence = "LOW"
        reason = (
            f"p-value = {p_value:.4f} ≥ α = {ALPHA}. "
            "We cannot reject the null hypothesis. The observed lift "
            "is within the range of random chance."
        )
        what_changes = (
            f"Statistical significance (p < {ALPHA}) in a clean re-run "
            "would change this recommendation."
        )

    elif h_obs >= h_pre:
        decision = "SHIP ✅"
        confidence = "HIGH"
        reason = (
            f"Statistically significant (p = {p_value:.4f} < {ALPHA}). "
            f"Observed Cohen's h = {h_obs:.4f} meets or exceeds "
            f"pre-registered MDE (h = {h_pre:.4f}). "
            f"Estimated annual uplift ₹{impact_crore:.2f} Cr (50% rollout). "
            "NOTE: SRM flag means this is directionally informative only."
        )
        what_changes = (
            "A clean re-run confirming this effect size would make this "
            "a high-confidence ship."
        )

    else:  # significant but below pre-registered MDE
        decision = "SHIP WITH CAUTION ⚠️"
        confidence = "MODERATE"
        reason = (
            f"Statistically significant (p = {p_value:.4f}) but observed "
            f"effect size (h = {h_obs:.4f}) is below pre-registered MDE "
            f"(h = {h_pre:.4f}). Lift exists but may be smaller than "
            f"the business-meaningful threshold."
        )
        what_changes = (
            "A clean re-run with effect size ≥ pre-registered MDE "
            "would change this to a full SHIP."
        )

    # ── PM-facing paragraph ────────────────────────────────────────────────
    pm_paragraph = _pm_paragraph(decision, reason, what_changes,
                                 lift, ci, impact, srm_detected)

    return {
        "decision"              : decision,
        "confidence_level"      : confidence,
        "srm_detected"          : srm_detected,
        "reasoning"             : reason,
        "what_would_change_rec" : what_changes,
        "pm_recommendation"     : pm_paragraph,
        "phase6_save": {
            "observed_cohens_h"    : h_obs,
            "observed_lift_abs"    : round(lift_abs, 6),
            "observed_lift_rel_pct": lift["relative_lift_pct"],
            "prereg_cohens_h"      : h_pre,
            "prereg_mde_abs"       : PRE_REGISTERED["mde_absolute"],
            "significant"          : significant,
            "p_value"              : p_value,
            "note": ("Phase 6 power retrospective will compare these "
                     "observed values to pre-registered values.")
        },
    }


def _pm_paragraph(decision, reason, what_changes,
                  lift, ci, impact, srm_detected) -> str:
    impact_crore = impact["50pct_rollout"]["annual_revenue_impact_crore"]
    lift_rel     = lift["relative_lift_pct"]
    ci_lo        = ci["ci_lower"]
    ci_hi        = ci["ci_upper"]
    trt_rate     = lift["treatment_rate"]
    ctrl_rate    = lift["control_rate"]

    srm_note = (
        "\n\n⚠️  CRITICAL: A Sample Ratio Mismatch was detected, meaning "
        "the experiment's randomization was broken. The numbers below are "
        "reported for completeness but CANNOT be used as the basis for a "
        "shipping decision. The experiment must be restarted."
    ) if srm_detected else ""

    return (
        f"RECOMMENDATION: {decision}{srm_note}\n\n"
        f"The checkout redesign experiment ran for 22 days on ShopSmart "
        f"India's checkout funnel (40,000 daily visitors). "
        f"The treatment group showed a conversion rate of "
        f"{trt_rate:.2%} vs. {ctrl_rate:.2%} in control — a relative "
        f"lift of {lift_rel:+.2f}% (95% CI: {ci_lo:+.4f} to {ci_hi:+.4f}). "
        f"At our assumed AOV of ₹1,200 and 50% traffic allocation, "
        f"this translates to a projected annual revenue impact of "
        f"₹{impact_crore:.2f} Crore.\n\n"
        f"{reason}\n\n"
        f"What would change this recommendation: {what_changes}"
    )