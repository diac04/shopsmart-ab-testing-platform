# phase3/phase3_run.py
"""
Phase 3 Master Runner — Experiment 1: Checkout Redesign
ShopSmart India A/B Testing Platform

Run from project root:
    python -m phase3.phase3_run

Produces:
    phase3/phase3_results.json
"""

import json
import os
import sys
import datetime

# ── Make project root importable ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase3.config      import OUTPUT_JSON, ANALYSIS_VARIANTS
from phase3.data_loader import load_exp1, split_groups, get_conversions
from phase3.statistical_tests import run_all_tests
from phase3.business_impact   import compute_impact
from phase3.recommendation    import generate_recommendation

DIVIDER = "═" * 70


def run_variant(remove_bots: bool,
                late_period_only: bool,
                label: str) -> tuple[dict, dict]:
    """Load data for one analysis variant and run all tests."""
    print(f"\n{'─'*60}")
    print(f"  VARIANT: {label}")
    print(f"{'─'*60}")

    df = load_exp1(remove_bots=remove_bots,
                   late_period_only=late_period_only)
    ctrl, trt = split_groups(df)

    ctrl_n, ctrl_conv, ctrl_rate = get_conversions(ctrl)
    trt_n,  trt_conv,  trt_rate  = get_conversions(trt)

    print(f"  Control  : {ctrl_n:>8,} users | {ctrl_conv:>6,} conversions "
          f"| rate = {ctrl_rate:.4%}")
    print(f"  Treatment: {trt_n:>8,} users | {trt_conv:>6,} conversions "
          f"| rate = {trt_rate:.4%}")

    results = run_all_tests(ctrl_n, ctrl_conv, trt_n, trt_conv, label)
    impact  = compute_impact(ctrl_rate, trt_rate, label)

    return results, impact


def pretty_print_results(results: dict, impact: dict,
                          rec: dict, label: str):
    """Console-friendly summary for the primary variant."""
    print(f"\n{DIVIDER}")
    print(f"  RESULTS SUMMARY — {label}")
    print(DIVIDER)

    desc = results["descriptive"]
    print(f"\n{'DESCRIPTIVE STATS':─<50}")
    for grp in ("control", "treatment"):
        g = desc[grp]
        print(f"  {grp.capitalize():<12}: n={g['n']:>8,}  "
              f"conv={g['conversions']:>6,}  "
              f"rate={g['conversion_rate']:.4%}")

    z   = results["ztest"]
    chi = results["chi_square"]
    h   = results["cohens_h"]
    ci  = results["confidence_interval"]
    lft = results["lift"]

    print(f"\n{'STATISTICAL TESTS':─<50}")
    print(f"  Z-statistic : {z['z_statistic']:>10.4f}")
    print(f"  p-value     : {z['p_value']:>10.6f}   "
          f"({'SIGNIFICANT ✅' if z['significant'] else 'NOT SIGNIFICANT ❌'})")
    print(f"  Chi2 stat   : {chi['chi2_stat']:>10.4f}   "
          f"p = {chi['p_value']:.6f}")
    print(f"  Cohen's h   : {h['cohens_h']:>10.4f}   "
          f"({h['magnitude']}) | pre-reg = {h['prereg_cohens_h']}")
    print(f"  95% CI diff : [{ci['ci_lower']:+.6f}, {ci['ci_upper']:+.6f}]  "
          f"excludes zero: {ci['ci_excludes_zero']}")
    print(f"  Absolute lift: {lft['absolute_lift']:+.6f}")
    print(f"  Relative lift: {lft['relative_lift_pct']:+.3f}%  "
          f"(pre-reg MDE: {lft['prereg_mde_rel_pct']}%)")

    print(f"\n{'BUSINESS IMPACT (50% rollout)':─<50}")
    imp = impact["50pct_rollout"]
    print(f"  Daily visitors     : {impact['assumptions']['daily_checkout_visitors']:>10,}")
    print(f"  AOV assumed        : ₹{impact['assumptions']['aov_inr']:>9,}")
    print(f"  Add. daily conv.   : {imp['additional_daily_conversions']:>10.1f}")
    print(f"  Daily rev. impact  : ₹{imp['daily_revenue_impact_inr']:>12,.2f}")
    print(f"  Annual rev. impact : ₹{imp['annual_revenue_impact_inr']:>12,.2f}")
    print(f"  Annual rev. impact : ₹{imp['annual_revenue_impact_crore']:.3f} Crore")
    print(f"  ⚠️   {impact['risk_note']}")

    print(f"\n{'RECOMMENDATION':─<50}")
    print(f"  Decision    : {rec['decision']}")
    print(f"  Confidence  : {rec['confidence_level']}")
    print(f"\n{rec['pm_recommendation']}")

    print(f"\n{'PHASE 6 SAVE (power retrospective)':─<50}")
    p6 = rec["phase6_save"]
    print(f"  Observed Cohen's h : {p6['observed_cohens_h']}")
    print(f"  Observed lift abs  : {p6['observed_lift_abs']}")
    print(f"  Observed lift rel% : {p6['observed_lift_rel_pct']}%")
    print(f"  p-value            : {p6['p_value']}")
    print(f"  Significant        : {p6['significant']}")
    print(DIVIDER)


def main():
    print(f"\n{DIVIDER}")
    print("  PHASE 3 — Statistical Analysis: Experiment 1 (Checkout)")
    print("  ShopSmart India A/B Testing Platform")
    print(f"  Run at: {datetime.datetime.now().isoformat()}")
    print(DIVIDER)
    print("\n  ⚠️   SRM FLAG: Exp1 has a detected Sample Ratio Mismatch.")
    print("  Results are reported for transparency and Phase 6 retrospective.")
    print("  SHIP DECISION: Blocked until clean re-run.\n")

    all_results = {
        "metadata": {
            "phase"       : 3,
            "project"     : "ShopSmart India AB Platform",
            "experiment"  : "Exp1: Checkout Redesign",
            "generated_at": datetime.datetime.now().isoformat(),
            "srm_warning" : ("SRM detected in Phase 2. T/C ratio=1.0833. "
                             "Results are analytical only — not decision-grade."),
        },
        "variants": {},
        "primary_recommendation": {},
    }

    # ── Run all three variants ─────────────────────────────────────────────
    variant_configs = [
        # (remove_bots, late_period_only, label)
        (True,  False, ANALYSIS_VARIANTS["primary"]),    # headline
        (False, False, ANALYSIS_VARIANTS["secondary"]),  # sensitivity
        (True,  True,  ANALYSIS_VARIANTS["tertiary"]),   # novelty check
    ]

    primary_results = None
    primary_impact  = None

    for remove_bots, late_period, label in variant_configs:
        results, impact = run_variant(remove_bots, late_period, label)

        # Recommendation only for primary variant
        rec = generate_recommendation(
            primary_results=results,
            impact=impact,
            srm_detected=True,
        )

        all_results["variants"][label] = {
            "results"       : results,
            "impact"        : impact,
            "recommendation": rec,
        }

        if label == ANALYSIS_VARIANTS["primary"]:
            primary_results = results
            primary_impact  = impact
            primary_rec     = rec
            all_results["primary_recommendation"] = rec

    # ── Pretty print primary variant ───────────────────────────────────────
    pretty_print_results(primary_results, primary_impact,
                          primary_rec, ANALYSIS_VARIANTS["primary"])

    # ── Sensitivity comparison table ───────────────────────────────────────
    print(f"\n{'SENSITIVITY ANALYSIS — Lift across variants':─<70}")
    print(f"  {'Variant':<45} {'Lift Abs':>10} {'Lift Rel%':>10} {'p-value':>10}")
    print(f"  {'─'*45} {'─'*10} {'─'*10} {'─'*10}")
    for label, v in all_results["variants"].items():
        lft = v["results"]["lift"]
        z   = v["results"]["ztest"]
        print(f"  {label:<45} "
              f"{lft['absolute_lift']:>+10.5f} "
              f"{lft['relative_lift_pct']:>+10.3f}% "
              f"{z['p_value']:>10.6f}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Results saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()