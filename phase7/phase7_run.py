# phase7/phase7_run.py
"""
Phase 7 Master Runner — ShopSmart India A/B Testing Platform

Reads:   phase3/, phase4/, phase5/, phase6/ result JSON files
Writes:  phase7/experiment_outputs/
           business_summary.json
           business_summary.csv
           final_decision_table.csv
           final_decision_table.json
           retro_paragraphs.txt
           phase7_written_report.txt
"""

import json
import csv
import os
import sys

# ── Make sure the project root is on sys.path ───────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase7.config import OUTPUT_DIR, BUSINESS
from phase7.data_loader import load_all
from phase7.business_summary import build_all_summaries
from phase7.decision_table import build_decision_table, save_decision_table
from phase7.retro_paragraphs import build_retro_text, save_retro


# ── Helpers ─────────────────────────────────────────────────────────────────
def save_json(obj: dict, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {filename} → {path}")


def save_summary_csv(summaries: dict) -> None:
    """Flatten each experiment summary to one row of a CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "business_summary.csv")

    rows = []
    for exp_key, s in summaries.items():
        row = {"experiment_key": exp_key}
        for k, v in s.items():
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        rows.append(row)

    if rows:
        # Collect ALL keys across ALL rows, not just the first row
        all_keys = ["experiment_key"]
        seen = {"experiment_key"}
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=all_keys,
                extrasaction="ignore",  # safety: skip any key not in fieldnames
                restval=""              # fill missing cells with empty string
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"  ✓ business_summary.csv → {path}")


def build_written_report(summaries: dict, table_rows: list, retro_text: str) -> str:
    """
    Combine everything into one readable text report.
    This is what you would paste into a README or present to leadership.
    """
    s1, s2, s3 = summaries["exp1"], summaries["exp2"], summaries["exp3"]

    lines = [
        "=" * 72,
        "SHOPSMART INDIA — A/B TESTING PLATFORM",
        "PHASE 7: EXECUTIVE BUSINESS SUMMARY & DECISION FRAMEWORK",
        "=" * 72,
        "",
        "THREE EXPERIMENTS | TWO SHIPS | ONE RESTART",
        "",
        "─" * 72,
        "SECTION 1 — EXPERIMENT BUSINESS SUMMARIES",
        "─" * 72,
        "",

        # ── Exp 1 ──
        "EXPERIMENT 1: UPI CHECKOUT REDESIGN",
        "Decision: DO NOT SHIP — Restart experiment after SRM fix",
        "",
        f"  Direction        : {s1['direction']}",
        f"  Avoided loss     : {s1['avoided_annual_loss_50pct_str']}/year (50% rollout)",
        f"                     {s1['avoided_annual_loss_100pct_str']}/year (100% rollout)",
        f"  Protected revenue: {s1['protected_revenue_monthly_str']}/month "
        f"(SPRT stopped early, Phase 6)",
        f"  Protected revenue: {s1['protected_revenue_annual_str']}/year",
        "",
        "  Cost-of-Delay Narrative:",
        f"  {s1['cost_of_delay_narrative']}",
        "",
        "  Assumptions:",
        *[f"    • {a}" for a in s1["assumptions"]],
        "",

        # ── Exp 2 ──
        "EXPERIMENT 2: PERSONALIZED RECOMMENDATIONS",
        "Decision: SHIP — Full rollout (100%)",
        "",
        f"  CUPED lift       : ₹{s2['cuped_lift_per_user_per_month_inr']:.2f}/user/month",
        f"  95% CI           : [₹{s2['cuped_lift_ci_lower_inr']}, "
        f"₹{s2['cuped_lift_ci_upper_inr']}]",
        f"  Monthly revenue  : {s2['monthly_rev_gain_full_rollout_str']}",
        f"  Annual (consv.)  : {s2['annual_rev_conservative_str']}  ← QUOTE TO LEADERSHIP",
        f"  Annual (point)   : {s2['annual_rev_point_str']}  (upper bound)",
        f"  Overstatement    : {s2['overstatement_avoided_annual_str']}/year avoided "
        f"(raw lift rejected)",
        f"  Dev cost         : {s2['dev_cost_str']}",
        f"  Payback period   : {s2['payback_str']}",
        f"  Payback detail   : Dev cost ₹{s2['dev_cost_inr']:,} recovered in "
        f"{s2['payback_days']} days at conservative monthly gain of "
        f"₹{BUSINESS['exp2_annual_conservative_inr'] // 12:,}/month",
        "",
        "  Outlier Robustness Caveat:",
        f"  {s2['outlier_caveat']}",
        "",
        "  Assumptions:",
        *[f"    • {a}" for a in s2["assumptions"]],
        "",

        # ── Exp 3 ──
        "EXPERIMENT 3: DISCOUNT BANNER PLACEMENT",
        f"Decision: {s3['decision']}",
        "",
        f"  Mobile CTR lift  : +{s3['mobile_ctr_lift_pp']} pp",
        f"  Desktop CTR lift : +{s3['desktop_ctr_lift_pp']} pp",
        f"  Add. conv./month : {s3['additional_conversions_mobile_month']:,} mobile + "
        f"{s3['additional_conversions_desktop_month']:,} desktop = "
        f"{s3['additional_conversions_total_month']:,} total",
        f"  Add. rev./month  : {s3['additional_revenue_total_month_str']}",
        f"  Annual revenue   : {s3['additional_revenue_annual_str']}",
        f"  Dev cost         : {s3['dev_cost_str']}",
        f"  Payback period   : {s3['payback_str']}",
        "",
        "  Multiple-Testing Note:",
        f"  {s3['multiple_testing_note']}",
        "",
        "  Segment Recommendations:",
        f"  Mobile  → {s3['segment_recommendation']['mobile']}",
        f"  Desktop → {s3['segment_recommendation']['desktop']}",
        f"  Tablet  → {s3['segment_recommendation']['tablet']}",
        "",
        "  Assumptions:",
        *[f"    • {a}" for a in s3["assumptions"]],
        "",

        # ── Decision table ──
        "─" * 72,
        "SECTION 2 — FINAL DECISION FRAMEWORK TABLE",
        "─" * 72,
        "",
        "(See final_decision_table.csv / .json for machine-readable version)",
        "",
    ]

    for i, row in enumerate(table_rows):
        lines += [
            f"  [{i+1}] {row['Experiment']}",
            f"      Result         : {row['Result']}",
            f"      Recommendation : {row['Recommendation']}",
            f"      Stat Justif.   : {row['Statistical Justification'][:120]}...",
            f"      ₹ Impact       : {row['₹ Impact']}",
            f"      Key Caveat     : {row['Key Caveat']}",
            "",
        ]

    lines += [
        "─" * 72,
        "SECTION 3 — PER-EXPERIMENT RETROSPECTIVES & POST-MORTEM",
        "─" * 72,
        "",
        retro_text,
    ]

    return "\n".join(lines)


def save_written_report(report: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "phase7_written_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✓ phase7_written_report.txt → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  PHASE 7 — ShopSmart A/B Platform")
    print("  Business Synthesis & Decision Framework")
    print("=" * 60 + "\n")

    # 1. Load all upstream results
    loaded = load_all()

    # 2. Build business summaries
    summaries = build_all_summaries(loaded)

    # 3. Save summaries
    print("[Phase 7] Saving business summaries...")
    save_json(summaries, "business_summary.json")
    save_summary_csv(summaries)

    # 4. Build + save decision table
    table_rows = build_decision_table(summaries)
    save_decision_table(table_rows)

    # 5. Build + save retro paragraphs
    print("[Phase 7] Building retrospective narratives...")
    retro_text = build_retro_text()
    save_retro(retro_text)

    # 6. Build + save written report
    print("[Phase 7] Building written report...")
    report = build_written_report(summaries, table_rows, retro_text)
    save_written_report(report)

    # ── Console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 7 COMPLETE")
    print("=" * 60)
    print()
    print("  THREE EXPERIMENTS | TWO SHIPS | ONE RESTART")
    print()

    s1, s2, s3 = summaries["exp1"], summaries["exp2"], summaries["exp3"]

    print("  EXP 1 — UPI Checkout         : DO NOT SHIP")
    print(f"    Avoided loss               : "
          f"{s1['avoided_annual_loss_50pct_str']}/year (50% rollout)")
    print(f"    Protected revenue          : "
          f"{s1['protected_revenue_annual_str']}/year")
    print()
    print("  EXP 2 — Personalized Recs    : SHIP ✓")
    print(f"    Conservative annual gain   : {s2['annual_rev_conservative_str']}")
    print(f"    Payback period             : {s2['payback_str']}")
    print(f"    Dev cost recovered in      : {s2['payback_days']} days")
    print(f"    Monthly conservative gain  : "
          f"₹{BUSINESS['exp2_annual_conservative_inr'] // 12:,}/month")
    print()
    print("  EXP 3 — Discount Banner      : SHIP ✓ (mobile + desktop)")
    print(f"    Annual revenue gain        : {s3['additional_revenue_annual_str']}")
    print(f"    Payback period             : {s3['payback_str']}")
    print()
    print("  OUTPUT FILES:")
    print(f"    phase7/experiment_outputs/business_summary.json")
    print(f"    phase7/experiment_outputs/business_summary.csv")
    print(f"    phase7/experiment_outputs/final_decision_table.csv")
    print(f"    phase7/experiment_outputs/final_decision_table.json")
    print(f"    phase7/experiment_outputs/retro_paragraphs.txt")
    print(f"    phase7/experiment_outputs/phase7_written_report.txt")
    print()


if __name__ == "__main__":
    main()