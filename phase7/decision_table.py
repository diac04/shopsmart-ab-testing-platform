# phase7/decision_table.py
"""
Builds the final decision framework table — the centrepiece of the
Phase 7 report.

Columns:
  Experiment | Result | Recommendation | Statistical Justification |
  ₹ Impact   | Key Caveat
"""

import csv
import json
import os
from phase7.config import OUTPUT_DIR


TABLE_ROWS = [
    {
        "Experiment":   "Exp 1 — UPI Checkout Redesign",
        "Result":       "NOT SIGNIFICANT (p=0.188) | Direction: NEGATIVE",
        "Recommendation": "DO NOT SHIP — Restart with SRM fix",
        "Statistical Justification": (
            "Frequentist: p=0.188, 95% CI [-0.40pp, +0.08pp] crosses zero. "
            "Bayesian: P(treatment>control)=9.4%, E[loss|ship]=0.1670pp (30x E[loss|hold]). "
            "SPRT: stopped at n=600, accepted H0. "
            "Novelty: steady-state lift still -0.135pp (treatment worse at end). "
            "Power retro: observed effect = 40.5% of pre-registered MDE; "
            "would need n=631,448/group to detect observed effect (16.8x planned)."
        ),
        "₹ Impact": (
            "Avoided loss: ₹1.42 Crore/yr (50% rollout) or ₹2.83 Crore/yr (100%). "
            "Protected revenue: ₹17 lakh/month (SPRT stopped early). "
            "Cost-of-delay: ₹2.52 lakh/month (counterfactual MDE scenario — "
            "moot because true effect is negative)."
        ),
        "Key Caveat": (
            "SRM present (T/C ratio=1.083, chi2=447, p≈0). "
            "ALL figures HIGH UNCERTAINTY. "
            "Do not re-run until traffic-split mechanism is audited and fixed."
        ),
    },
    {
        "Experiment":   "Exp 2 — Personalized Recommendations",
        "Result":       "SIGNIFICANT | CUPED lift = ₹18.90/user/month (p<0.001)",
        "Recommendation": "SHIP — Full rollout (100%)",
        "Statistical Justification": (
            "Bootstrap CI on CUPED-adjusted mean: [₹8.92, ₹28.91]. "
            "p<0.001. Cohen's d=0.065 (negligible but economically meaningful). "
            "Raw lift ₹56.35 REJECTED — pre-experiment group imbalance confirmed "
            "(control ₹54.59 vs treatment ₹59.10, p=0.000). "
            "CUPED removed ₹37.46 bias (rho=0.847, 71.7% variance reduction). "
            "Outlier robustness: lift stable ₹53–₹56 across 1–10% trimming → "
            "root cause is group composition, not extreme users. "
            "Analysis window: Days 7–14 only (novelty filter)."
        ),
        "₹ Impact": (
            "Conservative annual: ₹53.52 Crore (bootstrap lower CI). "
            "Point estimate annual: ₹113.37 Crore. "
            "Overstatement avoided vs raw lift: ₹225 Crore/yr. "
            "Dev cost: ₹20 lakh. Payback: <0.5 months (conservative)."
        ),
        "Key Caveat": (
            "Use ₹53.52 Crore figure with leadership, NOT ₹113 Crore. "
            "Outlier handling was decided post-hoc — pre-register protocol next time. "
            "CUPED requires pre-experiment covariate; ensure data pipeline captures "
            "this before future experiments."
        ),
    },
    {
        "Experiment":   "Exp 3 — Discount Banner Placement",
        "Result":       (
            "SIGNIFICANT overall (Z=8.02, p≈0, lift=+0.68pp, +20.4%). "
            "Mobile: +0.81pp Bonferroni p≈0. Desktop: +0.42pp Bonferroni p=0.0027."
        ),
        "Recommendation": (
            "SHIP (mobile + desktop). Device-level feature flags. "
            "Do NOT restrict to mobile-only."
        ),
        "Statistical Justification": (
            "Overall Z=8.02, p≈0. "
            "Bonferroni correction (confirmatory): mobile p≈0 (SHIP), "
            "desktop p=0.0027 (SHIP). No decision changed after correction. "
            "Interaction model: treat×mobile OR=1.073, p=0.18 → NOT significant. "
            "Cannot formally claim mobile benefits more than desktop. "
            "False positive cost this experiment: ₹0."
        ),
        "₹ Impact": (
            "Additional conversions: ~{mob}+{dsk}={tot}/month "
            "(mobile+desktop, 15% CTR→conversion, AOV ₹1,200). "
            "Additional revenue: see business_summary.json. "
            "Dev cost: ₹3 lakh. Payback: < 1 month."
        ).format(
            # These will be filled by the runner from business_summary
            mob="see JSON", dsk="see JSON", tot="see JSON"
        ),
        "Key Caveat": (
            "Click-to-conversion rate (15%) is an ASSUMPTION — validate with GA4. "
            "No tablet data; monitor tablet segment post-ship. "
            "Interaction non-significant (p=0.18): mobile-first deploy is fine "
            "operationally but do not advertise it as a mobile-specific win."
        ),
    },
]


def build_decision_table(summaries: dict) -> list:
    """
    Merge computed ₹ figures from business_summary into the table rows.
    Returns list of dicts (one per experiment).
    """
    print("[Phase 7] Building final decision table...")

    # Patch Exp 3 ₹ impact with real numbers from summaries
    s3 = summaries["exp3"]
    TABLE_ROWS[2]["₹ Impact"] = (
        f"Additional conversions: "
        f"{s3['additional_conversions_mobile_month']:,} mobile + "
        f"{s3['additional_conversions_desktop_month']:,} desktop = "
        f"{s3['additional_conversions_total_month']:,}/month. "
        f"Additional revenue: {s3['additional_revenue_total_month_str']}/month, "
        f"{s3['additional_revenue_annual_str']}/year. "
        f"Dev cost: {s3['dev_cost_str']}. "
        f"Payback: {s3['payback_str']}."
    )

    return TABLE_ROWS


def save_decision_table(rows: list) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "final_decision_table.csv")
    fieldnames = ["Experiment", "Result", "Recommendation",
                  "Statistical Justification", "₹ Impact", "Key Caveat"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # JSON
    json_path = os.path.join(OUTPUT_DIR, "final_decision_table.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"  ✓ final_decision_table.csv  → {csv_path}")
    print(f"  ✓ final_decision_table.json → {json_path}")