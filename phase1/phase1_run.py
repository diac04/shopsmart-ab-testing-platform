import numpy as np
import pandas as pd
import json
from datetime import datetime
from sample_size_calculator import (
    calculate_sample_size_proportion,
    calculate_sample_size_means,
    calculate_sample_size_segmented,
    calculate_experiment_duration
)

# ─────────────────────────────────────
# TRAFFIC ASSUMPTIONS
# ─────────────────────────────────────
# ShopSmart India modeled as growth-stage
# e-commerce with ~5M monthly active users
# (~200K daily active users)
#
# Checkout page  : 20% of DAU reach checkout = 40,000/day
# Homepage       : 80% of DAU see homepage   = 160,000/day
# Banner pages   : 60% of DAU see banner     = 120,000/day
# Mobile share   : 65% (IAMAI 2023 benchmark)
# Desktop share  : 35%
# ─────────────────────────────────────

DAILY_CHECKOUT  = 40_000
DAILY_HOMEPAGE  = 160_000
DAILY_BANNER    = 120_000
ALPHA           = 0.05
POWER           = 0.80

print("=" * 60)
print("ShopSmart India — Phase 1: Pre-Analysis Design")
print("=" * 60)


# ─────────────────────────────────────
# EXPERIMENT 1 — CHECKOUT REDESIGN
# Baseline conversion: 3.8%
# MDE: 0.4 pp absolute (3.8% -> 4.2%)
# Relative lift: ~10.5%
# ─────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 1 — CHECKOUT PAGE REDESIGN")
print("=" * 60)

exp1_power = calculate_sample_size_proportion(
    baseline_rate=0.038,
    mde=0.004,
    alpha=ALPHA,
    power=POWER,
    two_tailed=True,
    verbose=True
)

exp1_duration = calculate_experiment_duration(
    n_per_group=exp1_power['n_per_group'],
    daily_traffic=DAILY_CHECKOUT,
    experiment_name="Checkout Page Redesign"
)


# ─────────────────────────────────────
# EXPERIMENT 2 — PERSONALIZED RECS
# Baseline mean: Rs.850, std: Rs.420
# MDE: Rs.70 absolute (Rs.850 -> Rs.920)
# Relative lift: ~8.2%
# Note: right-skewed, zero-inflated
# ─────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 2 — PERSONALIZED RECOMMENDATIONS")
print("=" * 60)

exp2_power = calculate_sample_size_means(
    baseline_mean=850,
    baseline_std=420,
    mde_absolute=70,
    alpha=ALPHA,
    power=POWER,
    two_tailed=True,
    verbose=True
)

exp2_duration = calculate_experiment_duration(
    n_per_group=exp2_power['n_per_group'],
    daily_traffic=DAILY_HOMEPAGE,
    experiment_name="Personalized Recommendations"
)


# ─────────────────────────────────────
# EXPERIMENT 3 — BANNER PLACEMENT
# Overall CTR: 3.2% -> 3.8% (+0.6 pp)
# Mobile CTR : 3.5% -> 4.3% (+0.8 pp)
# Desktop CTR: 2.8% -> 3.2% (+0.4 pp)
# ─────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 3 — DISCOUNT BANNER PLACEMENT")
print("=" * 60)

print("\n--- Step A: Overall CTR ---")
exp3_overall = calculate_sample_size_proportion(
    baseline_rate=0.032,
    mde=0.006,
    alpha=ALPHA,
    power=POWER,
    two_tailed=True,
    verbose=True
)

print("\n--- Step B: Segmented (Mobile vs Desktop) ---")
exp3_segmented = calculate_sample_size_segmented(
    segments={
        'mobile': {
            'baseline_rate': 0.035,
            'mde': 0.008,
            'traffic_fraction': 0.65
        },
        'desktop': {
            'baseline_rate': 0.028,
            'mde': 0.004,
            'traffic_fraction': 0.35
        }
    },
    alpha=ALPHA,
    power=POWER,
    two_tailed=True,
    apply_bonferroni=True,
    verbose=True
)

exp3_n_final = max(
    exp3_overall['n_per_group'],
    exp3_segmented['n_per_group']
)

print(f"\n  Binding n per group (max of overall vs segmented): {exp3_n_final:,}")

exp3_duration = calculate_experiment_duration(
    n_per_group=exp3_n_final,
    daily_traffic=DAILY_BANNER,
    experiment_name="Discount Banner Placement"
)


# ─────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 1 SUMMARY TABLE")
print("=" * 60)

rows = [
    {
        'Experiment'      : 'Exp1: Checkout Redesign',
        'Metric'          : 'Conversion Rate',
        'Baseline'        : '3.80%',
        'MDE'             : '+0.40pp (+10.5%)',
        'n_per_group'     : exp1_power['n_per_group'],
        'total_n'         : exp1_power['total_n'],
        'Cohen'           : f"h={exp1_power['cohens_h']}",
        'Days_required'   : exp1_duration['days_needed'],
        'Min_run_days'    : 14,
        'Daily_traffic'   : DAILY_CHECKOUT,
    },
    {
        'Experiment'      : 'Exp2: Recs Revenue',
        'Metric'          : 'Revenue per User',
        'Baseline'        : 'Rs.850 (sd=420)',
        'MDE'             : '+Rs.70 (+8.24%)',
        'n_per_group'     : exp2_power['n_per_group'],
        'total_n'         : exp2_power['total_n'],
        'Cohen'           : f"d={exp2_power['cohens_d']}",
        'Days_required'   : exp2_duration['days_needed'],
        'Min_run_days'    : 14,
        'Daily_traffic'   : DAILY_HOMEPAGE,
    },
    {
        'Experiment'      : 'Exp3: Banner CTR',
        'Metric'          : 'CTR (Overall)',
        'Baseline'        : '3.20%',
        'MDE'             : '+0.60pp (+18.75%)',
        'n_per_group'     : exp3_overall['n_per_group'],
        'total_n'         : exp3_overall['total_n'],
        'Cohen'           : f"h={exp3_overall['cohens_h']}",
        'Days_required'   : int(np.ceil(exp3_overall['n_per_group'] / (DAILY_BANNER * 0.5))),
        'Min_run_days'    : 14,
        'Daily_traffic'   : DAILY_BANNER,
    },
    {
        'Experiment'      : 'Exp3: Banner Segmented',
        'Metric'          : 'CTR (Segmented)',
        'Baseline'        : 'M:3.5% D:2.8%',
        'MDE'             : 'M:+0.8pp D:+0.4pp',
        'n_per_group'     : exp3_n_final,
        'total_n'         : exp3_n_final * 2,
        'Cohen'           : 'Bonferroni a=0.025',
        'Days_required'   : exp3_duration['days_needed'],
        'Min_run_days'    : 14,
        'Daily_traffic'   : DAILY_BANNER,
    },
]

df = pd.DataFrame(rows)
print(df.to_string(index=False))


# ─────────────────────────────────────
# EFFECT SIZE SUMMARY
# ─────────────────────────────────────

print("\n" + "=" * 60)
print("EFFECT SIZE SUMMARY")
print("=" * 60)
print("Cohen benchmarks: Small=0.20  Medium=0.50  Large=0.80")
print()

for label, value, metric_type in [
    ("Exp1 Checkout ", exp1_power['cohens_h'],  "Cohen's h (proportion)"),
    ("Exp2 Revenue  ", exp2_power['cohens_d'],  "Cohen's d (means)    "),
    ("Exp3 Banner   ", exp3_overall['cohens_h'], "Cohen's h (proportion)"),
]:
    size_label = (
        "Small"  if value < 0.20 else
        "Medium" if value < 0.50 else
        "Large"
    )
    print(f"  {label}: {metric_type} = {value:.4f}  --> {size_label} effect")

print()
print("NOTE: Small effect sizes are NORMAL in e-commerce A/B testing.")
print("A 0.4pp conversion lift is tiny statistically but huge at scale.")


# ─────────────────────────────────────
# HYPOTHESES SUMMARY
# ─────────────────────────────────────

print("\n" + "=" * 60)
print("HYPOTHESES SUMMARY")
print("=" * 60)

hypotheses = {
    "Experiment 1 — Checkout Redesign": {
        "H0": "p_treatment = p_control (no effect on conversion rate)",
        "H1": "p_treatment != p_control (two-tailed)",
        "Why two-tailed": "Redesigns can backfire and hurt conversions",
        "Guardrails": [
            "Average Order Value — must not drop",
            "Page Load Time — must not increase >100ms",
            "Cart Abandonment Rate — must not rise",
            "Support Tickets (checkout) — must not spike"
        ]
    },
    "Experiment 2 — Personalized Recs": {
        "H0": "mu_treatment = mu_control (no effect on revenue per user)",
        "H1": "mu_treatment != mu_control (two-tailed)",
        "Why two-tailed": "Poor personalization model could REDUCE engagement",
        "Guardrails": [
            "Session Duration / Bounce Rate — must not worsen",
            "Rec Click-Through Rate — must stay healthy",
            "Return Rate / Refund Rate — must not rise",
            "Homepage Load Time — must not increase >200ms"
        ]
    },
    "Experiment 3 — Banner Placement": {
        "H0": "CTR_treatment = CTR_control (overall and per segment)",
        "H1": "CTR_treatment != CTR_control (two-tailed, Bonferroni for segments)",
        "Why two-tailed": "Banner position change could reduce visibility",
        "Guardrails": [
            "Conversion Rate (post-click) — must not drop",
            "Revenue per User (overall) — must not fall",
            "Banner Fatigue (CTR decay over time) — monitor weekly",
            "Profit Margin on banner purchases — must not erode"
        ]
    }
}

for exp_name, details in hypotheses.items():
    print(f"\n  {exp_name}")
    print(f"    H0         : {details['H0']}")
    print(f"    H1         : {details['H1']}")
    print(f"    Rationale  : {details['Why two-tailed']}")
    print(f"    Guardrails :")
    for g in details['Guardrails']:
        print(f"      - {g}")


# ─────────────────────────────────────
# SAVE TO JSON FOR PHASE 6
# ─────────────────────────────────────

phase1_save = {
    'metadata': {
        'phase': 1,
        'project': 'ShopSmart India AB Platform',
        'generated_at': datetime.now().isoformat(),
        'purpose': (
            'Pre-registered power analysis. '
            'Phase 6 retrospective will compare these pre-registered '
            'MDE and power values against actually observed effects.'
        )
    },
    'traffic_assumptions': {
        'daily_checkout_visitors': DAILY_CHECKOUT,
        'daily_homepage_visitors': DAILY_HOMEPAGE,
        'daily_banner_visitors'  : DAILY_BANNER,
        'mobile_share'           : 0.65,
        'desktop_share'          : 0.35,
        'source': 'IAMAI 2023 benchmarks, Baymard Institute e-commerce data'
    },
    'experiment_1_checkout': {
        'n_per_group'     : exp1_power['n_per_group'],
        'total_n'         : exp1_power['total_n'],
        'baseline_rate'   : exp1_power['baseline_rate'],
        'treatment_rate'  : exp1_power['treatment_rate'],
        'mde_absolute'    : exp1_power['mde_absolute'],
        'mde_relative_pct': exp1_power['mde_relative_pct'],
        'alpha'           : exp1_power['alpha'],
        'power'           : exp1_power['power'],
        'cohens_h'        : exp1_power['cohens_h'],
        'days_required'   : exp1_duration['days_needed'],
        'min_run_days'    : 14
    },
    'experiment_2_recommendations': {
        'n_per_group'     : exp2_power['n_per_group'],
        'total_n'         : exp2_power['total_n'],
        'baseline_mean'   : exp2_power['baseline_mean'],
        'treatment_mean'  : exp2_power['treatment_mean'],
        'baseline_std'    : exp2_power['baseline_std'],
        'mde_absolute'    : exp2_power['mde_absolute'],
        'mde_relative_pct': exp2_power['mde_relative_pct'],
        'alpha'           : exp2_power['alpha'],
        'power'           : exp2_power['power'],
        'cohens_d'        : exp2_power['cohens_d'],
        'days_required'   : exp2_duration['days_needed'],
        'min_run_days'    : 14,
        'skew_warning'    : exp2_power['skew_warning']
    },
    'experiment_3_banner': {
        'overall': {
            'n_per_group'     : exp3_overall['n_per_group'],
            'total_n'         : exp3_overall['total_n'],
            'baseline_rate'   : exp3_overall['baseline_rate'],
            'treatment_rate'  : exp3_overall['treatment_rate'],
            'mde_absolute'    : exp3_overall['mde_absolute'],
            'mde_relative_pct': exp3_overall['mde_relative_pct'],
            'cohens_h'        : exp3_overall['cohens_h'],
        },
        'segmented': {
            'n_per_group_binding'  : exp3_n_final,
            'total_n_binding'      : exp3_n_final * 2,
            'alpha_corrected'      : exp3_segmented['alpha_corrected'],
            'bonferroni_applied'   : exp3_segmented['bonferroni_applied'],
            'correction_note'      : exp3_segmented['correction_note'],
            'mobile_baseline'      : 0.035,
            'mobile_mde'           : 0.008,
            'desktop_baseline'     : 0.028,
            'desktop_mde'          : 0.004,
        },
        'days_required'  : exp3_duration['days_needed'],
        'min_run_days'   : 14,
    }
}


with open('phase1_power_analysis.json', 'w') as f:
    json.dump(phase1_save, f, indent=2)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print("  phase1_power_analysis.json  --> Load this in Phase 6")
print()
print("Phase 1 Complete.")
print("Copy the SUMMARY TABLE above and paste it into Phase 2.")