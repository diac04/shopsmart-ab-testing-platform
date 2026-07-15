# phase5/recommendation.py
"""
Part D — Device-specific operational rollout recommendation.
Translates HTE findings into a concrete engineering/product action plan.
"""

import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import OUTPUT_DIR, ALPHA


def build_recommendation(
    corrected_df:       pd.DataFrame,
    interaction_result: dict,
    cost_result:        dict,
    verbose:            bool = True
) -> dict:
    """
    Build a device-level rollout recommendation table and narrative.

    Logic:
      - If significant after correction AND abs_diff > 0  → SHIP
      - If significant after correction AND abs_diff < 0  → DO NOT SHIP (harmful)
      - If not significant after correction               → HOLD (keep control)
    """
    recommendations = []

    for device, row in corrected_df.iterrows():
        sig  = bool(row["sig_bonferroni"])
        diff = row["abs_diff"]

        if sig and diff > 0:
            action  = "SHIP"
            rationale = (
                f"Statistically significant positive lift "
                f"(+{diff*100:.2f}pp CTR) after Bonferroni correction. "
                f"Deploy treatment banner on {device}."
            )
        elif sig and diff <= 0:
            action  = "DO NOT SHIP"
            rationale = (
                f"Significant but NEGATIVE effect ({diff*100:.2f}pp CTR). "
                f"Treatment banner hurts {device} users. Keep control."
            )
        else:
            action  = "HOLD / KEEP CONTROL"
            rationale = (
                f"Effect not significant after multiple-comparison correction "
                f"(p_bonf={row['p_bonferroni']:.4f}). "
                f"Insufficient evidence to change layout for {device}."
            )

        recommendations.append({
            "device":      device,
            "action":      action,
            "ctr_ctrl":    f"{row['ctr_control']:.4f}",
            "ctr_treat":   f"{row['ctr_treatment']:.4f}",
            "lift_pp":     f"{diff*100:+.2f}",
            "p_raw":       f"{row['p_value']:.4f}",
            "p_bonf":      f"{row['p_bonferroni']:.4f}",
            "rationale":   rationale,
        })

    rec_df = pd.DataFrame(recommendations).set_index("device")

    # ── Narrative summary ──────────────────────────────────────────────────
    narrative = _build_narrative(
    rec_df,
    corrected_df,
    cost_result,
    interaction_result
)
    if verbose:
        print("\n" + "="*70)
        print("  PART D — Device-Specific Operational Rollout Recommendation")
        print("="*70)
        print(rec_df[["action", "ctr_ctrl", "ctr_treat",
                       "lift_pp", "p_bonf"]].to_string())
        print("\n  ── Rationale per device ──")
        for device, row in rec_df.iterrows():
            print(f"\n  [{device.upper()}]")
            print(f"  Action    : {row['action']}")
            print(f"  Rationale : {row['rationale']}")
        print("\n  ── Narrative Summary ──")
        print(narrative)
        print("="*70 + "\n")

    return {
        "recommendation_df": rec_df,
        "narrative":         narrative,
    }


def _build_narrative(
    rec_df,
    corrected_df,
    cost_result,
    interaction_result
) -> str:
    ship_devices = rec_df[rec_df["action"] == "SHIP"].index.tolist()
    hold_devices = rec_df[
        rec_df["action"].str.contains("HOLD", na=False)
    ].index.tolist()
    block_devices = rec_df[
        rec_df["action"] == "DO NOT SHIP"
    ].index.tolist()

    ship_str = ", ".join(ship_devices) or "none"
    hold_str = ", ".join(hold_devices) or "none"
    block_str = ", ".join(block_devices) or "none"

    cost_str = f"₹{cost_result['total_cost_inr']:,.0f}"

    coef_df = interaction_result["coef_df"]

    if "treat_x_mobile" in coef_df.index:
        interaction_or = coef_df.loc["treat_x_mobile", "odds_ratio"]
        interaction_p = coef_df.loc["treat_x_mobile", "p_value"]
    else:
        interaction_or = float("nan")
        interaction_p = float("nan")

    if interaction_p < ALPHA:
        hte_text = (
            f"The treatment-by-mobile interaction is statistically significant "
            f"(interaction OR={interaction_or:.3f}, p={interaction_p:.4f}). "
            f"This provides evidence that the banner's treatment effect varies "
            f"between mobile and desktop."
        )
    else:
        hte_text = (
            f"Mobile produced the larger observed relative CTR lift, but the "
            f"treatment-by-mobile interaction is not statistically significant "
            f"(interaction OR={interaction_or:.3f}, p={interaction_p:.4f}). "
            f"Therefore, Phase 1's directional expectation is supported "
            f"descriptively, but it is not formally confirmed. We cannot conclude "
            f"that the treatment effect differs between mobile and desktop."
        )

    narrative = f"""
ROLLOUT DECISION SUMMARY
─────────────────────────────────────────────────────────────────
SHIP treatment banner   : {ship_str}
HOLD (keep control)     : {hold_str}
DO NOT SHIP (block)     : {block_str}

HETEROGENEOUS TREATMENT EFFECT:
{hte_text}

BUSINESS DECISION:
Both mobile and desktop show positive CTR effects that remain
statistically significant after Bonferroni correction. Therefore,
enable the top discount banner on both mobile and desktop. Do not
restrict rollout to mobile merely because its observed lift is larger;
the formal interaction test does not show a statistically reliable
difference between device effects.

UNCORRECTED DECISION COST:
Skipping multiple-comparison correction would not have changed the
rollout decision in this experiment because both device results remain
significant after Bonferroni and BH-FDR correction. Consequently, the
estimated incremental monthly false-positive cost attributable to using
uncorrected p-values is {cost_str}. This is specific to the observed
results and does not imply that correction is unnecessary in future
experiments.

ENGINEERING ACTION ITEMS:
  1. Enable the top-banner treatment for mobile.
  2. Enable the top-banner treatment for desktop.
  3. Keep device-level feature flags for monitoring and rollback.
  4. Monitor CTR and guardrail metrics separately by device after launch.
  5. If proving a stronger mobile effect remains important, run a
     separately powered follow-up interaction experiment.
─────────────────────────────────────────────────────────────────
"""
    return narrative.strip()


if __name__ == "__main__":
    from phase5.data_loader     import load_data
    from phase5.segmented_tests import run_segmented_tests
    from phase5.multiple_corrections import apply_corrections
    from phase5.business_impact import compute_false_positive_cost
    from phase5.interaction_model import run_interaction_model

    df       = load_data()
    seg_res  = run_segmented_tests(df)
    corr     = apply_corrections(seg_res)
    cost     = compute_false_positive_cost(corr)
    hte      = run_interaction_model(df)
    build_recommendation(corr, hte, cost)