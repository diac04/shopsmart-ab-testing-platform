# phase5/business_impact.py
"""
Part C (continued) — ₹ Cost of a false positive.

If we had shipped to ALL devices based on uncorrected p-values,
and the treatment was actually harmful for some device segments,
what is the monthly revenue cost?
"""

import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    MONTHLY_ACTIVE_USERS, DEVICE_SHARE,
    REVENUE_PER_USER_PER_MONTH, REVENUE_LOSS_FRACTION,
    OUTPUT_DIR
)


def compute_false_positive_cost(
    corrected_df: pd.DataFrame,
    verbose: bool = True
) -> dict:
    """
    For each device where:
      - raw test said SHIP (significant)  but
      - corrected test says NO SHIP (not significant after correction)

    We treat that as a potential false positive.
    We estimate the monthly revenue at risk if we'd shipped anyway.

    If the treatment *actually* hurts CTR (abs_diff < 0),
    the revenue loss is concrete; if it's positive but uncertain,
    the cost is the expected loss weighted by false-positive probability.

    For conservative estimation we assume the null is true for
    those flagged segments (i.e., the observed uplift is noise).

    Returns dict with cost breakdown per device.
    """
    rows = []
    total_cost = 0.0

    for device, row in corrected_df.iterrows():
        raw_sig  = bool(row["significant"])      # uncorrected said ship?
        corr_sig = bool(row["sig_bonferroni"])   # corrected says ship?

        # Potential false positive = raw said yes, corrected says no
        is_fp_candidate = raw_sig and not corr_sig

        mau_segment  = MONTHLY_ACTIVE_USERS * DEVICE_SHARE.get(device, 0)
        revenue_base = mau_segment * REVENUE_PER_USER_PER_MONTH

        # If the banner hurts (negative diff), cost is direct loss
        # If the banner shows positive but uncorrected uplift, cost is
        # opportunity/trust cost — we use loss_fraction as the floor
        if row["abs_diff"] < 0:
            # Treatment makes things worse — shipping it costs us
            loss_per_user = abs(row["abs_diff"]) * REVENUE_PER_USER_PER_MONTH
            monthly_cost  = mau_segment * loss_per_user
        else:
            # Positive but possibly spurious — use conservative fraction
            monthly_cost  = revenue_base * REVENUE_LOSS_FRACTION if is_fp_candidate else 0.0

        rows.append({
            "device":           device,
            "mau_segment":      int(mau_segment),
            "raw_significant":  raw_sig,
            "corr_significant": corr_sig,
            "fp_candidate":     is_fp_candidate,
            "abs_diff":         row["abs_diff"],
            "monthly_cost_inr": round(monthly_cost, 2),
        })

        if is_fp_candidate:
            total_cost += monthly_cost

    result_df = pd.DataFrame(rows).set_index("device")

    if verbose:
        print("\n" + "="*65)
        print("  PART C — ₹ Cost of False Positive (uncorrected shipping)")
        print("="*65)
        print(result_df.to_string())
        print(f"\n  ▶ Total estimated monthly cost of uncorrected")
        print(f"    decision: ₹{total_cost:,.2f}")
        print("="*65 + "\n")

    return {
        "breakdown_df":     result_df,
        "total_cost_inr":   round(total_cost, 2),
    }


if __name__ == "__main__":
    from phase5.data_loader import load_data
    from phase5.segmented_tests import run_segmented_tests
    from phase5.multiple_corrections import apply_corrections

    df      = load_data()
    seg_res = run_segmented_tests(df)
    corr    = apply_corrections(seg_res)
    compute_false_positive_cost(corr)