# phase5/overall_test.py
"""
Part A (i) — Overall CTR test across ALL devices combined.
Uses a two-proportion z-test.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    TREATMENT_COL, OUTCOME_COL,
    CONTROL_LABEL, TREATMENT_LABEL, ALPHA, OUTPUT_DIR
)


def proportion_z_test(
    n1: int, x1: int,   # control:   n users, x clicked
    n2: int, x2: int,   # treatment: n users, x clicked
    two_sided: bool = True
) -> dict:
    """
    Two-proportion z-test (pooled).

    Returns dict with p1, p2, diff, z_stat, p_value, ci_low, ci_high.
    """
    p1   = x1 / n1
    p2   = x2 / n2
    diff = p2 - p1

    # pooled proportion
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    z = diff / se_pool
    p = (2 * stats.norm.sf(abs(z))) if two_sided else stats.norm.sf(z)

    # 95 % CI on the difference (unpooled SE for CI)
    se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z_crit  = stats.norm.ppf(1 - ALPHA/2)
    ci_low  = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    return {
        "ctr_control":   round(p1,   6),
        "ctr_treatment": round(p2,   6),
        "abs_diff":      round(diff, 6),
        "rel_lift_pct":  round(100 * diff / p1, 3) if p1 > 0 else None,
        "z_stat":        round(z, 4),
        "p_value":       round(p, 6),
        "ci_low":        round(ci_low,  6),
        "ci_high":       round(ci_high, 6),
        "significant":   bool(p < ALPHA),
        "n_control":     n1,
        "n_treatment":   n2,
    }


def run_overall_test(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Run the overall (all-device) CTR proportion z-test.

    Parameters
    ----------
    df : full experiment DataFrame

    Returns
    -------
    dict of test results
    """
    ctrl = df[df[TREATMENT_COL] == CONTROL_LABEL]
    trt  = df[df[TREATMENT_COL] == TREATMENT_LABEL]

    result = proportion_z_test(
        n1=len(ctrl), x1=ctrl[OUTCOME_COL].sum(),
        n2=len(trt),  x2=trt[OUTCOME_COL].sum(),
    )
    result["segment"] = "ALL_DEVICES"

    if verbose:
        print("\n" + "="*55)
        print("  PART A — Overall CTR Test (all devices combined)")
        print("="*55)
        print(f"  Control   : n={result['n_control']:,}  "
              f"CTR={result['ctr_control']:.4f}")
        print(f"  Treatment : n={result['n_treatment']:,}  "
              f"CTR={result['ctr_treatment']:.4f}")
        print(f"  Abs diff  : {result['abs_diff']:+.4f}  "
              f"({result['rel_lift_pct']:+.2f}%)")
        print(f"  Z-stat    : {result['z_stat']:.4f}")
        print(f"  P-value   : {result['p_value']:.6f}")
        print(f"  95% CI    : [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
        print(f"  Significant at α={ALPHA}: {result['significant']}")
        print("="*55 + "\n")

    return result


if __name__ == "__main__":
    from phase5.data_loader import load_data
    df = load_data()
    run_overall_test(df)