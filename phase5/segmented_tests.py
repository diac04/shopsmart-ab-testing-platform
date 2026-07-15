# phase5/segmented_tests.py
"""
Part A (ii) — Per-device segmented CTR tests.
Runs a separate proportion z-test for each device segment.
Results are collected into a DataFrame for use in correction step.
"""

import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    TREATMENT_COL, OUTCOME_COL, DEVICE_COL,
    CONTROL_LABEL, TREATMENT_LABEL,
    DEVICE_SEGMENTS, ALPHA, OUTPUT_DIR
)
from phase5.overall_test import proportion_z_test


def run_segmented_tests(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Run one proportion z-test per device segment.

    Returns
    -------
    pd.DataFrame  one row per segment, with all test statistics.
    """
    rows = []
    for device in DEVICE_SEGMENTS:
        seg   = df[df[DEVICE_COL] == device]
        ctrl  = seg[seg[TREATMENT_COL] == CONTROL_LABEL]
        trt   = seg[seg[TREATMENT_COL] == TREATMENT_LABEL]

        if len(ctrl) == 0 or len(trt) == 0:
            print(f"  [WARN] No data for device={device}, skipping.")
            continue

        res = proportion_z_test(
            n1=len(ctrl), x1=ctrl[OUTCOME_COL].sum(),
            n2=len(trt),  x2=trt[OUTCOME_COL].sum(),
        )
        res["segment"] = device
        rows.append(res)

    results_df = pd.DataFrame(rows).set_index("segment")

    if verbose:
        print("\n" + "="*65)
        print("  PART A — Segmented CTR Tests (per device, UNCORRECTED)")
        print("="*65)
        display_cols = [
            "n_control", "n_treatment",
            "ctr_control", "ctr_treatment",
            "abs_diff", "rel_lift_pct",
            "z_stat", "p_value", "significant"
        ]
        print(results_df[display_cols].to_string())
        print("="*65 + "\n")

    return results_df


if __name__ == "__main__":
    from phase5.data_loader import load_data
    df = load_data()
    run_segmented_tests(df)