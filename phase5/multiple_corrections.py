# phase5/multiple_corrections.py
"""
Part C — Multiple comparison correction.
Applies Bonferroni and Benjamini-Hochberg FDR to the
per-device segmented p-values.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import ALPHA, OUTPUT_DIR


def apply_corrections(segmented_results: pd.DataFrame,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Take raw per-segment test results and apply:
      1. Bonferroni correction
      2. Benjamini-Hochberg FDR

    Parameters
    ----------
    segmented_results : DataFrame from segmented_tests.run_segmented_tests()
                        Must have a 'p_value' column, indexed by device.

    Returns
    -------
    pd.DataFrame with original columns plus correction columns added.
    """
    df = segmented_results.copy()
    raw_pvals = df["p_value"].values
    n         = len(raw_pvals)

    # ── Bonferroni ─────────────────────────────────────────────────────────
    bonf_reject, bonf_pvals, _, _ = multipletests(
        raw_pvals, alpha=ALPHA, method="bonferroni"
    )
    df["p_bonferroni"]       = bonf_pvals.round(6)
    df["sig_bonferroni"]     = bonf_reject

    # ── Benjamini-Hochberg FDR ─────────────────────────────────────────────
    bh_reject, bh_pvals, _, _ = multipletests(
        raw_pvals, alpha=ALPHA, method="fdr_bh"
    )
    df["p_bh_fdr"]           = bh_pvals.round(6)
    df["sig_bh_fdr"]         = bh_reject

    # ── Summary table ──────────────────────────────────────────────────────
    summary_cols = [
        "ctr_control", "ctr_treatment", "abs_diff",
        "p_value", "significant",
        "p_bonferroni", "sig_bonferroni",
        "p_bh_fdr",    "sig_bh_fdr"
    ]

    if verbose:
        print("\n" + "="*75)
        print("  PART C — Multiple Comparison Correction")
        print(f"  Raw α={ALPHA},  Bonferroni α'={ALPHA/n:.4f},  BH-FDR q={ALPHA}")
        print("="*75)
        print(df[summary_cols].to_string())
        print("="*75 + "\n")

    return df


def correction_decision_table(corrected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact decision table showing what changed after correction.
    """
    rows = []
    for device, row in corrected_df.iterrows():
        rows.append({
            "device":              device,
            "raw_p":               row["p_value"],
            "raw_decision":        "SHIP" if row["significant"] else "NO SHIP",
            "bonferroni_p":        row["p_bonferroni"],
            "bonferroni_decision": "SHIP" if row["sig_bonferroni"] else "NO SHIP",
            "bh_fdr_p":            row["p_bh_fdr"],
            "bh_fdr_decision":     "SHIP" if row["sig_bh_fdr"] else "NO SHIP",
        })
    return pd.DataFrame(rows).set_index("device")


if __name__ == "__main__":
    from phase5.data_loader import load_data
    from phase5.segmented_tests import run_segmented_tests
    df      = load_data()
    seg_res = run_segmented_tests(df)
    corr    = apply_corrections(seg_res)
    print(correction_decision_table(corr))