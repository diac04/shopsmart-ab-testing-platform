# phase4/outlier_analysis.py
# ============================================================
# Part D — Outlier Robustness Analysis
#
# Phase 2 found: treatment kurtosis=6.84, max revenue=Rs.8,565
# vs control kurtosis=0.84, max=Rs.4,017
# Question: is the lift driven by a small cluster of high-value
# outliers, or is it present across the full distribution?
#
# Approaches:
#   1. Winsorization at (1%, 99%) and (5%, 95%)
#   2. Trimmed mean (5% each tail)
#   3. Percentile sub-analysis (lift by revenue band)
#   4. Top-N outlier removal (remove top 0.1%, 0.5%, 1%)
#   5. CUPED-adjusted outlier analysis (the trusted version)
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
import os

from phase4.config import (
    OUTPUT_DIR, ALPHA, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
    BOOTSTRAP_CI, CONTROL_LABEL, TREATMENT_LABEL,
    WINSOR_LOWER, WINSOR_UPPER, TRIM_FRAC
)


# ── Utility: bootstrap mean diff ─────────────────────────────────────────────

def _bootstrap_mean_diff(ctrl: np.ndarray,
                          treat: np.ndarray,
                          n_boot: int = BOOTSTRAP_RESAMPLES,
                          seed: int   = BOOTSTRAP_SEED) -> dict:
    rng        = np.random.default_rng(seed)
    n_c, n_t   = len(ctrl), len(treat)
    alpha_tail = (1 - BOOTSTRAP_CI) / 2
    boot_diffs = np.empty(n_boot)

    for i in range(n_boot):
        s_c = rng.choice(ctrl,  size=n_c, replace=True)
        s_t = rng.choice(treat, size=n_t, replace=True)
        boot_diffs[i] = np.mean(s_t) - np.mean(s_c)

    obs_diff = np.mean(treat) - np.mean(ctrl)
    ci_lo    = np.percentile(boot_diffs, 100 * alpha_tail)
    ci_hi    = np.percentile(boot_diffs, 100 * (1 - alpha_tail))

    shifted = boot_diffs - np.mean(boot_diffs)
    p_boot  = 2 * min(
        np.mean(shifted >= abs(obs_diff)),
        np.mean(shifted <= -abs(obs_diff))
    )
    p_boot  = max(p_boot, 1 / n_boot)   # floor at 1/n_boot

    return {
        "obs_diff"   : round(float(obs_diff), 2),
        "ci_lo"      : round(float(ci_lo),    2),
        "ci_hi"      : round(float(ci_hi),    2),
        "p_value"    : round(float(p_boot),   6),
        "significant": p_boot < ALPHA,
    }


# ── 1. Winsorization ──────────────────────────────────────────────────────────

def winsorize_array(arr: np.ndarray,
                    lower_q: float,
                    upper_q: float) -> np.ndarray:
    """Clip values at the lower_q and upper_q percentiles."""
    lo = np.percentile(arr, 100 * lower_q)
    hi = np.percentile(arr, 100 * upper_q)
    return np.clip(arr, lo, hi)


def run_winsorization(ctrl: np.ndarray,
                       treat: np.ndarray) -> list:
    """Run bootstrap on winsorized data at two thresholds."""
    results = []
    for lo, hi, label in [
        (0.01, 0.99, "Winsorized (1%–99%)"),
        (0.05, 0.95, "Winsorized (5%–95%)"),
    ]:
        ctrl_w  = winsorize_array(ctrl,  lo, hi)
        treat_w = winsorize_array(treat, lo, hi)
        boot    = _bootstrap_mean_diff(ctrl_w, treat_w)
        results.append({
            "scenario"   : label,
            "ctrl_mean"  : round(float(np.mean(ctrl_w)),  2),
            "treat_mean" : round(float(np.mean(treat_w)), 2),
            **boot
        })
    return results


# ── 2. Trimmed mean ───────────────────────────────────────────────────────────

def run_trimmed_mean(ctrl: np.ndarray,
                      treat: np.ndarray,
                      trim: float = TRIM_FRAC) -> dict:
    """
    Remove the bottom and top `trim` fraction from each group
    independently, then compare means.
    """
    ctrl_t  = stats.trim_mean(ctrl,  trim)
    treat_t = stats.trim_mean(treat, trim)
    lift    = treat_t - ctrl_t
    lift_pct= 100 * lift / ctrl_t if ctrl_t != 0 else np.nan

    # Winsorise for the bootstrap (trimming changes n, so winsorise instead)
    ctrl_w  = winsorize_array(ctrl,  trim, 1 - trim)
    treat_w = winsorize_array(treat, trim, 1 - trim)
    boot    = _bootstrap_mean_diff(ctrl_w, treat_w)

    return {
        "scenario"      : f"Trimmed mean ({int(trim*100)}% each tail)",
        "ctrl_trimmed"  : round(float(ctrl_t),  2),
        "treat_trimmed" : round(float(treat_t), 2),
        "obs_diff"      : round(float(lift),    2),
        "lift_pct"      : round(float(lift_pct),4),
        **boot
    }


# ── 3. Percentile sub-analysis ────────────────────────────────────────────────

def run_percentile_bands(ctrl: np.ndarray,
                          treat: np.ndarray) -> list:
    """
    Split purchasers into revenue bands and check lift within each.
    If the overall lift is driven by outliers, the top band will
    show a huge lift while lower bands show nothing.
    """
    # Use purchasers only for band analysis
    ctrl_buy  = ctrl[ctrl   > 0]
    treat_buy = treat[treat > 0]

    # Compute percentile thresholds on pooled purchasers
    pooled = np.concatenate([ctrl_buy, treat_buy])
    bands  = [
        ("Bottom 25%",   0,   25),
        ("25th–50th",   25,   50),
        ("50th–75th",   50,   75),
        ("75th–90th",   75,   90),
        ("90th–99th",   90,   99),
        ("Top 1%",      99,  100),
    ]

    results = []
    for label, lo_pct, hi_pct in bands:
        lo_val = np.percentile(pooled, lo_pct) if lo_pct > 0  else 0
        hi_val = np.percentile(pooled, hi_pct) if hi_pct < 100 else np.inf

        c_band = ctrl_buy[ (ctrl_buy  >= lo_val) & (ctrl_buy  < hi_val)]
        t_band = treat_buy[(treat_buy >= lo_val) & (treat_buy < hi_val)]

        if len(c_band) < 10 or len(t_band) < 10:
            results.append({
                "band": label, "n_ctrl": len(c_band),
                "n_treat": len(t_band),
                "ctrl_mean": None, "treat_mean": None,
                "lift_rs": None, "lift_pct": None,
                "note": "Too few observations"
            })
            continue

        ctrl_m  = np.mean(c_band)
        treat_m = np.mean(t_band)
        lift    = treat_m - ctrl_m
        lift_pct= 100 * lift / ctrl_m

        results.append({
            "band"      : label,
            "n_ctrl"    : len(c_band),
            "n_treat"   : len(t_band),
            "ctrl_mean" : round(float(ctrl_m),   2),
            "treat_mean": round(float(treat_m),  2),
            "lift_rs"   : round(float(lift),     2),
            "lift_pct"  : round(float(lift_pct), 2),
            "note"      : ""
        })
    return results


# ── 4. Top-N outlier removal ──────────────────────────────────────────────────

def run_topn_removal(ctrl: np.ndarray,
                      treat: np.ndarray) -> list:
    """
    Progressively remove the top 0.1%, 0.5%, 1% of users
    (by revenue) from BOTH groups and re-run the bootstrap.
    If the lift collapses, it was outlier-driven.
    """
    results = []
    for pct, label in [
        (0.001, "Remove top 0.1%"),
        (0.005, "Remove top 0.5%"),
        (0.010, "Remove top 1.0%"),
    ]:
        threshold_c = np.percentile(ctrl,  (1 - pct) * 100)
        threshold_t = np.percentile(treat, (1 - pct) * 100)

        ctrl_trim  = ctrl[ctrl   <= threshold_c]
        treat_trim = treat[treat <= threshold_t]

        boot = _bootstrap_mean_diff(ctrl_trim, treat_trim)
        results.append({
            "scenario"       : label,
            "threshold_ctrl" : round(float(threshold_c),  2),
            "threshold_treat": round(float(threshold_t),  2),
            "n_ctrl"         : len(ctrl_trim),
            "n_treat"        : len(treat_trim),
            "ctrl_mean"      : round(float(np.mean(ctrl_trim)),  2),
            "treat_mean"     : round(float(np.mean(treat_trim)), 2),
            **boot
        })
    return results


# ── Master runner ─────────────────────────────────────────────────────────────

def run_outlier_analysis(ctrl_late: np.ndarray,
                          treat_late: np.ndarray,
                          ctrl_cuped: np.ndarray  = None,
                          treat_cuped: np.ndarray = None) -> dict:
    """
    Full Part D outlier robustness pipeline.
    ctrl_cuped / treat_cuped: optional CUPED-adjusted arrays
    for comparison.
    """

    print("\n" + "="*60)
    print("  PART D — OUTLIER ROBUSTNESS ANALYSIS")
    print("="*60)

    # ── Baseline (raw, no adjustment) ────────────────────────
    raw_boot = _bootstrap_mean_diff(ctrl_late, treat_late)
    print(f"\n  BASELINE (raw, no outlier treatment)")
    print(f"    ctrl mean  : Rs.{np.mean(ctrl_late):.2f}")
    print(f"    treat mean : Rs.{np.mean(treat_late):.2f}")
    print(f"    lift       : Rs.{raw_boot['obs_diff']}  "
          f"CI [{raw_boot['ci_lo']}, {raw_boot['ci_hi']}]")
    print(f"    ctrl max   : Rs.{np.max(ctrl_late):.2f}")
    print(f"    treat max  : Rs.{np.max(treat_late):.2f}")
    print(f"    ctrl p99   : Rs.{np.percentile(ctrl_late, 99):.2f}")
    print(f"    treat p99  : Rs.{np.percentile(treat_late, 99):.2f}")

    # ── Winsorization ─────────────────────────────────────────
    print(f"\n  WINSORIZATION RESULTS")
    winsor_results = run_winsorization(ctrl_late, treat_late)
    for r in winsor_results:
        print(f"    {r['scenario']:<30} "
              f"lift Rs.{r['obs_diff']:>7}  "
              f"CI [{r['ci_lo']}, {r['ci_hi']}]  "
              f"p={r['p_value']:.4f}  "
              f"sig={r['significant']}")

    # ── Trimmed mean ──────────────────────────────────────────
    print(f"\n  TRIMMED MEAN RESULTS")
    trim_result = run_trimmed_mean(ctrl_late, treat_late)
    print(f"    {trim_result['scenario']:<30} "
          f"lift Rs.{trim_result['obs_diff']:>7}  "
          f"CI [{trim_result['ci_lo']}, {trim_result['ci_hi']}]  "
          f"p={trim_result['p_value']:.4f}  "
          f"sig={trim_result['significant']}")

    # ── Top-N removal ─────────────────────────────────────────
    print(f"\n  TOP-N OUTLIER REMOVAL")
    topn_results = run_topn_removal(ctrl_late, treat_late)
    for r in topn_results:
        print(f"    {r['scenario']:<25} "
              f"threshold ctrl Rs.{r['threshold_ctrl']:>8}  "
              f"treat Rs.{r['threshold_treat']:>8}  "
              f"lift Rs.{r['obs_diff']:>7}  "
              f"sig={r['significant']}")

    # ── Percentile band analysis ──────────────────────────────
    print(f"\n  PERCENTILE BAND ANALYSIS (purchasers only)")
    print(f"  {'Band':<18} {'n_ctrl':>7} {'n_treat':>8} "
          f"{'ctrl_mean':>10} {'treat_mean':>11} "
          f"{'lift_rs':>9} {'lift_pct':>9}")
    print("  " + "-"*78)
    band_results = run_percentile_bands(ctrl_late, treat_late)
    for r in band_results:
        if r["ctrl_mean"] is None:
            print(f"  {r['band']:<18} {'—':>7} {'—':>8} "
                  f"{'—':>10} {'—':>11} {'—':>9}  {r['note']}")
        else:
            print(f"  {r['band']:<18} {r['n_ctrl']:>7} {r['n_treat']:>8} "
                  f"{r['ctrl_mean']:>10.2f} {r['treat_mean']:>11.2f} "
                  f"{r['lift_rs']:>+9.2f} {r['lift_pct']:>+8.2f}%")

    # ── CUPED comparison (if provided) ───────────────────────
    cuped_boot = None
    if ctrl_cuped is not None and treat_cuped is not None:
        cuped_boot = _bootstrap_mean_diff(ctrl_cuped, treat_cuped)
        print(f"\n  CUPED-ADJUSTED (pre-exp imbalance corrected)")
        print(f"    lift       : Rs.{cuped_boot['obs_diff']}  "
              f"CI [{cuped_boot['ci_lo']}, {cuped_boot['ci_hi']}]  "
              f"p={cuped_boot['p_value']}  "
              f"sig={cuped_boot['significant']}")

    # ── Summary comparison table ──────────────────────────────
    print(f"\n" + "="*60)
    print(f"  OUTLIER SENSITIVITY SUMMARY TABLE")
    print(f"="*60)
    print(f"  {'Scenario':<35} {'Lift (Rs.)':>10} "
          f"{'CI Lo':>8} {'CI Hi':>8} {'Sig':>5}")
    print("  " + "-"*70)

    all_scenarios = [
        ("Raw (no treatment)",          raw_boot),
        *[(r["scenario"], r)            for r in winsor_results],
        (trim_result["scenario"],       trim_result),
        *[(r["scenario"], r)            for r in topn_results],
    ]
    if cuped_boot:
        all_scenarios.append(("CUPED-adjusted", cuped_boot))

    for label, r in all_scenarios:
        print(f"  {label:<35} {r['obs_diff']:>10}  "
              f"{r['ci_lo']:>8}  {r['ci_hi']:>8}  "
              f"{'✅' if r['significant'] else '❌':>5}")

    # ── Save CSV ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for label, r in all_scenarios:
        rows.append({
            "scenario"   : label,
            "lift_rs"    : r["obs_diff"],
            "ci_lo"      : r["ci_lo"],
            "ci_hi"      : r["ci_hi"],
            "p_value"    : r["p_value"],
            "significant": r["significant"],
        })
    df_bands = pd.DataFrame(band_results)
    df_out   = pd.DataFrame(rows)

    csv_path1 = os.path.join(OUTPUT_DIR, "outlier_sensitivity.csv")
    csv_path2 = os.path.join(OUTPUT_DIR, "percentile_bands.csv")
    df_out.to_csv(csv_path1, index=False)
    df_bands.to_csv(csv_path2, index=False)
    print(f"\n  ✅ outlier_sensitivity.csv → {csv_path1}")
    print(f"  ✅ percentile_bands.csv    → {csv_path2}")

    return {
        "raw_bootstrap"  : raw_boot,
        "winsorization"  : winsor_results,
        "trimmed_mean"   : trim_result,
        "topn_removal"   : topn_results,
        "bands"          : band_results,
        "cuped_bootstrap": cuped_boot,
    }