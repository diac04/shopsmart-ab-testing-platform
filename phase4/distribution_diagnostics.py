# phase4/distribution_diagnostics.py
# ============================================================
# Part A — Distribution Diagnostics
# - Histogram (revenue distribution, control vs treatment)
# - Q-Q plot (normality check)
# - Shapiro-Wilk test (confirm non-normality)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

from phase4.config import (
    OUTPUT_DIR, CONTROL_LABEL, TREATMENT_LABEL,
    ALPHA
)


# ── Helper ───────────────────────────────────────────────────────────────────

def _shapiro(arr: np.ndarray, label: str, max_n: int = 5000) -> dict:
    """
    Shapiro-Wilk is unreliable and very slow for n > 5,000.
    We subsample for the test but report the full-data descriptives.
    """
    rng    = np.random.default_rng(42)
    sample = rng.choice(arr, size=min(len(arr), max_n), replace=False)
    stat, p = stats.shapiro(sample)

    result = {
        "group"       : label,
        "n_full"      : len(arr),
        "n_tested"    : len(sample),
        "W_statistic" : round(float(stat), 6),
        "p_value"     : float(p),
        "normal"      : p > ALPHA,
        "verdict"     : "NORMAL" if p > ALPHA else "NON-NORMAL"
    }
    return result


def _descriptives(arr: np.ndarray, label: str) -> dict:
    purchase_mask = arr > 0
    return {
        "group"         : label,
        "n"             : len(arr),
        "mean"          : round(float(np.mean(arr)), 4),
        "median"        : round(float(np.median(arr)), 4),
        "std"           : round(float(np.std(arr, ddof=1)), 4),
        "min"           : round(float(np.min(arr)), 4),
        "max"           : round(float(np.max(arr)), 4),
        "skewness"      : round(float(stats.skew(arr)), 4),
        "kurtosis"      : round(float(stats.kurtosis(arr)), 4),
        "zero_pct"      : round(100 * np.mean(arr == 0), 4),
        "purchase_rate" : round(100 * np.mean(purchase_mask), 4),
        "p95"           : round(float(np.percentile(arr, 95)), 4),
        "p99"           : round(float(np.percentile(arr, 99)), 4),
    }


# ── Main diagnostic function ─────────────────────────────────────────────────

def run_distribution_diagnostics(ctrl: np.ndarray,
                                  treat: np.ndarray,
                                  period_label: str = "Late Period (Days 7–14)"):
    """
    Run full Part A diagnostics on control and treatment revenue arrays.

    Parameters
    ----------
    ctrl         : revenue array for control group
    treat        : revenue array for treatment group
    period_label : string label printed on plots

    Returns
    -------
    summary : dict with descriptives + Shapiro results
    """

    print("\n" + "="*60)
    print("  PART A — DISTRIBUTION DIAGNOSTICS")
    print(f"  Period : {period_label}")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Descriptive statistics ─────────────────────────────
    desc_ctrl  = _descriptives(ctrl,  CONTROL_LABEL)
    desc_treat = _descriptives(treat, TREATMENT_LABEL)

    print("\n  DESCRIPTIVE STATISTICS")
    print(f"  {'Metric':<18} {'Control':>12} {'Treatment':>12}")
    print("  " + "-"*44)
    for key in ["n", "mean", "median", "std", "skewness",
                "kurtosis", "zero_pct", "purchase_rate", "p95", "p99"]:
        print(f"  {key:<18} {desc_ctrl[key]:>12} {desc_treat[key]:>12}")

    # ── 2. Shapiro-Wilk test ──────────────────────────────────
    print("\n  SHAPIRO-WILK TEST (subsample n=5,000 per group)")
    sw_ctrl  = _shapiro(ctrl,  CONTROL_LABEL)
    sw_treat = _shapiro(treat, TREATMENT_LABEL)

    for sw in [sw_ctrl, sw_treat]:
        print(f"\n  Group     : {sw['group']}")
        print(f"  n (full)  : {sw['n_full']:,}")
        print(f"  n (tested): {sw['n_tested']:,}")
        print(f"  W stat    : {sw['W_statistic']:.6f}")
        print(f"  p-value   : {sw['p_value']:.2e}")
        print(f"  Verdict   : {'✅ ' if sw['normal'] else '❌ '}{sw['verdict']}")

    # ── 3. Plots ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Experiment 2 — Personalized Recommendations\n"
        f"Distribution Diagnostics | {period_label}",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.35)

    colors = {CONTROL_LABEL: "#4C72B0", TREATMENT_LABEL: "#DD8452"}

    # ── 3a. Histogram — all users (including zeros) ───────────
    ax1 = fig.add_subplot(gs[0, :])
    bins = np.linspace(0, np.percentile(
                            np.concatenate([ctrl, treat]), 99), 80)
    ax1.hist(ctrl,  bins=bins, alpha=0.55, color=colors[CONTROL_LABEL],
             label=f"Control  (n={len(ctrl):,})",  density=True)
    ax1.hist(treat, bins=bins, alpha=0.55, color=colors[TREATMENT_LABEL],
             label=f"Treatment (n={len(treat):,})", density=True)
    ax1.axvline(np.mean(ctrl),  color=colors[CONTROL_LABEL],
                linestyle="--", linewidth=1.8,
                label=f"Control mean  Rs.{np.mean(ctrl):.0f}")
    ax1.axvline(np.mean(treat), color=colors[TREATMENT_LABEL],
                linestyle="--", linewidth=1.8,
                label=f"Treatment mean Rs.{np.mean(treat):.0f}")
    ax1.set_title("Revenue Distribution (all users, clipped at p99)",
                  fontsize=11)
    ax1.set_xlabel("Revenue per User (Rs.)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9)

    # ── 3b. Histogram — purchasers only ──────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ctrl_buy  = ctrl[ctrl   > 0]
    treat_buy = treat[treat > 0]
    bins_buy  = np.linspace(0, np.percentile(
                    np.concatenate([ctrl_buy, treat_buy]), 99), 60)
    ax2.hist(ctrl_buy,  bins=bins_buy, alpha=0.55,
             color=colors[CONTROL_LABEL],
             label=f"Control  (n={len(ctrl_buy):,})", density=True)
    ax2.hist(treat_buy, bins=bins_buy, alpha=0.55,
             color=colors[TREATMENT_LABEL],
             label=f"Treatment (n={len(treat_buy):,})", density=True)
    ax2.set_title("Revenue Distribution — Purchasers Only", fontsize=11)
    ax2.set_xlabel("Revenue per User (Rs.)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)

    # ── 3c. Log-scale histogram ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    log_bins = np.logspace(
        np.log10(max(1, np.min(np.concatenate([ctrl_buy, treat_buy])))),
        np.log10(np.max(np.concatenate([ctrl_buy, treat_buy]))),
        60
    )
    ax3.hist(ctrl_buy,  bins=log_bins, alpha=0.55,
             color=colors[CONTROL_LABEL],
             label="Control",   density=True)
    ax3.hist(treat_buy, bins=log_bins, alpha=0.55,
             color=colors[TREATMENT_LABEL],
             label="Treatment", density=True)
    ax3.set_xscale("log")
    ax3.set_title("Log-Scale Revenue (Purchasers Only)", fontsize=11)
    ax3.set_xlabel("Revenue per User (Rs., log scale)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=9)

    # ── 3d. Q-Q plot — control ────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    rng    = np.random.default_rng(42)
    sample_ctrl = rng.choice(ctrl, size=min(len(ctrl), 3000), replace=False)
    (osm, osr), (slope, intercept, r) = stats.probplot(
        sample_ctrl, dist="norm", fit=True)
    ax4.scatter(osm, osr, s=6, alpha=0.4,
                color=colors[CONTROL_LABEL], label="Control data")
    ax4.plot(osm, slope * np.array(osm) + intercept,
             color="red", linewidth=1.5, label="Normal reference line")
    ax4.set_title(f"Q-Q Plot — Control\n"
                  f"Skew={desc_ctrl['skewness']:.2f}  "
                  f"Kurt={desc_ctrl['kurtosis']:.2f}", fontsize=10)
    ax4.set_xlabel("Theoretical Quantiles")
    ax4.set_ylabel("Sample Quantiles")
    ax4.legend(fontsize=9)

    # ── 3e. Q-Q plot — treatment ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    sample_treat = rng.choice(treat, size=min(len(treat), 3000),
                               replace=False)
    (osm2, osr2), (slope2, intercept2, r2) = stats.probplot(
        sample_treat, dist="norm", fit=True)
    ax5.scatter(osm2, osr2, s=6, alpha=0.4,
                color=colors[TREATMENT_LABEL], label="Treatment data")
    ax5.plot(osm2, slope2 * np.array(osm2) + intercept2,
             color="red", linewidth=1.5, label="Normal reference line")
    ax5.set_title(f"Q-Q Plot — Treatment\n"
                  f"Skew={desc_treat['skewness']:.2f}  "
                  f"Kurt={desc_treat['kurtosis']:.2f}", fontsize=10)
    ax5.set_xlabel("Theoretical Quantiles")
    ax5.set_ylabel("Sample Quantiles")
    ax5.legend(fontsize=9)

    # ── Save ──────────────────────────────────────────────────
    plot_path = os.path.join(OUTPUT_DIR, "distributions.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✅ Plot saved → {plot_path}")

    # ── 4. Summary dict ───────────────────────────────────────
    summary = {
        "period"            : period_label,
        "descriptives"      : {"control": desc_ctrl, "treatment": desc_treat},
        "shapiro_wilk"      : {"control": sw_ctrl,   "treatment": sw_treat},
        "non_normal_verdict": (not sw_ctrl["normal"]) and
                              (not sw_treat["normal"])
    }

    print("\n  PART A COMPLETE")
    return summary