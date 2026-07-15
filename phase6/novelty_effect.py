# phase6/novelty_effect.py
"""
Part C — Long-term effect analysis / Novelty effect decomposition
- Daily conversion rates per group
- Rolling average
- Statistical test: first N days vs remainder
- Novelty effect quantification
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
from scipy import stats

from phase6.config import NOVELTY, PRE_REG, PLOT_STYLE, BRAND_COLORS, PHASE6_OUT


def compute_daily_rates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily conversion rate per group.
    """
    daily = (data.groupby(["day", "group"])["converted"]
               .agg(conversions=("sum"), n=("count"))
               .reset_index())
    daily["rate"] = daily["conversions"] / daily["n"]
    # 95% CI using Wilson score interval
    daily["ci_lo"] = daily.apply(
        lambda r: proportion_ci(r["conversions"], r["n"])[0], axis=1)
    daily["ci_hi"] = daily.apply(
        lambda r: proportion_ci(r["conversions"], r["n"])[1], axis=1)
    return daily


def proportion_ci(k: int, n: int, z: float = 1.96):
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    p    = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1-p) / n + z**2 / (4 * n**2))) / denom
    return (max(0, centre - margin), min(1, centre + margin))


def rolling_rate(daily_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add rolling average per group."""
    result = []
    for grp, sub in daily_df.groupby("group"):
        sub = sub.sort_values("day").copy()
        sub["rolling_rate"] = sub["rate"].rolling(window, min_periods=1).mean()
        result.append(sub)
    return pd.concat(result).sort_values(["day","group"])


def quantify_novelty_effect(daily_df: pd.DataFrame,
                             warmup_days: int) -> dict:
    """
    Compare treatment rate in first warmup_days vs remaining days.
    Uses two-sample t-test on daily rates (treatment group only).
    """
    treat = daily_df[daily_df["group"] == "treatment"].sort_values("day")
    ctrl  = daily_df[daily_df["group"] == "control"].sort_values("day")

    treat_early = treat[treat["day"] <= warmup_days]["rate"].values
    treat_late  = treat[treat["day"]  > warmup_days]["rate"].values
    ctrl_early  = ctrl[ctrl["day"]   <= warmup_days]["rate"].values
    ctrl_late   = ctrl[ctrl["day"]    > warmup_days]["rate"].values

    # Treatment: early vs late
    if len(treat_early) > 1 and len(treat_late) > 1:
        t_stat, p_val = stats.ttest_ind(treat_early, treat_late,
                                         equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan

    results = {
        "warmup_days"          : warmup_days,
        "treat_early_mean"     : float(np.mean(treat_early)) if len(treat_early) else np.nan,
        "treat_late_mean"      : float(np.mean(treat_late))  if len(treat_late)  else np.nan,
        "ctrl_early_mean"      : float(np.mean(ctrl_early))  if len(ctrl_early)  else np.nan,
        "ctrl_late_mean"       : float(np.mean(ctrl_late))   if len(ctrl_late)   else np.nan,
        "treat_early_vs_late_t": float(t_stat) if not np.isnan(t_stat) else None,
        "treat_early_vs_late_p": float(p_val)  if not np.isnan(p_val)  else None,
        "n_early_days"         : int(len(treat_early)),
        "n_late_days"          : int(len(treat_late)),
    }

    novelty_magnitude = (results["treat_early_mean"] or 0) - \
                        (results["treat_late_mean"]  or 0)
    results["novelty_magnitude_abs"] = float(novelty_magnitude)
    results["novelty_significant"]   = (
        p_val is not None and p_val < 0.05
        and abs(novelty_magnitude) > 0.001
    )

    # Steady-state lift vs control
    if not np.isnan(results["treat_late_mean"]) and \
       not np.isnan(results["ctrl_late_mean"]):
        results["steady_state_lift"] = (results["treat_late_mean"] -
                                        results["ctrl_late_mean"])
    else:
        results["steady_state_lift"] = None

    return results


def plot_novelty_effect(daily_df: pd.DataFrame,
                         rolling_df: pd.DataFrame,
                         novelty_stats: dict,
                         warmup_days: int,
                         out_dir: str):
    """
    Two-panel figure:
      Left : Daily conversion rates with CIs + warmup shading
      Right: Rolling average + steady-state annotation
    """
    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for grp, color, lbl in [
        ("control",   BRAND_COLORS["control"],   "Control"),
        ("treatment", BRAND_COLORS["treatment"], "Treatment"),
    ]:
        sub  = daily_df[daily_df["group"] == grp].sort_values("day")
        days = sub["day"].values
        rate = sub["rate"].values
        lo   = sub["ci_lo"].values
        hi   = sub["ci_hi"].values

        # Daily rates + CI ribbon
        axes[0].plot(days, rate * 100, color=color, lw=2.5,
                     marker="o", ms=4, label=lbl)
        axes[0].fill_between(days, lo * 100, hi * 100,
                              alpha=0.15, color=color)

        # Rolling
        sub_r   = rolling_df[rolling_df["group"] == grp].sort_values("day")
        axes[1].plot(sub_r["day"], sub_r["rolling_rate"] * 100,
                     color=color, lw=2.5, label=f"{lbl} (rolling avg)")
        axes[1].scatter(sub["day"], sub["rate"] * 100,
                        color=color, alpha=0.3, s=20, zorder=3)

    # Warmup shading
    for ax in axes:
        ax.axvspan(0.5, warmup_days + 0.5, alpha=0.08, color="orange",
                   label=f"Initial period (days 1–{warmup_days})")
        ax.axvline(warmup_days + 0.5, color="orange",
                   lw=1.5, ls="--", alpha=0.7)

    # Steady-state lift annotation
    if novelty_stats["steady_state_lift"] is not None:
        sl = novelty_stats["steady_state_lift"] * 100
        axes[1].annotate(
            f"Steady-state Δ = {sl:+.3f}pp",
            xy=(daily_df["day"].max() * 0.75,
                (novelty_stats["treat_late_mean"] or 0) * 100),
            fontsize=10, color="purple",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.8)
        )

    axes[0].set_xlabel("Experiment Day", fontsize=11)
    axes[0].set_ylabel("Conversion Rate (%)", fontsize=11)
    axes[0].set_title("Daily Conversion Rates with 95% CI\n"
                       "Novelty Effect: Initial vs Steady-State", fontsize=11)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Experiment Day", fontsize=11)
    axes[1].set_ylabel("Conversion Rate (%)", fontsize=11)
    axes[1].set_title(f"Rolling {NOVELTY['rolling_window']}-Day Average\n"
                       "Separating Novelty Spike from Steady State", fontsize=11)
    axes[1].legend(fontsize=9)

    plt.suptitle("Novelty Effect Analysis — Experiment 1 Checkout\n"
                 "ShopSmart India A/B Platform", fontsize=13)
    plt.tight_layout()

    fname = os.path.join(out_dir, "novelty_effect.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved → {fname}")


def run_novelty_analysis(data: pd.DataFrame, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "═"*60)
    print("  PART C — NOVELTY EFFECT ANALYSIS")
    print("═"*60)

    warmup  = NOVELTY["warmup_days"]
    window  = NOVELTY["rolling_window"]

    daily_df   = compute_daily_rates(data)
    rolling_df = rolling_rate(daily_df, window)
    novelty    = quantify_novelty_effect(daily_df, warmup)

    print(f"\n  Warmup period        : days 1–{warmup}")
    print(f"  Treatment early mean : {novelty['treat_early_mean']:.5f}"
          if novelty["treat_early_mean"] else "  Treatment early mean : N/A")
    print(f"  Treatment late mean  : {novelty['treat_late_mean']:.5f}"
          if novelty["treat_late_mean"] else "  Treatment late mean : N/A")
    print(f"  Novelty magnitude    : {novelty['novelty_magnitude_abs']*100:.4f} pp")
    print(f"  Novelty significant  : {novelty['novelty_significant']}")
    if novelty["treat_early_vs_late_p"] is not None:
        print(f"  t-test p-value       : {novelty['treat_early_vs_late_p']:.4f}")
    if novelty["steady_state_lift"] is not None:
        print(f"  Steady-state lift    : {novelty['steady_state_lift']*100:.4f} pp "
              f"(treatment vs control, late period)")

    plot_novelty_effect(daily_df, rolling_df, novelty, warmup, out_dir)

    # Save outputs
    daily_path = os.path.join(out_dir, "daily_conversion_rates.csv")
    daily_df.drop(columns=["ci_lo","ci_hi"], errors="ignore").to_csv(
        daily_path, index=False)

    json_path = os.path.join(out_dir, "novelty_results.json")
    with open(json_path, "w") as f:
        json.dump(novelty, f, indent=2, default=str)
    print(f"  [output] Novelty results → {json_path}")

    return novelty