# phase6/power_retrospective.py
"""
Part D — Power Analysis Retrospective
Compares Phase 1 pre-registered MDE/power against
Phase 3 observed effect size.

Outputs:
  - Power curve chart
  - Retrospective paragraph (text file)
  - JSON summary
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
from scipy import stats

from phase6.config import PRE_REG, OBSERVED, PLOT_STYLE, BRAND_COLORS, PHASE6_OUT


# ─────────────────────────────────────────────────────────────────────────────
# Power calculation helpers
# ─────────────────────────────────────────────────────────────────────────────

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))


def power_for_effect(h: float, n_per_group: int,
                     alpha: float = 0.05) -> float:
    """
    Approximate power for two-proportion z-test given Cohen's h.
    Uses normal approximation.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ncp     = abs(h) * np.sqrt(n_per_group / 2)
    power   = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    return float(np.clip(power, 0, 1))


def mde_for_power(target_power: float,
                  n_per_group: int,
                  baseline: float,
                  alpha: float = 0.05) -> float:
    """
    Find the minimum detectable absolute effect for a given n and power.
    Binary search over possible treatment rates.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(target_power)

    lo, hi = 1e-6, 0.5
    for _ in range(200):
        mid  = (lo + hi) / 2
        h    = abs(cohens_h(baseline, baseline + mid))
        ncp  = h * np.sqrt(n_per_group / 2)
        pw   = 1 - stats.norm.cdf(z_alpha - ncp)
        if pw < target_power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def power_curve(baseline: float,
                n_per_group: int,
                alpha: float = 0.05,
                n_points: int = 300) -> tuple:
    """
    Return arrays (absolute_effects, powers) for a power curve.
    """
    effects = np.linspace(0.0001, 0.020, n_points)
    powers  = []
    for eff in effects:
        h  = abs(cohens_h(baseline, baseline + eff))
        pw = power_for_effect(h, n_per_group, alpha)
        powers.append(pw)
    return effects, np.array(powers)


# ─────────────────────────────────────────────────────────────────────────────
# Retrospective calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_retrospective(ctrl_conv: int, ctrl_n: int,
                           treat_conv: int, treat_n: int) -> dict:
    """
    Full retrospective comparison:
    Phase 1 design vs Phase 3 outcome.
    """
    # ── Phase 1 design ────────────────────────────────────────────────────
    design_n         = PRE_REG["n_per_group"]
    design_mde_abs   = PRE_REG["mde_absolute"]
    design_mde_rel   = PRE_REG["mde_relative_pct"]
    design_power     = PRE_REG["power"]
    design_alpha     = PRE_REG["alpha"]
    design_h         = PRE_REG["cohens_h"]
    baseline         = PRE_REG["baseline_rate"]

    # ── Phase 3 actual ────────────────────────────────────────────────────
    actual_ctrl_rate  = ctrl_conv  / ctrl_n
    actual_treat_rate = treat_conv / treat_n
    actual_lift_abs   = actual_treat_rate - actual_ctrl_rate
    actual_lift_rel   = actual_lift_abs / actual_ctrl_rate * 100
    actual_h          = abs(cohens_h(actual_ctrl_rate, actual_treat_rate))

    # ── Power the experiment HAD for the observed effect ─────────────────
    power_for_observed = power_for_effect(actual_h, design_n, design_alpha)

    # ── Power you would need to detect observed effect at 80% ─────────────
    if actual_h > 0:
        z_alpha = stats.norm.ppf(1 - design_alpha / 2)
        z_beta  = stats.norm.ppf(0.80)
        n_needed_for_observed = int(
            np.ceil(2 * ((z_alpha + z_beta) / actual_h) ** 2)
        )
    else:
        n_needed_for_observed = None

    # ── What effect could you detect at 80% power with actual n? ─────────
    actual_mde = mde_for_power(0.80, min(ctrl_n, treat_n),
                                baseline, design_alpha)

    return {
        "design_n_per_group"     : design_n,
        "actual_n_per_group"     : min(ctrl_n, treat_n),
        "design_mde_abs"         : design_mde_abs,
        "design_mde_rel_pct"     : design_mde_rel,
        "design_power"           : design_power,
        "design_alpha"           : design_alpha,
        "design_cohens_h"        : design_h,
        "observed_ctrl_rate"     : round(actual_ctrl_rate, 6),
        "observed_treat_rate"    : round(actual_treat_rate, 6),
        "observed_lift_abs"      : round(actual_lift_abs, 6),
        "observed_lift_rel_pct"  : round(actual_lift_rel, 4),
        "observed_cohens_h"      : round(actual_h, 6),
        "power_for_observed"     : round(power_for_observed, 5),
        "n_needed_for_observed"  : n_needed_for_observed,
        "actual_mde_abs"         : round(actual_mde, 6),
        "adequately_powered"     : power_for_observed >= 0.80,
        "observed_vs_mde_ratio"  : round(abs(actual_lift_abs) / design_mde_abs, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_retrospective(retro: dict, out_dir: str):
    """
    Power curve showing:
      - Pre-registered MDE (vertical line, annotated)
      - Observed effect (vertical line, annotated)
      - 80% power threshold (horizontal line)
      - Area where experiment is underpowered (shaded)
    """
    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        pass

    baseline   = PRE_REG["baseline_rate"]
    n          = retro["actual_n_per_group"]
    effects, powers = power_curve(baseline, n)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(effects * 100, powers * 100,
            color=BRAND_COLORS["control"], lw=2.5,
            label=f"Power curve (n={n:,}/group, α=0.05)")

    # 80% power line
    ax.axhline(80, color="black", lw=1.5, ls="--",
               label="80% power threshold")

    # MDE from Phase 1
    ax.axvline(retro["design_mde_abs"] * 100, color=BRAND_COLORS["success"],
               lw=2, ls="-.",
               label=f"Phase 1 MDE = {retro['design_mde_abs']*100:.2f}pp "
                     f"(power={retro['design_power']*100:.0f}%)")

    # Observed effect
    obs_abs = retro["observed_lift_abs"]
    obs_power = retro["power_for_observed"] * 100
    ax.axvline(abs(obs_abs) * 100, color=BRAND_COLORS["treatment"],
               lw=2, ls=":",
               label=f"Observed effect = {obs_abs*100:.3f}pp "
                     f"(power={obs_power:.1f}%)")

    # Shaded: underpowered region (left of MDE)
    mask = effects <= retro["design_mde_abs"]
    ax.fill_between(effects[mask] * 100, 0, powers[mask] * 100,
                    alpha=0.10, color=BRAND_COLORS["treatment"],
                    label="Underpowered region")

    # Annotations
    ax.annotate(
        f"Pre-registered\nMDE = {retro['design_mde_abs']*100:.2f}pp\n"
        f"Power = {retro['design_power']*100:.0f}%",
        xy=(retro["design_mde_abs"] * 100, 80),
        xytext=(retro["design_mde_abs"] * 100 + 0.1, 55),
        fontsize=9, color=BRAND_COLORS["success"],
        arrowprops=dict(arrowstyle="->", color=BRAND_COLORS["success"]),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=BRAND_COLORS["success"], alpha=0.8),
    )

    ax.annotate(
        f"Observed\neffect = {obs_abs*100:.3f}pp\n"
        f"Power = {obs_power:.1f}%",
        xy=(abs(obs_abs) * 100, obs_power),
        xytext=(abs(obs_abs) * 100 + 0.15, obs_power + 15),
        fontsize=9, color=BRAND_COLORS["treatment"],
        arrowprops=dict(arrowstyle="->", color=BRAND_COLORS["treatment"]),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=BRAND_COLORS["treatment"], alpha=0.8),
    )

    ax.set_xlabel("True Absolute Effect Size (percentage points)", fontsize=12)
    ax.set_ylabel("Statistical Power (%)", fontsize=12)
    ax.set_title(
        "Power Analysis Retrospective — Experiment 1 Checkout\n"
        "Phase 1 Pre-registered Design vs Phase 3 Observed Outcome",
        fontsize=12
    )
    ax.set_ylim(0, 105)
    ax.set_xlim(0, effects[-1] * 100)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fname = os.path.join(out_dir, "power_retrospective.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Written paragraph
# ─────────────────────────────────────────────────────────────────────────────

def write_power_retrospective(retro: dict) -> str:
    obs     = retro["observed_lift_abs"]
    mde     = retro["design_mde_abs"]
    ratio   = retro["observed_vs_mde_ratio"]
    pw_obs  = retro["power_for_observed"]
    n_need  = retro["n_needed_for_observed"]
    act_mde = retro["actual_mde_abs"]

    paragraphs = [
        "=" * 65,
        "POWER ANALYSIS RETROSPECTIVE — Experiment 1 Checkout",
        "ShopSmart India A/B Testing Platform",
        "=" * 65,
        "",
        "1. DESIGN vs OUTCOME COMPARISON",
        "-" * 40,
        f"  Phase 1 pre-registered the experiment to detect a minimum",
        f"  detectable effect (MDE) of {mde*100:.2f} percentage points (pp),",
        f"  which corresponds to a +{PRE_REG['mde_relative_pct']:.1f}% relative lift on the",
        f"  {PRE_REG['baseline_rate']*100:.1f}% baseline conversion rate.",
        f"  This was designed at 80% power and α=0.05, requiring",
        f"  {retro['design_n_per_group']:,} users per group ({retro['design_n_per_group']*2:,} total).",
        "",
        f"  Phase 3 analysis revealed an observed absolute lift of",
        f"  {obs*100:+.3f} pp — approximately {ratio:.2f}× the pre-registered MDE.",
        f"  The observed Cohen's h was {retro['observed_cohens_h']:.5f}, compared",
        f"  to the pre-registered h = {retro['design_cohens_h']:.4f}.",
        "",
        "2. WAS THE EXPERIMENT ADEQUATELY POWERED?",
        "-" * 40,
        f"  No. The experiment was powered for effects ≥ {mde*100:.2f} pp.",
        f"  For the effect we actually observed ({obs*100:+.3f} pp),",
        f"  the achieved statistical power was only {pw_obs*100:.1f}%.",
        f"  This means even if a true effect of this magnitude existed,",
        f"  the experiment had only a {pw_obs*100:.1f}% chance of detecting it,",
        f"  far below the 80% pre-registered target.",
        "",
        "3. WHAT WOULD WE HAVE MISSED?",
        "-" * 40,
        f"  If the true effect is {obs*100:+.3f} pp (as observed), the",
        f"  experiment is grossly underpowered to distinguish this from",
        f"  zero. To achieve 80% power for an effect this small would",
        f"  require approximately {n_need:,} users per group",
        f"  ({n_need*2:,} total) — about {n_need/retro['design_n_per_group']:.1f}× the",
        f"  pre-registered sample size.",
        "",
        f"  With the actual collected n of {retro['actual_n_per_group']:,}/group,",
        f"  the smallest detectable effect at 80% power is {act_mde*100:.3f} pp —",
        f"  which is {act_mde/mde:.2f}× the Phase 1 MDE.",
        "",
        "4. DESIGN LOOP CLOSURE",
        "-" * 40,
        f"  The null result (p=0.188, frequentist Phase 3) is consistent",
        f"  with two interpretations: (a) there is truly no effect, or",
        f"  (b) there is a small effect below our MDE that we cannot",
        f"  distinguish from noise. The Bayesian analysis (Part A) also",
        f"  finds P(treatment > control) ≈ 0.5, reinforcing (a).",
        "",
        f"  The practical implication: the UPI checkout redesign does not",
        f"  appear to move conversion by the economically meaningful",
        f"  +{mde*100:.2f} pp threshold. A future experiment targeting a",
        f"  smaller effect ({act_mde*100:.3f}–{mde*100:.2f} pp range) would require",
        f"  substantially more traffic or a longer run window than the",
        f"  {PRE_REG['min_run_days']}-day minimum set in Phase 1.",
        "",
        "=" * 65,
    ]
    return "\n".join(paragraphs)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_power_retrospective(ctrl_conv: int, ctrl_n: int,
                             treat_conv: int, treat_n: int,
                             out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "═"*60)
    print("  PART D — POWER ANALYSIS RETROSPECTIVE")
    print("═"*60)

    retro = compute_retrospective(ctrl_conv, ctrl_n, treat_conv, treat_n)

    print(f"\n  Phase 1 MDE (pre-registered): {retro['design_mde_abs']*100:.2f} pp")
    print(f"  Phase 3 observed lift       : {retro['observed_lift_abs']*100:+.4f} pp")
    print(f"  Observed vs MDE ratio       : {retro['observed_vs_mde_ratio']:.4f}×")
    print(f"  Power for observed effect   : {retro['power_for_observed']*100:.2f}%")
    print(f"  Adequately powered?         : {retro['adequately_powered']}")
    if retro["n_needed_for_observed"]:
        print(f"  N needed for observed effect: {retro['n_needed_for_observed']:,}/group")

    plot_power_retrospective(retro, out_dir)

    paragraph = write_power_retrospective(retro)
    txt_path  = os.path.join(out_dir, "power_retrospective.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(paragraph)
    print(f"\n  [output] Power retrospective → {txt_path}")
    print("\n" + paragraph)

    json_path = os.path.join(out_dir, "power_retrospective.json")
    with open(json_path, "w") as f:
        json.dump(retro, f, indent=2)
    print(f"  [output] Power retrospective JSON → {json_path}")

    return retro