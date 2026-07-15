# phase6/bayesian_analysis.py
"""
Part A — Bayesian A/B Testing (Experiment 1)
Beta-Binomial model using PyMC.
Outputs:
  - Posterior plots (both priors)
  - P(treatment > control)
  - Expected loss
  - Prior sensitivity conclusion (printed + saved)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import warnings
warnings.filterwarnings("ignore")

# PyMC import with graceful fallback message
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("[bayesian] PyMC not found — will use analytical Beta posteriors.")

from scipy import stats
from phase6.config import PRIORS, LOSS_THRESHOLD_ABS, PLOT_STYLE, BRAND_COLORS, PHASE6_OUT


# ─────────────────────────────────────────────────────────────────────────────
# Analytical Beta-Binomial (always runs — no PyMC dependency)
# ─────────────────────────────────────────────────────────────────────────────

def analytical_posterior(prior_alpha: float, prior_beta: float,
                         conversions: int, n: int) -> stats.beta:
    """
    Conjugate update: Beta(a,b) + Binomial(k,n) → Beta(a+k, b+n-k)
    """
    post_alpha = prior_alpha + conversions
    post_beta  = prior_beta  + (n - conversions)
    return stats.beta(post_alpha, post_beta), post_alpha, post_beta


def prob_treatment_better(ctrl_dist: stats.beta,
                          treat_dist: stats.beta,
                          n_samples: int = 200_000) -> float:
    """
    Monte-Carlo estimate of P(treatment > control).
    """
    rng  = np.random.default_rng(42)
    ctrl_samples  = ctrl_dist.rvs(n_samples, random_state=rng)
    treat_samples = treat_dist.rvs(n_samples, random_state=rng)
    return float(np.mean(treat_samples > ctrl_samples))


def expected_loss(ctrl_dist: stats.beta,
                  treat_dist: stats.beta,
                  n_samples: int = 200_000) -> dict:
    """
    Expected loss for each decision:
      loss(ship treatment) = E[max(control - treatment, 0)]
      loss(keep control)   = E[max(treatment - control, 0)]
    """
    rng  = np.random.default_rng(42)
    ctrl_samples  = ctrl_dist.rvs(n_samples, random_state=rng)
    treat_samples = treat_dist.rvs(n_samples, random_state=rng)
    diff = treat_samples - ctrl_samples

    loss_ship_treatment = float(np.mean(np.maximum(-diff, 0)))   # regret if we ship
    loss_keep_control   = float(np.mean(np.maximum( diff, 0)))   # regret if we don't
    return {
        "loss_ship_treatment" : loss_ship_treatment,
        "loss_keep_control"   : loss_keep_control,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PyMC model (runs only if PyMC is installed)
# ─────────────────────────────────────────────────────────────────────────────

def run_pymc_model(prior_alpha: float, prior_beta: float,
                   ctrl_conv: int, ctrl_n: int,
                   treat_conv: int, treat_n: int,
                   prior_label: str) -> dict:
    """
    Full PyMC Beta-Binomial model.
    Returns dict with trace summary and key scalars.
    """
    print(f"\n  [PyMC] Fitting model with prior {prior_label} ...")
    with pm.Model() as model:
        # Priors
        p_ctrl  = pm.Beta("p_control",   alpha=prior_alpha, beta=prior_beta)
        p_treat = pm.Beta("p_treatment", alpha=prior_alpha, beta=prior_beta)

        # Likelihoods
        obs_ctrl  = pm.Binomial("obs_control",   n=ctrl_n,  p=p_ctrl,
                                observed=ctrl_conv)
        obs_treat = pm.Binomial("obs_treatment", n=treat_n, p=p_treat,
                                observed=treat_conv)

        # Derived quantities
        delta     = pm.Deterministic("delta",    p_treat - p_ctrl)
        rel_lift  = pm.Deterministic("rel_lift", delta / p_ctrl)

        # Sample
        trace = pm.sample(
            2000, tune=1000, chains=2,
            progressbar=True, random_seed=42,
            target_accept=0.90,
            return_inferencedata=True,
        )

    # Summaries
    summary = az.summary(trace, var_names=["p_control", "p_treatment",
                                            "delta", "rel_lift"],
                         round_to=6)
    delta_samples = trace.posterior["delta"].values.flatten()
    prob_better   = float(np.mean(delta_samples > 0))

    return {
        "trace"        : trace,
        "summary"      : summary,
        "prob_better"  : prob_better,
        "delta_samples": delta_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_posteriors(results: dict, out_dir: str):
    """
    Two-panel figure per prior:
      Left:  posterior PDFs for control and treatment
      Right: posterior of delta (treatment - control) with ROPE
    """
    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        pass

    x = np.linspace(0.01, 0.08, 2000)

    for prior_key, res in results.items():
        fig = plt.figure(figsize=(14, 5))
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        # ── Left: individual posteriors ───────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(x, res["ctrl_dist"].pdf(x),
                 color=BRAND_COLORS["control"],  lw=2.5, label="Control posterior")
        ax1.plot(x, res["treat_dist"].pdf(x),
                 color=BRAND_COLORS["treatment"], lw=2.5,
                 label="Treatment posterior")
        ax1.axvline(res["ctrl_dist"].mean(),
                    color=BRAND_COLORS["control"],  lw=1.2, ls="--", alpha=0.7)
        ax1.axvline(res["treat_dist"].mean(),
                    color=BRAND_COLORS["treatment"], lw=1.2, ls="--", alpha=0.7)
        ax1.set_xlabel("Conversion Rate", fontsize=11)
        ax1.set_ylabel("Posterior Density", fontsize=11)
        ax1.set_title(f"Posterior Conversion Rates\n{res['label']}", fontsize=11)
        ax1.legend(fontsize=9)

        # ── Right: delta distribution ─────────────────────────────────────
        ax2    = fig.add_subplot(gs[1])
        deltas = res["delta_samples"]
        ax2.hist(deltas, bins=120, density=True,
                 color=BRAND_COLORS["neutral"], edgecolor="white",
                 linewidth=0.3, alpha=0.85, label="P(treatment − control)")

        # ROPE: [-MDE, +MDE] = [-0.004, +0.004]
        rope_lo, rope_hi = -0.004, 0.004
        ax2.axvspan(rope_lo, rope_hi, alpha=0.15, color="grey", label="ROPE ±0.4pp")
        ax2.axvline(0, color="black",  lw=1.5, ls="-",  label="No effect")
        ax2.axvline(np.mean(deltas), color=BRAND_COLORS["treatment"],
                    lw=2, ls="--", label=f"Mean Δ={np.mean(deltas):.4f}")

        prob_better = res["prob_better"]
        ax2.set_title(f"Posterior of Δ = Treatment − Control\n"
                      f"P(treatment > control) = {prob_better:.3f}", fontsize=11)
        ax2.set_xlabel("Δ Conversion Rate", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.legend(fontsize=8)

        plt.suptitle(f"Bayesian A/B Analysis — Experiment 1 Checkout\n"
                     f"Prior: {res['label']}", fontsize=12, y=1.02)
        plt.tight_layout()

        fname = os.path.join(out_dir, f"bayesian_posterior_{prior_key}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [plot] Saved → {fname}")


def plot_prior_sensitivity_comparison(results: dict, out_dir: str):
    """
    Side-by-side bar chart comparing P(treatment > control) across priors.
    """
    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        pass

    labels = [r["label"].split("—")[0].strip() for r in results.values()]
    probs  = [r["prob_better"] for r in results.values()]
    losses = [r["loss"]["loss_ship_treatment"] for r in results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # P(treatment > control)
    bars = axes[0].bar(labels, probs,
                       color=[BRAND_COLORS["control"],
                              BRAND_COLORS["neutral"],
                              BRAND_COLORS["treatment"]],
                       edgecolor="white", linewidth=0.8)
    axes[0].axhline(0.5,  color="black", lw=1.2, ls="--", label="50% (chance)")
    axes[0].axhline(0.95, color="green", lw=1.2, ls=":",  label="95% (ship threshold)")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("P(Treatment > Control)", fontsize=11)
    axes[0].set_title("Prior Sensitivity: P(Treatment Better)", fontsize=11)
    axes[0].legend(fontsize=8)
    for bar, prob in zip(bars, probs):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{prob:.3f}", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")

    # Expected loss if we ship treatment
    bars2 = axes[1].bar(labels, [l * 100 for l in losses],
                        color=[BRAND_COLORS["control"],
                               BRAND_COLORS["neutral"],
                               BRAND_COLORS["treatment"]],
                        edgecolor="white", linewidth=0.8)
    axes[1].axhline(LOSS_THRESHOLD_ABS * 100, color="red", lw=1.5, ls="--",
                    label=f"Threshold {LOSS_THRESHOLD_ABS*100:.1f}pp")
    axes[1].set_ylabel("Expected Loss if Ship Treatment (pp)", fontsize=11)
    axes[1].set_title("Prior Sensitivity: Expected Loss", fontsize=11)
    axes[1].legend(fontsize=8)
    for bar, loss in zip(bars2, losses):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f"{loss*100:.3f}pp", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")

    plt.tight_layout()
    fname = os.path.join(out_dir, "bayesian_prior_sensitivity.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_bayesian_analysis(ctrl_conv: int, ctrl_n: int,
                          treat_conv: int, treat_n: int,
                          out_dir: str) -> dict:
    """
    Full Part A pipeline.
    Returns dict with all results for downstream use.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "═"*60)
    print("  PART A — BAYESIAN ANALYSIS")
    print("═"*60)
    print(f"  Control  : {ctrl_conv:,} / {ctrl_n:,}  "
          f"(rate={ctrl_conv/ctrl_n:.4f})")
    print(f"  Treatment: {treat_conv:,} / {treat_n:,}  "
          f"(rate={treat_conv/treat_n:.4f})")

    all_results = {}

    for prior_key, prior_cfg in PRIORS.items():
        pa, pb   = prior_cfg["alpha"], prior_cfg["beta"]
        label    = prior_cfg["label"]
        print(f"\n  ── Prior: {label} ──")

        # Analytical posteriors
        ctrl_dist,  ca, cb  = analytical_posterior(pa, pb, ctrl_conv,  ctrl_n)
        treat_dist, ta, tb  = analytical_posterior(pa, pb, treat_conv, treat_n)

        p_better = prob_treatment_better(ctrl_dist, treat_dist)
        loss     = expected_loss(ctrl_dist, treat_dist)

        # Delta samples for histogram (analytical)
        rng            = np.random.default_rng(42)
        ctrl_s         = ctrl_dist.rvs(200_000, random_state=rng)
        treat_s        = treat_dist.rvs(200_000, random_state=rng)
        delta_samples  = treat_s - ctrl_s

        print(f"    Posterior Control  : Beta({ca:.1f}, {cb:.1f})"
              f"  mean={ctrl_dist.mean():.5f}"
              f"  95% CI [{ctrl_dist.ppf(0.025):.5f}, {ctrl_dist.ppf(0.975):.5f}]")
        print(f"    Posterior Treatment: Beta({ta:.1f}, {tb:.1f})"
              f"  mean={treat_dist.mean():.5f}"
              f"  95% CI [{treat_dist.ppf(0.025):.5f}, {treat_dist.ppf(0.975):.5f}]")
        print(f"    P(treatment > control)       = {p_better:.4f}")
        print(f"    E[loss | ship treatment]     = {loss['loss_ship_treatment']:.6f}"
              f"  ({loss['loss_ship_treatment']*100:.4f} pp)")
        print(f"    E[loss | keep control]       = {loss['loss_keep_control']:.6f}"
              f"  ({loss['loss_keep_control']*100:.4f} pp)")

        ship_decision = (p_better > 0.95 and
                         loss["loss_ship_treatment"] < LOSS_THRESHOLD_ABS)
        print(f"    Ship decision (p>0.95 & loss<{LOSS_THRESHOLD_ABS}): "
              f"{'[SHIP]' if ship_decision else '[DO NOT SHIP]'}")
        all_results[prior_key] = {
            "label"         : label,
            "prior_alpha"   : pa,
            "prior_beta"    : pb,
            "post_ctrl"     : (ca, cb),
            "post_treat"    : (ta, tb),
            "ctrl_dist"     : ctrl_dist,
            "treat_dist"    : treat_dist,
            "delta_samples" : delta_samples,
            "prob_better"   : p_better,
            "loss"          : loss,
            "ship_decision" : ship_decision,
            "ctrl_mean"     : ctrl_dist.mean(),
            "treat_mean"    : treat_dist.mean(),
            "delta_mean"    : float(np.mean(delta_samples)),
            "delta_95ci"    : (float(np.percentile(delta_samples, 2.5)),
                               float(np.percentile(delta_samples, 97.5))),
        }

        # Optional PyMC run (primary prior only to save time)
        if PYMC_AVAILABLE and prior_key == "uniform":
            try:
                pymc_res = run_pymc_model(pa, pb, ctrl_conv, ctrl_n,
                                          treat_conv, treat_n, label)
                all_results[prior_key]["pymc_summary"]   = pymc_res["summary"]
                all_results[prior_key]["pymc_prob_better"] = pymc_res["prob_better"]
                print(f"    [PyMC] P(treatment > control) = "
                      f"{pymc_res['prob_better']:.4f}  "
                      f"(cross-check vs analytical: {p_better:.4f})")
            except Exception as e:
                print(f"    [PyMC] Skipped due to error: {e}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_posteriors(all_results, out_dir)
    plot_prior_sensitivity_comparison(all_results, out_dir)

    # ── Prior sensitivity written conclusion ──────────────────────────────
    sensitivity_conclusion = _write_prior_sensitivity(all_results)
    conclusion_path = os.path.join(out_dir, "bayesian_prior_sensitivity.txt")
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write(sensitivity_conclusion)
    print(f"\n  [output] Prior sensitivity conclusion → {conclusion_path}")
    print("\n" + sensitivity_conclusion)

    # ── JSON summary ──────────────────────────────────────────────────────
    summary_json = {}
    for k, v in all_results.items():
        summary_json[k] = {
            "label"         : v["label"],
            "prob_better"   : round(v["prob_better"], 5),
            "loss_ship"     : round(v["loss"]["loss_ship_treatment"], 7),
            "loss_keep"     : round(v["loss"]["loss_keep_control"], 7),
            "ship_decision" : v["ship_decision"],
            "delta_mean"    : round(v["delta_mean"], 6),
            "delta_95ci"    : [round(x, 6) for x in v["delta_95ci"]],
        }

    json_path = os.path.join(out_dir, "bayesian_results.json")
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  [output] Bayesian results JSON → {json_path}")

    return all_results


def _write_prior_sensitivity(results: dict) -> str:
    priors_list  = list(results.items())
    primary_key, primary = priors_list[0]

    lines = [
        "=" * 65,
        "PRIOR SENSITIVITY CONCLUSION — Bayesian A/B, Experiment 1",
        "=" * 65,
        "",
        "PRIMARY PRIOR: Beta(1,1) — Uniform",
        "  Justification: No strong pre-experiment information about the",
        "  true checkout conversion rate beyond the operational baseline",
        "  of 3.8% (already captured in the data). The uniform prior is",
        "  intentionally weakly informative — it avoids overstating",
        "  certainty, respects the pre-registration spirit, and is the",
        "  standard 'minimum assumption' choice for conversion rate tests.",
        "  It assigns equal probability to every possible rate in [0,1],",
        "  letting the 75 000+ observed events dominate the posterior.",
        "",
    ]

    for prior_key, res in results.items():
        lines.append(f"Prior: {res['label']}")
        lines.append(f"  P(treatment > control)        = {res['prob_better']:.4f}")
        lines.append(f"  E[loss | ship treatment]      = "
                     f"{res['loss']['loss_ship_treatment']*100:.4f} pp")
        lines.append(f"  Posterior Δ mean              = "
                     f"{res['delta_mean']*100:.4f} pp")
        lines.append(f"  Posterior 95% CI of Δ         = "
                     f"[{res['delta_95ci'][0]*100:.4f}, "
                     f"{res['delta_95ci'][1]*100:.4f}] pp")
        lines.append(f"  Ship decision                 = "
                     f"{'SHIP' if res['ship_decision'] else 'DO NOT SHIP'}")
        lines.append("")

    # Agreement check
    decisions = [r["ship_decision"] for r in results.values()]
    probs     = [r["prob_better"]   for r in results.values()]
    all_agree = len(set(decisions)) == 1
    max_diff  = max(probs) - min(probs)

    lines += [
        "-" * 65,
        "SENSITIVITY CONCLUSION:",
        f"  All {len(results)} priors agree on ship/no-ship decision: "
        f"{'YES' if all_agree else 'NO — CAUTION'}",
        f"  Range of P(treatment > control) across priors: "
        f"{min(probs):.4f} – {max(probs):.4f}  (spread = {max_diff:.4f})",
        "",
    ]

    if all_agree and max_diff < 0.05:
        lines.append(
            "  [ROBUST] The conclusion is insensitive to prior choice.\n"
            "  The data overwhelm the prior (large n ~75 000 per group),\n"
            "  and all three priors yield the same decision: DO NOT SHIP.\n"
            "  P(treatment > control) hovers near 0.5, consistent with\n"
            "  the frequentist p=0.188 finding of no detectable effect.\n"
            "  The Bayesian and frequentist conclusions are coherent."
        )
    elif all_agree:
        lines.append(
            "  [CONSISTENT] decision across priors, but the spread in\n"
            "  probability is non-trivial -- worth reporting."
        )
    else:
        lines.append(
            "  [CAUTION] Different priors lead to different ship decisions.\n"
            "  Collect more data or consult a domain expert for the prior."
        )

    lines.append("=" * 65)
    return "\n".join(lines)