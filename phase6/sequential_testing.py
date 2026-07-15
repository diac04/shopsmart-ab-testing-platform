# phase6/sequential_testing.py
"""
Part B — Sequential Testing
1. Naive peeking simulation → inflated FPR
2. SPRT (Wald's sequential probability ratio test)
3. O'Brien-Fleming alpha-spending boundaries
4. Cost-of-delay analysis in ₹
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import json
import os
from scipy import stats

from phase6.config import (SEQUENTIAL, BUSINESS, PRE_REG,
                            PLOT_STYLE, BRAND_COLORS, PHASE6_OUT)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Naive peeking simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_peeking_fpr(n_simulations: int,
                         peek_every: int,
                         max_n: int,
                         alpha: float,
                         p0: float,
                         seed: int = 42) -> dict:
    """
    Under H0 (no difference, both groups have rate p0),
    simulate repeated peeking and record whether any peek
    yields p < alpha.  Computes the inflated FPR.
    """
    rng = np.random.default_rng(seed)
    false_positives = 0
    peek_counts     = []

    for _ in range(n_simulations):
        ctrl_conv  = 0
        treat_conv = 0
        found_sig  = False

        for n_so_far in range(peek_every, max_n + peek_every, peek_every):
            # Draw incremental observations
            n_new     = peek_every
            ctrl_conv  += rng.binomial(n_new, p0)
            treat_conv += rng.binomial(n_new, p0)
            n_total    = n_so_far

            # Two-proportion z-test
            p1 = ctrl_conv  / n_total
            p2 = treat_conv / n_total
            p_pool = (ctrl_conv + treat_conv) / (2 * n_total)

            if p_pool in (0, 1):
                continue
            se = np.sqrt(2 * p_pool * (1 - p_pool) / n_total)
            if se == 0:
                continue
            z  = (p2 - p1) / se
            pv = 2 * (1 - stats.norm.cdf(abs(z)))

            if pv < alpha:
                false_positives += 1
                found_sig        = True
                peek_counts.append(n_so_far)
                break

        if not found_sig:
            peek_counts.append(max_n)

    inflated_fpr = false_positives / n_simulations
    return {
        "inflated_fpr"   : inflated_fpr,
        "n_simulations"  : n_simulations,
        "peek_every"     : peek_every,
        "nominal_alpha"  : alpha,
        "peek_counts"    : peek_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPRT — Wald's Sequential Probability Ratio Test
# ─────────────────────────────────────────────────────────────────────────────

def run_sprt(data: pd.DataFrame,
             p0: float,
             p1: float,
             alpha: float,
             beta: float) -> dict:
    """
    SPRT for two proportions (binomial).
    H0: p = p0  vs  H1: p = p1  (treatment group only, relative to control)

    We compute cumulative log-likelihood ratio as data arrives.
    Boundaries: A = (1-beta)/alpha  (upper, reject H0)
                B = beta/(1-alpha)  (lower, accept H0)

    Returns dict with stopping info and full LLR trajectory.
    """
    A = np.log((1 - beta) / alpha)   # upper boundary (reject H0 → ship)
    B = np.log(beta / (1 - alpha))   # lower boundary (accept H0 → no ship)

    # Use treatment group rows sorted by day then sequential
    ctrl  = data[data["group"] == "control"].copy().reset_index(drop=True)
    treat = data[data["group"] == "treatment"].copy().reset_index(drop=True)

    # Pair observations sequentially
    n_pairs = min(len(ctrl), len(treat))

    llr_trajectory = []
    cumulative_llr = 0.0
    stopped_at     = None
    decision       = "continue"

    for i in range(n_pairs):
        c_obs = int(ctrl.iloc[i]["converted"])
        t_obs = int(treat.iloc[i]["converted"])

        # Log-likelihood ratio for one paired observation
        # LLR += log[ P(obs|H1) / P(obs|H0) ] for treatment arm
        def log_binom_pmf(k, p):
            if p <= 0 or p >= 1:
                return 0.0
            return k * np.log(p) + (1 - k) * np.log(1 - p)

        llr_step = (log_binom_pmf(t_obs, p1) - log_binom_pmf(t_obs, p0))
        cumulative_llr += llr_step
        llr_trajectory.append(cumulative_llr)

        if cumulative_llr >= A:
            stopped_at = i + 1
            decision   = "reject_H0_ship"
            break
        elif cumulative_llr <= B:
            stopped_at = i + 1
            decision   = "accept_H0_no_ship"
            break

    if stopped_at is None:
        stopped_at = n_pairs
        decision   = "inconclusive"

    return {
        "A"              : A,
        "B"              : B,
        "stopped_at_n"   : stopped_at,
        "decision"       : decision,
        "llr_trajectory" : llr_trajectory,
        "n_total"        : n_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. O'Brien-Fleming boundaries
# ─────────────────────────────────────────────────────────────────────────────

def obrien_fleming_boundaries(n_looks: int,
                               total_n: int,
                               alpha: float = 0.05) -> pd.DataFrame:
    """
    Approximate O'Brien-Fleming alpha-spending boundaries.
    Uses the canonical approximation: z_k ≈ z_alpha/2 * sqrt(K/k)
    where K = total looks, k = current look.

    Returns DataFrame with look number, sample size at look,
    critical z, critical p-value.
    """
    rows = []
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)

    for k in range(1, n_looks + 1):
        n_at_look = int(total_n * k / n_looks)
        z_k       = z_alpha2 * np.sqrt(n_looks / k)
        p_k       = 2 * (1 - stats.norm.cdf(z_k))
        rows.append({
            "look"      : k,
            "n_at_look" : n_at_look,
            "fraction"  : round(k / n_looks, 3),
            "z_boundary": round(z_k, 4),
            "p_boundary": round(p_k, 6),
            "alpha_spent": round(p_k, 6),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cost-of-delay analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost_of_delay(sprt_result: dict,
                           total_n_planned: int,
                           daily_visitors: int,
                           daily_revenue_impact_inr: float) -> dict:
    """
    Convert days saved (SPRT early stop vs full run) into ₹ value.
    """
    n_at_stop    = sprt_result["stopped_at_n"]
    n_full       = total_n_planned

    # Days at SPRT stop vs full run (per group)
    visitors_per_day_per_group = daily_visitors / 2
    days_at_sprt = n_at_stop / visitors_per_day_per_group
    days_full    = n_full    / visitors_per_day_per_group

    days_saved   = max(days_full - days_at_sprt, 0)
    inr_saved    = days_saved * daily_revenue_impact_inr

    return {
        "n_at_sprt_stop"         : n_at_stop,
        "n_full_run"             : n_full,
        "days_at_sprt"           : round(days_at_sprt, 2),
        "days_full_run"          : round(days_full, 2),
        "days_saved"             : round(days_saved, 2),
        "daily_revenue_impact_inr": daily_revenue_impact_inr,
        "total_inr_saved"        : round(inr_saved, 0),
        "decision"               : sprt_result["decision"],
        "note": (
            "Cost-of-delay is hypothetical: it assumes the treatment "
            "had achieved the pre-registered MDE (+0.4pp). In reality, "
            "Phase 3 showed near-zero lift, so SPRT would NOT stop early "
            "under H1 — it would accumulate evidence toward H0."
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_peeking_vs_sequential(peeking_result: dict,
                                sprt_result: dict,
                                ob_df: pd.DataFrame,
                                out_dir: str):
    """
    Three-panel figure:
      Top-left  : Peeking FPR vs number of peeks
      Top-right : SPRT LLR trajectory with stop boundaries
      Bottom    : O'Brien-Fleming boundary p-values across looks
    """
    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        pass

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel 1: Peeking FPR illustration ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    # Show FPR as function of max peeks (re-simulate for different n_peeks)
    peek_counts_range = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    fprs = []
    rng  = np.random.default_rng(99)
    p0   = SEQUENTIAL["h0_rate"]
    n_sim_quick = 2_000   # quick simulation for plot

    for n_peeks_max in peek_counts_range:
        fp = 0
        max_n_per_arm = PRE_REG["n_per_group"]
        peek_every    = max_n_per_arm // n_peeks_max
        peek_every    = max(peek_every, 1)
        for _ in range(n_sim_quick):
            ctrl_c = treat_c = 0
            sig = False
            for k in range(1, n_peeks_max + 1):
                ctrl_c  += rng.binomial(peek_every, p0)
                treat_c += rng.binomial(peek_every, p0)
                n_now    = k * peek_every
                pp       = (ctrl_c + treat_c) / (2 * n_now)
                if 0 < pp < 1:
                    se  = np.sqrt(2 * pp * (1 - pp) / n_now)
                    if se > 0:
                        z  = abs(treat_c/n_now - ctrl_c/n_now) / se
                        pv = 2 * (1 - stats.norm.cdf(z))
                        if pv < SEQUENTIAL["alpha"]:
                            fp  += 1
                            sig  = True
                            break
        fprs.append(fp / n_sim_quick)

    ax1.plot(peek_counts_range, fprs,
             color=BRAND_COLORS["treatment"], marker="o", lw=2.5,
             label="Peeking FPR")
    ax1.axhline(SEQUENTIAL["alpha"], color="black",
                lw=1.5, ls="--", label=f"Nominal α={SEQUENTIAL['alpha']}")
    ax1.fill_between(peek_counts_range, SEQUENTIAL["alpha"],
                     fprs, alpha=0.15, color=BRAND_COLORS["treatment"])
    ax1.set_xlabel("Number of Interim Peeks", fontsize=11)
    ax1.set_ylabel("False Positive Rate", fontsize=11)
    ax1.set_title("Naive Peeking Inflates False Positive Rate\n"
                  "(Under H₀: no true difference)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 0.5)

    # ── Panel 2: SPRT trajectory ──────────────────────────────────────────
    ax2  = fig.add_subplot(gs[0, 1])
    traj = sprt_result["llr_trajectory"]
    n_pts = len(traj)
    xs    = list(range(1, n_pts + 1))

    ax2.plot(xs, traj, color=BRAND_COLORS["neutral"], lw=1.5,
             label="Cumulative LLR", alpha=0.9)
    ax2.axhline(sprt_result["A"], color=BRAND_COLORS["success"],
                lw=2, ls="--", label=f"Upper (ship): A={sprt_result['A']:.2f}")
    ax2.axhline(sprt_result["B"], color=BRAND_COLORS["treatment"],
                lw=2, ls="--", label=f"Lower (no-ship): B={sprt_result['B']:.2f}")
    ax2.axhline(0, color="grey", lw=1, ls=":")

    stop_n = sprt_result["stopped_at_n"]
    if stop_n <= n_pts:
        ax2.axvline(stop_n, color="purple", lw=1.5, ls=":",
                    label=f"Stop at n={stop_n:,}")

    ax2.set_xlabel("Observations (per group)", fontsize=11)
    ax2.set_ylabel("Log-Likelihood Ratio", fontsize=11)
    ax2.set_title(f"SPRT Trajectory\nDecision: {sprt_result['decision']}",
                  fontsize=11)
    ax2.legend(fontsize=8)

    # ── Panel 3: O'Brien-Fleming boundaries ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.step(ob_df["look"], ob_df["p_boundary"],
             color=BRAND_COLORS["control"], lw=2.5, where="mid",
             label="O'Brien-Fleming boundary p-value")
    ax3.axhline(SEQUENTIAL["alpha"], color="black",
                lw=1.5, ls="--", label=f"Nominal α={SEQUENTIAL['alpha']}")
    ax3.fill_between(ob_df["look"], ob_df["p_boundary"],
                     SEQUENTIAL["alpha"], alpha=0.15,
                     color=BRAND_COLORS["control"], step="mid")
    for _, row in ob_df.iterrows():
        ax3.annotate(f"z={row['z_boundary']:.2f}",
                     (row["look"], row["p_boundary"]),
                     textcoords="offset points", xytext=(0, 8),
                     fontsize=8, ha="center")
    ax3.set_xlabel("Interim Look Number", fontsize=11)
    ax3.set_ylabel("Alpha-Spending Boundary (p-value)", fontsize=11)
    ax3.set_title("O'Brien-Fleming Alpha-Spending Boundaries\n"
                  "(5 planned looks)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.set_xticks(ob_df["look"])

    # ── Panel 4: Summary table ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    table_data = [
        ["Method",            "FPR / Control",     "When stops"],
        ["Naive α=0.05",      f"~{fprs[-1]:.2%} (inflated)",
                              "At arbitrary peek"],
        ["SPRT",              "≤ α by design",
                              f"n={sprt_result['stopped_at_n']:,}"],
        ["O'B-F (look 1)",    f"p={ob_df.iloc[0]['p_boundary']:.5f}",
                              f"n={ob_df.iloc[0]['n_at_look']:,}"],
        ["O'B-F (final)",     f"p={ob_df.iloc[-1]['p_boundary']:.5f}",
                              f"n={ob_df.iloc[-1]['n_at_look']:,}"],
    ]

    tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc="center", loc="center",
                    bbox=[0.0, 0.2, 1.0, 0.7])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(BRAND_COLORS["control"])
            cell.set_text_props(color="white", fontweight="bold")
    ax4.set_title("Peeking vs Sequential: Summary",
                  fontsize=11, pad=10)

    plt.suptitle("Sequential Testing Analysis — Experiment 1 Checkout\n"
                 "ShopSmart India A/B Platform", fontsize=13, y=1.01)

    fname = os.path.join(out_dir, "sequential_testing.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_sequential_analysis(data: pd.DataFrame, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "═"*60)
    print("  PART B — SEQUENTIAL TESTING")
    print("═"*60)

    cfg = SEQUENTIAL
    biz = BUSINESS

    # 1. Peeking simulation
    print("\n  1. Simulating naive peeking FPR ...")
    peek_res = simulate_peeking_fpr(
        n_simulations = cfg["n_simulations"],
        peek_every    = cfg["peek_every"],
        max_n         = PRE_REG["n_per_group"],
        alpha         = cfg["alpha"],
        p0            = cfg["h0_rate"],
    )
    print(f"     Nominal α            = {cfg['alpha']:.3f}")
    print(f"     Inflated FPR (peeking)= {peek_res['inflated_fpr']:.4f}  "
          f"({peek_res['inflated_fpr']*100:.1f}%)")
    print(f"     Peek frequency       = every {cfg['peek_every']:,} observations")

    # 2. SPRT
    print("\n  2. Running SPRT on actual Experiment 1 data ...")
    sprt_res = run_sprt(
        data  = data,
        p0    = cfg["h0_rate"],
        p1    = cfg["h0_rate"] + cfg["h1_lift"],
        alpha = cfg["alpha"],
        beta  = cfg["beta"],
    )
    print(f"     SPRT upper boundary A = {sprt_res['A']:.4f}")
    print(f"     SPRT lower boundary B = {sprt_res['B']:.4f}")
    print(f"     Stopped at n          = {sprt_res['stopped_at_n']:,} "
          f"(of {sprt_res['n_total']:,})")
    print(f"     Decision              = {sprt_res['decision']}")

    # 3. O'Brien-Fleming
    print("\n  3. Computing O'Brien-Fleming boundaries ...")
    ob_df = obrien_fleming_boundaries(
        n_looks  = cfg["ob_flemming_looks"],
        total_n  = PRE_REG["n_per_group"],
        alpha    = cfg["alpha"],
    )
    print(ob_df.to_string(index=False))
    ob_path = os.path.join(out_dir, "obrien_fleming_boundaries.csv")
    ob_df.to_csv(ob_path, index=False)
    print(f"     Saved → {ob_path}")

    # 4. Cost-of-delay
    print("\n  4. Cost-of-delay analysis ...")
    cod = compute_cost_of_delay(
        sprt_result              = sprt_res,
        total_n_planned          = PRE_REG["n_per_group"],
        daily_visitors           = biz["daily_visitors"],
        daily_revenue_impact_inr = biz["daily_revenue_impact_inr"],
    )
    print(f"     Days at SPRT stop    = {cod['days_at_sprt']:.1f}")
    print(f"     Days full planned run= {cod['days_full_run']:.1f}")
    print(f"     Days saved           = {cod['days_saved']:.1f}")
    print(f"     ₹ saved/earned sooner= ₹{cod['total_inr_saved']:,.0f}")
    print(f"     NOTE: {cod['note']}")

    # Plots
    plot_peeking_vs_sequential(peek_res, sprt_res, ob_df, out_dir)

    # Save JSON
    output = {
        "peeking_fpr"     : round(peek_res["inflated_fpr"], 5),
        "nominal_alpha"   : cfg["alpha"],
        "sprt"            : {k: v for k, v in sprt_res.items()
                              if k != "llr_trajectory"},
        "obrien_fleming"  : ob_df.to_dict(orient="records"),
        "cost_of_delay"   : cod,
    }
    json_path = os.path.join(out_dir, "sequential_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [output] Sequential results JSON → {json_path}")

    return output