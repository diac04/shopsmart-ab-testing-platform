# phase6/phase6_run.py
"""
Phase 6 Master Runner — ShopSmart India A/B Platform
Runs all four sub-analyses in sequence:
  A. Bayesian Analysis
  B. Sequential Testing
  C. Novelty Effect
  D. Power Retrospective
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import sys
import os
import json
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase6.config          import PHASE6_OUT
from phase6.data_loader     import load_experiment1, get_group_arrays, get_daily_rates
from phase6.bayesian_analysis   import run_bayesian_analysis
from phase6.sequential_testing  import run_sequential_analysis
from phase6.novelty_effect      import run_novelty_analysis
from phase6.power_retrospective import run_power_retrospective


def main():
    print("\n" + "█"*65)
    print("  PHASE 6 — ADVANCED STATISTICAL LAYER")
    print("  ShopSmart India A/B Testing Platform")
    print("█"*65)

    start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading Experiment 1 data ...")
    data = load_experiment1()
    ctrl_conv, ctrl_n, treat_conv, treat_n = get_group_arrays(data)

    out = PHASE6_OUT
    os.makedirs(out, exist_ok=True)

    # ── Part A: Bayesian ──────────────────────────────────────────────────
    print("\n[2/5] Running Part A: Bayesian Analysis ...")
    bayes_results = run_bayesian_analysis(
        ctrl_conv, ctrl_n, treat_conv, treat_n, out
    )

    # ── Part B: Sequential ────────────────────────────────────────────────
    print("\n[3/5] Running Part B: Sequential Testing ...")
    seq_results = run_sequential_analysis(data, out)

    # ── Part C: Novelty effect ────────────────────────────────────────────
    print("\n[4/5] Running Part C: Novelty Effect Analysis ...")
    novelty_results = run_novelty_analysis(data, out)

    # ── Part D: Power retrospective ───────────────────────────────────────
    print("\n[5/5] Running Part D: Power Retrospective ...")
    retro_results = run_power_retrospective(
        ctrl_conv, ctrl_n, treat_conv, treat_n, out
    )

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start

    print("\n" + "═"*65)
    print("  PHASE 6 COMPLETE — OUTPUT SUMMARY")
    print("═"*65)

    outputs = [
        ("bayesian_posterior_uniform.png",        "Posterior plot (primary prior)"),
        ("bayesian_posterior_weakly.png",         "Posterior plot (sensitivity prior)"),
        ("bayesian_posterior_informed.png",       "Posterior plot (informed prior)"),
        ("bayesian_prior_sensitivity.png",        "Prior sensitivity comparison"),
        ("bayesian_prior_sensitivity.txt",        "Prior sensitivity conclusion"),
        ("bayesian_results.json",                 "Bayesian results JSON"),
        ("sequential_testing.png",               "Peeking vs sequential chart"),
        ("obrien_fleming_boundaries.csv",         "O'Brien-Fleming boundaries"),
        ("sequential_results.json",              "Sequential testing JSON"),
        ("novelty_effect.png",                   "Novelty effect chart"),
        ("daily_conversion_rates.csv",            "Daily conversion rates"),
        ("novelty_results.json",                  "Novelty analysis JSON"),
        ("power_retrospective.png",               "Power retrospective chart"),
        ("power_retrospective.txt",               "Power retrospective paragraph"),
        ("power_retrospective.json",              "Power retrospective JSON"),
    ]

    all_present = True
    for fname, desc in outputs:
        fpath  = os.path.join(out, fname)
        exists = os.path.exists(fpath)
        status = "✓" if exists else "✗ MISSING"
        if not exists:
            all_present = False
        print(f"  {status}  {fname:<45} {desc}")

    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"  All outputs in: {out}")
    print(f"  All files present: {'✓ YES' if all_present else '✗ CHECK ABOVE'}")

    # ── Key numbers printout ──────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  KEY NUMBERS AT A GLANCE")
    print("─"*65)

    # Bayesian
    primary = bayes_results.get("uniform", {})
    print(f"  [Bayesian] P(treatment > control) [uniform prior] : "
          f"{primary.get('prob_better', 'N/A'):.4f}")
    print(f"  [Bayesian] E[loss | ship treatment]               : "
          f"{primary.get('loss', {}).get('loss_ship_treatment', 0)*100:.5f} pp")
    print(f"  [Bayesian] Ship decision                          : "
          f"{'SHIP' if primary.get('ship_decision') else 'DO NOT SHIP'}")

    # Sequential
    cod = seq_results.get("cost_of_delay", {})
    print(f"  [Sequential] Inflated FPR (peeking)               : "
          f"{seq_results.get('peeking_fpr', 0)*100:.1f}%  (nominal 5%)")
    print(f"  [Sequential] Cost-of-delay ₹ figure               : "
          f"₹{cod.get('total_inr_saved', 0):,.0f}  "
          f"({cod.get('days_saved', 0):.1f} days × "
          f"₹{cod.get('daily_revenue_impact_inr', 0):,.0f}/day)")

    # Novelty
    print(f"  [Novelty] Novelty magnitude                       : "
          f"{novelty_results.get('novelty_magnitude_abs', 0)*100:.4f} pp")
    print(f"  [Novelty] Novelty statistically significant       : "
          f"{novelty_results.get('novelty_significant', False)}")
    sl = novelty_results.get("steady_state_lift")
    if sl is not None:
        print(f"  [Novelty] Steady-state lift                       : "
              f"{sl*100:+.4f} pp")

    # Power
    print(f"  [Power] Power for observed effect                 : "
          f"{retro_results.get('power_for_observed', 0)*100:.1f}%")
    print(f"  [Power] Adequately powered for observed effect?   : "
          f"{retro_results.get('adequately_powered', False)}")
    nn = retro_results.get("n_needed_for_observed")
    if nn:
        print(f"  [Power] N needed to detect observed effect @ 80%  : "
              f"{nn:,}/group")

    print("─"*65 + "\n")


if __name__ == "__main__":
    main()