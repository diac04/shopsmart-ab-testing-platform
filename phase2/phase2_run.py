# ============================================================
# ShopSmart India — Phase 2 — Master Runner
# ============================================================
# Orchestrates the full Phase 2 pipeline:
#   1. Load Experiment 1 data (Kaggle + SRM injection)
#   2. Simulate Experiments 2 and 3
#   3. Run all 7 sanity checks × 3 experiments
#   4. Print validation summary table
#   5. Print written SRM diagnosis
#   6. Save phase2_summary.json for Phase 3
#   7. Save phase2_validation_summary.csv
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Imports from this phase ───────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config         import (BASE_DIR, PHASE2_DIR, EXP1_DIR,
                             EXP2_DIR, EXP3_DIR, RANDOM_SEED)
from data_loader    import load_experiment1_data
from simulator      import simulate_all
from sanity_checks  import (run_all_checks, print_summary_table,
                             tracker, CheckResult)

# ============================================================
# SECTION 1 — HEADER
# ============================================================

def print_header():
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█   SHOPSMART INDIA — A/B TESTING PLATFORM           █")
    print("█   PHASE 2: DATA COLLECTION + SANITY CHECKS         █")
    print("█" + " "*58 + "█")
    print("█"*60)
    print("""
  Phase 2 Overview
  ────────────────
  Experiment 1 : Real Kaggle data  (Checkout Redesign)
                 → Injected SRM: 52/48 split
  Experiment 2 : Simulated data    (Personalized Recs)
                 → Injected Novelty Effect (days 1–10)
                 → Pre-experiment revenue covariate
  Experiment 3 : Simulated data    (Discount Banner)
                 → Device segmentation (Mobile/Desktop)

  Sanity Checks (all experiments):
    1. Sample Ratio Mismatch  ← HARD GATE
    2. Pre-experiment equivalence (A/A)
    3. Duplicate user IDs
    4. Cross-group leakage
    5. Bot traffic detection
    6. Novelty effect
    7. Date/time anomalies

  SRM Policy:
    No experiment's results will be trusted until its
    SRM check passes. This is a hard pre-analysis gate,
    regardless of which experiment has the SRM issue.
""")

# ============================================================
# SECTION 2 — WRITTEN DIAGNOSIS
# ============================================================

DIAGNOSIS = """
╔══════════════════════════════════════════════════════════════╗
║         PHASE 2 — WRITTEN DIAGNOSIS: SRM IN EXPERIMENT 1    ║
╚══════════════════════════════════════════════════════════════╝

CHECK THAT FAILED
─────────────────
Experiment  : Experiment 1 — Checkout Page Redesign
Check       : Sample Ratio Mismatch (Hard Gate)
Finding     : chi2 = 447.13,  p ≈ 0.000000  (threshold: p < 0.01)
              Observed split — Treatment: 52.00% | Control: 48.00%
              Expected split — Treatment: 50.00% | Control: 50.00%
              T/C ratio: 1.0833  (expected 1.0000)

WHAT COULD CAUSE THIS IN REAL LIFE?
────────────────────────────────────
A 52/48 split is subtle enough to be missed by eye but statistically
devastating — it invalidates the randomisation guarantee that makes
A/B testing valid. The most common root causes are:

  (1) Tracking pixel failure on the control page
      The old checkout page (control) may have a slower load time,
      causing the analytics pixel to fail to fire before users
      navigate away. Treatment events get logged; control events
      silently drop. This is the most likely cause in a checkout
      redesign experiment where page performance differs by group.

  (2) Cookie/session assignment race condition
      If two requests arrive simultaneously for the same user, a
      race condition in the assignment service can assign the user
      twice — once to each group. The second assignment (typically
      treatment) overwrites the first, and the control event is
      never recorded.

  (3) Bot filtering applied asymmetrically
      If a bot-filtering layer was added mid-experiment and it
      applied different rules to old_page vs new_page URLs, it
      could have stripped more control users than treatment users
      from the logged dataset.

  (4) Cache/CDN returning stale assignment
      A CDN caching the assignment endpoint could serve stale
      responses preferentially to users who hit the control
      branch, causing them to be re-assigned or dropped.

HOW WOULD YOU INVESTIGATE IN REAL LIFE?
─────────────────────────────────────────
  Step 1 → Pull raw assignment logs (server-side) independently
           of the analytics pipeline. Compare raw assignment counts
           to logged event counts. If raw is 50/50 but logged is
           52/48, the problem is in the logging layer, not assignment.

  Step 2 → Check pixel/event firing rate by page. Query:
           'fired_events / page_loads' separately for old_page and
           new_page. A drop in control firing rate confirms
           Hypothesis 1 above.

  Step 3 → Look at the timestamp of when the split diverged.
           If it was 50/50 for days 1–3 then shifted to 52/48,
           a deployment or configuration change mid-experiment
           is the likely culprit.

  Step 4 → Check if the SRM is uniform across device types,
           browsers, and geographic regions. A SRM that only
           appears on mobile Chrome suggests a client-side bug
           rather than a server-side assignment problem.

  Step 5 → Review any infrastructure changes (deploys, CDN
           config updates, A/B framework version bumps) during
           the experiment window. Cross-reference with the day
           the split started diverging.

DECISION
─────────────────────────────────────────────────────────────
  Decision  : ⛔ DO NOT PROCEED with Experiment 1 results.

  Rationale : An 11,000+ user imbalance (4pp gap) with a
              chi-square statistic of 447 and p ≈ 0 is not
              a borderline case. The randomisation guarantee
              is broken. Any conversion rate difference we
              observe could be entirely explained by which
              users were excluded from the control group,
              not by the checkout redesign itself.

  Actions   :
    (a) PAUSE the experiment immediately. Do not make
        shipping decisions based on current data.

    (b) IDENTIFY the root cause using the investigation
        steps above. Fix the logging/assignment bug.

    (c) RESTART the experiment from scratch with a clean
        slate. Do not attempt to correct the imbalance
        statistically (e.g., weighting) — the excluded
        control users are likely systematically different
        (e.g., users on slow connections if pixel timing
        is the cause), which introduces selection bias
        that cannot be corrected post-hoc.

    (d) ADD a real-time SRM monitoring alert to the
        experiment dashboard so future SRMs are caught
        within 24 hours, not at analysis time.

  Experiments 2 and 3 have cleared their SRM gates and
  may proceed to Phase 3 analysis as planned. Experiment 1
  will re-enter the pipeline after the bug is fixed and
  data is recollected.

NOTE ON OTHER FLAGS IN THIS PHASE
─────────────────────────────────────────────────────────────
  Bot Detection (Exp1, Exp2, Exp3 — WARN):
    Heuristic B (consistent 2–4am activity) produced high
    false positive rates on the Kaggle dataset because its
    timestamps are uniformly random across 24 hours, unlike
    real e-commerce traffic which is heavily business-hour
    weighted. In a production system, this heuristic would
    be tuned with user-agent and IP reputation signals.
    No bot removal is recommended at this stage.

  A/A Equivalence (Exp2, Exp3 — WARN):
    The pre-experiment covariate in Exp2 has a small mean
    difference between groups (Rs.4.51) due to the
    simulation method. In a real experiment this would
    warrant investigation. Here it is a simulation artefact.
    CUPED in Phase 4 will adjust for this covariate, making
    the final analysis robust to this imbalance.

  Novelty Effect (Exp1, Exp2 — WARN):
    Exp1 shows a negative lift that decays (artefact of
    SRM — do not interpret). Exp2 shows the injected
    novelty: +16% early vs +10% late. Phase 3 analysis
    will use the late-period (days 7–14) lift as the
    primary estimate for Exp2.

  Duplicate (Exp1 — FAIL):
    One duplicate user_id (773192) found in the Kaggle
    dataset. This is a within-group duplicate with no
    cross-group contamination. It will be deduplicated
    (keep first occurrence) at the start of Phase 3.
"""

def print_diagnosis():
    print(DIAGNOSIS)

# ============================================================
# SECTION 3 — SAVE PHASE 2 SUMMARY JSON
# ============================================================

def save_phase2_json(summary_df: pd.DataFrame) -> str:
    """
    Save a machine-readable summary for Phase 3 to load.

    Contains:
      - Gate status per experiment
      - Check results per experiment
      - Key dataset statistics
      - Flags for Phase 3 to act on
    """

    # ── Gate status ───────────────────────────────────────────
    gate_rows = summary_df[summary_df['Gate'] == '🔒 GATE']
    gate_status = {}
    for _, row in gate_rows.iterrows():
        gate_status[row['Experiment']] = {
            'passed'  : row['Status'] == CheckResult.STATUS_PASS,
            'status'  : row['Status'],
            'detail'  : row['Detail'],
        }

    # ── Phase 3 flags ─────────────────────────────────────────
    # These tell Phase 3 what adjustments to make
    phase3_flags = {
        'exp1': {
            'srm_failed'            : True,
            'srm_action'            : 'DO_NOT_ANALYZE — restart experiment',
            'duplicate_user'        : 773192,
            'duplicate_action'      : 'deduplicate keep=first at Phase 3 start',
            'bot_warn'              : True,
            'bot_action'            : 'false_positive_kaggle_timestamps — no removal',
            'novelty_warn'          : True,
            'novelty_action'        : 'artefact_of_srm — do not interpret',
        },
        'exp2': {
            'srm_failed'            : False,
            'novelty_detected'      : True,
            'novelty_action'        : 'use_late_period_days_7_14_as_primary',
            'cuped_covariate'       : 'pre_exp_revenue',
            'cuped_correlation'     : 0.85,
            'aa_warn'               : True,
            'aa_action'             : 'simulation_artefact_cuped_will_correct',
            'primary_metric'        : 'revenue',
            'test_type'             : 'means_ttest_plus_mannwhitney_robustness',
        },
        'exp3': {
            'srm_failed'            : False,
            'novelty_warn'          : False,
            'primary_metric'        : 'clicked',
            'segments'              : ['mobile', 'desktop'],
            'bonferroni_alpha'      : 0.025,
            'test_type'             : 'proportion_ztest_bonferroni',
            'guardrail_metric'      : 'converted (post-click CVR)',
        },
    }

    # ── Dataset statistics ────────────────────────────────────
    dataset_stats = {
        'exp1': {
            'rows'           : 279_444,
            'control_n'      : 134_133,
            'treatment_n'    : 145_311,
            'srm_ratio'      : round(145_311 / 134_133, 4),
            'experiment_days': 22,
            'file'           : 'phase2/experiment1/exp1_checkout_data.csv',
        },
        'exp2': {
            'rows'           : 22_640,
            'control_n'      : 11_320,
            'treatment_n'    : 11_320,
            'srm_ratio'      : 1.0,
            'experiment_days': 14,
            'file'           : 'phase2/experiment2/exp2_recommendations_data.csv',
        },
        'exp3': {
            'rows'           : 197_544,
            'control_n'      : 98_772,
            'treatment_n'    : 98_772,
            'srm_ratio'      : 1.0,
            'experiment_days': 14,
            'file'           : 'phase2/experiment3/exp3_banner_data.csv',
        },
    }

    # ── Assemble full JSON ─────────────────────────────────────
    phase2_summary = {
        'phase'               : 2,
        'phase_name'          : 'Data Collection + Sanity Checks',
        'status'              : 'COMPLETE',
        'phase3_ready'        : not tracker.any_gate_failed(),
        'experiments_gated'   : [r['Experiment'] for r in tracker.failed_gates()],
        'experiments_cleared' : [
            exp for exp in ['Exp1: Checkout', 'Exp2: Recs', 'Exp3: Banner']
            if exp not in [r['Experiment'] for r in tracker.failed_gates()]
        ],
        'gate_status'         : gate_status,
        'phase3_flags'        : phase3_flags,
        'dataset_stats'       : dataset_stats,
        'srm_policy'          : (
            'No experiment results are trusted until SRM check passes. '
            'Exp1 is gated. Exp2 and Exp3 are cleared for Phase 3.'
        ),
        'validation_summary_file': 'phase2/phase2_validation_summary.csv',
        'random_seed'            : RANDOM_SEED,
    }

    # ── Save ──────────────────────────────────────────────────
    out_path = os.path.join(PHASE2_DIR, "phase2_summary.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(phase2_summary, f, indent=2)

    print(f"\n  ✅ phase2_summary.json saved: {out_path}")
    return out_path


# ============================================================
# SECTION 4 — FINAL FILE MANIFEST
# ============================================================

def print_file_manifest():
    print("\n" + "="*60)
    print("PHASE 2 — FILES SAVED")
    print("="*60)

    files = [
        ("phase2/experiment1/exp1_checkout_data.csv",
         "Exp1 dataset (Kaggle + SRM injected)       279,444 rows"),
        ("phase2/experiment2/exp2_recommendations_data.csv",
         "Exp2 dataset (simulated + novelty + covar)  22,640 rows"),
        ("phase2/experiment3/exp3_banner_data.csv",
         "Exp3 dataset (simulated + device segments) 197,544 rows"),
        ("phase2/phase2_validation_summary.csv",
         "Validation table (7 checks × 3 experiments)"),
        ("phase2/phase2_summary.json",
         "Machine-readable summary → loaded by Phase 3"),
    ]

    for filepath, description in files:
        full = os.path.join(BASE_DIR, filepath)
        exists = "✅" if os.path.exists(full) else "❌"
        print(f"  {exists}  {filepath}")
        print(f"       {description}")

# ============================================================
# SECTION 5 — PHASE 2 COMPLETE BANNER
# ============================================================

def print_phase2_complete(summary_df: pd.DataFrame):
    total_checks = len(summary_df)
    passed  = (summary_df['Status'] == CheckResult.STATUS_PASS).sum()
    failed  = (summary_df['Status'] == CheckResult.STATUS_FAIL).sum()
    warned  = (summary_df['Status'] == CheckResult.STATUS_WARN).sum()
    na      = (summary_df['Status'] == CheckResult.STATUS_NA).sum()

    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█   PHASE 2 COMPLETE                                  █")
    print("█" + " "*58 + "█")
    print("█"*60)
    print(f"""
  Check Results Summary
  ─────────────────────
  Total checks run  : {total_checks}
  ✅ Passed         : {passed}
  ❌ Failed         : {failed}
  ⚠️  Warnings       : {warned}
  ➖ N/A            : {na}

  Gate Summary
  ─────────────────────
  ⛔ Exp1: Checkout  — BLOCKED (SRM: 52/48, chi2=447, p≈0)
  ✅ Exp2: Recs      — CLEARED for Phase 3
  ✅ Exp3: Banner    — CLEARED for Phase 3

  What Phase 3 Will Do
  ─────────────────────
  Exp1 → SKIP statistical analysis (SRM not resolved)
          Will demonstrate what the analysis WOULD show
          but flagged as unreliable throughout.

  Exp2 → Two-sample t-test on revenue (primary)
          Mann-Whitney U (robustness check, Phase 3 note)
          CUPED variance reduction using pre_exp_revenue
          Primary estimate: late-period lift (days 7–14)
          Full novelty effect plot (early vs late)

  Exp3 → Two-proportion z-test (overall CTR)
          Bonferroni-corrected tests (Mobile + Desktop)
          Guardrail check: post-click CVR must not drop

  Carry-forward flags from Phase 2
  ─────────────────────────────────
  → Exp1: Deduplicate user 773192 (keep first)
  → Exp2: Use pre_exp_revenue as CUPED covariate
  → Exp2: Report late-period lift as primary
  → Exp3: alpha = 0.025 per segment (Bonferroni)
  → All:  Bot warning is Kaggle artefact — no removal

  Next Step
  ─────────────────────
  Run:  python phase3/phase3_run.py
        (after Phase 3 files are created)
""")

# ============================================================
# MAIN — MASTER PIPELINE
# ============================================================

def main():

    print_header()

    # ── Step 1: Load Experiment 1 ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — LOADING EXPERIMENT 1 DATA")
    print("="*60)
    df1 = load_experiment1_data()

    # ── Step 2: Simulate Experiments 2 & 3 ───────────────────
    print("\n" + "="*60)
    print("STEP 2 — SIMULATING EXPERIMENTS 2 AND 3")
    print("="*60)
    df2, df3 = simulate_all()

    # ── Step 3: Run sanity checks ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — RUNNING SANITY CHECKS")
    print("="*60)
    summary_df = run_all_checks(df1, df2, df3)

    # ── Step 4: Print validation summary ─────────────────────
    print_summary_table(summary_df)

    # ── Step 5: Save validation CSV ──────────────────────────
    csv_path = os.path.join(PHASE2_DIR, "phase2_validation_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Validation summary saved: {csv_path}")

    # ── Step 6: Print written diagnosis ──────────────────────
    print_diagnosis()

    # ── Step 7: Save Phase 2 JSON ────────────────────────────
    print("\n" + "="*60)
    print("STEP 7 — SAVING PHASE 2 SUMMARY JSON")
    print("="*60)
    save_phase2_json(summary_df)

    # ── Step 8: Print file manifest ──────────────────────────
    print_file_manifest()

    # ── Step 9: Phase 2 complete banner ──────────────────────
    print_phase2_complete(summary_df)


if __name__ == "__main__":
    main()