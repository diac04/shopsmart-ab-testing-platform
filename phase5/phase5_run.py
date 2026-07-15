# phase5/phase5_run.py
"""
Phase 5 Master Runner — Experiment 3: Discount Banner Placement
Orchestrates all sub-modules and saves outputs.
"""

import os, sys, json
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase5.config              import OUTPUT_DIR, EXPERIMENT_NAME
from phase5.data_loader         import load_data
from phase5.overall_test        import run_overall_test
from phase5.segmented_tests     import run_segmented_tests
from phase5.interaction_model   import run_interaction_model
from phase5.multiple_corrections import apply_corrections, correction_decision_table
from phase5.business_impact     import compute_false_positive_cost
from phase5.recommendation      import build_recommendation
from phase5.written_paragraphs  import generate_paragraphs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'█'*65}")
    print(f"  PHASE 5 — {EXPERIMENT_NAME}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*65}")

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_data(verbose=True)

    # ── PART A: Overall test ───────────────────────────────────────────────
    overall_result = run_overall_test(df, verbose=True)

    # ── PART A: Segmented tests (uncorrected) ──────────────────────────────
    segmented_df = run_segmented_tests(df, verbose=True)

    # ── PART B: Interaction model (HTE) ───────────────────────────────────
    hte_result = run_interaction_model(df, verbose=True)

    # ── PART C: Multiple comparison corrections ────────────────────────────
    corrected_df   = apply_corrections(segmented_df, verbose=True)
    decision_table = correction_decision_table(corrected_df)
    print("\n  ── Decision Table (Before vs After Correction) ──")
    print(decision_table.to_string())

    # ── PART C: ₹ Cost of false positive ──────────────────────────────────
    cost_result = compute_false_positive_cost(corrected_df, verbose=True)

    # ── PART D: Recommendation ────────────────────────────────────────────
    rec_result = build_recommendation(
        corrected_df, hte_result, cost_result, verbose=True
    )

    # ── Written paragraphs ─────────────────────────────────────────────────
    para_path = os.path.join(OUTPUT_DIR, "written_paragraphs.txt")
    full_text  = generate_paragraphs(
        recommendation_narrative=rec_result["narrative"],
        cost_inr=cost_result["total_cost_inr"],
        output_path=para_path,
    )

    # ── Save all outputs ───────────────────────────────────────────────────
    _save_outputs(
        overall_result, segmented_df, corrected_df,
        decision_table, hte_result, cost_result, rec_result
    )

    print(f"\n{'█'*65}")
    print(f"  PHASE 5 COMPLETE")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print(f"{'█'*65}\n")


def _save_outputs(overall, segmented_df, corrected_df,
                  decision_table, hte_result, cost_result, rec_result):
    """Save all artefacts to phase5/experiment3/"""

    # 1. Overall test (JSON)
    with open(os.path.join(OUTPUT_DIR, "overall_test.json"), "w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"  [SAVED] overall_test.json")

    # 2. Segmented results (CSV)
    segmented_df.to_csv(os.path.join(OUTPUT_DIR, "segmented_tests_raw.csv"))
    print(f"  [SAVED] segmented_tests_raw.csv")

    # 3. Corrected results (CSV)
    corrected_df.to_csv(os.path.join(OUTPUT_DIR, "segmented_tests_corrected.csv"))
    print(f"  [SAVED] segmented_tests_corrected.csv")

    # 4. Decision table (CSV)
    decision_table.to_csv(os.path.join(OUTPUT_DIR, "decision_table.csv"))
    print(f"  [SAVED] decision_table.csv")

    # 5. Interaction model coefficients (CSV)
    hte_result["coef_df"].to_csv(
        os.path.join(OUTPUT_DIR, "interaction_model_coefs.csv")
    )
    print(f"  [SAVED] interaction_model_coefs.csv")

        # 6. Interaction model summary (TXT)
    with open(os.path.join(OUTPUT_DIR, "interaction_model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(hte_result["summary_text"])
        f.write("\n\nINTERPRETATIONS:\n")
        for line in hte_result["interpretations"]:
            # Strip unicode symbols that cause cp1252 encoding errors on Windows
            clean_line = (
                line
                .replace("✓", "[YES]")
                .replace("✗", "[NO]")
                .replace("•", "-")
                .replace("→", "->")
            )
            f.write(f"  - {clean_line}\n")
    print(f"  [SAVED] interaction_model_summary.txt")

    # 7. False positive cost breakdown (CSV)
    cost_result["breakdown_df"].to_csv(
        os.path.join(OUTPUT_DIR, "false_positive_cost.csv")
    )
    print(f"  [SAVED] false_positive_cost.csv")

    # 8. Recommendations (CSV)
    rec_result["recommendation_df"].to_csv(
        os.path.join(OUTPUT_DIR, "recommendations.csv")
    )
    print(f"  [SAVED] recommendations.csv")

    # 9. Master summary JSON
    summary = {
        "experiment":           "Experiment 3 - Discount Banner Placement",
        "phase":                5,
        "run_timestamp":        datetime.now().isoformat(),
        "overall_ctr_result":   overall,
        "false_positive_cost":  cost_result["total_cost_inr"],
        "device_actions": {
            device: row["action"]
            for device, row in rec_result["recommendation_df"].iterrows()
        },
    }
        # 9. Master summary JSON
    with open(os.path.join(OUTPUT_DIR, "phase5_exp3_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [SAVED] phase5_exp3_results.json")


if __name__ == "__main__":
    main()