# phase5/interaction_model.py
"""
Part B — Heterogeneous Treatment Effect (HTE) via logistic regression
         with a treatment × device interaction term.

Model:
  clicked ~ treatment_flag + mobile_flag + treatment_flag:mobile_flag

We use manual binary encoding instead of pd.Categorical to avoid
the singular matrix issue that arises with only 2 device categories.

Reference cell:
  treatment_flag = 0  → control
  mobile_flag    = 0  → desktop

Coefficients:
  Intercept                    → log-odds of clicking for desktop/control
  treatment_flag               → treatment effect ON DESKTOP
  mobile_flag                  → mobile vs desktop difference IN CONTROL
  treatment_flag:mobile_flag   → EXTRA treatment effect on mobile vs desktop
                                 (this is the HTE term we care about)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    TREATMENT_COL, OUTCOME_COL, DEVICE_COL, OUTPUT_DIR,
    CONTROL_LABEL, TREATMENT_LABEL
)


def run_interaction_model(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Fit logistic regression with treatment x device interaction.
    Uses manual binary encoding to avoid singular matrix errors.

    Returns
    -------
    dict with:
        summary_text      : full statsmodels summary as string
        coef_df           : DataFrame of coefficients, OR, p-values
        interpretations   : list of plain-language interpretation strings
        model             : fitted model object
    """

    # ── Build binary features ──────────────────────────────────────────────
    model_df = df[[OUTCOME_COL, TREATMENT_COL, DEVICE_COL]].copy()

    # Binary encode: 1 = treatment, 0 = control
    model_df["treatment_flag"] = (
        model_df[TREATMENT_COL] == TREATMENT_LABEL
    ).astype(int)

    # Binary encode: 1 = mobile, 0 = desktop (reference)
    model_df["mobile_flag"] = (
        model_df[DEVICE_COL] == "mobile"
    ).astype(int)

    # Interaction term
    model_df["treat_x_mobile"] = (
        model_df["treatment_flag"] * model_df["mobile_flag"]
    )

    # ── Design matrix ──────────────────────────────────────────────────────
    X = model_df[["treatment_flag", "mobile_flag", "treat_x_mobile"]]
    X = sm.add_constant(X)          # adds intercept column
    y = model_df[OUTCOME_COL]

    # ── Fit logistic regression ────────────────────────────────────────────
    model = sm.Logit(y, X).fit(disp=False)

    # ── Extract coefficients ───────────────────────────────────────────────
    coef_df = pd.DataFrame({
        "coefficient": model.params,
        "std_err":     model.bse,
        "z_stat":      model.tvalues,
        "p_value":     model.pvalues,
        "odds_ratio":  np.exp(model.params),
        "or_ci_low":   np.exp(model.conf_int()[0]),
        "or_ci_high":  np.exp(model.conf_int()[1]),
    }).round(6)

    # ── Plain-language interpretations ─────────────────────────────────────
    interpretations = _interpret(coef_df)

    if verbose:
        print("\n" + "="*70)
        print("  PART B — Logistic Regression: Treatment × Device Interaction")
        print("="*70)
        print(model.summary())
        print("\n  ── Odds Ratios ──")
        print(coef_df[["odds_ratio", "or_ci_low", "or_ci_high", "p_value"]].to_string())
        print("\n  ── Plain-Language Interpretations ──")
        for line in interpretations:
            print(f"  • {line}")
        print("="*70 + "\n")

    return {
        "summary_text":     model.summary().as_text(),
        "coef_df":          coef_df,
        "interpretations":  interpretations,
        "model":            model,
        "aic":              model.aic,
        "llr_pvalue":       model.llr_pvalue,
    }


def _interpret(coef_df: pd.DataFrame) -> list:
    """
    Translate each coefficient into plain English.

    Reference cell = desktop + control group.
    """
    lines = []

    # ── Intercept ──────────────────────────────────────────────────────────
    if "const" in coef_df.index:
        OR = coef_df.loc["const", "odds_ratio"]
        p  = coef_df.loc["const", "p_value"]
        lines.append(
            f"BASELINE (desktop, control): odds of clicking = {OR:.4f} "
            f"(p={p:.4f}). This is the reference cell."
        )

    # ── Treatment effect on desktop ────────────────────────────────────────
    if "treatment_flag" in coef_df.index:
        OR  = coef_df.loc["treatment_flag", "odds_ratio"]
        p   = coef_df.loc["treatment_flag", "p_value"]
        pct = (OR - 1) * 100
        dir_word = "higher" if OR > 1 else "lower"
        sig = "✓ significant" if p < 0.05 else "✗ not significant"
        lines.append(
            f"TREATMENT effect on DESKTOP: the banner increases click odds "
            f"by {abs(pct):.1f}% on desktop (OR={OR:.3f}, p={p:.4f}) — {sig}."
        )

    # ── Mobile vs desktop in control ───────────────────────────────────────
    if "mobile_flag" in coef_df.index:
        OR  = coef_df.loc["mobile_flag", "odds_ratio"]
        p   = coef_df.loc["mobile_flag", "p_value"]
        pct = (OR - 1) * 100
        dir_word = "higher" if OR > 1 else "lower"
        lines.append(
            f"MOBILE baseline vs DESKTOP baseline (control group only): "
            f"mobile users have {abs(pct):.1f}% {dir_word} click odds "
            f"even without treatment (OR={OR:.3f}, p={p:.4f})."
        )

    # ── Interaction: extra treatment effect on mobile vs desktop ───────────
    if "treat_x_mobile" in coef_df.index:
        OR  = coef_df.loc["treat_x_mobile", "odds_ratio"]
        p   = coef_df.loc["treat_x_mobile", "p_value"]
        pct = abs((OR - 1) * 100)
        dir_word = "stronger" if OR > 1 else "weaker"
        sig = "✓ significant" if p < 0.05 else "✗ not significant"
        lines.append(
            f"INTERACTION (treatment × mobile): the treatment effect is "
            f"{dir_word} on mobile than on desktop by a factor of {OR:.3f} "
            f"in odds (OR={OR:.3f}, p={p:.4f}) — {sig}."
        )
        if p < 0.05:
            lines.append(
                f"  → This confirms HETEROGENEOUS treatment effects: "
                f"the banner placement works differently on mobile vs desktop. "
                f"Phase 1 hypothesis VALIDATED."
            )
        else:
            lines.append(
                f"  → The difference between mobile and desktop effects "
                f"is NOT statistically significant. The treatment effect "
                f"may be uniform across devices despite the raw difference "
                f"in segment-level CTR lifts."
            )

    return lines


if __name__ == "__main__":
    from phase5.data_loader import load_data
    df = load_data()
    run_interaction_model(df)