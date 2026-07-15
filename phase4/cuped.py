# phase4/cuped.py
# ============================================================
# Part C — CUPED (Controlled-experiment Using Pre-Experiment Data)
#
# Key idea: residualise the outcome on the pre-experiment covariate
# to remove variance explained by pre-existing user differences.
# The CUPED-adjusted metric has the same expectation as the raw
# metric but lower variance → narrower CIs → more power.
#
# Formula:
#   Y_cuped = Y - theta * (X - E[X])
#   theta   = Cov(Y, X) / Var(X)      ← pooled across both groups
#
# The adjustment is computed on the POOLED dataset so theta is
# not contaminated by treatment assignment.
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
import os

from phase4.config import (
    OUTPUT_DIR, ALPHA, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
    BOOTSTRAP_CI, CONTROL_LABEL, TREATMENT_LABEL,
    N_PER_GROUP_P1, BASELINE_STD,
    GROUP_COL, REVENUE_COL, PRE_EXP_COL
)


# ── Core CUPED adjustment ─────────────────────────────────────────────────────

def compute_theta(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Compute the CUPED coefficient theta = Cov(Y,X) / Var(X).
    Pooled across both groups (Y and X are full-dataset arrays).
    """
    cov_matrix = np.cov(Y, X, ddof=1)
    theta = cov_matrix[0, 1] / cov_matrix[1, 1]
    return float(theta)


def apply_cuped(Y: np.ndarray,
                X: np.ndarray,
                theta: float,
                X_mean_global: float) -> np.ndarray:
    """
    Return CUPED-adjusted outcome for one group.
    Y_cuped = Y - theta * (X - X_mean_global)
    """
    return Y - theta * (X - X_mean_global)


# ── Variance reduction calculator ────────────────────────────────────────────

def variance_reduction_stats(var_raw_ctrl: float,
                              var_raw_treat: float,
                              var_cuped_ctrl: float,
                              var_cuped_treat: float) -> dict:
    """
    Compute variance reduction % and the equivalent user saving.

    Power formula: n ∝ sigma^2
    If variance drops by r%, you need (1-r%) as many users
    for the same power.
    """
    var_raw_avg   = (var_raw_ctrl   + var_raw_treat)   / 2
    var_cuped_avg = (var_cuped_ctrl + var_cuped_treat) / 2

    reduction_pct = 100 * (1 - var_cuped_avg / var_raw_avg)

    # Phase 1 required n per group = 566
    # New required n = 566 * (1 - reduction_pct/100)
    n_p1             = N_PER_GROUP_P1
    n_cuped_required = int(np.ceil(n_p1 * (1 - reduction_pct / 100)))
    users_saved      = n_p1 - n_cuped_required

    return {
        "var_raw_ctrl"      : round(float(var_raw_ctrl),   2),
        "var_raw_treat"     : round(float(var_raw_treat),  2),
        "var_raw_avg"       : round(float(var_raw_avg),    2),
        "var_cuped_ctrl"    : round(float(var_cuped_ctrl), 2),
        "var_cuped_treat"   : round(float(var_cuped_treat),2),
        "var_cuped_avg"     : round(float(var_cuped_avg),  2),
        "reduction_pct"     : round(float(reduction_pct),  2),
        "n_phase1_required" : n_p1,
        "n_cuped_required"  : n_cuped_required,
        "users_saved"       : users_saved,
        "users_saved_pct"   : round(100 * users_saved / n_p1, 2),
    }


# ── Bootstrap CI on CUPED-adjusted mean difference ───────────────────────────

def bootstrap_cuped_ci(ctrl_cuped: np.ndarray,
                        treat_cuped: np.ndarray) -> dict:
    """
    Bootstrap CI on the CUPED-adjusted mean difference.
    This is the trusted interval we carry forward to business impact.
    """
    rng        = np.random.default_rng(BOOTSTRAP_SEED)
    n_c, n_t   = len(ctrl_cuped), len(treat_cuped)
    n_boot     = BOOTSTRAP_RESAMPLES
    alpha_tail = (1 - BOOTSTRAP_CI) / 2

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        s_c = rng.choice(ctrl_cuped,  size=n_c, replace=True)
        s_t = rng.choice(treat_cuped, size=n_t, replace=True)
        boot_diffs[i] = np.mean(s_t) - np.mean(s_c)

    obs_diff = np.mean(treat_cuped) - np.mean(ctrl_cuped)
    ci_lo    = np.percentile(boot_diffs, 100 * alpha_tail)
    ci_hi    = np.percentile(boot_diffs, 100 * (1 - alpha_tail))

    shifted = boot_diffs - np.mean(boot_diffs)
    p_boot  = 2 * min(
        np.mean(shifted >= abs(obs_diff)),
        np.mean(shifted <= -abs(obs_diff))
    )

    return {
        "obs_cuped_diff_rs" : round(float(obs_diff), 2),
        "ci_lo"             : round(float(ci_lo),    2),
        "ci_hi"             : round(float(ci_hi),    2),
        "bootstrap_p_value" : round(float(p_boot),   6),
        "significant"       : p_boot < ALPHA,
        "ci_excludes_zero"  : ci_lo > 0,
    }


# ── Welch t-test on CUPED-adjusted values ────────────────────────────────────

def ttest_cuped(ctrl_cuped: np.ndarray,
                treat_cuped: np.ndarray) -> dict:
    """Welch t-test on CUPED-adjusted means."""
    t_stat, p_val = stats.ttest_ind(treat_cuped, ctrl_cuped,
                                    equal_var=False)
    obs_diff = np.mean(treat_cuped) - np.mean(ctrl_cuped)

    # SE of difference
    n_c = len(ctrl_cuped)
    n_t = len(treat_cuped)
    se  = np.sqrt(np.var(ctrl_cuped,  ddof=1) / n_c +
                  np.var(treat_cuped, ddof=1) / n_t)
    df  = min(n_c, n_t) - 1
    t_c = stats.t.ppf(1 - ALPHA / 2, df=df)
    ci_lo = obs_diff - t_c * se
    ci_hi = obs_diff + t_c * se

    return {
        "t_statistic" : round(float(t_stat), 4),
        "p_value"     : float(p_val),
        "significant" : p_val < ALPHA,
        "obs_diff_rs" : round(float(obs_diff), 2),
        "ci_lo"       : round(float(ci_lo),    2),
        "ci_hi"       : round(float(ci_hi),    2),
        "se"          : round(float(se),        4),
    }


# ── Master runner ─────────────────────────────────────────────────────────────

def run_cuped(df_full: pd.DataFrame,
              ctrl_late: np.ndarray,
              treat_late: np.ndarray) -> dict:
    """
    Full CUPED pipeline.

    Parameters
    ----------
    df_full    : full dataset (all days) — needed for pre-exp covariate
    ctrl_late  : raw revenue, control,   late period (days 7–14)
    treat_late : raw revenue, treatment, late period (days 7–14)

    Returns
    -------
    dict with all CUPED results
    """
    print("\n" + "="*60)
    print("  PART C — CUPED VARIANCE REDUCTION")
    print("="*60)

    # ── Step 1: get pre-experiment covariate (full dataset) ───
    # Each user appears once in df_full — covariate is stable
    pre_ctrl  = df_full.loc[
        df_full[GROUP_COL] == CONTROL_LABEL,   PRE_EXP_COL].values
    pre_treat = df_full.loc[
        df_full[GROUP_COL] == TREATMENT_LABEL, PRE_EXP_COL].values

    print(f"\n  Pre-exp covariate loaded:")
    print(f"    Control   n={len(pre_ctrl):,} | "
          f"mean=Rs.{np.mean(pre_ctrl):.2f} | "
          f"std=Rs.{np.std(pre_ctrl, ddof=1):.2f}")
    print(f"    Treatment n={len(pre_treat):,} | "
          f"mean=Rs.{np.mean(pre_treat):.2f} | "
          f"std=Rs.{np.std(pre_treat, ddof=1):.2f}")

    # ── Step 2: theta computed on FULL pooled dataset ─────────
    # We pool all users' revenue (late period) + covariate (full)
    # Note: late period has fewer rows than full — we match by
    # re-extracting the late-period subset with its covariate.
    #
    # The cleanest approach: use only users present in late period
    # Pull their pre_exp_revenue from df_full by user_id

    from phase4.config import DAY_COL, NOVELTY_CUTOFF_DAY
    df_late = df_full[df_full[DAY_COL] >= NOVELTY_CUTOFF_DAY].copy()

    # For each user in late period, get their pre_exp_revenue
    # (already in the dataframe — same value for each user)
    pre_ctrl_late  = df_late.loc[
        df_late[GROUP_COL] == CONTROL_LABEL,   PRE_EXP_COL].values
    pre_treat_late = df_late.loc[
        df_late[GROUP_COL] == TREATMENT_LABEL, PRE_EXP_COL].values

    print(f"\n  Late-period covariate (matched to outcome):")
    print(f"    Control   n={len(pre_ctrl_late):,} | "
          f"mean=Rs.{np.mean(pre_ctrl_late):.2f}")
    print(f"    Treatment n={len(pre_treat_late):,} | "
          f"mean=Rs.{np.mean(pre_treat_late):.2f}")

    # Verify lengths match
    assert len(ctrl_late)  == len(pre_ctrl_late),  \
        f"Length mismatch ctrl: {len(ctrl_late)} vs {len(pre_ctrl_late)}"
    assert len(treat_late) == len(pre_treat_late), \
        f"Length mismatch treat: {len(treat_late)} vs {len(pre_treat_late)}"

    # Pool for theta
    Y_pool = np.concatenate([ctrl_late,      treat_late])
    X_pool = np.concatenate([pre_ctrl_late,  pre_treat_late])

    theta       = compute_theta(Y_pool, X_pool)
    X_mean_glob = np.mean(X_pool)

    # Pearson correlation (diagnostic)
    rho, rho_p  = stats.pearsonr(Y_pool, X_pool)

    print(f"\n  Pooled correlation (rho)  : {rho:.4f}  (p={rho_p:.2e})")
    print(f"  Global pre-exp mean (X̄)  : Rs.{X_mean_glob:.4f}")
    print(f"  Theta (Cov/Var)           : {theta:.6f}")
    print(f"  Theoretical var reduction : {100*rho**2:.2f}%  (= rho²×100)")

    # ── Step 3: apply CUPED adjustment ───────────────────────
    ctrl_cuped  = apply_cuped(ctrl_late,  pre_ctrl_late,  theta, X_mean_glob)
    treat_cuped = apply_cuped(treat_late, pre_treat_late, theta, X_mean_glob)

    # ── Step 4: variance reduction stats ─────────────────────
    vr = variance_reduction_stats(
        var_raw_ctrl   = np.var(ctrl_late,   ddof=1),
        var_raw_treat  = np.var(treat_late,  ddof=1),
        var_cuped_ctrl = np.var(ctrl_cuped,  ddof=1),
        var_cuped_treat= np.var(treat_cuped, ddof=1),
    )

    print(f"\n  VARIANCE REDUCTION")
    print(f"  {'Metric':<35} {'Control':>12} {'Treatment':>12}")
    print("  " + "-"*60)
    print(f"  {'Raw variance':<35} "
          f"{vr['var_raw_ctrl']:>12.2f} "
          f"{vr['var_raw_treat']:>12.2f}")
    print(f"  {'CUPED-adjusted variance':<35} "
          f"{vr['var_cuped_ctrl']:>12.2f} "
          f"{vr['var_cuped_treat']:>12.2f}")
    print(f"  {'Variance reduction (%)':<35} "
          f"{vr['reduction_pct']:>12.2f}%")
    print(f"\n  Phase 1 required n per group   : {vr['n_phase1_required']:,}")
    print(f"  CUPED-equivalent n per group   : {vr['n_cuped_required']:,}")
    print(f"  Users saved per group          : {vr['users_saved']:,}"
          f"  ({vr['users_saved_pct']}%)")

    # ── Step 5: inference on CUPED-adjusted metric ────────────
    tt   = ttest_cuped(ctrl_cuped, treat_cuped)
    boot = bootstrap_cuped_ci(ctrl_cuped, treat_cuped)

    print(f"\n  CUPED-ADJUSTED INFERENCE")
    print(f"  Welch t-test:")
    print(f"    t-statistic  : {tt['t_statistic']}")
    print(f"    p-value      : {tt['p_value']:.4e}")
    print(f"    Obs diff     : Rs.{tt['obs_diff_rs']}")
    print(f"    95% CI       : [Rs.{tt['ci_lo']}, Rs.{tt['ci_hi']}]")
    print(f"    Significant  : {tt['significant']}")
    print(f"\n  Bootstrap (10,000 resamples):")
    print(f"    Obs diff     : Rs.{boot['obs_cuped_diff_rs']}")
    print(f"    95% CI       : [Rs.{boot['ci_lo']}, Rs.{boot['ci_hi']}]")
    print(f"    p-value      : {boot['bootstrap_p_value']}")
    print(f"    Significant  : {boot['significant']}")

    # ── Step 6: raw vs CUPED comparison ──────────────────────
    raw_mean_diff   = np.mean(treat_late)  - np.mean(ctrl_late)
    cuped_mean_diff = np.mean(treat_cuped) - np.mean(ctrl_cuped)

    print(f"\n  RAW vs CUPED MEAN DIFFERENCE")
    print(f"    Raw mean diff   : Rs.{raw_mean_diff:.2f}")
    print(f"    CUPED mean diff : Rs.{cuped_mean_diff:.2f}")
    print(f"    Difference      : Rs.{cuped_mean_diff - raw_mean_diff:.2f}"
          f"  (CUPED corrects for pre-exp imbalance)")

    # ── Step 7: save CSV ──────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_rows = [
        {"metric": "Raw variance — control",          "value": vr["var_raw_ctrl"]},
        {"metric": "Raw variance — treatment",        "value": vr["var_raw_treat"]},
        {"metric": "CUPED variance — control",        "value": vr["var_cuped_ctrl"]},
        {"metric": "CUPED variance — treatment",      "value": vr["var_cuped_treat"]},
        {"metric": "Variance reduction (%)",          "value": vr["reduction_pct"]},
        {"metric": "Phase 1 n per group",             "value": vr["n_phase1_required"]},
        {"metric": "CUPED-equivalent n per group",    "value": vr["n_cuped_required"]},
        {"metric": "Users saved per group",           "value": vr["users_saved"]},
        {"metric": "Theta",                           "value": round(theta, 6)},
        {"metric": "Pooled correlation rho",          "value": round(rho, 4)},
        {"metric": "Raw mean diff (Rs.)",             "value": round(raw_mean_diff, 2)},
        {"metric": "CUPED mean diff (Rs.)",           "value": round(cuped_mean_diff, 2)},
        {"metric": "CUPED bootstrap CI lo",           "value": boot["ci_lo"]},
        {"metric": "CUPED bootstrap CI hi",           "value": boot["ci_hi"]},
        {"metric": "CUPED bootstrap p-value",         "value": boot["bootstrap_p_value"]},
    ]
    df_out   = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUTPUT_DIR, "cuped_results.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  ✅ cuped_results.csv saved → {csv_path}")

    return {
        "theta"           : theta,
        "rho"             : rho,
        "X_mean_global"   : X_mean_glob,
        "variance_stats"  : vr,
        "ttest"           : tt,
        "bootstrap"       : boot,
        "ctrl_cuped"      : ctrl_cuped,
        "treat_cuped"     : treat_cuped,
        "raw_mean_diff"   : round(float(raw_mean_diff),   2),
        "cuped_mean_diff" : round(float(cuped_mean_diff), 2),
    }