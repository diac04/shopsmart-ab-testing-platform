# phase4/statistical_methods.py
# ============================================================
# Part B — Method Comparison
# A. Log transform + t-test
# B. Mann-Whitney U test
# C. Bootstrap CI (10,000 resamples)
# D. Delta method — applicability decision
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
import os

from phase4.config import (
    OUTPUT_DIR, ALPHA, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
    BOOTSTRAP_CI, CONTROL_LABEL, TREATMENT_LABEL, IS_RATIO_METRIC
)


# ── A. Log transform + t-test ─────────────────────────────────────────────────

def log_transform_ttest(ctrl: np.ndarray,
                        treat: np.ndarray) -> dict:
    """
    Log-transform revenue (purchasers only), run Welch t-test on
    log scale, then back-transform.

    IMPORTANT CAVEAT — back-transformation:
      exp(mean_log) estimates the GEOMETRIC mean, NOT the arithmetic
      mean. The arithmetic mean on the original scale is
      exp(mu + sigma^2/2) under log-normality. We report both so the
      reader is never misled into thinking the back-transformed
      difference equals the Rs. lift on the arithmetic mean.
    """
    # Use purchasers only — log(0) is undefined
    ctrl_buy  = ctrl[ctrl   > 0]
    treat_buy = treat[treat > 0]

    log_ctrl  = np.log(ctrl_buy)
    log_treat = np.log(treat_buy)

    t_stat, p_val = stats.ttest_ind(log_ctrl, log_treat,
                                    equal_var=False)   # Welch

    # Back-transform: geometric means
    geo_mean_ctrl  = np.exp(np.mean(log_ctrl))
    geo_mean_treat = np.exp(np.mean(log_treat))
    geo_lift       = geo_mean_treat - geo_mean_ctrl
    geo_lift_pct   = 100 * geo_lift / geo_mean_ctrl

    # Back-transform: arithmetic means (log-normal formula)
    arith_ctrl  = np.exp(np.mean(log_ctrl)  + 0.5 * np.var(log_ctrl,  ddof=1))
    arith_treat = np.exp(np.mean(log_treat) + 0.5 * np.var(log_treat, ddof=1))
    arith_lift  = arith_treat - arith_ctrl

    # 95% CI on log-scale difference, back-transformed
    n_c, n_t    = len(log_ctrl), len(log_treat)
    se_diff     = np.sqrt(np.var(log_ctrl,  ddof=1) / n_c +
                          np.var(log_treat, ddof=1) / n_t)
    t_crit      = stats.t.ppf(1 - ALPHA / 2, df=min(n_c, n_t) - 1)
    log_diff    = np.mean(log_treat) - np.mean(log_ctrl)
    ci_log_lo   = log_diff - t_crit * se_diff
    ci_log_hi   = log_diff + t_crit * se_diff

    # Back-transformed CI is on RATIO (multiplicative), not difference
    ci_ratio_lo = np.exp(ci_log_lo)
    ci_ratio_hi = np.exp(ci_log_hi)

    result = {
        "method"            : "Log-transform + Welch t-test",
        "n_ctrl_purchasers" : n_c,
        "n_treat_purchasers": n_t,
        "log_mean_ctrl"     : round(float(np.mean(log_ctrl)),  6),
        "log_mean_treat"    : round(float(np.mean(log_treat)), 6),
        "log_diff"          : round(float(log_diff), 6),
        "t_statistic"       : round(float(t_stat), 4),
        "p_value"           : float(p_val),
        "significant"       : p_val < ALPHA,
        "geo_mean_ctrl"     : round(float(geo_mean_ctrl),  2),
        "geo_mean_treat"    : round(float(geo_mean_treat), 2),
        "geo_lift_rs"       : round(float(geo_lift),       2),
        "geo_lift_pct"      : round(float(geo_lift_pct),   4),
        "ci_ratio_lo"       : round(float(ci_ratio_lo),    4),
        "ci_ratio_hi"       : round(float(ci_ratio_hi),    4),
        "arith_mean_ctrl"   : round(float(arith_ctrl),     2),
        "arith_mean_treat"  : round(float(arith_treat),    2),
        "arith_lift_rs"     : round(float(arith_lift),     2),
        "caveat"            : (
            "Back-transformed CI is a ratio (multiplicative), not an "
            "additive Rs. difference. exp(log_mean) = geometric mean, "
            "not arithmetic mean. Arithmetic mean requires the "
            "log-normal correction exp(mu + sigma^2/2)."
        )
    }
    return result


# ── B. Mann-Whitney U test ────────────────────────────────────────────────────

def mann_whitney_test(ctrl: np.ndarray,
                      treat: np.ndarray) -> dict:
    """
    Distribution-free test of stochastic dominance.
    H0: P(treatment > control) = 0.5
    Reports the common language effect size (CLES) = U / (n1*n2).
    Does NOT test mean equality directly — important limitation.
    """
    u_stat, p_val = stats.mannwhitneyu(treat, ctrl,
                                       alternative="two-sided")
    n_c = len(ctrl)
    n_t = len(treat)

    # Common Language Effect Size
    cles = u_stat / (n_c * n_t)

    # Rank-biserial correlation (effect size for MW)
    rank_biserial = 2 * cles - 1

    result = {
        "method"          : "Mann-Whitney U (distribution-free)",
        "n_ctrl"          : n_c,
        "n_treat"         : n_t,
        "U_statistic"     : float(u_stat),
        "p_value"         : float(p_val),
        "significant"     : p_val < ALPHA,
        "CLES"            : round(float(cles), 4),
        "rank_biserial_r" : round(float(rank_biserial), 4),
        "interpretation"  : (
            f"CLES={cles:.3f}: a randomly chosen treatment user has a "
            f"{100*cles:.1f}% chance of higher revenue than a randomly "
            f"chosen control user. Tests distributional shift, "
            f"not arithmetic mean difference."
        )
    }
    return result


# ── C. Bootstrap CI ───────────────────────────────────────────────────────────

def bootstrap_ci(ctrl: np.ndarray,
                 treat: np.ndarray) -> dict:
    """
    Percentile bootstrap on mean difference and median difference.
    10,000 resamples. Makes no distributional assumptions.
    Works on all users (including zeros) — no exclusion needed.
    """
    rng        = np.random.default_rng(BOOTSTRAP_SEED)
    n_c, n_t   = len(ctrl), len(treat)
    n_boot     = BOOTSTRAP_RESAMPLES
    alpha_tail = (1 - BOOTSTRAP_CI) / 2

    boot_mean_diff   = np.empty(n_boot)
    boot_median_diff = np.empty(n_boot)

    for i in range(n_boot):
        s_c = rng.choice(ctrl,  size=n_c, replace=True)
        s_t = rng.choice(treat, size=n_t, replace=True)
        boot_mean_diff[i]   = np.mean(s_t)   - np.mean(s_c)
        boot_median_diff[i] = np.median(s_t) - np.median(s_c)

    obs_mean_diff   = np.mean(treat)   - np.mean(ctrl)
    obs_median_diff = np.median(treat) - np.median(ctrl)

    ci_mean_lo, ci_mean_hi     = np.percentile(
        boot_mean_diff,   [100 * alpha_tail, 100 * (1 - alpha_tail)])
    ci_median_lo, ci_median_hi = np.percentile(
        boot_median_diff, [100 * alpha_tail, 100 * (1 - alpha_tail)])

    # Bootstrap p-value (two-tailed, shift under H0)
    shifted = boot_mean_diff - np.mean(boot_mean_diff)
    p_boot  = 2 * min(
        np.mean(shifted >= abs(obs_mean_diff)),
        np.mean(shifted <= -abs(obs_mean_diff))
    )

    result = {
        "method"              : f"Bootstrap CI ({n_boot:,} resamples)",
        "n_ctrl"              : n_c,
        "n_treat"             : n_t,
        "n_resamples"         : n_boot,
        "obs_mean_diff_rs"    : round(float(obs_mean_diff),   2),
        "boot_ci_mean_lo"     : round(float(ci_mean_lo),      2),
        "boot_ci_mean_hi"     : round(float(ci_mean_hi),      2),
        "obs_median_diff_rs"  : round(float(obs_median_diff), 2),
        "boot_ci_median_lo"   : round(float(ci_median_lo),    2),
        "boot_ci_median_hi"   : round(float(ci_median_hi),    2),
        "bootstrap_p_value"   : round(float(p_boot),          6),
        "significant"         : p_boot < ALPHA,
        "ci_excludes_zero_mean"  : ci_mean_lo   > 0 or ci_mean_hi   < 0,
        "ci_excludes_zero_median": ci_median_lo > 0 or ci_median_hi < 0,
    }
    return result


# ── D. Delta method — applicability decision ──────────────────────────────────

def delta_method_decision() -> dict:
    """
    Explicit written decision on whether the delta method applies.

    The delta method is needed when the metric is a RATIO of two
    random variables, e.g. total_revenue / total_visits, where both
    numerator and denominator vary by user.

    In this experiment, revenue_per_user = total_revenue / n_users,
    where n_users is a FIXED count (the number of users assigned to
    each group). The denominator is not random — it is set by the
    randomisation. Therefore this is a SIMPLE AVERAGE, not a ratio
    metric, and the delta method is not required.
    """
    verdict = {
        "method"      : "Delta Method",
        "applicable"  : IS_RATIO_METRIC,   # False — from config
        "explanation" : (
            "NOT APPLICABLE. Revenue-per-user in this design is "
            "total_group_revenue / n_assigned_users. The denominator "
            "(n_assigned_users) is fixed by randomisation — it is not "
            "a random variable. The delta method is required only when "
            "both numerator and denominator are random (e.g., "
            "revenue/visits where visits vary per user). Here, the "
            "metric is a simple arithmetic mean, and standard "
            "inference (t-test, bootstrap) applies directly. If the "
            "metric were redefined as revenue/sessions (where one user "
            "can have multiple sessions), the delta method would be "
            "necessary to account for within-user correlation."
        )
    }
    return verdict


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all_methods(ctrl: np.ndarray,
                    treat: np.ndarray) -> dict:
    """Run all four methods and print a comparison table."""

    print("\n" + "="*60)
    print("  PART B — METHOD COMPARISON")
    print("="*60)

    res_log  = log_transform_ttest(ctrl, treat)
    res_mw   = mann_whitney_test(ctrl, treat)
    res_boot = bootstrap_ci(ctrl, treat)
    res_dm   = delta_method_decision()

    # ── Print structured summary ──────────────────────────────
    print("\n  A. LOG TRANSFORM + WELCH T-TEST")
    print(f"     Purchasers only  : ctrl={res_log['n_ctrl_purchasers']:,}"
          f"  treat={res_log['n_treat_purchasers']:,}")
    print(f"     t-statistic      : {res_log['t_statistic']}")
    print(f"     p-value          : {res_log['p_value']:.4e}")
    print(f"     Significant      : {res_log['significant']}")
    print(f"     Geometric mean   : ctrl Rs.{res_log['geo_mean_ctrl']}"
          f"  →  treat Rs.{res_log['geo_mean_treat']}")
    print(f"     Geometric lift   : Rs.{res_log['geo_lift_rs']}"
          f"  ({res_log['geo_lift_pct']}%)")
    print(f"     CI on ratio      : [{res_log['ci_ratio_lo']:.4f},"
          f" {res_log['ci_ratio_hi']:.4f}]")
    print(f"     Arithmetic lift  : Rs.{res_log['arith_lift_rs']}"
          f"  (log-normal formula)")
    print(f"     ⚠️  Caveat        : {res_log['caveat']}")

    print("\n  B. MANN-WHITNEY U TEST")
    print(f"     U statistic      : {res_mw['U_statistic']:,.0f}")
    print(f"     p-value          : {res_mw['p_value']:.4e}")
    print(f"     Significant      : {res_mw['significant']}")
    print(f"     CLES             : {res_mw['CLES']}"
          f"  |  rank-biserial r: {res_mw['rank_biserial_r']}")
    print(f"     Interpretation   : {res_mw['interpretation']}")

    print("\n  C. BOOTSTRAP CI (10,000 resamples)")
    print(f"     Observed mean diff   : Rs.{res_boot['obs_mean_diff_rs']}")
    print(f"     95% CI mean diff     : "
          f"[Rs.{res_boot['boot_ci_mean_lo']}, "
          f"Rs.{res_boot['boot_ci_mean_hi']}]")
    print(f"     CI excludes zero     : {res_boot['ci_excludes_zero_mean']}")
    print(f"     Observed median diff : Rs.{res_boot['obs_median_diff_rs']}")
    print(f"     95% CI median diff   : "
          f"[Rs.{res_boot['boot_ci_median_lo']}, "
          f"Rs.{res_boot['boot_ci_median_hi']}]")
    print(f"     Bootstrap p-value    : {res_boot['bootstrap_p_value']}")
    print(f"     Significant          : {res_boot['significant']}")

    print("\n  D. DELTA METHOD")
    print(f"     Applicable       : {res_dm['applicable']}")
    print(f"     Decision         : {res_dm['explanation']}")

    # ── Comparison table ──────────────────────────────────────
    print("\n" + "="*60)
    print("  METHOD COMPARISON TABLE")
    print("="*60)
    rows = [
        ["Method",
         "Lift (Rs.)", "p-value", "Significant", "Notes"],
        ["─"*28, "─"*12, "─"*10, "─"*12, "─"*30],
        ["Log t-test (geo mean)",
         f"Rs.{res_log['geo_lift_rs']}",
         f"{res_log['p_value']:.4e}",
         str(res_log['significant']),
         "Geometric mean only; CI is ratio"],
        ["Log t-test (arith est.)",
         f"Rs.{res_log['arith_lift_rs']}",
         f"{res_log['p_value']:.4e}",
         str(res_log['significant']),
         "Log-normal arith correction"],
        ["Mann-Whitney U",
         "N/A (rank test)",
         f"{res_mw['p_value']:.4e}",
         str(res_mw['significant']),
         f"CLES={res_mw['CLES']}"],
        ["Bootstrap mean diff",
         f"Rs.{res_boot['obs_mean_diff_rs']}",
         f"{res_boot['bootstrap_p_value']}",
         str(res_boot['significant']),
         f"CI [{res_boot['boot_ci_mean_lo']},"
         f"{res_boot['boot_ci_mean_hi']}]"],
        ["Bootstrap median diff",
         f"Rs.{res_boot['obs_median_diff_rs']}",
         "—",
         str(res_boot['ci_excludes_zero_median']),
         f"CI [{res_boot['boot_ci_median_lo']},"
         f"{res_boot['boot_ci_median_hi']}]"],
        ["Delta method",
         "N/A",
         "N/A",
         "N/A",
         "Not applicable — simple average"],
    ]
    for row in rows:
        print(f"  {row[0]:<28} {row[1]:<16} {row[2]:<12}"
              f" {row[3]:<14} {row[4]}")

    # ── Save CSV ──────────────────────────────────────────────
    csv_rows = [
        {"method": "Log t-test (geometric mean)",
         "lift_rs": res_log["geo_lift_rs"],
         "p_value": res_log["p_value"],
         "significant": res_log["significant"],
         "ci_lo": res_log["ci_ratio_lo"],
         "ci_hi": res_log["ci_ratio_hi"],
         "notes": "CI is ratio, not additive Rs."},
        {"method": "Log t-test (arithmetic est.)",
         "lift_rs": res_log["arith_lift_rs"],
         "p_value": res_log["p_value"],
         "significant": res_log["significant"],
         "ci_lo": None, "ci_hi": None,
         "notes": "Log-normal arith correction"},
        {"method": "Mann-Whitney U",
         "lift_rs": None,
         "p_value": res_mw["p_value"],
         "significant": res_mw["significant"],
         "ci_lo": None, "ci_hi": None,
         "notes": f"CLES={res_mw['CLES']}"},
        {"method": "Bootstrap mean diff",
         "lift_rs": res_boot["obs_mean_diff_rs"],
         "p_value": res_boot["bootstrap_p_value"],
         "significant": res_boot["significant"],
         "ci_lo": res_boot["boot_ci_mean_lo"],
         "ci_hi": res_boot["boot_ci_mean_hi"],
         "notes": "10,000 resamples"},
        {"method": "Bootstrap median diff",
         "lift_rs": res_boot["obs_median_diff_rs"],
         "p_value": None,
         "significant": res_boot["ci_excludes_zero_median"],
         "ci_lo": res_boot["boot_ci_median_lo"],
         "ci_hi": res_boot["boot_ci_median_hi"],
         "notes": "10,000 resamples"},
        {"method": "Delta method",
         "lift_rs": None, "p_value": None,
         "significant": None,
         "ci_lo": None, "ci_hi": None,
         "notes": "Not applicable — simple average metric"},
    ]
    df_out = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUTPUT_DIR, "method_comparison.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  ✅ method_comparison.csv saved → {csv_path}")

    return {
        "log_ttest"  : res_log,
        "mann_whitney": res_mw,
        "bootstrap"  : res_boot,
        "delta"      : res_dm,
    }