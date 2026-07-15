# phase3/statistical_tests.py
"""
All statistical tests for Experiment 1 (binary conversion metric).

Tests implemented
─────────────────
1. Descriptive statistics per group
2. Two-proportion Z-test  (primary)
3. Chi-square test of independence (confirmatory)
4. Cohen's h effect size
5. 95 % CI on the difference in proportions
6. Relative lift
7. Pre-registered vs observed comparison
"""

import numpy as np
import scipy.stats as stats
from  statsmodels.stats.proportion import (
    proportions_ztest,
    proportion_confint,
)
from phase3.config import ALPHA, CI_LEVEL, PRE_REGISTERED


# ══════════════════════════════════════════════════════════════════════════════
# 1. Descriptive statistics
# ══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(ctrl_n, ctrl_conv, trt_n, trt_conv) -> dict:
    ctrl_rate = ctrl_conv / ctrl_n
    trt_rate  = trt_conv  / trt_n
    return {
        "control": {
            "n"              : ctrl_n,
            "conversions"    : ctrl_conv,
            "conversion_rate": round(ctrl_rate, 6),
            "non_conversions": ctrl_n - ctrl_conv,
        },
        "treatment": {
            "n"              : trt_n,
            "conversions"    : trt_conv,
            "conversion_rate": round(trt_rate, 6),
            "non_conversions": trt_n - trt_conv,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Two-proportion Z-test
# ══════════════════════════════════════════════════════════════════════════════

def two_prop_ztest(ctrl_n, ctrl_conv, trt_n, trt_conv) -> dict:
    """
    H0: p_treatment == p_control
    H1: p_treatment != p_control  (two-sided)
    """
    counts = np.array([trt_conv,  ctrl_conv])
    nobs   = np.array([trt_n,     ctrl_n])

    z_stat, p_value = proportions_ztest(counts, nobs, alternative="two-sided")

    return {
        "test"       : "Two-proportion Z-test",
        "z_statistic": round(float(z_stat), 4),
        "p_value"    : round(float(p_value), 6),
        "alpha"      : ALPHA,
        "significant": bool(p_value < ALPHA),
        "direction"  : "treatment > control" if z_stat > 0 else
                       "treatment < control",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Chi-square test of independence
# ══════════════════════════════════════════════════════════════════════════════

def chi_square_test(ctrl_n, ctrl_conv, trt_n, trt_conv) -> dict:
    """
    Contingency table:
                 Converted   Not Converted
    Control        a              b
    Treatment      c              d
    """
    a = ctrl_conv;   b = ctrl_n - ctrl_conv
    c = trt_conv;    d = trt_n  - trt_conv

    contingency = np.array([[a, b], [c, d]])
    chi2, p, dof, expected = stats.chi2_contingency(contingency,
                                                     correction=False)
    return {
        "test"       : "Chi-square test of independence",
        "chi2_stat"  : round(float(chi2), 4),
        "p_value"    : round(float(p), 6),
        "dof"        : int(dof),
        "alpha"      : ALPHA,
        "significant": bool(p < ALPHA),
        "contingency_table": {
            "control_converted"    : int(a),
            "control_not_converted": int(b),
            "treatment_converted"  : int(c),
            "treatment_not_converted": int(d),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Cohen's h
# ══════════════════════════════════════════════════════════════════════════════

def cohens_h(p1: float, p2: float) -> dict:
    """
    Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    p1 = treatment rate, p2 = control rate
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    h    = abs(phi1 - phi2)

    # Interpretation thresholds (Cohen 1988)
    if h < 0.20:
        magnitude = "small"
    elif h < 0.50:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "cohens_h"        : round(float(h), 4),
        "magnitude"       : magnitude,
        "prereg_cohens_h" : PRE_REGISTERED["cohens_h_prereg"],
        "vs_prereg"       : round(float(h - PRE_REGISTERED["cohens_h_prereg"]),
                                  4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. 95 % Confidence interval on difference in proportions
# ══════════════════════════════════════════════════════════════════════════════

def ci_difference(ctrl_n, ctrl_conv, trt_n, trt_conv) -> dict:
    """
    Newcombe's method (recommended for proportion differences).
    Falls back to normal approximation explicitly.
    """
    p1 = trt_conv  / trt_n
    p2 = ctrl_conv / ctrl_n
    diff = p1 - p2

    # Normal-approximation CI
    se   = np.sqrt(p1*(1-p1)/trt_n + p2*(1-p2)/ctrl_n)
    z_cv = stats.norm.ppf(1 - (1 - CI_LEVEL) / 2)   # 1.96 for 95%
    lo   = diff - z_cv * se
    hi   = diff + z_cv * se

    # Wilson interval for each proportion (used in display)
    ctrl_lo, ctrl_hi = proportion_confint(ctrl_conv, ctrl_n,
                                          alpha=1-CI_LEVEL, method="wilson")
    trt_lo,  trt_hi  = proportion_confint(trt_conv,  trt_n,
                                          alpha=1-CI_LEVEL, method="wilson")

    return {
        "point_estimate_diff"         : round(diff, 6),
        "ci_lower"                    : round(lo,   6),
        "ci_upper"                    : round(hi,   6),
        "ci_level"                    : CI_LEVEL,
        "method"                      : "Normal approximation",
        "control_wilson_ci"           : [round(ctrl_lo,6), round(ctrl_hi,6)],
        "treatment_wilson_ci"         : [round(trt_lo,6),  round(trt_hi,6)],
        "ci_excludes_zero"            : bool(lo > 0 or hi < 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Relative lift
# ══════════════════════════════════════════════════════════════════════════════

def relative_lift(ctrl_rate: float, trt_rate: float) -> dict:
    lift_abs = trt_rate - ctrl_rate
    lift_rel = lift_abs / ctrl_rate if ctrl_rate > 0 else 0.0
    return {
        "control_rate"        : round(ctrl_rate, 6),
        "treatment_rate"      : round(trt_rate,  6),
        "absolute_lift"       : round(lift_abs,  6),
        "relative_lift_pct"   : round(lift_rel * 100, 3),
        "prereg_mde_abs"      : PRE_REGISTERED["mde_absolute"],
        "prereg_mde_rel_pct"  : PRE_REGISTERED["mde_relative_pct"],
        "observed_vs_mde"     : "above MDE" if abs(lift_abs) >=
                                PRE_REGISTERED["mde_absolute"] else "below MDE",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Master runner — returns full results dict for one analysis variant
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests(ctrl_n, ctrl_conv, trt_n, trt_conv,
                  variant_label: str) -> dict:
    ctrl_rate = ctrl_conv / ctrl_n
    trt_rate  = trt_conv  / trt_n

    desc   = descriptive_stats(ctrl_n, ctrl_conv, trt_n, trt_conv)
    ztest  = two_prop_ztest(ctrl_n, ctrl_conv, trt_n, trt_conv)
    chi2   = chi_square_test(ctrl_n, ctrl_conv, trt_n, trt_conv)
    h      = cohens_h(trt_rate, ctrl_rate)
    ci     = ci_difference(ctrl_n, ctrl_conv, trt_n, trt_conv)
    lift   = relative_lift(ctrl_rate, trt_rate)

    return {
        "variant"            : variant_label,
        "descriptive"        : desc,
        "ztest"              : ztest,
        "chi_square"         : chi2,
        "cohens_h"           : h,
        "confidence_interval": ci,
        "lift"               : lift,
    }