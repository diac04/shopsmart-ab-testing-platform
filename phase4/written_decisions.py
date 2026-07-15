# phase4/written_decisions.py
# ============================================================
# The two written paragraphs required by Phase 4 spec:
#   1. Method justification — why Bootstrap on CUPED-adjusted
#      values is the primary method
#   2. Outlier verdict — is the lift robust or an artifact
#
# These are the answers to the questions senior interviewers
# actually ask. They are written as reportable paragraphs,
# not just bullet points.
# ============================================================

import os
import json
from phase4.config import OUTPUT_DIR


# ── Decision 1: Method justification ─────────────────────────────────────────

METHOD_JUSTIFICATION = """
WRITTEN DECISION 1 — METHOD JUSTIFICATION
==========================================

For the final report, the primary method is the bootstrap confidence
interval on the CUPED-adjusted mean difference, yielding a lift of
Rs.18.90 (95% CI: Rs.8.92 to Rs.28.91, p < 0.001).

Here is the reasoning for choosing this over the alternatives:

The log-transform t-test was rejected as the primary method for two
reasons. First, it operates on purchasers only, silently discarding
the 34% of users with zero revenue. Since zero-revenue users are a
meaningful economic signal — a user who did not purchase is genuinely
different from one who did — excluding them biases the estimate upward
and misrepresents the population-level effect. Second, and more
critically, the back-transformation caveat is non-trivial: exp(mu) on
the log scale gives the geometric mean, not the arithmetic mean.
Reporting a geometric-mean lift to a business stakeholder who is
thinking in terms of total revenue is technically incorrect. The
log-normal arithmetic correction (exp(mu + sigma^2/2)) exists but
introduces additional assumptions about distributional form that are
not necessary when a better method is available. The log t-test is a
useful robustness check — and it confirms significance — but it should
not be the headline number.

The Mann-Whitney U test was rejected as the primary method because it
does not produce an estimate of the mean revenue difference in rupees.
The CLES of 0.528 tells us that a treatment user has a 52.8% chance of
higher revenue than a control user, which is a valid finding, but it
cannot be directly converted to a business impact figure. A product
manager cannot take "CLES=0.528" to a budget conversation. Mann-Whitney
is retained as a robustness check confirming that the distributional
shift is real (p=1.97e-08), but it is not the reportable estimate.

The delta method was explicitly determined to be inapplicable. Revenue-
per-user in this design is a simple arithmetic mean — total group
revenue divided by n_assigned_users, where the denominator is fixed by
randomisation. The delta method is required only when both numerator and
denominator are random variables (for example, revenue-per-session where
one user may have multiple sessions). That condition does not hold here.

The bootstrap CI on the CUPED-adjusted mean difference is the correct
choice for three reasons. First, it makes no distributional assumption —
it is valid for right-skewed, zero-inflated, heavy-tailed data by
construction. Second, it operates on all users including zeros, so it
estimates the true population-level effect. Third, and most importantly,
it is applied after CUPED adjustment, which removes the Rs.37.46 of
pre-experiment revenue imbalance that was inflating the raw lift. The
CUPED step is not optional here — Phase 2 flagged a statistically
significant pre-experiment difference (p=0.000, permutation test), and
without correction we would have reported a lift of Rs.56.35 that is
nearly 3x the true causal effect. The bootstrap CI then provides honest
uncertainty quantification around the corrected estimate. Both the Welch
t-test and bootstrap agree on Rs.18.90 with overlapping CIs, which
gives us confidence the result is not a computational artifact.
"""


# ── Decision 2: Outlier verdict ───────────────────────────────────────────────

OUTLIER_VERDICT = """
WRITTEN DECISION 2 — OUTLIER VERDICT
======================================

The lift is ROBUST. It is not an artifact of a small cluster of
high-value outlier accounts.

The evidence is as follows. When we remove the top 0.1% of users by
revenue from both groups, the lift moves from Rs.56.35 to Rs.55.22 —
a change of Rs.1.13, or 2.0%. Removing the top 1.0% moves it to
Rs.53.89 — a change of Rs.2.46, or 4.4%. Winsorising at the 1st and
99th percentile produces Rs.55.08. Across all six outlier-treatment
scenarios tested, the lift remains in the range Rs.53.47 to Rs.55.22,
always statistically significant, always with a confidence interval
that excludes zero. A result that is driven by a handful of whale
accounts would collapse under even mild trimming. This one does not.

However, the band analysis reveals something more nuanced. Splitting
purchasers into revenue bands, the lift across bands 1 through 5 (the
bottom 99% of purchasers) is small and inconsistent in direction:
+Rs.2.55 in the bottom quartile, +Rs.4.10 in the second quartile,
+Rs.6.56 in the third, +Rs.0.20 in the 75th-90th percentile band, and
-Rs.11.77 in the 90th-99th band. None of these are economically
meaningful individually. The top 1% band shows +Rs.111.79, but with
only n=30 control and n=56 treatment users — sample sizes too small
to be reliable, and the imbalance in counts (30 vs 56) is itself a
symptom of the pre-experiment group composition difference that CUPED
corrects for.

The conclusion is that the raw lift of Rs.56.35 is not driven by
outliers in the statistical sense — it survives trimming and
winsorisation robustly. It is driven by pre-experiment group
composition: the treatment group contained users who were systematically
higher spenders before the experiment began. That is a selection bias
issue, not an outlier issue. CUPED correctly identifies and removes
this bias, reducing the lift to the honest causal estimate of Rs.18.90.
This is the number we stand behind. It is statistically significant
(p < 0.001), robust to distributional assumptions (confirmed by both
t-test and bootstrap), and corrected for the pre-experiment imbalance
flagged in Phase 2.
"""


# ── Print and save ────────────────────────────────────────────────────────────

def run_written_decisions() -> dict:
    print("\n" + "="*60)
    print("  PHASE 4 — WRITTEN DECISIONS")
    print("="*60)

    print(METHOD_JUSTIFICATION)
    print(OUTLIER_VERDICT)

    # Save to text file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_path = os.path.join(OUTPUT_DIR, "written_decisions.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(METHOD_JUSTIFICATION)
        f.write("\n\n")
        f.write(OUTLIER_VERDICT)

    print(f"  Saved -> {txt_path}")

    return {
        "method_justification": METHOD_JUSTIFICATION.strip(),
        "outlier_verdict"     : OUTLIER_VERDICT.strip(),
    }