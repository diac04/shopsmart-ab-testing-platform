# phase7/retro_paragraphs.py
"""
Three short per-experiment retrospective paragraphs +
one full post-mortem narrative (300-400 words, VP-level).
"""

import os
from phase7.config import OUTPUT_DIR


# ── Short retros ────────────────────────────────────────────────────────────

RETRO_EXP1 = """
EXPERIMENT 1 RETROSPECTIVE — UPI Checkout Redesign
---------------------------------------------------
The defining complication was the Sample Ratio Mismatch (T/C=1.083,
chi2=447, p≈0), which invalidated all inferential statistics before
analysis even began. If I ran this again I would:

1. Add a real-time SRM dashboard (alert threshold: chi2 p<0.01) so the
   experiment is paused within 24 hours of launch, not discovered at
   analysis time after 279,000 users have already been enrolled.

2. Pre-register a guardrail metric for traffic-split fidelity alongside
   the primary conversion metric. Guardrail breach = automatic stop.

3. Reduce the planned sample size from n=37,671/group to n=25,000/group
   for the pilot, running a 3-day ramp (1%→10%→50%) to catch
   infrastructure bugs early.

4. Add a holdout log for bot traffic: bots inflated the treatment arm
   differently than control — a pre-analysis bot filter rule (based on
   session duration < 3 s) should be part of the data cleaning script,
   not an optional sensitivity check.

5. For the metric itself: conversion rate is correct, but adding a
   secondary guardrail on checkout completion time (ms) would help
   distinguish UX friction from genuine preference change.
"""

RETRO_EXP2 = """
EXPERIMENT 2 RETROSPECTIVE — Personalized Recommendations
----------------------------------------------------------
The defining complication was pre-experiment group composition bias:
treatment users had ₹59.10 baseline revenue vs control ₹54.59 (p=0.000),
making the raw lift of ₹56.35/user look enormous and unreliable.
CUPED corrected this, but the correction was applied post-hoc.
If I ran this again I would:

1. Pre-register the outlier-handling protocol BEFORE running the
   experiment: specify trimming thresholds (e.g., 99th percentile cap),
   the CUPED covariate window (prior 30-day revenue), and which method
   is primary (bootstrap on CUPED-adjusted means). No post-hoc choices.

2. Enforce randomisation balance checks at assignment time — if the
   pre-experiment covariate mean differs by >5% between arms at n=500,
   halt and re-randomise before scaling.

3. Use stratified randomisation (or re-randomisation) on prior-30-day
   revenue buckets (₹0, ₹1–₹100, ₹100+) so group imbalance is
   structurally impossible, reducing reliance on CUPED as a rescue tool.

4. Extend the analysis window: Days 7–14 is only 8 days. A 21-day window
   (dropping Days 1–6 for novelty) would give a more stable estimate,
   especially for a recommendation engine where personalisation quality
   improves with interaction history.

5. Add a guardrail on recommendation diversity (e.g., % of users shown
   the same top-1 product) to catch model collapse or filter-bubble risk.
"""

RETRO_EXP3 = """
EXPERIMENT 3 RETROSPECTIVE — Discount Banner Placement
-------------------------------------------------------
The defining complication was the absence of tablet data and the
temptation to restrict rollout to mobile-only based on segment sizes,
despite the interaction term being non-significant (p=0.18).
If I ran this again I would:

1. Pre-register the device segmentation plan — explicitly name mobile,
   desktop, AND tablet as pre-registered subgroups, with a plan for
   what to do if one segment has insufficient data (minimum n per
   segment = 10,000 per arm).

2. Use a higher multiple-testing budget for subgroup analysis: instead
   of Bonferroni (conservative), pre-register Holm-Bonferroni with
   a primary/secondary hierarchy (overall CTR = primary;
   device-level = secondary confirmatory). This recovers some power
   while maintaining FWER control.

3. Instrument the click-to-conversion funnel before the experiment,
   so we have a direct revenue metric rather than estimating revenue
   from CTR × assumed 15% conversion rate. That 15% assumption is the
   single biggest source of uncertainty in the ₹ impact figures.

4. Run the experiment for a full 4-week cycle (not just 2 weeks) to
   capture weekday/weekend variation in banner-click behaviour, which
   is likely larger on mobile than desktop.

5. Add a guardrail on banner fatigue: measure CTR decay over time
   (day-by-day rolling average) to ensure the lift is not driven
   purely by novelty in the first 3 days.
"""


# ── Full post-mortem narrative (Experiment 2 — VP level) ───────────────────

POSTMORTEM_EXP2 = """
FULL POST-MORTEM — Experiment 2: Personalized Recommendations
Audience: VP of Product / Chief Analytics Officer
Word count: ~370 words
==============================================================

HYPOTHESIS
We hypothesised that replacing the current "top-sellers" recommendation
widget with a personalised ML model (collaborative filtering on 90-day
purchase history) would increase revenue per user. The pre-registered
minimum detectable effect was ₹10/user/month, based on an internal
benchmark from a comparable recommendations rollout at a peer company.

WHAT WE FOUND
The personalised model works — but the headline number is not what it
first appeared. Our initial analysis produced a lift of ₹56.35/user/month,
which would have translated to ₹338 Crore/year at full rollout. That
figure was wrong, and catching the error before shipping is one of the
most valuable things this A/B testing platform has done.

THE SURPRISE
When we ran CUPED (Control using Pre-Experiment Data) as a variance
reduction step, we discovered that treatment users had ₹4.51 higher
average revenue than control users in the 30 days BEFORE the experiment
began (₹59.10 vs ₹54.59, p<0.001). The randomisation algorithm had
created groups that were not comparable at baseline. This imbalance —
not the recommendation engine — was responsible for ₹37.46 of the
observed ₹56.35 lift. After correcting for pre-experiment bias, the
true CUPED-adjusted lift is ₹18.90/user/month (95% CI: ₹8.92–₹28.91).

We also ran an outlier sensitivity check, trimming the top 1%, 5%, and
10% of revenue values. The raw lift remained stable at ₹53–₹56 across
all trimming thresholds. This ruled out extreme users as the cause —
confirming that the root cause was group composition, not a handful of
high-spenders inflating the treatment mean.

BUSINESS IMPACT
The correct conservative figure is ₹53.52 Crore/year. This is still a
large, statistically robust, and economically significant result. The
payback on the estimated ₹20 lakh ML infrastructure investment is under
three weeks on the conservative estimate.

WHAT I WOULD DO DIFFERENTLY
Three things: First, pre-register the CUPED covariate and the
outlier-handling thresholds before unblinding — we made those decisions
with knowledge of the data, which is an analytical conflict of interest
even if the decisions were defensible. Second, enforce a randomisation
balance check at n=500: if pre-experiment revenue differs by >5% between
arms, halt and re-randomise before scaling to 13,000 users. Third, use
stratified randomisation on revenue buckets (₹0 / ₹1–₹100 / ₹100+) to
make group imbalance structurally impossible, so we need CUPED as a
precision tool rather than a bias-correction rescue.

The recommendation is clear: SHIP. But quote ₹53 Crore to leadership,
not ₹113 Crore, and explain why — it builds more credibility than a
headline number that actuals will underperform.
"""


def build_retro_text() -> str:
    sections = [
        "=" * 70,
        "PHASE 7 — PER-EXPERIMENT RETROSPECTIVES",
        "=" * 70,
        "",
        RETRO_EXP1.strip(),
        "",
        "-" * 70,
        "",
        RETRO_EXP2.strip(),
        "",
        "-" * 70,
        "",
        RETRO_EXP3.strip(),
        "",
        "=" * 70,
        "PHASE 7 — FULL POST-MORTEM NARRATIVE (VP LEVEL)",
        "=" * 70,
        "",
        POSTMORTEM_EXP2.strip(),
        "",
    ]
    return "\n".join(sections)


def save_retro(text: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "retro_paragraphs.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  ✓ retro_paragraphs.txt → {path}")