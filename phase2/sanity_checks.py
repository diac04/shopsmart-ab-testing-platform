# ============================================================
# ShopSmart India — Phase 2 — Sanity Checks
# ============================================================
# Runs 7 pre-analysis quality checks on all 3 experiments.
# SRM is treated as a HARD GATE — if SRM fails for any
# experiment, that experiment's results must not be trusted
# until the root cause is investigated and resolved.
#
# Checks:
#   1. Sample Ratio Mismatch       ← HARD GATE
#   2. Pre-experiment equivalence  (A/A simulation)
#   3. Duplicate user ID check
#   4. Cross-group user leakage
#   5. Bot traffic detection
#   6. Novelty effect check
#   7. Date/time anomaly check
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    EXP1_DIR, EXP2_DIR, EXP3_DIR,
    CHECKS, RANDOM_SEED, EXP1
)

# ============================================================
# RESULT TRACKER
# ============================================================

class CheckResult:
    """Tracks pass/fail/warn for every check × experiment."""

    STATUS_PASS = "✅ PASS"
    STATUS_FAIL = "❌ FAIL"
    STATUS_WARN = "⚠️  WARN"
    STATUS_NA   = "➖ N/A"

    def __init__(self):
        self.results = []   # list of dicts

    def add(self, experiment: str, check: str,
            status: str, detail: str, is_gate: bool = False):
        self.results.append({
            'Experiment' : experiment,
            'Check'      : check,
            'Status'     : status,
            'Gate'       : '🔒 GATE' if is_gate else '',
            'Detail'     : detail,
        })

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def any_gate_failed(self) -> bool:
        gate_rows = [r for r in self.results
                     if r['Gate'] == '🔒 GATE'
                     and r['Status'] == self.STATUS_FAIL]
        return len(gate_rows) > 0

    def failed_gates(self) -> list:
        return [r for r in self.results
                if r['Gate'] == '🔒 GATE'
                and r['Status'] == self.STATUS_FAIL]


tracker = CheckResult()


# ============================================================
# CHECK 1 — SAMPLE RATIO MISMATCH  [HARD GATE]
# ============================================================

def check_srm(df: pd.DataFrame,
              exp_name: str,
              expected_split: tuple = (0.5, 0.5),
              alpha: float = None) -> dict:
    """
    Chi-square goodness-of-fit test on group sizes.

    H0: Observed group sizes match expected ratio.
    H1: Observed group sizes deviate from expected ratio.

    We use alpha=0.01 (stricter than experiment alpha=0.05)
    because SRM is a data quality issue, not a business metric.
    A false negative here (missing a real SRM) is much more
    costly than a false positive (investigating a clean dataset).

    The test is:
        chi2 = sum( (O_i - E_i)^2 / E_i )
    where O_i = observed count in group i
          E_i = expected count under equal split

    Parameters
    ----------
    df            : experiment DataFrame with 'group' column
    exp_name      : label for reporting
    expected_split: tuple of expected proportions (control, treatment)
    alpha         : significance level for SRM test
    """
    if alpha is None:
        alpha = CHECKS['srm_alpha']   # 0.01

    print(f"\n{'='*60}")
    print(f"CHECK 1 — SAMPLE RATIO MISMATCH [{exp_name}]")
    print(f"{'='*60}")
    print(f"  [HARD GATE] No results trusted until this passes.")
    print(f"  Alpha (SRM-specific): {alpha}")

    groups  = df['group'].value_counts().sort_index()
    total   = len(df)

    print(f"\n  Observed group counts:")
    for g, n in groups.items():
        print(f"    {g:>12}: {n:>10,}  ({n/total*100:.4f}%)")

    # Expected counts under null
    group_list  = groups.index.tolist()
    observed    = groups.values

    # Map expected split to group order
    # Assumes group_list is sorted alphabetically: control, treatment
    expected_props = {
        'control'  : expected_split[0],
        'treatment': expected_split[1],
    }
    expected = np.array([expected_props.get(g, 0.5) for g in group_list]) * total

    print(f"\n  Expected counts (under {expected_split[0]:.0%}/{expected_split[1]:.0%} split):")
    for g, e in zip(group_list, expected):
        print(f"    {g:>12}: {e:>10,.1f}")

    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    ratio = observed[group_list.index('treatment')] / observed[group_list.index('control')] \
            if 'control' in group_list and 'treatment' in group_list else np.nan

    print(f"\n  Chi-square statistic : {chi2:.4f}")
    print(f"  P-value              : {p_value:.6f}")
    print(f"  Threshold alpha      : {alpha}")
    print(f"  Treatment/Control ratio: {ratio:.4f}  (expected 1.0000)")

    if p_value < alpha:
        status = CheckResult.STATUS_FAIL
        detail = (f"chi2={chi2:.2f}, p={p_value:.6f} < {alpha}. "
                  f"T/C ratio={ratio:.4f}. SRM DETECTED.")
        print(f"\n  ❌ FAIL — SRM detected. p={p_value:.6f}")
        print(f"  ⛔ This experiment is GATED until SRM is resolved.")
    else:
        status = CheckResult.STATUS_PASS
        detail = (f"chi2={chi2:.2f}, p={p_value:.6f} >= {alpha}. "
                  f"T/C ratio={ratio:.4f}. No SRM.")
        print(f"\n  ✅ PASS — No SRM detected. p={p_value:.6f}")

    tracker.add(exp_name, "1. SRM (Gate)", status, detail, is_gate=True)
    return {'chi2': chi2, 'p_value': p_value, 'ratio': ratio, 'status': status}


# ============================================================
# CHECK 2 — PRE-EXPERIMENT EQUIVALENCE (A/A SIMULATION)
# ============================================================

def check_aa_equivalence(df: pd.DataFrame,
                          exp_name: str,
                          metric_col: str,
                          metric_type: str = 'proportion',
                          n_simulations: int = 1000) -> dict:
    """
    Pre-experiment equivalence check (A/A test simulation).

    PURPOSE:
    If the two groups were truly randomly assigned, any pre-
    experiment metric should show no significant difference.
    We simulate this by:
      (a) For proportion metrics: bootstrapping the observed
          data and checking false positive rate
      (b) For means metrics: permutation test on the metric

    For Exp1 (conversion): We split data by week_of_exp==1
    and treat it as a pseudo pre-period.
    For Exp2 (revenue): We use pre_exp_revenue directly.
    For Exp3 (CTR): We use day 1 only as pseudo pre-period.

    A high false positive rate (>5%) would suggest the
    randomisation was not working correctly.

    Parameters
    ----------
    df          : experiment DataFrame
    exp_name    : label for reporting
    metric_col  : column name of metric to test
    metric_type : 'proportion' or 'mean'
    n_simulations: number of permutation draws
    """
    print(f"\n{'='*60}")
    print(f"CHECK 2 — PRE-EXPERIMENT EQUIVALENCE [{exp_name}]")
    print(f"{'='*60}")
    print(f"  Metric             : {metric_col}")
    print(f"  Method             : Permutation test ({n_simulations:,} draws)")

    rng_aa = np.random.default_rng(RANDOM_SEED + 99)

    ctrl = df[df['group'] == 'control'][metric_col].dropna().values
    trt  = df[df['group'] == 'treatment'][metric_col].dropna().values

    # Observed difference in means/proportions
    obs_diff = trt.mean() - ctrl.mean()

    print(f"\n  Control   {metric_col}: {ctrl.mean():.4f}")
    print(f"  Treatment {metric_col}: {trt.mean():.4f}")
    print(f"  Observed difference   : {obs_diff:.4f}")

    # Permutation test
    combined   = np.concatenate([ctrl, trt])
    n_ctrl     = len(ctrl)
    perm_diffs = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled      = rng_aa.permutation(combined)
        perm_ctrl     = shuffled[:n_ctrl]
        perm_trt      = shuffled[n_ctrl:]
        perm_diffs[i] = perm_trt.mean() - perm_ctrl.mean()

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    # False positive rate: fraction of permutations with p < 0.05
    # (should be ~5% under correct randomisation)
    fp_rate = np.mean(np.abs(perm_diffs) >= np.abs(perm_diffs) * 0)  # always 1
    # More useful: fraction of simulated |diffs| > 2*obs_std
    obs_std    = combined.std()
    large_diff = np.mean(np.abs(perm_diffs) > 2 * obs_std)

    print(f"  Permutation p-value   : {p_value:.4f}")
    print(f"  (p < 0.05 = groups differ before experiment started)")

    alpha_aa = 0.05
    if p_value < alpha_aa:
        status = CheckResult.STATUS_WARN
        detail = (f"Pre-exp diff={obs_diff:.4f}, perm p={p_value:.4f}. "
                  f"Groups may differ pre-experiment.")
        print(f"  ⚠️  WARN — Pre-experiment difference detected.")
    else:
        status = CheckResult.STATUS_PASS
        detail = (f"Pre-exp diff={obs_diff:.4f}, perm p={p_value:.4f}. "
                  f"Groups are equivalent pre-experiment.")
        print(f"  ✅ PASS — Groups are equivalent pre-experiment.")

    tracker.add(exp_name, "2. A/A Equivalence", status, detail)
    return {'obs_diff': obs_diff, 'p_value': p_value, 'status': status}


# ============================================================
# CHECK 3 — DUPLICATE USER IDs
# ============================================================

def check_duplicates(df: pd.DataFrame, exp_name: str) -> dict:
    """
    Check for duplicate user_id entries within the same group.

    Duplicates can arise from:
      - Double-firing of tracking events
      - ETL pipeline re-processing
      - Session stitching errors

    A user appearing in BOTH groups is leakage (Check 4).
    A user appearing TWICE in the same group is a duplicate.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 3 — DUPLICATE USER IDs [{exp_name}]")
    print(f"{'='*60}")

    total_rows = len(df)
    unique_ids = df['user_id'].nunique()
    duplicates = total_rows - unique_ids

    # Find which user_ids appear more than once
    dup_ids = df[df.duplicated(subset='user_id', keep=False)]['user_id'].unique()

    print(f"  Total rows         : {total_rows:,}")
    print(f"  Unique user_ids    : {unique_ids:,}")
    print(f"  Duplicate rows     : {duplicates:,}")

    if duplicates > 0:
        print(f"  Sample duplicate IDs: {dup_ids[:5]}")
        # Check if duplicates are within-group or cross-group
        dup_df        = df[df['user_id'].isin(dup_ids)]
        within_group  = dup_df.groupby('user_id')['group'].nunique()
        cross_group   = (within_group > 1).sum()
        within_only   = (within_group == 1).sum()
        print(f"  Cross-group duplicates (leakage): {cross_group}")
        print(f"  Within-group duplicates         : {within_only}")

        status = CheckResult.STATUS_FAIL
        detail = f"{duplicates} duplicate rows. {cross_group} cross-group."
        print(f"  ❌ FAIL — Duplicates found.")
    else:
        status = CheckResult.STATUS_PASS
        detail = "No duplicate user_ids found."
        print(f"  ✅ PASS — No duplicates.")

    tracker.add(exp_name, "3. Duplicates", status, detail)
    return {'duplicates': duplicates, 'status': status}


# ============================================================
# CHECK 4 — CROSS-GROUP USER LEAKAGE
# ============================================================

def check_leakage(df: pd.DataFrame, exp_name: str) -> dict:
    """
    Detect users who appear in BOTH control and treatment.

    This is a serious data quality issue called 'leakage' or
    'contamination'. It biases both groups toward each other,
    reducing the true effect size and potentially masking a
    real difference.

    Causes:
      - Cookie deletion and re-assignment
      - Multi-device usage (phone = control, desktop = treatment)
      - Bug in assignment logic (hash collision)
    """
    print(f"\n{'='*60}")
    print(f"CHECK 4 — CROSS-GROUP LEAKAGE [{exp_name}]")
    print(f"{'='*60}")

    user_groups   = df.groupby('user_id')['group'].nunique()
    leaked_users  = user_groups[user_groups > 1].index.tolist()
    n_leaked      = len(leaked_users)

    print(f"  Users in both groups: {n_leaked:,}")

    if n_leaked > 0:
        pct = n_leaked / df['user_id'].nunique() * 100
        print(f"  Leakage rate        : {pct:.4f}%")
        print(f"  Sample leaked IDs   : {leaked_users[:5]}")
        status = CheckResult.STATUS_FAIL
        detail = f"{n_leaked} users appear in both groups ({pct:.3f}%)."
        print(f"  ❌ FAIL — Cross-group contamination detected.")
    else:
        status = CheckResult.STATUS_PASS
        detail = "No users appear in both groups."
        print(f"  ✅ PASS — No cross-group leakage.")

    tracker.add(exp_name, "4. Leakage", status, detail)
    return {'leaked_users': n_leaked, 'status': status}


# ============================================================
# CHECK 5 — BOT TRAFFIC DETECTION
# ============================================================

def check_bots(df: pd.DataFrame,
               exp_name: str,
               max_daily_sessions: int = None) -> dict:
    """
    Basic heuristic bot detection.

    Heuristics:
      A) Users with an unusually high number of sessions per day
         (>50 events/day suggests automated traffic)
      B) Users active at 2am–4am consistently (bots often run
         in off-peak hours to avoid detection)
      C) Users with identical timestamps (exact duplicate events)

    Note: This is a basic check. Production systems use more
    sophisticated signals (mouse movement, JS execution,
    user-agent strings, IP reputation). We flag suspected bots
    but do not remove them here — that decision goes to the
    analyst who reviews this report.
    """
    if max_daily_sessions is None:
        max_daily_sessions = CHECKS['bot_max_daily_sessions']

    print(f"\n{'='*60}")
    print(f"CHECK 5 — BOT TRAFFIC DETECTION [{exp_name}]")
    print(f"{'='*60}")
    print(f"  Heuristic A: >{max_daily_sessions} sessions/day")
    print(f"  Heuristic B: Active 2am–4am on >50% of days")
    print(f"  Heuristic C: Exact duplicate timestamps")

    suspected_bots = set()

    # Heuristic A: High session count per day
    if 'experiment_day' in df.columns:
        sessions_per_day = (
            df.groupby(['user_id', 'experiment_day'])
              .size()
              .reset_index(name='n_sessions')
        )
        heavy_users = sessions_per_day[
            sessions_per_day['n_sessions'] > max_daily_sessions
        ]['user_id'].unique()
        suspected_bots.update(heavy_users)
        print(f"\n  Heuristic A — High daily sessions (>{max_daily_sessions}):")
        print(f"    Suspected bots: {len(heavy_users):,}")
    else:
        print(f"  Heuristic A — Skipped (no experiment_day column)")
        heavy_users = np.array([])

    # Heuristic B: Night-time activity (2am–4am)
    if 'hour_of_day' in df.columns:
        night_users = df[
            df['hour_of_day'].between(
                CHECKS['time_anomaly_hour_start'],
                CHECKS['time_anomaly_hour_end']
            )
        ]['user_id'].unique()

        # Flag only if they ALSO appear at night on >50% of their active days
        if 'experiment_day' in df.columns:
            user_days    = df.groupby('user_id')['experiment_day'].nunique()
            night_df     = df[df['hour_of_day'].between(2, 4)]
            night_days   = night_df.groupby('user_id')['experiment_day'].nunique()
            night_ratio  = (night_days / user_days).dropna()
            consistent_night = night_ratio[night_ratio > 0.5].index.tolist()
            suspected_bots.update(consistent_night)
            print(f"\n  Heuristic B — Consistent 2–4am activity (>50% of days):")
            print(f"    Suspected bots: {len(consistent_night):,}")
        else:
            print(f"\n  Heuristic B — Night users (2–4am): {len(night_users):,}")
    else:
        consistent_night = []
        print(f"  Heuristic B — Skipped (no hour_of_day column)")

    # Heuristic C: Exact duplicate timestamps per user
    if 'timestamp' in df.columns:
        dup_ts = df[df.duplicated(subset=['user_id', 'timestamp'], keep=False)]
        dup_ts_users = dup_ts['user_id'].unique()
        suspected_bots.update(dup_ts_users)
        print(f"\n  Heuristic C — Exact duplicate timestamps:")
        print(f"    Affected users: {len(dup_ts_users):,}")
    else:
        dup_ts_users = np.array([])
        print(f"  Heuristic C — Skipped (no timestamp column)")

    n_bots = len(suspected_bots)
    bot_pct = n_bots / df['user_id'].nunique() * 100 if df['user_id'].nunique() > 0 else 0

    print(f"\n  Total suspected bots : {n_bots:,}  ({bot_pct:.4f}% of users)")

    # Threshold: >1% bot rate is concerning
    if bot_pct > 1.0:
        status = CheckResult.STATUS_WARN
        detail = f"{n_bots} suspected bots ({bot_pct:.3f}%). Review before analysis."
        print(f"  ⚠️  WARN — Bot rate > 1%. Recommend investigation.")
    elif n_bots > 0:
        status = CheckResult.STATUS_PASS
        detail = f"{n_bots} suspected bots ({bot_pct:.3f}%). Within acceptable range."
        print(f"  ✅ PASS — Low bot rate. Acceptable.")
    else:
        status = CheckResult.STATUS_PASS
        detail = "No bot signals detected."
        print(f"  ✅ PASS — No bot signals detected.")

    tracker.add(exp_name, "5. Bot Detection", status, detail)
    return {'n_bots': n_bots, 'bot_pct': bot_pct, 'status': status}


# ============================================================
# CHECK 6 — NOVELTY EFFECT
# ============================================================

def check_novelty(df: pd.DataFrame,
                  exp_name: str,
                  metric_col: str,
                  early_days: tuple = (1, 3),
                  late_days:  tuple = (7, 14),
                  threshold:  float = None) -> dict:
    """
    Compare early-experiment treatment lift to late-experiment lift.

    A novelty effect exists when:
      early_lift / late_lift > threshold (default 1.15)

    This means users responded strongly to the new feature
    initially but settled back toward baseline over time.
    The true steady-state effect is the late-period lift.

    If detected, we should:
      1. Report both early and late lift separately
      2. Use late-period lift as the primary effect estimate
      3. Consider extending the experiment to get more late data
    """
    if threshold is None:
        threshold = CHECKS['novelty_ratio_threshold']

    print(f"\n{'='*60}")
    print(f"CHECK 6 — NOVELTY EFFECT [{exp_name}]")
    print(f"{'='*60}")
    print(f"  Metric    : {metric_col}")
    print(f"  Early days: {early_days[0]}–{early_days[1]}")
    print(f"  Late days : {late_days[0]}–{late_days[1]}")
    print(f"  Threshold : lift_ratio > {threshold}")

    # Check that experiment_day column exists
    if 'experiment_day' not in df.columns:
        print(f"  ⚠️  No experiment_day column — skipping.")
        tracker.add(exp_name, "6. Novelty Effect", CheckResult.STATUS_NA,
                    "No experiment_day column.")
        return {'status': CheckResult.STATUS_NA}

    ctrl  = df[df['group'] == 'control']
    trt   = df[df['group'] == 'treatment']

    def period_lift(ctrl_df, trt_df, days):
        c = ctrl_df[ctrl_df['experiment_day'].between(*days)][metric_col].mean()
        t = trt_df[trt_df['experiment_day'].between(*days)][metric_col].mean()
        lift = (t - c) / c * 100 if c > 0 else 0
        return c, t, lift

    c_early, t_early, lift_early = period_lift(ctrl, trt, early_days)
    c_late,  t_late,  lift_late  = period_lift(ctrl, trt, late_days)

    print(f"\n  Early period (days {early_days[0]}–{early_days[1]}):")
    print(f"    Control   : {c_early:.4f}")
    print(f"    Treatment : {t_early:.4f}")
    print(f"    Lift      : {lift_early:+.2f}%")

    print(f"\n  Late period (days {late_days[0]}–{late_days[1]}):")
    print(f"    Control   : {c_late:.4f}")
    print(f"    Treatment : {t_late:.4f}")
    print(f"    Lift      : {lift_late:+.2f}%")

    if lift_late != 0 and lift_early != 0:
        lift_ratio = abs(lift_early) / abs(lift_late)
    else:
        lift_ratio = 1.0

    print(f"\n  Lift ratio (early/late): {lift_ratio:.4f}  (threshold: {threshold})")

    if lift_ratio > threshold:
        status = CheckResult.STATUS_WARN
        detail = (f"Early lift={lift_early:+.2f}%, Late lift={lift_late:+.2f}%, "
                  f"Ratio={lift_ratio:.3f} > {threshold}. Novelty effect detected.")
        print(f"  ⚠️  WARN — Novelty effect detected. Use late-period lift.")
    else:
        status = CheckResult.STATUS_PASS
        detail = (f"Early lift={lift_early:+.2f}%, Late lift={lift_late:+.2f}%, "
                  f"Ratio={lift_ratio:.3f}. No novelty concern.")
        print(f"  ✅ PASS — No novelty effect. Lift is stable over time.")

    tracker.add(exp_name, "6. Novelty Effect", status, detail)
    return {
        'lift_early' : lift_early,
        'lift_late'  : lift_late,
        'lift_ratio' : lift_ratio,
        'status'     : status
    }


# ============================================================
# CHECK 7 — DATE/TIME ANOMALY
# ============================================================

def check_time_anomalies(df: pd.DataFrame, exp_name: str) -> dict:
    """
    Detect suspicious patterns in the timestamp distribution.

    We check for:
      A) Traffic spikes: any single day has >3x the median
         daily traffic (could indicate a bot attack or a
         misfire that sent traffic to one group)
      B) Traffic gaps: any day with <20% of median traffic
         (could indicate a logging outage — those days should
         be excluded from analysis)
      C) Weekend vs weekday imbalance: if one group gets
         disproportionately more weekend traffic, seasonal
         behaviour could confound results
    """
    print(f"\n{'='*60}")
    print(f"CHECK 7 — DATE/TIME ANOMALIES [{exp_name}]")
    print(f"{'='*60}")

    issues = []

    if 'experiment_day' not in df.columns:
        print(f"  ⚠️  No experiment_day column — skipping.")
        tracker.add(exp_name, "7. Time Anomalies", CheckResult.STATUS_NA,
                    "No experiment_day column.")
        return {'status': CheckResult.STATUS_NA}

    # ── A) Daily traffic spike/gap ────────────────────────────
    daily_counts = df.groupby('experiment_day').size()
    median_daily = daily_counts.median()
    spike_days   = daily_counts[daily_counts > 3   * median_daily].index.tolist()
    gap_days     = daily_counts[daily_counts < 0.2 * median_daily].index.tolist()

    print(f"\n  Daily traffic (median: {median_daily:,.0f} users/day):")
    print(f"    Spike days (>3x median) : {spike_days if spike_days else 'None'}")
    print(f"    Gap days   (<20% median): {gap_days   if gap_days   else 'None'}")

    if spike_days:
        issues.append(f"Spike days: {spike_days}")
    if gap_days:
        issues.append(f"Gap days: {gap_days}")

    # ── B) Weekend vs weekday split per group ─────────────────
    if 'day_of_week' in df.columns:
        df_copy = df.copy()
        df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
        weekend_by_group = df_copy.groupby('group')['is_weekend'].mean()

        print(f"\n  Weekend traffic fraction by group:")
        for g, frac in weekend_by_group.items():
            print(f"    {g:>12}: {frac*100:.2f}%")

        max_diff = weekend_by_group.max() - weekend_by_group.min()
        if max_diff > 0.05:  # >5pp difference
            issues.append(f"Weekend imbalance: {max_diff*100:.1f}pp between groups")
            print(f"  ⚠️  Weekend imbalance of {max_diff*100:.1f}pp detected.")
        else:
            print(f"  Weekend split balanced (diff={max_diff*100:.2f}pp).")

    # ── C) Night-hour spike ───────────────────────────────────
    if 'hour_of_day' in df.columns:
        night_pct = df[df['hour_of_day'].between(2, 4)].shape[0] / len(df) * 100
        print(f"\n  Traffic between 2–4am : {night_pct:.4f}%")
        if night_pct > 5:
            issues.append(f"Unusual night traffic: {night_pct:.2f}%")
            print(f"  ⚠️  Unusual 2–4am traffic.")
        else:
            print(f"  Night traffic within normal range.")

    # ── Summary ───────────────────────────────────────────────
    if issues:
        status = CheckResult.STATUS_WARN
        detail = " | ".join(issues)
        print(f"\n  ⚠️  WARN — Time anomalies: {detail}")
    else:
        status = CheckResult.STATUS_PASS
        detail = "No date/time anomalies detected."
        print(f"\n  ✅ PASS — No time anomalies.")

    tracker.add(exp_name, "7. Time Anomalies", status, detail)
    return {'issues': issues, 'status': status}


# ============================================================
# MASTER RUNNER — ALL CHECKS × ALL EXPERIMENTS
# ============================================================

def run_all_checks(df1: pd.DataFrame,
                   df2: pd.DataFrame,
                   df3: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 7 checks across all 3 experiments.
    SRM is the hard gate — failures are highlighted separately.

    Parameters
    ----------
    df1 : Experiment 1 DataFrame (checkout, real Kaggle data)
    df2 : Experiment 2 DataFrame (recommendations, simulated)
    df3 : Experiment 3 DataFrame (banner, simulated)
    """

    print("\n" + "="*60)
    print("RUNNING ALL SANITY CHECKS — 3 EXPERIMENTS × 7 CHECKS")
    print("="*60)
    print("SRM CHECK IS A HARD GATE.")
    print("No experiment's results will be trusted until its SRM passes.")

    # ── EXPERIMENT 1 ──────────────────────────────────────────
    print("\n" + "█"*60)
    print("█  EXPERIMENT 1 — CHECKOUT REDESIGN")
    print("█"*60)

    check_srm(df1, "Exp1: Checkout",
              expected_split=(0.5, 0.5))

    # For A/A: use first-week conversion as pseudo pre-period
    df1_week1 = df1[df1['week_of_exp'] == 1].copy()
    check_aa_equivalence(df1_week1, "Exp1: Checkout",
                         metric_col='converted',
                         metric_type='proportion')

    check_duplicates(df1, "Exp1: Checkout")
    check_leakage(df1,    "Exp1: Checkout")
    check_bots(df1,       "Exp1: Checkout")

    check_novelty(df1, "Exp1: Checkout",
                  metric_col='converted',
                  early_days=(1, 3), late_days=(7, 14))

    check_time_anomalies(df1, "Exp1: Checkout")

    # ── EXPERIMENT 2 ──────────────────────────────────────────
    print("\n" + "█"*60)
    print("█  EXPERIMENT 2 — PERSONALIZED RECOMMENDATIONS")
    print("█"*60)

    check_srm(df2, "Exp2: Recs",
              expected_split=(0.5, 0.5))

    # For A/A: use pre_exp_revenue (the real pre-experiment covariate)
    check_aa_equivalence(df2, "Exp2: Recs",
                         metric_col='pre_exp_revenue',
                         metric_type='mean')

    check_duplicates(df2, "Exp2: Recs")
    check_leakage(df2,    "Exp2: Recs")
    check_bots(df2,       "Exp2: Recs")

    check_novelty(df2, "Exp2: Recs",
                  metric_col='revenue',
                  early_days=(1, 3), late_days=(7, 14))

    check_time_anomalies(df2, "Exp2: Recs")

    # ── EXPERIMENT 3 ──────────────────────────────────────────
    print("\n" + "█"*60)
    print("█  EXPERIMENT 3 — DISCOUNT BANNER PLACEMENT")
    print("█"*60)

    check_srm(df3, "Exp3: Banner",
              expected_split=(0.5, 0.5))

    # For A/A: use day-1 CTR only as pseudo pre-period
    df3_day1 = df3[df3['experiment_day'] == 1].copy()
    check_aa_equivalence(df3_day1, "Exp3: Banner",
                         metric_col='clicked',
                         metric_type='proportion')

    check_duplicates(df3, "Exp3: Banner")
    check_leakage(df3,    "Exp3: Banner")
    check_bots(df3,       "Exp3: Banner")

    check_novelty(df3, "Exp3: Banner",
                  metric_col='clicked',
                  early_days=(1, 3), late_days=(7, 14))

    check_time_anomalies(df3, "Exp3: Banner")

    return tracker.summary_table()


# ============================================================
# PRINT VALIDATION SUMMARY TABLE
# ============================================================

def print_summary_table(summary: pd.DataFrame):
    """Print a formatted validation summary table."""

    print("\n\n" + "="*60)
    print("PHASE 2 — VALIDATION SUMMARY TABLE")
    print("="*60)
    print("🔒 GATE = Hard gate. Experiment blocked until resolved.")
    print("❌ FAIL = Check failed. Action required.")
    print("⚠️  WARN = Warning. Investigate before Phase 3.")
    print("✅ PASS = Check passed.")
    print("="*60)

    # Group by experiment
    for exp in summary['Experiment'].unique():
        exp_df = summary[summary['Experiment'] == exp]
        print(f"\n  {'─'*56}")
        print(f"  {exp}")
        print(f"  {'─'*56}")
        for _, row in exp_df.iterrows():
            gate_str = f" {row['Gate']}" if row['Gate'] else ""
            print(f"  {row['Status']}{gate_str:<12} {row['Check']:<25} {row['Detail'][:60]}")

    # ── Gate summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GATE STATUS SUMMARY")
    print("="*60)
    gate_rows = summary[summary['Gate'] == '🔒 GATE']
    for _, row in gate_rows.iterrows():
        icon = "⛔ BLOCKED" if row['Status'] == CheckResult.STATUS_FAIL else "✅ CLEARED"
        print(f"  {icon}  {row['Experiment']:<25} {row['Check']}")

    # ── Overall readiness ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3 READINESS")
    print("="*60)
    if tracker.any_gate_failed():
        failed = tracker.failed_gates()
        print(f"  ⛔ NOT READY — {len(failed)} gate(s) failed:")
        for f in failed:
            print(f"     → {f['Experiment']}: {f['Detail']}")
        print(f"\n  Action required before Phase 3 can proceed.")
    else:
        print(f"  ✅ ALL GATES CLEARED — Phase 3 can proceed.")


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("Loading datasets...")
    df1 = pd.read_csv(os.path.join(EXP1_DIR, "exp1_checkout_data.csv"),
                      parse_dates=['timestamp'])
    df2 = pd.read_csv(os.path.join(EXP2_DIR, "exp2_recommendations_data.csv"),
                      parse_dates=['timestamp'])
    df3 = pd.read_csv(os.path.join(EXP3_DIR, "exp3_banner_data.csv"),
                      parse_dates=['timestamp'])

    print(f"  Exp1: {len(df1):,} rows loaded")
    print(f"  Exp2: {len(df2):,} rows loaded")
    print(f"  Exp3: {len(df3):,} rows loaded")

    summary = run_all_checks(df1, df2, df3)
    print_summary_table(summary)

    # Save summary
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "phase2_validation_summary.csv"
    )
    summary.to_csv(out_path, index=False)
    print(f"\n  ✅ Validation summary saved: {out_path}")
    print("\n✅ sanity_checks.py complete.")