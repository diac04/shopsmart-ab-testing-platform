# ============================================================
# ShopSmart India — Phase 2 — Data Simulator
# ============================================================
# Generates realistic simulated datasets for:
#   Experiment 2 — Personalized Recommendations
#     - Right-skewed revenue (log-normal base)
#     - Novelty effect: early lift decays over ~10 days
#     - Pre-experiment revenue covariate (corr ~0.65 with during-exp)
#   Experiment 3 — Discount Banner Placement
#     - Overall CTR (binary)
#     - Segmented CTR (Mobile vs Desktop, Bonferroni-corrected)
# ============================================================

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    EXP2_DIR, EXP3_DIR,
    EXP2, EXP3,
    RANDOM_SEED
)

rng = np.random.default_rng(RANDOM_SEED)

# ============================================================
# SHARED UTILITY — Timestamp Generator
# ============================================================

def make_timestamps(n: int,
                    start: str = "2024-01-01",
                    days: int = 14,
                    seed: int = RANDOM_SEED) -> pd.Series:
    """
    Generate n random timestamps spread over `days` days,
    weighted toward business hours (8am–10pm IST) to reflect
    realistic Indian e-commerce traffic patterns.

    Traffic shape:
      - Low    : 0am–7am   (10% of daily traffic)
      - Medium : 7am–11am  (20%)
      - Peak   : 11am–9pm  (60%)
      - Low    : 9pm–12am  (10%)
    """
    rng_ts = np.random.default_rng(seed)

    start_ts  = pd.Timestamp(start)
    end_ts    = start_ts + pd.Timedelta(days=days)
    total_sec = int((end_ts - start_ts).total_seconds())

    # Random uniform seconds within window
    seconds_offset = rng_ts.integers(0, total_sec, size=n)

    # Hour-of-day weighting
    hours = (seconds_offset // 3600) % 24
    weight = np.where(hours < 7,  0.10 / 7,
             np.where(hours < 11, 0.20 / 4,
             np.where(hours < 21, 0.60 / 10,
                                  0.10 / 3)))

    # Rejection sampling for weighted hours
    accept_prob  = weight / weight.max()
    accept_mask  = rng_ts.random(size=n) < accept_prob
    n_accepted   = accept_mask.sum()

    # Fill rejected slots with uniform (good enough for simulation)
    final_seconds        = seconds_offset.copy()
    n_rejected           = n - n_accepted
    fill_seconds         = rng_ts.integers(
                               int(11 * 3600), int(21 * 3600), size=n_rejected
                           )
    final_seconds[~accept_mask] = (
        rng_ts.integers(0, days, size=n_rejected) * 86400
        + fill_seconds
    )

    timestamps = pd.to_datetime(
        [start_ts + pd.Timedelta(seconds=int(s)) for s in final_seconds]
    )
    return pd.Series(timestamps)


# ============================================================
# EXPERIMENT 2 — Personalized Recommendations
# ============================================================

def simulate_experiment2() -> pd.DataFrame:
    """
    Simulate Experiment 2: Revenue per User A/B test.

    Design decisions:
    -----------------
    1. LOG-NORMAL revenue base
       Real e-commerce revenue is right-skewed. We use log-normal
       so the bulk of users spend Rs.200–500 but a tail spends
       Rs.5000+. This matches Indian e-commerce patterns.

    2. NOVELTY EFFECT injection
       Treatment users in days 1–10 receive an additional boost
       that decays geometrically each day. By day 10 the boost
       is near zero. This simulates users initially excited by
       the new recommendation widget but returning to baseline
       behaviour as novelty wears off.

       Boost on day d = novelty_boost * (novelty_decay ^ (d-1))
       Day 1 boost: Rs.120   Day 5: Rs.32   Day 10: Rs.6

    3. PRE-EXPERIMENT COVARIATE
       Each user has a pre-experiment revenue (last 30 days).
       This is correlated with their during-experiment revenue
       (rho ~ 0.65) but not identical — it captures stable
       user-level spending habits. CUPED in Phase 4 will use
       this to reduce variance and increase test sensitivity.

    4. ZERO-REVENUE users
       ~35% of users visit but do not purchase. We model this
       with a Bernoulli purchase indicator, then multiply by
       the log-normal spend amount.
    """

    print("\n" + "="*60)
    print("SIMULATING EXPERIMENT 2 — PERSONALIZED RECOMMENDATIONS")
    print("="*60)

    n_per_group  = EXP2['n_per_group']          # 566 → we'll use more for realism
    # Use 14 days × 160k daily / 2 groups for a fuller dataset
    # but cap at a reasonable simulation size
    n_sim        = max(n_per_group * 20, 10_000)  # ~11,320 users per group minimum
    # For realism use 14 days of traffic
    n_control    = n_sim
    n_treatment  = n_sim

    print(f"  Users per group    : {n_sim:,}")
    print(f"  Total users        : {n_sim * 2:,}")
    print(f"  Experiment days    : 14")
    print(f"  Novelty boost day1 : Rs.{EXP2['novelty_boost']:.0f}")
    print(f"  Novelty decay rate : {EXP2['novelty_decay']} per day")
    print(f"  Pre-exp corr (rho) : {EXP2['pre_exp_corr']}")

    # ── User IDs ──────────────────────────────────────────────
    control_ids   = np.arange(100_001, 100_001 + n_control)
    treatment_ids = np.arange(200_001, 200_001 + n_treatment)

    # ── Timestamps ────────────────────────────────────────────
    print("\n  Generating timestamps...")
    control_ts   = make_timestamps(n_control,   days=14, seed=RANDOM_SEED)
    treatment_ts = make_timestamps(n_treatment, days=14, seed=RANDOM_SEED + 1)

    # ── Experiment day (1-indexed) ────────────────────────────
    start = pd.Timestamp("2024-01-01")
    control_day   = ((control_ts   - start).dt.days + 1).clip(1, 14)
    treatment_day = ((treatment_ts - start).dt.days + 1).clip(1, 14)

    # ── Log-normal revenue parameters ────────────────────────
    # We want median spend ≈ Rs.400, mean ≈ Rs.850 for control
    # For log-normal: mean = exp(mu + sigma^2/2)
    # We tune mu and sigma to match baseline_mean and baseline_std
    target_mean = EXP2['baseline_mean']   # 850
    target_std  = EXP2['baseline_std']    # 420

    # Solve for log-normal params
    # variance = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    cv2   = (target_std / target_mean) ** 2   # coefficient of variation squared
    sigma = np.sqrt(np.log(1 + cv2))
    mu    = np.log(target_mean) - sigma**2 / 2

    print(f"\n  Log-normal params  : mu={mu:.4f}  sigma={sigma:.4f}")
    print(f"  Implied mean       : Rs.{np.exp(mu + sigma**2/2):.2f}")

    # ── Purchase indicator (Bernoulli) ────────────────────────
    purchase_prob = 0.65   # 65% of users make a purchase

    # ── CONTROL group revenue ─────────────────────────────────
    purchase_control = (rng.random(n_control) < purchase_prob).astype(int)
    raw_spend_ctrl   = rng.lognormal(mu, sigma, size=n_control)
    revenue_control  = purchase_control * raw_spend_ctrl

    # ── TREATMENT group revenue ───────────────────────────────
    # Treatment mean = Rs.920 → scale log-normal mean up
    lift_ratio    = EXP2['treatment_mean'] / EXP2['baseline_mean']  # 920/850
    mu_treat      = mu + np.log(lift_ratio)   # shift log-mean

    purchase_treat = (rng.random(n_treatment) < purchase_prob + 0.02).astype(int)
    raw_spend_trt  = rng.lognormal(mu_treat, sigma, size=n_treatment)
    revenue_treat  = purchase_treat * raw_spend_trt

    # ── INJECT NOVELTY EFFECT ─────────────────────────────────
    print("\n  Injecting novelty effect into treatment group...")
    novelty_boost = EXP2['novelty_boost']    # Rs.120 on day 1
    novelty_decay = EXP2['novelty_decay']    # 0.75 per day

    day_boost = np.zeros(n_treatment)
    for d in range(1, EXP2['novelty_days'] + 1):
        mask         = (treatment_day == d).values
        boost_today  = novelty_boost * (novelty_decay ** (d - 1))
        day_boost[mask] = boost_today

    # Apply boost only to purchasers (non-zero revenue)
    purchaser_mask         = revenue_treat > 0
    revenue_treat_novelty  = revenue_treat.copy()
    revenue_treat_novelty[purchaser_mask] += day_boost[purchaser_mask]

    print(f"  Day 1  boost: Rs.{novelty_boost * (novelty_decay**0):.2f}")
    print(f"  Day 5  boost: Rs.{novelty_boost * (novelty_decay**4):.2f}")
    print(f"  Day 10 boost: Rs.{novelty_boost * (novelty_decay**9):.2f}")
    print(f"  Day 11+boost: Rs.0.00 (novelty worn off)")

    # ── PRE-EXPERIMENT COVARIATE ──────────────────────────────
    print("\n  Generating pre-experiment revenue covariate...")
    # We want pre_exp_revenue correlated with during-experiment revenue
    # Method: Cholesky decomposition to create correlated log-normals
    # For simplicity: pre = rho * during_revenue + sqrt(1-rho^2) * noise
    # We work in log-space for the correlation to be meaningful

    rho = EXP2['pre_exp_corr']   # 0.65

    def make_pre_exp(revenue_during: np.ndarray,
                     rho: float,
                     seed_offset: int) -> np.ndarray:
           """
           Create pre-experiment revenue correlated with
           during-experiment revenue at approximately rho.

           Fix: Separate purchasers from non-purchasers.
           Non-purchasers get pre_exp = 0 (they are habitual
           non-buyers). Purchasers get a correlated log-normal.
           This avoids the zero-inflation destroying correlation.
           """
           
           rng_pre    = np.random.default_rng(RANDOM_SEED + seed_offset)
           pre_exp    = np.zeros(len(revenue_during))

           # Only correlate among purchasers (non-zero revenue)
           buyer_mask = revenue_during > 0
           r_buyers   = revenue_during[buyer_mask]

           log_during = np.log(r_buyers)                        # safe — all > 0
           noise      = rng_pre.normal(
                         loc   = 0,
                         scale = log_during.std(),
                         size  = buyer_mask.sum()
                     )
           log_pre    = rho * log_during + np.sqrt(1 - rho**2) * noise
           pre_buyers = np.exp(log_pre).clip(0)

           pre_exp[buyer_mask] = pre_buyers
           return pre_exp

    pre_exp_control   = make_pre_exp(revenue_control,       rho, seed_offset=10)
    pre_exp_treatment = make_pre_exp(revenue_treat_novelty, rho, seed_offset=20)

    # ── Assemble DataFrames ───────────────────────────────────
    df_control = pd.DataFrame({
        'user_id'          : control_ids,
        'timestamp'        : control_ts.values,
        'group'            : 'control',
        'experiment_day'   : control_day.values,
        'revenue'          : np.round(revenue_control, 2),
        'pre_exp_revenue'  : np.round(pre_exp_control, 2),
        'purchased'        : purchase_control,
    })

    df_treatment = pd.DataFrame({
        'user_id'          : treatment_ids,
        'timestamp'        : treatment_ts.values,
        'group'            : 'treatment',
        'experiment_day'   : treatment_day.values,
        'revenue'          : np.round(revenue_treat_novelty, 2),
        'pre_exp_revenue'  : np.round(pre_exp_treatment, 2),
        'purchased'        : purchase_treat,
    })

    df = pd.concat([df_control, df_treatment], ignore_index=True)
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['week_of_exp'] = ((df['experiment_day'] - 1) // 7 + 1).astype(int)

    # ── Validation report ─────────────────────────────────────
    print("\n" + "─"*50)
    print("  EXPERIMENT 2 — REVENUE SUMMARY")
    print("─"*50)
    for g, gdf in df.groupby('group'):
        rev  = gdf['revenue']
        pct  = gdf['purchased'].mean() * 100
        print(f"\n  {g.upper()}")
        print(f"    n              : {len(gdf):,}")
        print(f"    Purchase rate  : {pct:.2f}%")
        print(f"    Mean revenue   : Rs.{rev.mean():.2f}")
        print(f"    Median revenue : Rs.{rev.median():.2f}")
        print(f"    Std revenue    : Rs.{rev.std():.2f}")
        print(f"    Max revenue    : Rs.{rev.max():.2f}")
        print(f"    Zero-revenue   : {(rev == 0).sum():,} users")

    # Novelty check: early vs late treatment revenue
    print("\n" + "─"*50)
    print("  NOVELTY EFFECT — EARLY vs LATE TREATMENT REVENUE")
    print("─"*50)
    trt = df[df['group'] == 'treatment']
    ctrl = df[df['group'] == 'control']

    for period, days in [("Days 1–3", (1, 3)),
                          ("Days 4–7", (4, 7)),
                          ("Days 8–14",(8, 14))]:
        t_rev = trt[trt['experiment_day'].between(*days)]['revenue'].mean()
        c_rev = ctrl[ctrl['experiment_day'].between(*days)]['revenue'].mean()
        lift  = (t_rev - c_rev) / c_rev * 100 if c_rev > 0 else 0
        print(f"  {period:12} | Control: Rs.{c_rev:7.2f} | "
              f"Treatment: Rs.{t_rev:7.2f} | Lift: {lift:+.2f}%")

    # Pre-experiment correlation check
    merged = df[['revenue', 'pre_exp_revenue']].copy()
    actual_corr = merged.corr().iloc[0, 1]
    print(f"\n  Pre-exp vs during-exp correlation : {actual_corr:.4f} "
          f"(target ~{rho})")

    return df


# ============================================================
# EXPERIMENT 3 — Discount Banner Placement
# ============================================================

def simulate_experiment3() -> pd.DataFrame:
    """
    Simulate Experiment 3: Discount Banner Placement CTR test.

    Design decisions:
    -----------------
    1. DEVICE SEGMENTATION
       65% mobile, 35% desktop — matches Indian internet usage.
       Mobile and desktop have different baseline CTRs and MDEs,
       requiring Bonferroni correction for multiple comparisons.

    2. BINARY OUTCOME
       Each user either clicks the banner (1) or doesn't (0).
       CTR = proportion of users who click.

    3. POST-CLICK CONVERSION
       We also simulate whether a click leads to a purchase
       (post-click conversion rate ~18%). This is a guardrail
       metric: if post-click CVR drops, the clicks may be
       low-quality (curiosity clicks, not intent clicks).

    4. NO injected problem in Exp3 — the SRM in Exp1 is the
       single injected issue for this phase.
    """

    print("\n" + "="*60)
    print("SIMULATING EXPERIMENT 3 — DISCOUNT BANNER PLACEMENT")
    print("="*60)

    n_per_group = EXP3['n_per_group']    # 98,772
    total_n     = EXP3['total_n']        # 197,544

    print(f"  n per group        : {n_per_group:,}")
    print(f"  Total n            : {total_n:,}")
    print(f"  Mobile share       : {EXP3['mobile_share']*100:.0f}%")
    print(f"  Desktop share      : {EXP3['desktop_share']*100:.0f}%")

    n_control   = n_per_group
    n_treatment = n_per_group

    # ── User IDs ──────────────────────────────────────────────
    control_ids   = np.arange(300_001, 300_001 + n_control)
    treatment_ids = np.arange(400_001, 400_001 + n_treatment)

    # ── Device assignment ─────────────────────────────────────
    def assign_devices(n: int, mobile_share: float, seed: int) -> np.ndarray:
        rng_dev = np.random.default_rng(seed)
        return np.where(rng_dev.random(n) < mobile_share, 'mobile', 'desktop')

    ctrl_device  = assign_devices(n_control,   EXP3['mobile_share'], seed=RANDOM_SEED + 30)
    treat_device = assign_devices(n_treatment, EXP3['mobile_share'], seed=RANDOM_SEED + 31)

    # ── CTR by device and group ───────────────────────────────
    def generate_ctr(devices: np.ndarray,
                     mobile_ctr: float,
                     desktop_ctr: float,
                     seed: int) -> np.ndarray:
        rng_ctr = np.random.default_rng(seed)
        n       = len(devices)
        probs   = np.where(devices == 'mobile', mobile_ctr, desktop_ctr)
        return (rng_ctr.random(n) < probs).astype(int)

    ctrl_clicked = generate_ctr(
        ctrl_device,
        EXP3['baseline_ctr_mobile'],
        EXP3['baseline_ctr_desktop'],
        seed=RANDOM_SEED + 40
    )
    treat_clicked = generate_ctr(
        treat_device,
        EXP3['treatment_ctr_mobile'],
        EXP3['treatment_ctr_desktop'],
        seed=RANDOM_SEED + 41
    )

    # ── Post-click conversion (guardrail) ─────────────────────
    post_click_cvr = 0.18   # 18% of clicks lead to purchase

    def generate_post_click(clicked: np.ndarray,
                            cvr: float, seed: int) -> np.ndarray:
        rng_pc  = np.random.default_rng(seed)
        convert = np.zeros(len(clicked), dtype=int)
        click_idx = np.where(clicked == 1)[0]
        conversions = (rng_pc.random(len(click_idx)) < cvr).astype(int)
        convert[click_idx] = conversions
        return convert

    ctrl_converted  = generate_post_click(ctrl_clicked,  post_click_cvr, seed=RANDOM_SEED + 50)
    treat_converted = generate_post_click(treat_clicked, post_click_cvr, seed=RANDOM_SEED + 51)

    # ── Timestamps ────────────────────────────────────────────
    print("\n  Generating timestamps...")
    ctrl_ts  = make_timestamps(n_control,   days=14, seed=RANDOM_SEED + 60)
    treat_ts = make_timestamps(n_treatment, days=14, seed=RANDOM_SEED + 61)

    # ── Experiment day ────────────────────────────────────────
    start = pd.Timestamp("2024-01-01")
    ctrl_day  = ((ctrl_ts  - start).dt.days + 1).clip(1, 14)
    treat_day = ((treat_ts - start).dt.days + 1).clip(1, 14)

    # ── Assemble DataFrames ───────────────────────────────────
    df_control = pd.DataFrame({
        'user_id'         : control_ids,
        'timestamp'       : ctrl_ts.values,
        'group'           : 'control',
        'device'          : ctrl_device,
        'experiment_day'  : ctrl_day.values,
        'clicked'         : ctrl_clicked,
        'converted'       : ctrl_converted,
    })

    df_treatment = pd.DataFrame({
        'user_id'         : treatment_ids,
        'timestamp'       : treat_ts.values,
        'group'           : 'treatment',
        'device'          : treat_device,
        'experiment_day'  : treat_day.values,
        'clicked'         : treat_clicked,
        'converted'       : treat_converted,
    })

    df = pd.concat([df_control, df_treatment], ignore_index=True)
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['week_of_exp'] = ((df['experiment_day'] - 1) // 7 + 1).astype(int)

    # ── Validation report ─────────────────────────────────────
    print("\n" + "─"*50)
    print("  EXPERIMENT 3 — CTR SUMMARY")
    print("─"*50)
    for g, gdf in df.groupby('group'):
        ctr     = gdf['clicked'].mean() * 100
        pc_cvr  = (gdf['converted'].sum() / gdf['clicked'].sum() * 100
                   if gdf['clicked'].sum() > 0 else 0)
        print(f"\n  {g.upper()}")
        print(f"    n                  : {len(gdf):,}")
        print(f"    Overall CTR        : {ctr:.4f}%")
        print(f"    Post-click CVR     : {pc_cvr:.4f}%")

        for dev in ['mobile', 'desktop']:
            seg     = gdf[gdf['device'] == dev]
            seg_ctr = seg['clicked'].mean() * 100
            print(f"    {dev.capitalize()} CTR ({len(seg):,} users): {seg_ctr:.4f}%")

    print("\n  Group split:")
    grp = df['group'].value_counts()
    total = len(df)
    for g, n in grp.items():
        print(f"    {g:>10}: {n:>8,}  ({n/total*100:.2f}%)")

    return df


# ============================================================
# SAVE UTILITIES
# ============================================================

def save_experiment2(df: pd.DataFrame) -> str:
    os.makedirs(EXP2_DIR, exist_ok=True)
    filepath = os.path.join(EXP2_DIR, "exp2_recommendations_data.csv")
    df.to_csv(filepath, index=False)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"\n  ✅ Saved Experiment 2: {filepath}")
    print(f"     Rows : {len(df):,}  |  Size : {size_mb:.2f} MB")
    return filepath


def save_experiment3(df: pd.DataFrame) -> str:
    os.makedirs(EXP3_DIR, exist_ok=True)
    filepath = os.path.join(EXP3_DIR, "exp3_banner_data.csv")
    df.to_csv(filepath, index=False)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"\n  ✅ Saved Experiment 3: {filepath}")
    print(f"     Rows : {len(df):,}  |  Size : {size_mb:.2f} MB")
    return filepath


# ============================================================
# MASTER FUNCTION
# ============================================================

def simulate_all() -> tuple:
    """Run both simulators and save outputs. Returns (df2, df3)."""
    df2 = simulate_experiment2()
    save_experiment2(df2)

    df3 = simulate_experiment3()
    save_experiment3(df3)

    return df2, df3


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    df2, df3 = simulate_all()

    print("\n" + "="*60)
    print("EXP2 SAMPLE ROWS")
    print("="*60)
    print(df2.head(5).to_string(index=False))

    print("\n" + "="*60)
    print("EXP3 SAMPLE ROWS")
    print("="*60)
    print(df3.head(5).to_string(index=False))

    print("\n✅ simulator.py complete.")