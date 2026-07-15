import numpy as np
import scipy.stats as stats
import pandas as pd
import warnings
from typing import Dict

warnings.filterwarnings('ignore')


def calculate_sample_size_proportion(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
    verbose: bool = True
) -> Dict:

    assert 0 < baseline_rate < 1
    assert 0 < mde < 1
    assert baseline_rate + mde <= 1
    assert 0 < alpha < 1
    assert 0 < power < 1

    p1 = baseline_rate
    p2 = baseline_rate + mde

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    variance_p1 = p1 * (1 - p1)
    variance_p2 = p2 * (1 - p2)

    numerator = (z_alpha + z_beta) ** 2 * (variance_p1 + variance_p2)
    denominator = (p2 - p1) ** 2

    n_per_group = int(np.ceil(numerator / denominator))
    total_n = n_per_group * 2

    relative_lift = mde / baseline_rate * 100
    cohens_h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

    result = {
        'experiment_type': 'proportion',
        'n_per_group': n_per_group,
        'total_n': total_n,
        'baseline_rate': p1,
        'treatment_rate': p2,
        'mde_absolute': mde,
        'mde_relative_pct': round(relative_lift, 2),
        'alpha': alpha,
        'power': power,
        'z_alpha': round(z_alpha, 4),
        'z_beta': round(z_beta, 4),
        'cohens_h': round(abs(cohens_h), 4),
        'two_tailed': two_tailed
    }

    if verbose:
        tail_str = "two-tailed" if two_tailed else "one-tailed"
        print(f"\n{'='*60}")
        print(f"SAMPLE SIZE — PROPORTION TEST ({tail_str})")
        print(f"{'='*60}")
        print(f"  Baseline rate (control)  : {p1:.4f}  ({p1*100:.2f}%)")
        print(f"  Treatment rate (expected): {p2:.4f}  ({p2*100:.2f}%)")
        print(f"  MDE (absolute)           : {mde:.4f}  ({mde*100:.3f} pp)")
        print(f"  MDE (relative)           : {relative_lift:.2f}%")
        print(f"  Alpha                    : {alpha}")
        print(f"  Power                    : {power}")
        print(f"  Z_alpha                  : {z_alpha:.4f}")
        print(f"  Z_beta                   : {z_beta:.4f}")
        print(f"  Cohen's h                : {abs(cohens_h):.4f}")
        print(f"  ----------------------------------------")
        print(f"  Required n per group     : {n_per_group:,}")
        print(f"  Required total n         : {total_n:,}")
        print(f"{'='*60}")

    return result


def calculate_sample_size_means(
    baseline_mean: float,
    baseline_std: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
    verbose: bool = True
) -> Dict:

    assert baseline_std > 0
    assert mde_absolute > 0
    assert 0 < alpha < 1
    assert 0 < power < 1

    mu2 = baseline_mean + mde_absolute
    sigma = baseline_std
    cohens_d = abs(mde_absolute) / sigma

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    n_per_group = int(np.ceil(2 * ((z_alpha + z_beta) / cohens_d) ** 2))
    total_n = n_per_group * 2
    mde_relative_pct = mde_absolute / baseline_mean * 100

    result = {
        'experiment_type': 'means',
        'n_per_group': n_per_group,
        'total_n': total_n,
        'baseline_mean': baseline_mean,
        'treatment_mean': mu2,
        'baseline_std': baseline_std,
        'mde_absolute': mde_absolute,
        'mde_relative_pct': round(mde_relative_pct, 2),
        'cohens_d': round(cohens_d, 4),
        'alpha': alpha,
        'power': power,
        'z_alpha': round(z_alpha, 4),
        'z_beta': round(z_beta, 4),
        'two_tailed': two_tailed,
        'skew_warning': (
            'Revenue data is right-skewed. '
            't-test valid for large n via CLT. '
            'Mann-Whitney U will be applied in Phase 3 as robustness check.'
        )
    }

    if verbose:
        tail_str = "two-tailed" if two_tailed else "one-tailed"
        print(f"\n{'='*60}")
        print(f"SAMPLE SIZE — MEANS TEST ({tail_str})")
        print(f"{'='*60}")
        print(f"  Baseline mean (control)  : Rs.{baseline_mean:,.0f}")
        print(f"  Treatment mean (expected): Rs.{mu2:,.0f}")
        print(f"  Baseline std             : Rs.{sigma:,.0f}")
        print(f"  MDE (absolute)           : Rs.{mde_absolute:,.0f}")
        print(f"  MDE (relative)           : {mde_relative_pct:.2f}%")
        print(f"  Cohen's d                : {cohens_d:.4f}")
        print(f"  Alpha                    : {alpha}")
        print(f"  Power                    : {power}")
        print(f"  Z_alpha                  : {z_alpha:.4f}")
        print(f"  Z_beta                   : {z_beta:.4f}")
        print(f"  ----------------------------------------")
        print(f"  Required n per group     : {n_per_group:,}")
        print(f"  Required total n         : {total_n:,}")
        print(f"  ----------------------------------------")
        print(f"  WARNING: {result['skew_warning']}")
        print(f"{'='*60}")

    return result


def calculate_sample_size_segmented(
    segments: Dict,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
    apply_bonferroni: bool = True,
    verbose: bool = True
) -> Dict:

    n_segments = len(segments)

    if apply_bonferroni and n_segments > 1:
        alpha_corrected = alpha / n_segments
        correction_note = (
            f"Bonferroni: alpha={alpha} divided by "
            f"{n_segments} segments = {alpha_corrected:.4f}"
        )
    else:
        alpha_corrected = alpha
        correction_note = "No correction applied"

    segment_results = {}

    for seg_name, seg_params in segments.items():
        seg_result = calculate_sample_size_proportion(
            baseline_rate=seg_params['baseline_rate'],
            mde=seg_params['mde'],
            alpha=alpha_corrected,
            power=power,
            two_tailed=two_tailed,
            verbose=False
        )

        traffic_frac = seg_params['traffic_fraction']
        total_needed = int(np.ceil(seg_result['n_per_group'] / traffic_frac))

        segment_results[seg_name] = {
            **seg_result,
            'traffic_fraction': traffic_frac,
            'n_per_group_in_segment': seg_result['n_per_group'],
            'total_experiment_n_per_group_needed': total_needed,
            'baseline_rate': seg_params['baseline_rate'],
            'mde_absolute': seg_params['mde'],
        }

    max_n_per_group = max(
        v['total_experiment_n_per_group_needed']
        for v in segment_results.values()
    )

    if verbose:
        tail_str = "two-tailed" if two_tailed else "one-tailed"
        print(f"\n{'='*60}")
        print(f"SAMPLE SIZE — SEGMENTED TEST ({tail_str})")
        print(f"{'='*60}")
        print(f"  Number of segments       : {n_segments}")
        print(f"  Multiple comparison fix  : {correction_note}")
        print(f"  Alpha (corrected)        : {alpha_corrected:.4f}")
        print(f"  Power                    : {power}")
        print()
        for seg_name, seg_data in segment_results.items():
            frac_pct = seg_data['traffic_fraction'] * 100
            print(f"  Segment : {seg_name.upper()} ({frac_pct:.0f}% of traffic)")
            print(f"    Baseline rate          : {seg_data['baseline_rate']*100:.2f}%")
            print(f"    MDE                    : {seg_data['mde_absolute']*100:.3f} pp")
            print(f"    N needed in segment    : {seg_data['n_per_group_in_segment']:,} per group")
            print(f"    Total exp n needed     : {seg_data['total_experiment_n_per_group_needed']:,} per group")
            print()
        print(f"  ----------------------------------------")
        print(f"  Binding n per group      : {max_n_per_group:,}")
        print(f"  Total experiment n       : {max_n_per_group * 2:,}")
        print(f"{'='*60}")

    return {
        'experiment_type': 'segmented_proportion',
        'n_per_group': max_n_per_group,
        'total_n': max_n_per_group * 2,
        'alpha_original': alpha,
        'alpha_corrected': alpha_corrected,
        'power': power,
        'bonferroni_applied': apply_bonferroni,
        'segment_results': segment_results,
        'correction_note': correction_note
    }


def calculate_experiment_duration(
    n_per_group: int,
    daily_traffic: int,
    traffic_fraction_in_experiment: float = 1.0,
    group_split: float = 0.5,
    verbose: bool = True,
    experiment_name: str = "Experiment"
) -> Dict:

    daily_in_experiment = daily_traffic * traffic_fraction_in_experiment
    daily_per_group = daily_in_experiment * group_split
    days_needed = int(np.ceil(n_per_group / daily_per_group))
    weeks_needed = round(days_needed / 7, 1)

    result = {
        'experiment_name': experiment_name,
        'n_per_group_required': n_per_group,
        'daily_total_traffic': daily_traffic,
        'traffic_fraction_in_experiment': traffic_fraction_in_experiment,
        'daily_in_experiment': int(daily_in_experiment),
        'daily_per_group': int(daily_per_group),
        'days_needed': days_needed,
        'weeks_needed': weeks_needed,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"DURATION ESTIMATE — {experiment_name.upper()}")
        print(f"{'='*60}")
        print(f"  Required n per group     : {n_per_group:,}")
        print(f"  Daily total traffic      : {daily_traffic:,}")
        print(f"  In-experiment fraction   : {traffic_fraction_in_experiment*100:.0f}%")
        print(f"  Daily in experiment      : {int(daily_in_experiment):,}")
        print(f"  Daily per group (50/50)  : {int(daily_per_group):,}")
        print(f"  ----------------------------------------")
        print(f"  Days required            : {days_needed}")
        print(f"  Weeks required           : {weeks_needed}")
        print(f"{'='*60}")

    return result