"""Representation and demographic-parity differences with explicit support rules."""
import numpy as np
from .protocol import TARGET, PROTOCOL
from .safe_reporting import category_counts, count


def dpd(frame, attribute, groups):
    rates = []
    for group in groups:
        part = frame.loc[frame[attribute].eq(group), TARGET]
        if len(part) < 11:
            return np.nan
        rates.append(part.mean())
    return float(max(rates) - min(rates)) if len(rates) >= 2 else np.nan


def bootstrap_dpd(frame, attribute, groups, rng, resamples):
    """Ordinary record bootstrap via a multinomial over group-by-outcome cells."""
    cells = np.array([[((frame[attribute] == g) & (frame[TARGET] == y)).sum()
                       for y in [0, 1]] for g in groups])
    # Groups cover the declared support. Missing/unknown groups must be explicit.
    draws = rng.multinomial(len(frame), cells.ravel() / len(frame), size=resamples).reshape(resamples, len(groups), 2)
    denominators = draws.sum(axis=2)
    valid = (denominators >= 11).all(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rates = draws[:, :, 1] / denominators
    values = rates.max(axis=1) - rates.min(axis=1)
    values[~valid] = np.nan
    return values


def evaluate_fairness(real, synthetic, seed):
    result, representation, parity = {}, [], []
    rng = np.random.RandomState(seed)
    for attribute in ['ethnicity_grouped', 'gender']:
        groups = sorted(real[attribute].dropna().unique())
        real_counts, synth_counts = category_counts(real[attribute], groups), category_counts(synthetic[attribute], groups)
        for group in groups:
            nr, ns = int(real[attribute].eq(group).sum()), int(synthetic[attribute].eq(group).sum())
            ratio = (ns / len(synthetic)) / (nr / len(real))
            # Exact zero is an explicitly requested erasure statistic; small nonzero counts stay hidden.
            public_ratio = ratio if ns == 0 or ns >= 11 else '<11'
            representation.append({'attribute': attribute, 'group': group, 'real_n': real_counts[group],
                'synthetic_n': synth_counts[group], 'ratio': public_ratio,
                'erased': ns == 0, 'inflated_gt1_2': ratio > 1.2})
            result['fairness.representation.' + attribute + '.' + group] = public_ratio
        real_dpd, synth_dpd = dpd(real, attribute, groups), dpd(synthetic, attribute, groups)
        difference = synth_dpd - real_dpd
        rule_ok = np.isfinite(real_dpd) and np.isfinite(synth_dpd) and real_dpd >= PROTOCOL['bar_min_real_dpd']
        ratio = synth_dpd / real_dpd if rule_ok else np.nan
        reason = 'defined' if rule_ok else 'insufficient_group_support' if not np.isfinite(synth_dpd) or not np.isfinite(real_dpd) else 'real_dpd_below_0.01'
        boot_r = bootstrap_dpd(real, attribute, groups, rng, PROTOCOL['bootstrap_resamples'])
        boot_s = bootstrap_dpd(synthetic, attribute, groups, rng, PROTOCOL['bootstrap_resamples'])
        boot_diff = boot_s - boot_r
        valid = np.isfinite(boot_diff)
        bounds = np.quantile(boot_diff[valid], [.025, .975]) if valid.any() and np.isfinite(difference) else [np.nan, np.nan]
        with np.errstate(divide='ignore', invalid='ignore'):
            boot_ratio = boot_s / boot_r
        valid_ratio = np.isfinite(boot_ratio) & (boot_r >= PROTOCOL['bar_min_real_dpd'])
        ratio_bounds = np.quantile(boot_ratio[valid_ratio], [.025, .975]) if valid_ratio.any() and rule_ok else [np.nan, np.nan]
        parity.append({'attribute': attribute, 'dpd_real': real_dpd, 'dpd_synthetic': synth_dpd,
            'dpd_difference': difference, 'difference_ci_low': bounds[0], 'difference_ci_high': bounds[1],
            'bar': ratio, 'bar_ci_low': ratio_bounds[0], 'bar_ci_high': ratio_bounds[1], 'bar_status': reason,
            'valid_difference_resamples': count(valid.sum()), 'valid_bar_resamples': count(valid_ratio.sum())})
        result['fairness.dpd_difference.' + attribute] = difference
        result['fairness.bar.' + attribute] = ratio
    return result, representation, parity
