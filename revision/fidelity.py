"""Distributional fidelity, observed values and missingness kept distinct."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
from .protocol import FEATURES, CONTINUOUS_FEATURES


def association(frame, a, b):
    pair = frame[[a, b]].dropna()
    if len(pair) < 11:
        return np.nan
    numeric_a, numeric_b = a in CONTINUOUS_FEATURES, b in CONTINUOUS_FEATURES
    if numeric_a and numeric_b:
        return pair[a].corr(pair[b]) if pair[a].nunique() > 1 and pair[b].nunique() > 1 else np.nan
    if not numeric_a and not numeric_b:
        table = pd.crosstab(pair[a], pair[b])
        denom = len(pair) * (min(table.shape) - 1)
        return np.sqrt(chi2_contingency(table, correction=False)[0] / denom) if denom else np.nan
    category, value = (b, a) if numeric_a else (a, b)
    grand = pair[value].mean()
    total = ((pair[value] - grand) ** 2).sum()
    if total == 0 or pair[category].nunique() < 2:
        return np.nan
    grouped = pair.groupby(category, observed=True)[value].agg(['size', 'mean'])
    between = (grouped['size'] * (grouped['mean'] - grand) ** 2).sum()
    return np.sqrt(max(0, min(1, between / total)))


def evaluate_fidelity(real, synthetic):
    result = {}
    for c in FEATURES:
        result['fidelity.missingness_abs_error.' + c] = abs(real[c].isna().mean() - synthetic[c].isna().mean())
        if c in CONTINUOUS_FEATURES:
            x, y = real[c].dropna(), synthetic[c].dropna()
            result['fidelity.ks_observed.' + c] = ks_2samp(x, y).statistic if min(len(x), len(y)) >= 11 else np.nan
            fill = real[c].median()
            result['fidelity.ks_train_median_filled.' + c] = ks_2samp(real[c].fillna(fill), synthetic[c].fillna(fill)).statistic if pd.notna(fill) else np.nan
        else:
            x, y = real[c].fillna('__MISSING__').value_counts(), synthetic[c].fillna('__MISSING__').value_counts()
            categories = x.index.union(y.index)
            result['fidelity.js_divergence.' + c] = jensenshannon(x.reindex(categories, fill_value=0), y.reindex(categories, fill_value=0), base=2) ** 2
    for a, b in combinations(FEATURES, 2):
        kind = 'pearson' if a in CONTINUOUS_FEATURES and b in CONTINUOUS_FEATURES else 'cramers_v' if a not in CONTINUOUS_FEATURES and b not in CONTINUOUS_FEATURES else 'correlation_ratio'
        result['fidelity.association_abs_error.' + kind + '.' + a + '__' + b] = abs(association(real, a, b) - association(synthetic, a, b))
    for prefix in ['ks_observed', 'ks_train_median_filled', 'js_divergence', 'missingness_abs_error', 'association_abs_error']:
        all_values = [v for k, v in result.items() if k.startswith('fidelity.' + prefix + '.')]
        values = [v for v in all_values if np.isfinite(v)]
        # Do not silently improve a generator's mean by dropping unsupported
        # features or associations. Partial means are explicitly sensitivity outputs.
        result['fidelity.' + prefix + '.supported_fraction'] = len(values) / len(all_values) if all_values else 0.
        result['fidelity.' + prefix + '.available_mean'] = float(np.mean(values)) if values else np.nan
        result['fidelity.' + prefix + '.mean'] = float(np.mean(values)) if values and len(values) == len(all_values) else np.nan
    return result
