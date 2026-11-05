"""Across-seed descriptive uncertainty and paired comparisons, without row data."""
from itertools import combinations, product
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t, ttest_rel


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return {'n_seeds': n, 'mean': mean, 'sd': sd, 'ci_low': mean - half, 'ci_high': mean + half,
            'ci_scope': 'descriptive variation across overlapping splits of a fixed cohort'}


@lru_cache(maxsize=16)
def sign_patterns(n):
    return np.array(list(product([0, 1], repeat=n)), dtype=float)


def exact_wilcoxon(differences):
    """Exact two-sided sign enumeration of average absolute ranks; zeros removed.

    Unlike SciPy's classic exact table this also handles tied absolute ranks.
    The null requires exchangeable independent signs (symmetric differences).
    """
    differences = np.asarray(differences, dtype=float)
    values = differences[np.isfinite(differences) & (differences != 0)]
    if not len(values):
        return 0., 1., 0
    if len(values) > 16:
        raise ValueError('Exact enumeration is bounded to 16 nonzero seed pairs')
    ranks = rankdata(abs(values), method='average')
    positive = ranks[values > 0].sum()
    statistic = min(positive, ranks.sum() - positive)
    signs = sign_patterns(len(values))
    sums = (signs * ranks).sum(axis=1)
    distribution = np.minimum(sums, ranks.sum() - sums)
    return float(statistic), float(np.mean(distribution <= statistic + 1e-12)), len(values)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan)
    eligible = np.flatnonzero(np.isfinite(pvalues))
    order = eligible[np.argsort(pvalues[eligible], kind='stable')]
    if len(order):
        adjusted = np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1))
        out[order] = np.minimum(adjusted, 1.)
    return out


def paired(a, b):
    aligned = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    aligned = aligned[np.isfinite(aligned.to_numpy(dtype=float)).all(axis=1)]
    differences = (aligned.a - aligned.b).to_numpy(dtype=float)
    differences = differences[np.isfinite(differences)]
    result = summarize(differences)
    statistic, pvalue, nonzero = exact_wilcoxon(differences)
    result.update(wilcoxon_statistic=statistic if len(differences) else np.nan,
                  wilcoxon_p=pvalue if len(differences) else np.nan, wilcoxon_nonzero_pairs=nonzero,
                  wilcoxon_min_two_sided_p=2 / 2 ** nonzero if nonzero else 1.,
                  paired_t_statistic=np.nan, paired_t_p=np.nan,
                  test_status='insufficient_pairs' if len(differences) < 2 else 'defined')
    if len(differences) >= 2:
        if np.all(differences == 0):
            result.update(paired_t_statistic=0., paired_t_p=1.)
        elif np.ptp(differences) <= 32 * np.finfo(float).eps * max(1., np.max(abs(differences))):
            result['test_status'] = 'paired_t_undefined_zero_variance'
        else:
            test = ttest_rel(aligned.a, aligned.b)
            result.update(paired_t_statistic=float(test.statistic), paired_t_p=float(test.pvalue))
    return result


def tables(tidy):
    values = tidy.copy()
    values['numeric'] = pd.to_numeric(values.value, errors='coerce')
    summaries, comparisons = [], []
    for (generator, metric), part in values.groupby(['generator', 'metric'], sort=True):
        summaries.append(dict(generator=generator, metric=metric, **summarize(part.numeric)))
    for metric, part in values.groupby('metric', sort=True):
        pivot = part.pivot(index='seed', columns='generator', values='numeric')
        for a, b in combinations(sorted(pivot.columns), 2):
            comparisons.append(dict(metric=metric, generator_a=a, generator_b=b, **paired(pivot[a], pivot[b])))
    summary, comparison = pd.DataFrame(summaries), pd.DataFrame(comparisons)
    if not comparison.empty:
        for test in ['wilcoxon', 'paired_t']:
            comparison[test + '_p_holm_within_metric'] = comparison.groupby('metric')[test + '_p'].transform(
                lambda series: pd.Series(holm(series.to_numpy()), index=series.index))
    return summary, comparison
