"""Legacy distance-feature attack, refitted permutation null, fixed-range Gower.

No predictions, nearest records, attack features or row-level distances are public.
Missing numeric values use train medians for the legacy attack. Revised Gower
uses equal original-column weights, training ranges, distance 1 for exactly one
missing value and 0 for two missing values. Numeric contributions are capped at 1.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .protocol import FEATURES, CONTINUOUS_FEATURES, PROTOCOL
from .safe_reporting import event_summary, count

try:
    from numba import njit, prange, set_num_threads
except ImportError:
    njit = None


def attack_encoding(train, frame):
    out = np.empty((len(frame), len(FEATURES)))
    for i, c in enumerate(FEATURES):
        if c in CONTINUOUS_FEATURES:
            median = train[c].median()
            out[:, i] = frame[c].fillna(median if pd.notna(median) else 0)
        else:
            cats = sorted(train[c].dropna().unique(), key=str)
            out[:, i] = pd.Categorical(frame[c], categories=cats).codes
    return out


def membership_features(train, test, synth, seed):
    n = min(2000, len(train), len(test))
    positions_r = np.random.RandomState(seed).choice(len(train), n, replace=False)
    positions_t = np.random.RandomState(seed).choice(len(test), n, replace=False)
    scaler = StandardScaler().fit(attack_encoding(train, synth))
    reference = scaler.transform(attack_encoding(train, synth))[:5000]
    points = np.concatenate([scaler.transform(attack_encoding(train, train.iloc[positions_r])),
                             scaler.transform(attack_encoding(train, test.iloc[positions_t]))])
    features = np.empty((len(points), 4))
    for begin in range(0, len(points), 100):
        d = cdist(points[begin:begin + 100], reference)
        features[begin:begin + len(d)] = np.column_stack([d.min(axis=1), d.mean(axis=1),
                                                         np.percentile(d, 10, axis=1), d.std(axis=1)])
    return features, np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]


def evaluate_membership(train, test, synth, seed):
    features, labels = membership_features(train, test, synth, seed)
    attack = LogisticRegression(max_iter=1000, random_state=seed)
    rng = np.random.RandomState(seed)
    with threadpool_limits(limits=1):
        observed_folds = cross_val_score(attack, features, labels, cv=5, scoring='accuracy', n_jobs=1)
        observed = observed_folds.mean()
        null = np.array([cross_val_score(attack, features, rng.permutation(labels), cv=5,
                        scoring='accuracy', n_jobs=1).mean() for _ in range(PROTOCOL['permutation_resamples'])])
    return {'privacy.mia_accuracy': observed, 'privacy.mia_fold_sd': observed_folds.std(),
            'privacy.mia_null_mean': null.mean(), 'privacy.mia_null_p95': np.quantile(null, .95),
            'privacy.mia_permutation_p': (1 + np.count_nonzero(null >= observed)) / (len(null) + 1),
            'privacy.mia_excess_over_null': observed - null.mean()}


def gower_arrays(train, frame):
    out = np.empty((len(frame), len(FEATURES)))
    for i, c in enumerate(FEATURES):
        if c in CONTINUOUS_FEATURES:
            minimum, maximum = train[c].min(), train[c].max()
            span = maximum - minimum
            out[:, i] = (frame[c] - minimum) / span if pd.notna(span) and span > 0 else frame[c].where(frame[c].isna(), 0)
        else:
            cats = sorted(train[c].dropna().unique(), key=str)
            out[:, i] = pd.Categorical(frame[c], categories=cats).codes
            out[frame[c].isna(), i] = np.nan
    return out


def _nearest_loop(query, reference, categorical, exclude_self):
    out = np.empty(len(query))
    for i in prange(len(query)):
        best = np.inf
        for j in range(len(reference)):
            if exclude_self and i == j:
                continue
            distance = 0.
            for k in range(query.shape[1]):
                a, b = query[i, k], reference[j, k]
                if np.isnan(a) and np.isnan(b):
                    delta = 0.
                elif np.isnan(a) or np.isnan(b):
                    delta = 1.
                elif categorical[k]:
                    delta = 1.0 if a != b else 0.0
                else:
                    delta = min(abs(a - b), 1.)
                distance += delta
                if distance >= best:
                    break
            if distance < best:
                best = distance
        out[i] = best / query.shape[1]
    return out


if njit is not None:
    _nearest_compiled = njit(parallel=True, cache=True)(_nearest_loop)


def nearest_gower(query, reference, categorical, exclude_self=False):
    if njit is not None:
        set_num_threads(PROTOCOL['model_jobs'])
        return _nearest_compiled(query, reference, categorical, exclude_self)
    # Vectorized fallback for environments without numba. No metric fallback.
    out = np.empty(len(query))
    for start in range(0, len(query), 32):
        block = query[start:start + 32]
        total = np.zeros((len(block), len(reference)))
        for k in range(query.shape[1]):
            a, b = block[:, k, None], reference[None, :, k]
            missing_a, missing_b = np.isnan(a), np.isnan(b)
            d = (a != b).astype(float) if categorical[k] else np.minimum(np.abs(a - b), 1.)
            d = np.where(missing_a & missing_b, 0, np.where(missing_a | missing_b, 1, d))
            total += d
        if exclude_self:
            total[np.arange(len(block)), start + np.arange(len(block))] = np.inf
        out[start:start + len(block)] = total.min(axis=1) / query.shape[1]
    return out


def evaluate_dcr(train, test, synth, seed, cached_real=None):
    categorical = np.array([c not in CONTINUOUS_FEATURES for c in FEATURES])
    r, t, s = [gower_arrays(train, frame) for frame in [train, test, synth]]
    if cached_real is None:
        real_nearest = nearest_gower(r, r, categorical, exclude_self=True)
        test_nearest = nearest_gower(t, r, categorical)
    else:
        real_nearest, test_nearest = cached_real
    synth_nearest = nearest_gower(s, r, categorical)
    threshold = np.quantile(real_nearest, .05)
    near = synth_nearest < threshold
    # Ratios retain legacy query sample size but use one fixed train-range metric.
    rng = np.random.RandomState(seed)
    sample_s = rng.choice(len(s), min(2000, len(s)), replace=False)
    sample_t = np.random.RandomState(seed).choice(len(t), min(2000, len(t)), replace=False)
    med_s, med_t = np.median(synth_nearest[sample_s]), np.median(test_nearest[sample_t])
    result = {'privacy.dcr_synth_median': med_s, 'privacy.dcr_holdout_median': med_t,
        'privacy.dcr_ratio': med_s / med_t if med_t > 0 else np.nan,
        'privacy.near_duplicate_threshold': threshold,
        'privacy.near_duplicate_rate': event_summary(near.sum(), len(synth))['fraction']}
    rows = []
    for group in sorted(train.ethnicity_grouped.unique()):
        mask = synth.ethnicity_grouped.eq(group).to_numpy()
        summary = event_summary(near[mask].sum(), mask.sum())
        rows.append({'ethnicity': group, 'synthetic_n': count(mask.sum()),
                     'near_duplicate_n': summary['count'], 'near_duplicate_rate': summary['fraction']})
    return result, rows, (real_nearest, test_nearest)
