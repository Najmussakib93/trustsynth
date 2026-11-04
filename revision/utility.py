"""Mortality utility on untouched real test sets with training-only preprocessing."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from .protocol import FEATURES, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, TARGET, PROTOCOL
from .safe_reporting import count, event_summary

PREDICTORS = [c for c in FEATURES if c != TARGET]


def make_model(seed):
    transform = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), CONTINUOUS_FEATURES),
        ('category', Pipeline([('fill', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)])
    return Pipeline([('transform', transform), ('model', XGBClassifier(n_estimators=100, max_depth=4,
        learning_rate=0.1, random_state=seed, n_jobs=PROTOCOL['model_jobs'], tree_method='hist', eval_metric='logloss'))])


def metrics(y, prob):
    y, prob = np.asarray(y), np.asarray(prob)
    if len(np.unique(y)) < 2 or not np.isfinite(prob).all():
        return {'auroc': np.nan, 'auprc': np.nan, 'ece': np.nan}
    bins = np.minimum((prob * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(np.mean(y[bins == b]) - np.mean(prob[bins == b]))
              for b in range(10) if np.any(bins == b))
    return {'auroc': roc_auc_score(y, prob), 'auprc': average_precision_score(y, prob), 'ece': ece}


def bootstrap_auc_ap(y, prob, seed, resamples=1000):
    """Ordinary paired record bootstrap; weighted tie handling matches sklearn.

    Multinomial weights are exactly equivalent to sampling n rows with replacement.
    Batches bound memory; undefined one-class replicates are excluded and counted.
    """
    y, prob = np.asarray(y, dtype=int), np.asarray(prob)
    order = np.argsort(-prob, kind='stable')
    y, prob = y[order], prob[order]
    if len(np.unique(y)) < 2:
        return {'auroc_ci_low': np.nan, 'auroc_ci_high': np.nan,
                'auprc_ci_low': np.nan, 'auprc_ci_high': np.nan, 'valid_resamples': count(0)}
    starts = np.r_[0, np.flatnonzero(prob[1:] != prob[:-1]) + 1]
    rng = np.random.RandomState(seed)
    aucs, aps = [], []
    for offset in range(0, resamples, 50):
        w = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=min(50, resamples - offset))
        pos = np.add.reduceat(w * y, starts, axis=1)
        neg = np.add.reduceat(w * (1 - y), starts, axis=1)
        tp, fp = pos.cumsum(axis=1), neg.cumsum(axis=1)
        p, n = tp[:, -1], fp[:, -1]
        valid = (p > 0) & (n > 0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            aucs.extend(((pos * (n[:, None] - fp + .5 * neg)).sum(axis=1) / (p * n))[valid])
            aps.extend(((pos * precision).sum(axis=1) / p)[valid])
    result = {'valid_resamples': count(len(aucs))}
    for name, values in [('auroc', aucs), ('auprc', aps)]:
        bounds = np.quantile(values, [.025, .975]) if values else [np.nan, np.nan]
        result[name + '_ci_low'], result[name + '_ci_high'] = bounds
    return result


def subgroups(test):
    for attribute in ['ethnicity_grouped', 'gender']:
        for group in sorted(test[attribute].unique()):
            yield attribute, str(group), test[attribute].eq(group).to_numpy()
    # Age above 89 is encoded 90; use broad, prespecified bins.
    age_group = pd.cut(test.age, [18, 45, 65, 80, np.inf], right=False,
                       labels=['18-44', '45-64', '65-79', '80+'])
    for group in age_group.cat.categories:
        yield 'age', group, age_group.eq(group).to_numpy()
    # Supplementary pooled estimate, never replacing individual ethnicity rows.
    yield 'ethnicity_pooled', 'NON_WHITE', test.ethnicity_grouped.ne('WHITE').to_numpy()


def fit_predict(training, test, seed):
    model = make_model(seed)
    model.fit(training[PREDICTORS], training[TARGET])
    probability = model.predict_proba(test[PREDICTORS])[:, 1]
    importance = permutation_importance(model, test[PREDICTORS], test[TARGET], scoring='roc_auc',
        n_repeats=PROTOCOL['permutation_importance_repeats'], random_state=seed, n_jobs=1).importances_mean
    return model, probability, importance


def evaluate_utility(train, test, synthetic, seed, trtr=None):
    if trtr is None:
        _, real_prob, real_importance = fit_predict(train, test, seed)
        if seed == 0:
            _, repeat_prob, repeat_importance = fit_predict(train, test, seed)
            if not np.array_equal(real_prob, repeat_prob) or not np.array_equal(real_importance, repeat_importance):
                raise RuntimeError('Corrected TRTR independent repeat failed')
    else:
        real_prob, real_importance = trtr
    if synthetic[TARGET].nunique() < 2:
        synthetic_prob, synthetic_importance = np.full(len(test), np.nan), np.full(len(PREDICTORS), np.nan)
    else:
        _, synthetic_prob, synthetic_importance = fit_predict(synthetic, test, seed)
        if seed == 0:
            _, repeat_prob, repeat_importance = fit_predict(synthetic, test, seed)
            if not np.array_equal(synthetic_prob, repeat_prob) or not np.array_equal(synthetic_importance, repeat_importance):
                raise RuntimeError('Corrected TSTR independent repeat failed')
    result, subgroup_rows = {}, []
    for label, probability in [('trtr', real_prob), ('tstr', synthetic_prob)]:
        for metric, value in metrics(test[TARGET], probability).items():
            result['utility.' + label + '.' + metric] = value
        for attribute, group, mask in subgroups(test):
            y, probs = test.loc[mask, TARGET].to_numpy(), probability[mask]
            positives = int(y.sum())
            row = {'model': label, 'attribute': attribute, 'group': group, 'n': count(len(y)),
                'positive_count': event_summary(positives, len(y))['count'], 'unstable_lt50_positive': positives < 50}
            if len(y) >= 11:
                row.update(metrics(y, probs))
                if np.isfinite(probs).all():
                    row.update(bootstrap_auc_ap(y, probs, seed, PROTOCOL['bootstrap_resamples']))
            subgroup_rows.append(row)
    result['utility.auroc_gap_signed'] = result['utility.trtr.auroc'] - result['utility.tstr.auroc']
    result['utility.auroc_gap_absolute'] = abs(result['utility.auroc_gap_signed'])
    result['utility.auprc_gap_signed'] = result['utility.trtr.auprc'] - result['utility.tstr.auprc']
    result['utility.importance_spearman'] = spearmanr(real_importance, synthetic_importance).statistic
    result['utility.tstr_fit_available'] = bool(synthetic[TARGET].nunique() >= 2)
    for c, a, b in zip(PREDICTORS, real_importance, synthetic_importance):
        result['utility.importance_trtr.' + c] = a
        result['utility.importance_tstr.' + c] = b
    return result, subgroup_rows, (real_prob, real_importance)
