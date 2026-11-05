"""Declared normalization and sensitivity; missing components never become zero."""
from itertools import product
import numpy as np
import pandas as pd

from .legacy import config as legacy
from .protocol import PROTOCOL
from .statistics import summarize

DIMENSIONS = ['fidelity', 'privacy', 'fairness', 'utility', 'integrity']
REFERENCES = {
    'ks': legacy.KS_NORM_REF,
    'js_divergence_bits': legacy.JSD_NORM_REF / np.log(2),
    'association': legacy.CORR_NORM_REF,
    'mia_excess_over_chance': .5,
    'dcr_ratio': 1.,
    'representation_abs_deviation': 1.,
    'dpd_positive_difference': 1.,
    'auroc_gap': legacy.TSTR_GAP_NORM_REF,
    'importance_spearman': 1.,
    'hard_rule_csr': 1.,
}
FORMULAS = {
    'fidelity': 'mean(L(mean observed KS,ks), L(mean JS divergence in bits,js_divergence_bits), L(mean mixed-pair association absolute error,association))',
    'privacy': 'mean(L(max(MIA_accuracy-0.5,0),mia_excess_over_chance), U(DCR_ratio,dcr_ratio))',
    'fairness': 'mean(L(mean absolute representation-ratio deviation from 1,representation_abs_deviation), L(mean positive DPD_synth-minus-real across sex/ethnicity,dpd_positive_difference))',
    'utility': 'mean(L(abs(TRTR_AUROC-TSTR_AUROC),auroc_gap), U(permutation_importance_Spearman,importance_spearman))',
    'integrity': 'U(record-all-retained-rules CSR among fully applicable records,hard_rule_csr)',
    'TrustScore': 'sum(w_d * S_d), primary w_d=1/5; all five dimensions must be defined',
    'L': 'clip(1-x/reference,0,1)', 'U': 'clip(x/reference,0,1)',
}
MAIN_GENERATORS = ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']


def complete_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else np.nan


def component_values(metrics, references=None):
    r = REFERENCES if references is None else references
    get = lambda key: float(metrics.get(key, np.nan))
    lower = lambda value, name: float(np.clip(1 - value / r[name], 0, 1))
    upper = lambda value, name: float(np.clip(value / r[name], 0, 1))
    ratios = [float(value) for key, value in metrics.items() if key.startswith('fairness.representation.')]
    rep_error = complete_mean([abs(value - 1) for value in ratios])
    parity = [get('fairness.dpd_difference.' + attribute) for attribute in ['ethnicity_grouped', 'gender']]
    return {'fidelity.ks': lower(get('fidelity.ks_observed.mean'), 'ks'),
        'fidelity.jsd': lower(get('fidelity.js_divergence.mean'), 'js_divergence_bits'),
        'fidelity.association': lower(get('fidelity.association_abs_error.mean'), 'association'),
        'privacy.mia': lower(max(get('privacy.mia_accuracy') - .5, 0), 'mia_excess_over_chance'),
        'privacy.dcr': upper(get('privacy.dcr_ratio'), 'dcr_ratio'),
        'fairness.representation': lower(rep_error, 'representation_abs_deviation'),
        'fairness.parity': lower(complete_mean([max(value, 0) for value in parity]), 'dpd_positive_difference'),
        'utility.auroc_gap': lower(get('utility.auroc_gap_absolute'), 'auroc_gap'),
        'utility.importance': upper(get('utility.importance_spearman'), 'importance_spearman'),
        'integrity.record_csr': upper(get('integrity.csr_record_all'), 'hard_rule_csr')}


def score_cell(metrics, references=None):
    components = component_values(metrics, references)
    scores = {dimension: complete_mean([value for key, value in components.items() if key.startswith(dimension + '.')]) for dimension in DIMENSIONS}
    scores['TrustScore'] = complete_mean([scores[dimension] for dimension in DIMENSIONS])
    return scores


def calculate(tidy, references=None, components=False):
    rows = []
    numeric = tidy.copy()
    numeric['value'] = pd.to_numeric(numeric.value, errors='coerce')
    for (generator, seed), part in numeric.groupby(['generator', 'seed'], sort=True):
        metrics = dict(zip(part.metric, part.value))
        result = component_values(metrics, references) if components else score_cell(metrics, references)
        for metric, value in result.items():
            rows.append({'seed': seed, 'generator': generator, 'metric': ('component.' if components else 'score.') + metric, 'value': value})
    return pd.DataFrame(rows)


def eligible_profiles(scored):
    dimension_rows = scored[scored.metric.isin(['score.' + name for name in DIMENSIONS])]
    cube = dimension_rows.pivot(index=['generator', 'seed'], columns='metric', values='value')
    profiles, eligibility = {}, []
    for generator in sorted(set(scored.generator) - {'CTGAN_no_log', 'TVAE_900'}):
        available = cube.loc[generator].reindex(PROTOCOL['seeds']) if generator in cube.index.get_level_values(0) else pd.DataFrame()
        good = len(available) == len(PROTOCOL['seeds']) and len(available.columns) == 5 and np.isfinite(available.to_numpy()).all()
        missing = [dimension for dimension in DIMENSIONS if 'score.' + dimension not in available or not np.isfinite(available['score.' + dimension]).all()]
        eligibility.append({'generator': generator, 'eligible': bool(good), 'undefined_dimensions': ';'.join(missing),
                            'complete_seed_count': int(np.isfinite(available.to_numpy()).all(axis=1).sum()) if len(available) else 0,
                            'required_seed_count': len(PROTOCOL['seeds']),
                            'reason': 'complete_five_dimension_ten_seed_profile' if good else 'undefined_or_missing_required_components'})
        if good:
            profiles[generator] = available.reindex(columns=['score.' + name for name in DIMENSIONS]).mean().to_numpy()
    return profiles, eligibility


def weight_sensitivity(scored, samples=10000, seed=42):
    profiles, eligibility = eligible_profiles(scored)
    names = sorted(profiles)
    if not names:
        return pd.DataFrame(), pd.DataFrame(eligibility)
    weights = np.random.RandomState(seed).dirichlet(np.ones(len(DIMENSIONS)), size=samples)
    values = weights @ np.vstack([profiles[name] for name in names]).T
    ties = np.isclose(values, values.max(axis=1, keepdims=True), rtol=0, atol=1e-12)
    credits = ties / ties.sum(axis=1, keepdims=True)
    rows = []
    for index, name in enumerate(names):
        share = float(credits[:, index].mean())
        se = float(credits[:, index].std(ddof=1) / np.sqrt(samples))
        rows.append({'generator': name, 'winner_share': share, 'mc_se': se,
            'mc_ci_low': max(0., share - 1.96 * se), 'mc_ci_high': min(1., share + 1.96 * se),
            'weight_vectors': samples, 'seed': seed, 'ties_split_equally': True,
            'scope': 'conditional on eligible complete profiles; uncertainty is Monte Carlo, not across patients'})
    return pd.DataFrame(rows), pd.DataFrame(eligibility)


def normalization_sensitivity(tidy):
    rows = []
    scenarios = [('baseline', REFERENCES)]
    scenarios += [(name + '_x' + str(factor), dict(REFERENCES, **{name: reference * factor}))
                  for name, reference in REFERENCES.items() for factor in [.75, 1.25]]
    for scenario, references in scenarios:
        scored = calculate(tidy, references)
        profiles, eligibility = eligible_profiles(scored)
        scores = {name: float(np.mean(profile)) for name, profile in profiles.items()}
        ranks = pd.Series(scores, dtype=float).rank(ascending=False, method='min')
        for record in eligibility:
            name = record['generator']
            rows.append(dict(record, scenario=scenario, TrustScore=scores.get(name, np.nan),
                             rank=float(ranks.get(name, np.nan))))
    return pd.DataFrame(rows)


def threshold_sensitivity(tidy, scored):
    rows = []
    numeric = tidy.copy()
    numeric['value'] = pd.to_numeric(numeric.value, errors='coerce')
    definitions = [
        ('fidelity.ks_observed.mean', np.linspace(0, .5, 51), 'above'),
        ('fidelity.js_divergence.mean', np.linspace(0, .5, 51), 'above'),
        ('fidelity.association_abs_error.mean', np.linspace(0, .5, 51), 'above'),
        ('privacy.mia_accuracy', np.linspace(.5, .75, 51), 'above'),
        ('privacy.dcr_ratio', np.linspace(0, 2, 51), 'below'),
        ('utility.auroc_gap_absolute', np.linspace(0, .2, 51), 'above'),
        ('utility.importance_spearman', np.linspace(0, 1, 51), 'below'),
        ('integrity.csr_record_all', np.linspace(.8, 1, 51), 'below'),
        ('fairness.dpd_difference.ethnicity_grouped', np.linspace(0, .2, 51), 'above')]
    definitions += [('score.' + dimension, np.linspace(0, 1, 51), 'below') for dimension in DIMENSIONS]
    all_values = pd.concat([numeric, scored], ignore_index=True)
    for metric, thresholds, direction in definitions:
        part = all_values[all_values.metric.eq(metric)]
        for generator, group in part.groupby('generator'):
            values = pd.to_numeric(group.value, errors='coerce').dropna().to_numpy()
            for threshold in thresholds:
                flags = values > threshold if direction == 'above' else values < threshold
                interval = summarize(flags.astype(float))
                rows.append({'generator': generator, 'metric': metric, 'threshold': threshold, 'direction': direction,
                    'defined_seeds': len(values), 'flagged_seeds': int(flags.sum()),
                    'flagged_fraction': float(flags.mean()) if len(flags) else np.nan,
                    'flagged_ci_low': interval['ci_low'], 'flagged_ci_high': interval['ci_high']})
    # Representation thresholds are swept as a symmetric tolerance about 1.
    for generator, group in numeric[numeric.metric.str.startswith('fairness.representation.')].groupby('generator'):
        for delta in np.linspace(0, 1, 51):
            flags = []
            for seed, part in group.groupby('seed'):
                values = part.value.to_numpy(dtype=float)
                flags.append(True if (np.abs(values - 1) > delta).any() else False if np.isfinite(values).all() else np.nan)
            defined = np.asarray([value for value in flags if np.isfinite(value)], dtype=float)
            interval = summarize(defined)
            rows.append({'generator': generator, 'metric': 'fairness.representation_any_outside_1_plus_minus_delta',
                'threshold': delta, 'direction': 'outside', 'defined_seeds': len(defined),
                'flagged_seeds': int(defined.sum()), 'flagged_fraction': float(defined.mean()) if len(defined) else np.nan,
                'flagged_ci_low': interval['ci_low'], 'flagged_ci_high': interval['ci_high']})
    return pd.DataFrame(rows)
