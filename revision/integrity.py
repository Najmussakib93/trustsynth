"""Approved hard-rule CSR and separate descriptive population comparisons.

Input records and bootstrap draws remain local. Public exports contain aggregates
only. All-bootstrap intervals condition on the trained generator and train cutoffs.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import pandas as pd

from .constraint_proposal import original_checks
from .legacy.config import DOMAIN_CONSTRAINTS
from .evaluate import load_cell
from .generators import fingerprint
from .protocol import PROTOCOL
from .safe_reporting import ROOT, PUBLIC, count, private_path, write_csv, write_json

APPROVAL = json.loads((Path(__file__).with_name('structural_approval.json')).read_text())
AGE_LABELS = ['18-44', '45-64', '65-79', '80+']
UNITS = ['CARDIAC', 'MEDICAL', 'OTHER', 'SURGICAL']


def safe_rate(events, total):
    """Withhold a non-extreme rate if it would reveal a small event/complement."""
    events, total = int(events), int(total)
    if total < 11 or 0 < min(events, total - events) < 11:
        return np.nan
    return events / total


def hard_rules(frame, dropped):
    rows, applicable_all, valid_all, included_rates = [], np.ones(len(frame), bool), np.ones(len(frame), bool), []
    for name, applicable, valid in original_checks(frame):
        applicable, valid = applicable.to_numpy(), valid.to_numpy()
        n, passed = int(applicable.sum()), int((applicable & valid).sum())
        rate = safe_rate(passed, n)
        rows.append({'constraint': name, 'excluded_primary': name in dropped,
            'applicable_n': count(n), 'violations_n': count(n - passed),
            'all_applicable_records_valid': bool(n > 0 and passed == n),
            'csr': rate, 'coverage': safe_rate(n, len(frame))})
        if name not in dropped:
            applicable_all &= applicable
            valid_all &= valid
            included_rates.append(rate)
    n = int(applicable_all.sum())
    passed = int((applicable_all & valid_all).sum())
    result = {'integrity.csr_record_all': safe_rate(passed, n),
        'integrity.complete_rule_coverage': safe_rate(n, len(frame)),
        'integrity.csr_constraint_mean': float(np.mean(included_rates)) if np.isfinite(included_rates).all() else np.nan}
    result.update({'integrity.constraint.' + row['constraint'] + '.csr': row['csr'] for row in rows})
    result.update({'integrity.constraint.' + row['constraint'] + '.coverage': row['coverage'] for row in rows})
    return rows, result


def parameters(real):
    apache = real.apache_score.dropna().to_numpy()
    pair = real[['spo2_mean', 'resp_rate_mean']].dropna()
    return {'apache_quintile_edges': np.quantile(apache, [.2, .4, .6, .8]).tolist() if len(apache) >= 11 else None,
            'spo2_lower_quintile': float(pair.spo2_mean.quantile(.2)) if len(pair) >= 11 else None,
            'resp_rate_upper_quintile': float(pair.resp_rate_mean.quantile(.8)) if len(pair) >= 11 else None}


def inherited_rule_audit(train):
    """Classify inheritance from SDV bounds/support without exporting extrema."""
    rows = []
    for name, (column, operation, bound) in DOMAIN_CONSTRAINTS.items():
        values = train[column].dropna()
        if values.empty:
            inherited, reason = False, 'no_observed_training_support'
        elif operation == 'in':
            inherited = bool(values.isin(bound).all())
            reason = 'observed_categorical_support' if inherited else 'training_support_violates_rule'
        else:
            endpoint = values.min() if operation in ['>', '>='] else values.max()
            inherited = bool({'>': lambda: endpoint > bound, '>=': lambda: endpoint >= bound,
                              '<': lambda: endpoint < bound, '<=': lambda: endpoint <= bound}[operation]())
            reason = 'observed_bounds_imply_rule' if inherited else 'observed_bounds_do_not_imply_rule'
        rows.append({'constraint': name, 'inherited_when_observed': inherited, 'reason': reason,
                     'scope': 'SDV enforce_min_max_values/categorical support; restored missingness excluded'})
    rows.append({'constraint': 'sbp_mean_greater_than_dbp_mean', 'inherited_when_observed': False,
                 'reason': 'cross_column_relation_not_enforced', 'scope': 'SDV univariate bounds'})
    return rows


def numeric_bootstrap(values, population_size, rng, resamples):
    """Exact marginal ordinary-record bootstrap for a conditional distribution.

    Its observed group count is Binomial(N,n/N); observations within that group
    are then sampled from its empirical distribution. Group size is not fixed.
    """
    values = np.asarray(values, dtype=float)
    point = np.quantile(values, [.25, .5, .75]) if len(values) >= 11 else np.full(3, np.nan)
    boot = np.full((resamples, 3), np.nan)
    if len(values) >= 11:
        sizes = rng.binomial(population_size, len(values) / population_size, size=resamples)
        for b, size in enumerate(sizes):
            if size >= 11:
                boot[b] = np.quantile(values[rng.randint(len(values), size=size)], [.25, .5, .75])
    return point, boot


def population_profile(frame, cutoffs, seed, resamples):
    rng = np.random.RandomState(seed)
    output = {}
    n_total = len(frame)

    def store(family, component, units, n, point, draws, status=None):
        output[(family, component)] = {'units': units, 'observed_n': count(n),
            'coverage': safe_rate(n, n_total), 'estimate': float(point),
            'draws': np.asarray(draws, dtype=float),
            'status': status or ('defined' if np.isfinite(point) else 'insufficient_support')}

    # Quantile boundaries are learned from real train once and held fixed.
    valid = frame.apache_score.notna() & frame.hospital_mortality.notna()
    if cutoffs['apache_quintile_edges'] is not None:
        group = np.searchsorted(cutoffs['apache_quintile_edges'], frame.loc[valid, 'apache_score'].to_numpy(), side='right')
        y = frame.loc[valid, 'hospital_mortality'].to_numpy()
        ns = np.bincount(group, minlength=5)
        positives = np.bincount(group, weights=y, minlength=5)
        sizes = rng.multinomial(n_total, np.r_[ns, n_total - ns.sum()] / n_total, size=resamples)[:, :5]
        rates = np.divide(positives, ns, out=np.zeros(5), where=ns > 0)
        successes = rng.binomial(sizes, rates)
        draws = np.divide(successes, sizes, out=np.full_like(successes, np.nan, dtype=float), where=sizes >= 11)
        points = np.where((ns >= 11) & (positives >= 11) & (ns - positives >= 11), rates, np.nan)
        for i in range(5):
            store('apache_mortality', 'Q' + str(i + 1) + '_risk', 'probability', ns[i], points[i], draws[:, i])
        store('apache_mortality', 'Q5_minus_Q1_risk', 'probability_difference', min(ns[0], ns[4]),
              points[4] - points[0], draws[:, 4] - draws[:, 0])
    else:
        for component in ['Q' + str(i) + '_risk' for i in range(1, 6)] + ['Q5_minus_Q1_risk']:
            store('apache_mortality', component, 'probability', 0, np.nan, np.full(resamples, np.nan))

    pair = frame[['spo2_mean', 'resp_rate_mean']].dropna()
    if cutoffs['spo2_lower_quintile'] is not None and len(pair):
        event = (pair.spo2_mean.le(cutoffs['spo2_lower_quintile']) &
                 pair.resp_rate_mean.ge(cutoffs['resp_rate_upper_quintile']))
        successes, n = int(event.sum()), len(pair)
        sizes = rng.binomial(n_total, n / n_total, size=resamples)
        counts = rng.binomial(sizes, successes / n)
        draws = np.divide(counts, sizes, out=np.full(resamples, np.nan), where=sizes >= 11)
        point = successes / n if min(successes, n - successes) >= 11 else np.nan
        store('joint_respiratory', 'low_spo2_and_high_resp_rate', 'probability', n, point, draws)
    else:
        store('joint_respiratory', 'low_spo2_and_high_resp_rate', 'probability', 0, np.nan, np.full(resamples, np.nan))

    age = pd.cut(frame.age, [18, 45, 65, 80, np.inf], right=False, labels=AGE_LABELS)
    for family, column, labels, group, statistics, units in [
        ('los_by_unit', 'los_days', UNITS, frame.admission_type_grouped, ['median', 'q75'], 'days'),
        ('age_creatinine', 'creatinine_mean', AGE_LABELS, age, ['median', 'q75'], 'recorded_creatinine_units'),
        ('age_hemoglobin', 'hemoglobin_mean', AGE_LABELS, age, ['median', 'q25'], 'recorded_hemoglobin_units'),
        ('unit_severity', 'apache_score', UNITS, frame.admission_type_grouped, ['median', 'iqr'], 'APACHE_points')]:
        for label in labels:
            values = frame.loc[group.eq(label), column].dropna().to_numpy()
            point, boot = numeric_bootstrap(values, n_total, rng, resamples)
            for statistic in statistics:
                index = {'q25': 0, 'median': 1, 'q75': 2}.get(statistic)
                estimate = point[index] if index is not None else point[2] - point[0]
                draws = boot[:, index] if index is not None else boot[:, 2] - boot[:, 0]
                store(family, label + '.' + statistic, units, len(values), estimate, draws)

    # Joint multinomial group sizes preserve covariance for the outcome contrast.
    outcome_values = [frame.loc[frame.hospital_mortality.eq(y), 'los_days'].dropna().to_numpy() for y in [0, 1]]
    ns = np.array([len(value) for value in outcome_values])
    sizes = rng.multinomial(n_total, np.r_[ns, n_total - ns.sum()] / n_total, size=resamples)[:, :2]
    points, draws = [], []
    for i, values in enumerate(outcome_values):
        point = np.quantile(values, [.5, .75]) if len(values) >= 11 else np.full(2, np.nan)
        boot = np.full((resamples, 2), np.nan)
        if len(values) >= 11:
            for b, size in enumerate(sizes[:, i]):
                if size >= 11:
                    boot[b] = np.quantile(values[rng.randint(len(values), size=size)], [.5, .75])
        points.append(point)
        draws.append(boot)
    for i, statistic in enumerate(['median', 'q75']):
        store('los_by_mortality', 'deceased_minus_surviving.' + statistic, 'days', ns.min(),
              points[1][i] - points[0][i], draws[1][:, i] - draws[0][:, i])
    return output


def compare_profiles(real, synthetic):
    rows = []
    for (family, component), a in real.items():
        b = synthetic[(family, component)]
        point = b['estimate'] - a['estimate']
        differences = b['draws'] - a['draws']
        valid = np.isfinite(differences)
        bounds = np.quantile(differences[valid], APPROVAL['interval']) if np.isfinite(point) and valid.any() else [np.nan, np.nan]
        # Do not expose a small-event rate indirectly through the other rate + difference.
        defined = np.isfinite(point)
        rows.append({'family': family, 'component': component, 'units': a['units'],
            'real_observed_n': a['observed_n'], 'synthetic_observed_n': b['observed_n'],
            'real_coverage': a['coverage'], 'synthetic_coverage': b['coverage'],
            'real_estimate': a['estimate'] if defined else np.nan,
            'synthetic_estimate': b['estimate'] if defined else np.nan,
            'difference': point, 'ci_low': bounds[0], 'ci_high': bounds[1],
            'valid_bootstrap_resamples': count(valid.sum()) if defined else '<11',
            'status': 'defined' if defined else 'insufficient_support_or_disclosure',
            'interval_scope': 'pointwise_conditional_bootstrap; no preservation decision'})
    return rows


def validate_approval():
    if APPROVAL['status'] != 'approved_by_author':
        raise RuntimeError('The structural-check approval is missing')
    baseline = pd.read_csv(PUBLIC / 'integrity/original_real_train.csv')
    violated = set(baseline.loc[~baseline.all_applicable_records_valid.astype(str).str.lower().eq('true'), 'constraint'])
    if violated != set(APPROVAL['excluded_primary_rules']):
        raise RuntimeError('Real-rule violations differ from the approved baseline; audit the cohort before continuing')
    return violated


def signature(metadata, train, test, resamples):
    return hashlib.sha256((Path(__file__).read_text(encoding='utf-8') + json.dumps(APPROVAL, sort_keys=True) +
        metadata['synthetic_sha256'] + fingerprint(train) + fingerprint(test) + str(resamples) +
        np.__version__ + pd.__version__).encode()).hexdigest()


def run_cell(generator, seed, resamples=1000, pilot=False):
    train, test, synth, metadata = load_cell(generator, seed)
    key = signature(metadata, train, test, resamples)
    directory = PUBLIC / 'cells' / (generator + '-' + str(seed))
    stamp = directory / 'integrity_status.json'
    if not pilot and stamp.exists() and json.loads(stamp.read_text()).get('key') == key:
        return 0.
    dropped = validate_approval()
    start = time.perf_counter()
    cutoffs = parameters(train)
    # Cache only real bootstrap summaries, never export bootstrap draws publicly.
    real_key = hashlib.sha256((Path(__file__).read_text(encoding='utf-8') + fingerprint(train) + str(seed) + str(resamples)).encode()).hexdigest()
    cache = private_path('population_real/' + real_key + '.pkl')
    if cache.exists():
        with cache.open('rb') as stream:
            payload = pickle.load(stream)
            real, reference = payload['real'], payload['reference']
    else:
        real = population_profile(train, cutoffs, seed, resamples)
        reference = population_profile(train, cutoffs, seed + 10000, resamples)
        with cache.open('wb') as stream:
            pickle.dump({'real': real, 'reference': reference}, stream, protocol=4)
    profile = population_profile(synth, cutoffs, seed + 10000, resamples)
    population = compare_profiles(real, profile)
    repeated = None
    if seed == 0 and not pilot:
        again = compare_profiles(real, population_profile(synth, cutoffs, seed + 10000, resamples))
        repeated = pd.DataFrame(population).equals(pd.DataFrame(again))
        if not repeated:
            raise RuntimeError('Population bootstrap independent repeat failed')
    if pilot:
        return time.perf_counter() - start
    hard, metrics = hard_rules(synth, dropped)
    real_hard, real_metrics = hard_rules(train, dropped)
    for row in population:
        prefix = 'population.' + row['family'] + '.' + row['component'] + '.'
        for name in ['real_estimate', 'synthetic_estimate', 'difference', 'ci_low', 'ci_high', 'real_coverage', 'synthetic_coverage']:
            metrics[prefix + name] = row[name]
    write_csv(directory / 'integrity_constraints.csv', [dict(seed=seed, generator=generator, dataset='SYNTHETIC', **row) for row in hard])
    write_csv(directory / 'population_consistency.csv', [dict(seed=seed, generator=generator, **row) for row in population])
    write_csv(directory / 'integrity_metrics.csv', [{'seed': seed, 'generator': generator, 'metric': metric,
        'value': value if np.isfinite(value) else ''} for metric, value in metrics.items()])
    write_csv(PUBLIC / 'integrity' / ('real_seed' + str(seed) + '.csv'), real_hard)
    write_csv(PUBLIC / 'integrity' / ('sdv_inherited_rules_seed' + str(seed) + '.csv'), inherited_rule_audit(train))
    write_csv(PUBLIC / 'integrity' / ('population_self_reference_seed' + str(seed) + '.csv'), compare_profiles(real, reference))
    write_json(PUBLIC / 'integrity' / ('cutoffs_seed' + str(seed) + '.json'), cutoffs)
    write_json(stamp, {'key': key, 'approval': APPROVAL, 'bootstrap_resamples': resamples,
        'independent_repeat_identical': repeated, 'wall_seconds': time.perf_counter() - start})
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', default='GaussianCopula')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    try:
        if args.benchmark:
            seconds = run_cell(args.generator, args.seed, resamples=25, pilot=True)
            cells = len([p for p in (PUBLIC / 'generators').glob('*_seed*.json') if '_pilot' not in p.name])
            estimate = seconds * APPROVAL['bootstrap_resamples'] / 25 * (cells + len(PROTOCOL['seeds']))
            write_json(PUBLIC / 'finalization_runtime_estimate.json', {'pilot_seconds': seconds, 'pilot_resamples': 25,
                'estimated_integrity_seconds': estimate, 'cells': cells,
                'scope': 'conservative linear bootstrap extrapolation; excludes plots/statistics; runtime is provisional'})
            print('Estimated integrity/population evaluation minutes: ' + format(estimate / 60, '.1f'), flush=True)
        else:
            # Exceptions can contain records; the finalizer also captures subprocess output privately.
            with contextlib.redirect_stderr(io.StringIO()):
                run_cell(args.generator, args.seed)
            print(args.generator + ' seed ' + str(args.seed) + ': approved integrity aggregates saved', flush=True)
    except Exception as exc:
        private_path('logs/integrity_failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('Integrity evaluation stopped: ' + type(exc).__name__ + '; details retained privately.', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
