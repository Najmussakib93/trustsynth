"""Bounded AIM attempt with fixed, public preprocessing domains (no private fills).

Theoretical mechanism bounds assume protected random coins and a fixed training
table. Publicly seeded research runs, cohort construction and non-private
evaluation aggregates do not constitute a certified DP release.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import pickle
import random
import time
import traceback
import inspect

import numpy as np
import pandas as pd
from .generators import split_cohort, fingerprint
from .protocol import FEATURES, DATA, PROTOCOL, CONTINUOUS_FEATURES
from .safe_reporting import PUBLIC, private_path, write_json, count, category_counts

BOUNDS = {'age': [18, 120], 'heart_rate_mean': [20, 300], 'sbp_mean': [40, 300],
    'dbp_mean': [20, 200], 'temperature_mean': [25, 45], 'spo2_mean': [50, 100],
    'resp_rate_mean': [4, 60], 'glucose_mean': [0, 2000], 'creatinine_mean': [0, 30],
    'hemoglobin_mean': [0, 30], 'los_days': [0, 365], 'apache_score': [0, 300]}
CATEGORIES = {'gender': ['FEMALE', 'MALE'], 'ethnicity_grouped': ['ASIAN', 'BLACK', 'HISPANIC', 'OTHER', 'WHITE'],
    'admission_type_grouped': ['CARDIAC', 'MEDICAL', 'OTHER', 'SURGICAL'],
    'hospital_region': ['Midwest', 'Northeast', 'South', 'West', 'Unknown'], 'hospital_mortality': [0, 1]}
SETTINGS = {'bins': 12, 'epsilon': [1.0, 10.0], 'delta': 1e-6, 'degree': 2,
            'max_model_size': 20, 'max_cells': 10000, 'rounds': 50,
            'bounds': BOUNDS, 'categories': CATEGORIES, 'preprocessing_epsilon': 0.0,
            'adjacency': 'backend record adjacency on a public fixed-size table; cohort membership not privatized',
            'privacy_claim': 'nominal mechanism budgets for seeded research; public seeds and evaluation outputs are not an end-to-end DP release'}


def encode(frame):
    out = pd.DataFrame(index=frame.index)
    for c in FEATURES:
        if c in BOUNDS:
            lo, hi = BOUNDS[c]
            bins = np.floor((frame[c].clip(lo, hi) - lo) / (hi - lo) * SETTINGS['bins'])
            out[c] = bins.clip(0, SETTINGS['bins'] - 1).fillna(SETTINGS['bins']).astype(int)
        else:
            values = frame[c]
            if c == 'hospital_region':
                values = values.where(values.isin(CATEGORIES[c]), 'Unknown')
            codes = pd.Categorical(values, categories=CATEGORIES[c]).codes
            if (codes < 0).any():
                raise RuntimeError('Value outside the declared public categorical domain')
            out[c] = codes
    return out


def decode(codes):
    out = pd.DataFrame(index=codes.index)
    for c in FEATURES:
        values = pd.to_numeric(codes[c], errors='raise').to_numpy(dtype=int)
        if c in BOUNDS:
            lo, hi = BOUNDS[c]
            out[c] = lo + (values + .5) / SETTINGS['bins'] * (hi - lo)
            out.loc[values == SETTINGS['bins'], c] = np.nan
        else:
            out[c] = np.asarray(CATEGORIES[c])[values]
    return out


def fit_sample(train, epsilon, seed):
    from snsynth.aim import AIMSynthesizer
    from snsynth.transform import TableTransformer, LabelTransformer
    random.seed(seed)
    np.random.seed(seed)
    # Only declared domain values are used to fit category encoders.
    public = pd.DataFrame({c: np.arange(13) % (13 if c in BOUNDS else len(CATEGORIES[c])) for c in FEATURES})
    transform = TableTransformer([LabelTransformer(nullable=False) for _ in FEATURES])
    transform.fit(public)
    model = AIMSynthesizer(epsilon=float(epsilon), delta=SETTINGS['delta'], degree=2,
        max_model_size=SETTINGS['max_model_size'], max_cells=SETTINGS['max_cells'], rounds=SETTINGS['rounds'], verbose=False)
    start = time.perf_counter()
    model.fit(encode(train), transformer=transform, preprocessor_eps=0.0)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sample = model.sample(len(train))
    if not isinstance(sample, pd.DataFrame):
        sample = pd.DataFrame(sample, columns=FEATURES)
    return decode(sample), {'fit_seconds': fit_seconds, 'sample_seconds': time.perf_counter() - start}


def run(seed, epsilon):
    train, test = split_cohort(seed)
    name = 'AIM_eps' + str(int(epsilon))
    settings = {'generator': name, 'seed': seed, 'epsilon': epsilon, 'aim': SETTINGS,
                'protocol': PROTOCOL, 'train_sha256': fingerprint(train),
                'software': {p: importlib.metadata.version(p) for p in ['smartnoise-synth', 'numpy', 'pandas']}}
    settings['implementation_sha256'] = hashlib.sha256((inspect.getsource(fit_sample) + inspect.getsource(encode) + inspect.getsource(decode)).encode()).hexdigest()
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    artifact = DATA / 'synthetic' / (key + '.pkl')
    metadata_path = PUBLIC / 'generators' / (name + '_seed' + str(seed) + '.json')
    if artifact.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()).get('cache_key') == key:
        return
    captures = io.StringIO()
    with contextlib.redirect_stdout(captures), contextlib.redirect_stderr(captures):
        synth, timings = fit_sample(train, epsilon, seed)
        identical = None
        if seed == 0:
            other, _ = fit_sample(train, epsilon, seed)
            identical = fingerprint(synth) == fingerprint(other)
            if not identical:
                raise RuntimeError('AIM independent same-seed repeat failed')
    private_path('logs/' + name + '_seed' + str(seed) + '.log').write_text(captures.getvalue(), encoding='utf-8')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open('wb') as stream:
        pickle.dump({'synthetic': synth, 'settings': settings}, stream, protocol=4)
    write_json(metadata_path, {'settings': settings, 'cache_key': key,
        'private_artifact': str(artifact.relative_to(DATA)), 'synthetic_sha256': fingerprint(synth),
        'rows': count(len(synth)), 'timings': timings, 'same_seed_independent_fit_identical': identical,
        'gender': {'train': category_counts(train.gender), 'test': category_counts(test.gender),
                   'synthetic': category_counts(synth.gender)}})
    from .environment_lock import save_lock
    save_lock('dp')
    print(name + ' seed ' + str(seed) + ' complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-seeds', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    attempt_path = PUBLIC / 'dp_attempt.json'
    if attempt_path.exists() and not args.retry:
        if json.loads(attempt_path.read_text()).get('state') in ['failed', 'unavailable', 'budget_exhausted']:
            print('The bounded AIM attempt has already stopped; preserving its failure report.', flush=True)
            return
    try:
        # Dependency failures happen before private records are loaded and can be reported exactly.
        try:
            from snsynth.aim import AIMSynthesizer
        except ImportError as exc:
            write_json(PUBLIC / 'dp_attempt.json', {'state': 'unavailable', 'phase': 'dependency_import',
                'exception': type(exc).__name__, 'message': str(exc), 'settings': SETTINGS})
            print('AIM unavailable: ' + str(exc), flush=True)
            return
        for epsilon in [args.epsilon] if args.epsilon else SETTINGS['epsilon']:
            for seed in PROTOCOL['seeds'] if args.all_seeds else [args.seed]:
                if time.perf_counter() - started >= PROTOCOL['dp_attempt_limit_seconds']:
                    write_json(PUBLIC / 'dp_attempt.json', {'state': 'budget_exhausted', 'limit_seconds': PROTOCOL['dp_attempt_limit_seconds']})
                    return
                run(seed, epsilon)
        write_json(PUBLIC / 'dp_attempt.json', {'state': 'full_protocol_complete' if args.all_seeds and args.epsilon is None else 'requested_cells_complete',
            'wall_seconds': time.perf_counter() - started, 'settings': SETTINGS})
    except Exception as exc:
        private_path('logs/aim_failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(PUBLIC / 'dp_attempt.json', {'state': 'failed', 'phase': 'fit_or_sample',
            'exception': type(exc).__name__, 'wall_seconds': time.perf_counter() - started})
        print('AIM attempt stopped after ' + type(exc).__name__ + '; no further DP tuning attempted.', flush=True)


if __name__ == '__main__':
    main()
