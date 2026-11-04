"""Resumable aggregate evaluation cells; generated records never leave private paths."""
import argparse
import contextlib
import hashlib
import io
import json
import pickle
import time
import traceback
import importlib.metadata
from pathlib import Path

import numpy as np

from .generators import split_cohort, fingerprint
from .protocol import DATA, PROTOCOL
from .safe_reporting import PUBLIC, private_path, write_json, write_csv


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_cell(generator, seed):
    metadata = json.loads((PUBLIC / 'generators' / (generator + '_seed' + str(seed) + '.json')).read_text())
    path = (DATA / metadata['private_artifact']).resolve()
    if DATA.resolve() not in path.parents:
        raise RuntimeError('Unsafe synthetic cache path')
    with path.open('rb') as stream:
        bundle = pickle.load(stream)
    train, test = split_cohort(seed)
    if bundle['settings']['train_sha256'] != fingerprint(train):
        raise RuntimeError('Synthetic set belongs to a different training split')
    if fingerprint(bundle['synthetic']) != metadata['synthetic_sha256']:
        raise RuntimeError('Synthetic cache hash mismatch')
    return train, test, bundle['synthetic'], metadata


def evaluate_cell(generator, seed, phase):
    started = time.perf_counter()
    train, test, synth, metadata = load_cell(generator, seed)
    directory = PUBLIC / 'cells' / (generator + '-' + str(seed))
    directory.mkdir(parents=True, exist_ok=True)
    source_files = ['evaluate.py', 'protocol.py'] + (['fidelity.py', 'utility.py', 'fairness.py'] if phase == 'quality' else ['privacy.py'])
    source_hash = hashlib.sha256(b''.join((Path(__file__).parent / f).read_bytes() for f in source_files)).hexdigest()
    software = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost']}
    if phase == 'privacy':
        try:
            software['numba'] = importlib.metadata.version('numba')
        except importlib.metadata.PackageNotFoundError:
            software['numba'] = None
    real_key = hashlib.sha256((source_hash + fingerprint(train) + fingerprint(test) +
                              json.dumps(software, sort_keys=True)).encode()).hexdigest()
    stamp_path = directory / (phase + '_status.json')
    key = hashlib.sha256((real_key + metadata['synthetic_sha256']).encode()).hexdigest()
    if stamp_path.exists() and json.loads(stamp_path.read_text()).get('key') == key:
        print(generator + ' seed ' + str(seed) + ' ' + phase + ': validated aggregate cache', flush=True)
        return
    metrics = {}
    if phase == 'quality':
        from .fidelity import evaluate_fidelity
        from .utility import evaluate_utility, fit_predict
        from .fairness import evaluate_fairness
        metrics.update(evaluate_fidelity(train, synth))
        trtr_path = private_path('trtr/seed' + str(seed) + '-' + real_key + '.pkl')
        trtr = None
        if trtr_path.exists():
            with trtr_path.open('rb') as stream:
                trtr = pickle.load(stream)
        utility, subgroups, trtr = evaluate_utility(train, test, synth, seed, trtr)
        with trtr_path.open('wb') as stream:
            pickle.dump(trtr, stream, protocol=4)
        metrics.update(utility)
        fairness, representation, parity = evaluate_fairness(train, synth, seed)
        metrics.update(fairness)
        for name, rows in [('subgroup_utility', subgroups), ('representation', representation), ('parity', parity)]:
            write_csv(directory / (name + '.csv'), [dict(seed=seed, generator=generator, **row) for row in rows],
                      columns=list(dict.fromkeys(k for row in rows for k in dict(seed=seed, generator=generator, **row))))
        for name, value in metadata['timings'].items():
            metrics['runtime.' + name] = value
    elif phase == 'privacy':
        from .privacy import evaluate_membership, evaluate_dcr
        real_cache = private_path('dcr/real-seed' + str(seed) + '-' + real_key + '.pkl')
        distances = None
        if real_cache.exists():
            with real_cache.open('rb') as stream:
                distances = pickle.load(stream)
        dcr, duplicate_rows, distances = evaluate_dcr(train, test, synth, seed, distances)
        metrics.update(dcr)
        with real_cache.open('wb') as stream:
            pickle.dump(distances, stream, protocol=4)
        write_csv(directory / 'near_duplicates.csv', [dict(seed=seed, generator=generator, **row) for row in duplicate_rows])
        metrics.update(evaluate_membership(train, test, synth, seed))
    write_csv(directory / (phase + '_metrics.csv'), [
        {'seed': seed, 'generator': generator, 'metric': k, 'value': v if not isinstance(v, (float, np.floating)) or np.isfinite(v) else ''}
        for k, v in metrics.items()])
    write_json(stamp_path, {'key': key, 'source_sha256': source_hash,
        'real_split_sha256': real_key, 'software': software,
        'synthetic_sha256': metadata['synthetic_sha256'], 'wall_seconds': time.perf_counter() - started,
        'undefined_metrics': [k for k, v in metrics.items() if isinstance(v, (float, np.floating)) and not np.isfinite(v)]})
    from .environment_lock import save_lock
    save_lock('evaluation' if phase == 'quality' else 'privacy')
    print(generator + ' seed ' + str(seed) + ' ' + phase + ': aggregate outputs complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--phase', choices=['quality', 'privacy'], default='quality')
    args = parser.parse_args()
    captures = io.StringIO()
    try:
        # Third-party messages can include records in validation errors; keep them private.
        with contextlib.redirect_stderr(captures):
            evaluate_cell(args.generator, args.seed, args.phase)
    except Exception as exc:
        private_path('logs/evaluation_failure.log').write_text(captures.getvalue() + traceback.format_exc(), encoding='utf-8')
        print('Evaluation failed: ' + type(exc).__name__ + '; details retained privately.', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
