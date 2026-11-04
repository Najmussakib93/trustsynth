"""Deterministic generator worker. No record previews or unsanitized errors on stdout."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import random
import time
import traceback
import inspect

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .encoding import MissingnessCodec
from .protocol import COHORT, DATA, PROTOCOL, CATEGORICAL_FEATURES, TARGET
from .safe_reporting import PUBLIC, private_path, write_json, category_counts, count


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(PROTOCOL['torch_threads'])
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_cohort(seed):
    cohort = pd.read_csv(COHORT)
    train, test = train_test_split(cohort, test_size=PROTOCOL['test_fraction'],
                                   stratify=cohort[TARGET], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def fit_sample_sdv(train, name, seed, epochs, log_frequency=True, device='cpu'):
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
    seed_everything(seed)
    codec = MissingnessCodec().fit(train)
    encoded = codec.transform(train)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(encoded)
    for c in encoded:
        metadata.update_column(c, sdtype='categorical' if c in CATEGORICAL_FEATURES or c == TARGET or c.endswith('__missing') else 'numerical')
    kwargs = dict(metadata=metadata, enforce_min_max_values=True, enforce_rounding=True)
    if name == 'GaussianCopula':
        model = GaussianCopulaSynthesizer(**kwargs)
    elif name.startswith('CTGAN'):
        model = CTGANSynthesizer(**kwargs, epochs=epochs, log_frequency=log_frequency, cuda=device == 'cuda', verbose=False)
    elif name.startswith('TVAE'):
        model = TVAESynthesizer(**kwargs, epochs=epochs, cuda=device == 'cuda', verbose=False)
    else:
        raise ValueError('Unknown SDV generator')
    start = time.perf_counter()
    # CTGAN otherwise launches one Windows process per logical core. Its RDT
    # random-state contexts are unsafe in threads; serialize the transform.
    epoch_times = []
    from unittest.mock import patch
    import torch
    import importlib
    from ctgan import data_transformer
    original_parallel = data_transformer.Parallel
    def serial_parallel(*args, **kwargs):
        kwargs['n_jobs'] = 1
        return original_parallel(*args, **kwargs)
    training_module = importlib.import_module('ctgan.synthesizers.ctgan' if name.startswith('CTGAN') else 'ctgan.synthesizers.tvae')
    original_tqdm = training_module.tqdm
    class TimedIterator:
        def __init__(self, *args, **kwargs):
            self.inner = original_tqdm(*args, **kwargs)
        def __iter__(self):
            for value in self.inner:
                if device == 'cuda':
                    torch.cuda.synchronize()
                epoch_start = time.perf_counter()
                yield value
                if device == 'cuda':
                    torch.cuda.synchronize()
                epoch_times.append(time.perf_counter() - epoch_start)
        def __getattr__(self, name):
            return getattr(self.inner, name)
    with patch.object(data_transformer, 'Parallel', serial_parallel), patch.object(training_module, 'tqdm', TimedIterator):
        model.fit(encoded)
    fit_seconds = time.perf_counter() - start
    # SDV creates its model during fit; global NumPy/Torch seeds control training.
    # Its explicit model RNG controls sampling, overriding SDV's default fixed seed.
    model._set_random_state(seed)
    start = time.perf_counter()
    synthetic = codec.inverse_transform(model.sample(num_rows=len(train)))
    timings = {'fit_seconds': fit_seconds, 'sample_seconds': time.perf_counter() - start}
    if epoch_times:
        timings.update(epoch_median_seconds=float(np.median(epoch_times)),
                       epoch_p90_seconds=float(np.quantile(epoch_times, .9)),
                       preprocessing_seconds=fit_seconds - sum(epoch_times))
    return synthetic, timings


def run_cell(name, seed, epochs, repeat=False, pilot=False, device='cpu'):
    train, test = split_cohort(seed)
    versions = {p: importlib.metadata.version(p) for p in ['sdv', 'ctgan', 'rdt', 'copulas', 'torch', 'numpy', 'pandas']}
    settings = {'generator': name, 'seed': seed, 'epochs': epochs, 'log_frequency': name != 'CTGAN_no_log',
        'protocol': PROTOCOL, 'train_sha256': fingerprint(train), 'software': versions,
        'device': device,
        'implementation_sha256': hashlib.sha256((inspect.getsource(fit_sample_sdv) +
            inspect.getsource(MissingnessCodec) + inspect.getsource(seed_everything)).encode()).hexdigest()}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    location = DATA / 'synthetic' / (('pilot_' if pilot else '') + key + '.pkl')
    location.parent.mkdir(parents=True, exist_ok=True)
    label = name + '_seed' + str(seed) + ('_pilot_' + device + '_e' + str(epochs) if pilot else '')
    public = PUBLIC / 'generators' / (label + '.json')
    if location.exists() and public.exists():
        existing = json.loads(public.read_text())
        if existing.get('cache_key') == key and (not repeat or existing.get('same_seed_independent_fit_identical') is True):
            print(label + ': validated cache available', flush=True)
            return existing
    captures = io.StringIO()
    with contextlib.redirect_stdout(captures), contextlib.redirect_stderr(captures):
        synth, timings = fit_sample_sdv(train, name, seed, epochs, name != 'CTGAN_no_log', device)
        repeated = None
        if repeat:
            other, repeat_timings = fit_sample_sdv(train, name, seed, epochs, name != 'CTGAN_no_log', device)
            repeated = fingerprint(synth) == fingerprint(other)
            timings['repeat_fit_seconds'] = repeat_timings['fit_seconds']
            if not repeated:
                raise RuntimeError('Independent same-seed generator repeat failed')
    private_path('logs/' + label + '.log').write_text(captures.getvalue(), encoding='utf-8')
    if not synth.gender.isin(['MALE', 'FEMALE']).all() or not synth[TARGET].isin([0, 1]).all():
        raise RuntimeError('Synthetic categorical support invariant failed')
    with location.open('wb') as stream:
        pickle.dump({'synthetic': synth, 'settings': settings}, stream, protocol=4)
    result = {'settings': settings, 'cache_key': key, 'private_artifact': str(location.relative_to(DATA)),
              'timings': timings, 'same_seed_independent_fit_identical': repeated,
              'synthetic_sha256': fingerprint(synth), 'rows': count(len(synth)),
              'gender': {'train': category_counts(train.gender), 'test': category_counts(test.gender),
                         'synthetic': category_counts(synth.gender)}}
    write_json(public, result)
    from .environment_lock import save_lock
    save_lock('sdv')
    print(label + ': completed; fit seconds=' + format(timings['fit_seconds'], '.1f') +
          '; independent repeat identical=' + str(repeated), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', required=True, choices=['GaussianCopula', 'CTGAN', 'TVAE', 'CTGAN_no_log', 'TVAE_900'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    try:
        run_cell(args.generator, args.seed, args.epochs, args.repeat, args.pilot, args.device)
    except Exception as exc:
        private_path('logs/generator_failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('Generator worker failed: ' + type(exc).__name__ + '; details retained privately.', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
