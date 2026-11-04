"""Synthcity's actual ddpm plugin on the same train partitions and missingness codec."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import pickle
import time
import traceback
import inspect

from .encoding import MissingnessCodec
from .generators import split_cohort, seed_everything, fingerprint
from .protocol import DATA, TARGET, PROTOCOL
from .safe_reporting import PUBLIC, private_path, write_json, count, category_counts

SETTINGS = {'n_iter': 1000, 'batch_size': 1024, 'num_timesteps': 1000, 'lr': .002,
            'weight_decay': .0001, 'scheduler': 'cosine', 'model_type': 'mlp',
            'model_params': {'n_layers_hidden': 3, 'n_units_hidden': 256, 'dropout': 0.0},
            'is_classification': False, 'validation_size': 0, 'continuous_encoder': 'quantile'}


def fit_sample(train, seed, device, iterations=1000, sample_rows=None):
    # Import the requested plugin directly, avoiding discovery of unrelated models.
    from synthcity.plugins.generic.plugin_ddpm import plugin
    from synthcity.plugins.core.dataloader import GenericDataLoader
    import torch
    seed_everything(seed)
    codec = MissingnessCodec().fit(train)
    encoded = codec.transform(train)
    encoded[TARGET] = encoded[TARGET].astype(str)
    settings = dict(SETTINGS, n_iter=iterations)
    model = plugin(**settings, random_state=seed, device=torch.device(device),
        workspace=private_path('synthcity_workspace/seed' + str(seed)))
    start = time.perf_counter()
    import importlib
    from unittest.mock import patch
    module = importlib.import_module('synthcity.plugins.core.models.tabular_ddpm')
    original_trange = module.trange
    epoch_times = []
    class TimedEpochs:
        def __init__(self, *args, **kwargs):
            self.inner = original_trange(*args, **kwargs)
        def __iter__(self):
            for value in self.inner:
                epoch_start = time.perf_counter()
                yield value
                epoch_times.append(time.perf_counter() - epoch_start)
        def __getattr__(self, name):
            return getattr(self.inner, name)
    with patch.object(module, 'trange', TimedEpochs):
        model.fit(GenericDataLoader(encoded, target_column=TARGET, random_state=seed))
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    out = model.generate(len(train) if sample_rows is None else sample_rows, random_state=seed).dataframe()
    out[TARGET] = out[TARGET].astype(int)
    timings = {'fit_seconds': fit_seconds, 'sample_seconds': time.perf_counter() - start}
    if epoch_times:
        timings.update(epoch_median_seconds=float(__import__('numpy').median(epoch_times)),
            epoch_p90_seconds=float(__import__('numpy').quantile(epoch_times, .9)),
            preprocessing_seconds=fit_seconds - sum(epoch_times))
    return codec.inverse_transform(out), timings


def run(seed, device, pilot=False, repeat=False):
    train, test = split_cohort(seed)
    label = 'TabDDPM_seed' + str(seed) + ('_pilot' if pilot else '')
    versions = {name: importlib.metadata.version(name) for name in ['synthcity', 'torch', 'numpy', 'pandas', 'scikit-learn']}
    settings = dict(generator='TabDDPM', seed=seed, device=device, protocol=PROTOCOL, ddpm=SETTINGS,
                    n_iter=3 if pilot else SETTINGS['n_iter'], train_sha256=fingerprint(train), software=versions,
                    sample_rows=min(1024, len(train)) if pilot else len(train),
                    implementation_sha256=hashlib.sha256((inspect.getsource(fit_sample) + inspect.getsource(MissingnessCodec)).encode()).hexdigest(),
                    compatibility_patch_sha256=hashlib.sha256((PUBLIC / 'synthcity_import_compatibility.patch').read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    metadata_path = PUBLIC / 'generators' / (label + '.json')
    artifact = DATA / 'synthetic' / (key + '.pkl')
    if metadata_path.exists() and artifact.exists():
        existing = json.loads(metadata_path.read_text())
        if existing.get('cache_key') == key and (not repeat or existing.get('same_seed_independent_fit_identical') is True):
            return
    captures = io.StringIO()
    with contextlib.redirect_stdout(captures), contextlib.redirect_stderr(captures):
        synth, timings = fit_sample(train, seed, device, settings['n_iter'], settings['sample_rows'])
        identical = None
        if repeat:
            other, _ = fit_sample(train, seed, device, settings['n_iter'], settings['sample_rows'])
            identical = fingerprint(synth) == fingerprint(other)
            if not identical:
                raise RuntimeError('DDPM independent same-seed repeat failed')
    private_path('logs/' + label + '.log').write_text(captures.getvalue(), encoding='utf-8')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open('wb') as stream:
        pickle.dump({'synthetic': synth, 'settings': settings}, stream, protocol=4)
    write_json(metadata_path, {'settings': settings, 'cache_key': key,
        'private_artifact': str(artifact.relative_to(DATA)), 'synthetic_sha256': fingerprint(synth),
        'rows': count(len(synth)), 'timings': timings, 'same_seed_independent_fit_identical': identical,
        'gender': {'train': category_counts(train.gender), 'test': category_counts(test.gender),
                   'synthetic': category_counts(synth.gender)}})
    from .environment_lock import save_lock
    save_lock('synthcity')
    print(label + ': aggregate timing and reproducibility metadata saved', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-seeds', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    try:
        for seed in PROTOCOL['seeds'] if args.all_seeds else [args.seed]:
            run(seed, args.device, args.pilot, seed == 0)
    except Exception as exc:
        private_path('logs/ddpm_failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('DDPM failed: ' + type(exc).__name__ + '; details retained privately.', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
