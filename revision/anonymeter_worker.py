"""Declared Anonymeter attacks with real-test controls and seeded RNG factories."""
import argparse
import contextlib
import hashlib
import io
import json
import random
import time
import traceback
import inspect
import importlib.metadata
from unittest.mock import patch

import numpy as np
from .evaluate import load_cell
from .protocol import FEATURES
from .generators import fingerprint
from .safe_reporting import ROOT, PUBLIC, private_path, write_json, write_csv, count

CONFIG = json.loads((ROOT / 'revision' / 'attacker_config.json').read_text())


@contextlib.contextmanager
def seeded_numpy(seed):
    """Keep NumPy's Generator algorithm, supplying spawned seeds where absent.

    Anonymeter 1.0.0 does not expose random_state for all RNG paths. Patching
    the factory in this single-process context makes those paths reproducible;
    explicitly seeded generators retain their requested seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    original = np.random.default_rng
    sequence = np.random.SeedSequence(seed)
    def factory(seed=None):
        return original(sequence.spawn(1)[0] if seed is None else seed)
    from anonymeter.evaluators import singling_out_evaluator
    with patch('numpy.random.default_rng', factory), patch.object(singling_out_evaluator, 'rng', original(sequence.spawn(1)[0])):
        yield


def attack(train, test, synth, kind, seed):
    from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
    common = dict(ori=train, syn=synth, control=test, n_attacks=CONFIG['n_attacks'])
    start = time.perf_counter()
    generated_queries = []
    with seeded_numpy(seed):
        if kind == 'singling_out':
            from anonymeter.evaluators import singling_out_evaluator as module
            original = module._generate_singling_out_queries
            def capture_query_count(*args, **kwargs):
                queries = original(*args, **kwargs)
                generated_queries.append(len(queries))
                return queries
            evaluator = SinglingOutEvaluator(**common, n_cols=CONFIG['singling_out']['n_cols'],
                max_attempts=CONFIG['singling_out']['max_attempts'])
            with patch.object(module, '_generate_singling_out_queries', capture_query_count):
                evaluator.evaluate(mode='multivariate')
        elif kind == 'linkability':
            evaluator = LinkabilityEvaluator(**common,
                aux_cols=tuple(CONFIG['linkability']['aux_cols']), n_neighbors=CONFIG['linkability']['n_neighbors'])
            evaluator.evaluate(n_jobs=1)
        else:
            secret = kind.removeprefix('inference_')
            evaluator = InferenceEvaluator(**common, aux_cols=[c for c in FEATURES if c != secret],
                secret=secret, regression=False)
            evaluator.evaluate(n_jobs=1)
        risk = evaluator.risk(confidence_level=CONFIG['confidence_level'])
    result = {'risk': float(risk.value), 'ci_low': float(risk.ci[0]), 'ci_high': float(risk.ci[1]),
              'wall_seconds': time.perf_counter() - start}
    if generated_queries:
        result.update(generated_queries=count(generated_queries[0]),
                      query_generation_complete=generated_queries[0] == CONFIG['n_attacks'])
    return result


def run(generator, seed, kind):
    train, test, synth, metadata = load_cell(generator, seed)
    signature = hashlib.sha256((json.dumps(CONFIG, sort_keys=True) + metadata['synthetic_sha256'] +
        fingerprint(train) + fingerprint(test) + inspect.getsource(attack) + inspect.getsource(seeded_numpy) +
        json.dumps({p: importlib.metadata.version(p) for p in ['anonymeter', 'numpy', 'pandas', 'numba', 'scikit-learn']}, sort_keys=True)).encode()).hexdigest()
    path = PUBLIC / 'anonymeter' / (generator + '-' + str(seed) + '-' + kind + '.json')
    if path.exists() and json.loads(path.read_text()).get('key') == signature:
        return
    captures = io.StringIO()
    with contextlib.redirect_stdout(captures), contextlib.redirect_stderr(captures):
        result = attack(train, test, synth, kind, seed)
        identical = None
        if seed == 0:
            other = attack(train, test, synth, kind, seed)
            identical = all(result[k] == other[k] for k in ['risk', 'ci_low', 'ci_high'])
            if not identical:
                raise RuntimeError('Anonymeter independent seeded repeat failed')
    private_path('logs/anonymeter-' + generator + '-' + str(seed) + '-' + kind + '.log').write_text(captures.getvalue(), encoding='utf-8')
    write_json(path, dict(seed=seed, generator=generator, attack=kind, key=signature,
                         independent_repeat_identical=identical, attacker_config=CONFIG, **result))
    directory = PUBLIC / 'cells' / (generator + '-' + str(seed))
    write_csv(directory / ('anonymeter-' + kind + '_metrics.csv'), [
        {'seed': seed, 'generator': generator, 'metric': 'privacy.anonymeter.' + kind + '.' + metric,
         'value': float(value) if isinstance(value, bool) else value}
        for metric, value in result.items()])
    from .environment_lock import save_lock
    save_lock('anonymeter')
    print(generator + ' seed ' + str(seed) + ' ' + kind + ': aggregate risk and CI saved', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-cells', action='store_true')
    parser.add_argument('--generator', default='GaussianCopula')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--attack', default='singling_out')
    args = parser.parse_args()
    try:
        if not args.all_cells:
            run(args.generator, args.seed, args.attack)
            return
        names = sorted({p.name.split('_seed')[0] for p in (PUBLIC / 'generators').glob('*_seed*.json')
                        if '_pilot' not in p.name and not p.name.startswith(('CTGAN_no_log', 'TVAE_900'))})
        for generator in names:
            for seed in CONFIG['singling_out']['seeds']:
                run(generator, seed, 'singling_out')
                if seed in CONFIG['linkability']['seeds']:
                    run(generator, seed, 'linkability')
                    for secret in CONFIG['inference']['secrets']:
                        run(generator, seed, 'inference_' + secret)
    except Exception as exc:
        private_path('logs/anonymeter_failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('Anonymeter failed: ' + type(exc).__name__ + '; details retained privately.', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
