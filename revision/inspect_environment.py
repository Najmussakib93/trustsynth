"""Software-only smoke checks; no clinical data are accessed."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['sdv', 'synthcity', 'dp', 'anonymeter'])
    args = parser.parse_args()
    started = time.perf_counter()
    if args.kind == 'sdv':
        import torch
        print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA smoke check failed')
        import sdv
        print('SDV:', sdv.__version__, flush=True)
        a = torch.ones((32, 32), device='cuda')
        _ = a @ a
        print('CUDA matrix operation succeeded', flush=True)
    elif args.kind == 'synthcity':
        import torch
        from synthcity.plugins.generic.plugin_ddpm import plugin
        plugin(n_iter=1, device=torch.device('cpu'))
        print('DDPM plugin construction succeeded', flush=True)
    elif args.kind == 'dp':
        from snsynth.aim import AIMSynthesizer
        AIMSynthesizer(epsilon=1.0)
        print('AIM construction succeeded', flush=True)
    else:
        from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
        from numba import njit
        print('Anonymeter evaluators and Numba imported', flush=True)
    root = Path(__file__).resolve().parents[1]
    target = root / ('.venv-' + args.kind)
    (target / 'READY.json').write_text(json.dumps({'kind': args.kind, 'smoke_seconds': time.perf_counter() - started}))


if __name__ == '__main__':
    main()
