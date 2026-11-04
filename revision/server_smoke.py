"""Test the real generator paths on invented tables, with independent repeats."""
import argparse
import contextlib
import io
import os

import numpy as np
import pandas as pd

from .protocol import CONTINUOUS_FEATURES, FEATURES, TARGET
from .generators import fingerprint


def invented_table():
    rng = np.random.RandomState(17)
    table = pd.DataFrame({column: rng.normal(50, 5, 128) if column in CONTINUOUS_FEATURES
        else rng.choice(['A', 'B'], 128) for column in FEATURES})
    table['gender'] = rng.choice(['FEMALE', 'MALE'], len(table))
    table[TARGET] = rng.binomial(1, .2, len(table))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['sdv', 'synthcity'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; run setup on an allocated GPU node or select --device cpu.')
        _ = torch.ones((32, 32), device='cuda') @ torch.ones((32, 32), device='cuda')
        torch.cuda.synchronize()
    train = invented_table()
    names = ['GaussianCopula', 'CTGAN', 'TVAE'] if args.kind == 'sdv' else ['TabDDPM']
    for name in names:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if args.kind == 'sdv':
                from .generators import fit_sample_sdv
                def fit():
                    return fit_sample_sdv(train, name, 0, 3, device='cpu' if name == 'GaussianCopula' else args.device)[0]
            else:
                from .synthcity_generator import fit_sample
                def fit():
                    return fit_sample(train, 0, args.device, iterations=1, sample_rows=64)[0]
            first, second = fit(), fit()
        if fingerprint(first) != fingerprint(second):
            raise RuntimeError(name + ': independent same-seed fit differed')
        print(name + ': invented-data independent repeats identical (' + args.device + ')', flush=True)


if __name__ == '__main__':
    main()
