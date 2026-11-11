"""Verify AIM calibration and the production fit/sample path on invented data."""
import contextlib
import io
import math
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .dp_generator import BOUNDS, CATEGORIES, SETTINGS, fit_sample
from .patch_aim import provenance
from .protocol import FEATURES
from .safe_reporting import PUBLIC, write_json


def generated_table(n=128):
    rng = np.random.RandomState(42)
    table = pd.DataFrame({column: rng.uniform(*BOUNDS[column], n) if column in BOUNDS
                          else rng.choice(CATEGORIES[column], n) for column in FEATURES})
    for column in BOUNDS:
        table.loc[::17, column] = np.nan
    return table[FEATURES]


def main():
    started = time.perf_counter()
    compatibility = provenance()
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, timeout=120)
    from snsynth.utils import cdp_rho, gaussian_noise, laplace_noise
    rho = float(cdp_rho(1.0, 1e-6))
    if not math.isfinite(rho) or rho <= 0:
        raise RuntimeError('AIM calibration returned an invalid rho')
    print('AIM calibration passed: cdp_rho(1.0, 1e-6)=' + format(rho, '.12g'), flush=True)
    if not all(math.isfinite(value) for value in [gaussian_noise(1.), laplace_noise(1.)]):
        raise RuntimeError('SmartNoise scalar samplers returned non-finite values')
    np.random.seed(0)
    noise_first = gaussian_noise(1.)
    np.random.seed(0)
    noise_second = gaussian_noise(1.)
    # Use all 17 public domains and the unmodified production AIM settings.
    train = generated_table()
    print('Checking production AIM fit/sample on invented data only.', flush=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        synthetic, timing = fit_sample(train, epsilon=1.0, seed=0)
    if list(synthetic) != FEATURES or len(synthetic) != len(train):
        raise RuntimeError('Generated-data AIM fit/sample returned an invalid schema or size')
    for column in FEATURES:
        if column in BOUNDS:
            if not synthetic[column].dropna().between(*BOUNDS[column]).all():
                raise RuntimeError('AIM decoded values outside a public numeric domain')
        elif not synthetic[column].isin(CATEGORIES[column]).all():
            raise RuntimeError('AIM decoded values outside a public categorical domain')
    receipt = {'state': 'passed', 'input': 'invented_data_only', 'rows': len(train),
               'pip_check': 'passed', 'cdp_rho_1_1e_6': rho,
               'scalar_noise_checks': 'passed', 'fit_sample': 'passed',
               'same_numpy_seed_noise_equal': bool(noise_first == noise_second),
               'settings': SETTINGS, 'compatibility': compatibility, 'timing': timing,
               'wall_seconds': time.perf_counter() - started}
    from .environment_lock import save_lock
    save_lock('dp-aim-compatible')
    write_json(PUBLIC / 'aim_software_smoke.json', receipt)
    print('AIM pip check, calibration, scalar noise and generated-data fit/sample passed; rho=' + format(rho, '.12g'), flush=True)


if __name__ == '__main__':
    main()
