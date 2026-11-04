"""Software smoke check using invented independent tables, never eICU records."""
import contextlib
import io
import numpy as np
import pandas as pd
from .anonymeter_worker import attack
from .protocol import FEATURES, CONTINUOUS_FEATURES
from .safe_reporting import PUBLIC, write_json


def table(n, rng):
    frame = pd.DataFrame({c: rng.normal(50, 10, n) if c in CONTINUOUS_FEATURES else rng.choice(['A', 'B', 'C'], n) for c in FEATURES})
    frame['hospital_mortality'] = rng.binomial(1, .1, n)
    return frame


def main():
    rng = np.random.RandomState(17)
    train, test, synth = table(2400, rng), table(600, rng), table(2400, rng)
    result = {}
    for kind in ['singling_out', 'linkability', 'inference_ethnicity_grouped', 'inference_hospital_mortality']:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            first, second = attack(train, test, synth, kind, 0), attack(train, test, synth, kind, 0)
        equal = all(first[k] == second[k] for k in ['risk', 'ci_low', 'ci_high'])
        result[kind] = {'repeat_identical': equal, 'software_only_wall_seconds': first['wall_seconds']}
        if not equal:
            raise RuntimeError('Invented-data Anonymeter repeat failed')
        print('Invented-data ' + kind + ': exact repeated risk/CI match', flush=True)
    write_json(PUBLIC / 'anonymeter_software_smoke.json', result)


if __name__ == '__main__':
    main()
