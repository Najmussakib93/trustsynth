"""Validate optional generator cells and estimate their cost before full runs."""
import json
import subprocess
from .pipeline import run, interpreter
from .safe_reporting import PUBLIC, write_json


def main():
    run('synthcity_generator', ['--seed', 0, '--device', 'cpu', '--pilot'], 0, interpreter('synthcity'), timeout=1800)
    receipt = json.loads((PUBLIC / 'generators' / 'TabDDPM_seed0_pilot.json').read_text())
    timings = receipt['timings']
    path = PUBLIC / 'runtime_estimate.json'
    estimate = json.loads(path.read_text())
    estimate['TabDDPM'] = {'pilot': timings, 'independent_repeat_identical': receipt['same_seed_independent_fit_identical'],
        'pilot_sample_rows': receipt['settings']['sample_rows'],
        'sampling_extrapolation_multiplier': 16000 / receipt['settings']['sample_rows'],
        'estimated_full_cell_seconds': timings['preprocessing_seconds'] + 1000 * timings['epoch_p90_seconds'] +
            timings['sample_seconds'] * 16000 / receipt['settings']['sample_rows']}
    write_json(path, estimate)
    print('DDPM estimated hours for ten seeds plus an independent seed-0 repeat: ' +
          format(11 * estimate['TabDDPM']['estimated_full_cell_seconds'] / 3600, '.2f'), flush=True)
    try:
        run('dp_generator', ['--seed', 0, '--epsilon', 1], 0, interpreter('dp'), timeout=1200)
    except subprocess.TimeoutExpired:
        write_json(PUBLIC / 'dp_attempt.json', {'state': 'budget_exhausted', 'phase': 'first_cell_fit', 'limit_seconds': 1200})
        print('AIM pilot exceeded the bounded attempt; further DP runs will be skipped.', flush=True)


if __name__ == '__main__':
    main()
