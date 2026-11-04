"""Measure epoch cost before launching the full experiment, including repeated pilots."""
import argparse
import json
from .pipeline import run, interpreter
from .safe_reporting import PUBLIC, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    estimates = {}
    for name in ['CTGAN', 'TVAE']:
        run('generators', ['--generator', name, '--seed', 0, '--epochs', 3, '--device', args.device, '--repeat', '--pilot'],
            0, interpreter('sdv') if args.device == 'cuda' else None)
        result = json.loads((PUBLIC / 'generators' / (name + '_seed0_pilot_' + args.device + '_e3.json')).read_text())
        timing = result['timings']
        cost = timing['preprocessing_seconds'] + 300 * timing['epoch_p90_seconds']
        estimates[name] = {'pilot': timing, 'independent_repeat_identical': result['same_seed_independent_fit_identical'],
                          'estimated_300_epoch_fit_seconds': cost}
    ctgan, tvae = estimates['CTGAN'], estimates['TVAE']
    estimates['sdv_total_seconds'] = (14 * ctgan['estimated_300_epoch_fit_seconds'] +
        11 * tvae['estimated_300_epoch_fit_seconds'] +
        3 * (tvae['pilot']['preprocessing_seconds'] + 900 * tvae['pilot']['epoch_p90_seconds']))
    estimates['scope'] = 'SDV main 10 seeds, independent full seed0 repeats, CTGAN mechanism 3 seeds, TVAE mechanism 3 seeds; excludes DDPM, AIM and evaluation'
    estimates['device'] = args.device
    write_json(PUBLIC / 'runtime_estimate.json', estimates)
    print('Estimated SDV training hours, excluding other generators and evaluation: ' + format(estimates['sdv_total_seconds'] / 3600, '.2f'), flush=True)


if __name__ == '__main__':
    main()
