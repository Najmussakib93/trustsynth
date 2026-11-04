"""Run approved stages with all subprocess output retained in ignored private logs."""
import argparse
import json
import os
import subprocess
import sys
import time

from .runtime_paths import ROOT, require_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    os.umask(0o077)
    ready = ROOT / '.venv-sdv/SERVER_READY.json'
    if not ready.is_file() or json.loads(ready.read_text())['device'] != args.device:
        parser.error('Run server/setup.py for the selected device before launching the job.')
    sdv, synthcity = require_environment('sdv'), require_environment('synthcity')
    require_environment('anonymeter')
    require_environment('dp')
    directory = ROOT / 'results/private/revision/logs'
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / ('server-run-' + time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '.log')
    env = dict(os.environ, PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='4')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    commands = [[sdv, '01_extract_cohort.py'], [sdv, '01_extract_cohort.py'],
                [sdv, '-m', 'revision.benchmark', '--device', args.device],
                [synthcity, '-m', 'revision.synthcity_generator', '--pilot', '--device', args.device],
                [sdv, '02_run_evaluation.py', '--stage', 'through-constraints', '--device', args.device]]
    print('Starting approved stages. Private log: ' + str(log_path.relative_to(ROOT)), flush=True)
    print('Aggregate progress: results/revision/RUN_STATUS.json', flush=True)
    with log_path.open('w', encoding='utf-8') as stream:
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print('Job stopped; inspect the private run log and results/revision/RUN_STATUS.json.', file=sys.stderr)
                return result.returncode
    print('Stopped at the structural-constraint approval gate. See results/revision/REVISION_REPORT.md.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
