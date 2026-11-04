"""Sequential, resumable revision runner. Stops at the author's Task 8 approval gate."""
import argparse
import json
import os
import subprocess
import sys
import time
import atexit
from datetime import datetime, timezone

from .protocol import PROTOCOL
from .safe_reporting import ROOT, PUBLIC, private_path, write_json
from .runtime_paths import require_environment


def interpreter(environment):
    return require_environment(environment)


def run(module, arguments, seed, executable=None, timeout=None):
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(seed), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_NUM_THREADS=str(PROTOCOL['torch_threads']), LOKY_MAX_CPU_COUNT='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [executable or sys.executable, '-m', 'revision.' + module] + list(map(str, arguments))
    write_json(PUBLIC / 'RUN_STATUS.json', {'state': 'running', 'module': module, 'arguments': arguments,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'supervisor_pid': os.getpid()})
    log = private_path('logs/' + module + '-' + str(seed) + '-' + str(time.time_ns()) + '.log')
    print('Running ' + module + ' ' + ' '.join(map(str, arguments)), flush=True)
    try:
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        write_json(PUBLIC / 'RUN_STATUS.json', {'state': 'worker_timeout', 'module': module,
            'arguments': arguments, 'limit_seconds': timeout, 'private_log': str(log.relative_to(ROOT))})
        raise
    if result.returncode:
        write_json(PUBLIC / 'RUN_STATUS.json', {'state': 'failed', 'module': module, 'arguments': arguments,
                   'exit_code': result.returncode, 'private_log': str(log.relative_to(ROOT))})
        raise RuntimeError('Worker failed; error log is private')
    print('Completed ' + module + ' ' + ' '.join(map(str, arguments)), flush=True)
    from .report_progress import main as refresh_report
    refresh_report()


def available_generators():
    return sorted({p.name.split('_seed')[0] for p in (PUBLIC / 'generators').glob('*_seed*.json') if '_pilot' not in p.name})


def main():
    if os.name == 'nt':
        import ctypes
        # Keep long authorized computations active; do not change display or
        # permanent power settings. Windows releases this request on exit.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        atexit.register(lambda: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000))
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['gaussian', 'train', 'quality', 'privacy', 'anonymeter', 'through-constraints'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    full = args.stage == 'through-constraints'
    if args.stage in ['gaussian', 'train'] or full:
        names = ['GaussianCopula'] if args.stage == 'gaussian' else ['GaussianCopula', 'CTGAN', 'TVAE']
        for name in names:
            for seed in PROTOCOL['seeds']:
                device = 'cpu' if name == 'GaussianCopula' else args.device
                command = ['--generator', name, '--seed', seed, '--epochs', 300, '--device', device]
                if seed == 0:
                    # Full-length independent repeats, in addition to pilot checks.
                    command += ['--repeat']
                run('generators', command, seed, sys.executable if device == 'cpu' else interpreter('sdv'))
        if args.stage != 'gaussian':
            for name, epochs in [('CTGAN_no_log', 300), ('TVAE_900', 900)]:
                for seed in PROTOCOL['mechanism_seeds']:
                    run('generators', ['--generator', name, '--seed', seed, '--epochs', epochs, '--device', args.device], seed,
                        sys.executable if args.device == 'cpu' else interpreter('sdv'))
            run('synthcity_generator', ['--all-seeds', '--device', args.device], 0, interpreter('synthcity'))
            try:
                run('dp_generator', ['--all-seeds'], 0, interpreter('dp'), timeout=PROTOCOL['dp_attempt_limit_seconds'])
            except subprocess.TimeoutExpired:
                write_json(PUBLIC / 'dp_attempt.json', {'state': 'budget_exhausted',
                    'phase': 'full_run', 'limit_seconds': PROTOCOL['dp_attempt_limit_seconds']})
                print('Bounded AIM run timed out; preserving completed cells and continuing.', flush=True)
    if args.stage == 'quality' or full:
        for name in available_generators():
            for seed in PROTOCOL['seeds']:
                if not (PUBLIC / 'generators' / (name + '_seed' + str(seed) + '.json')).exists():
                    continue
                run('evaluate', ['--generator', name, '--seed', seed, '--phase', 'quality'], seed, sys.executable)
    if args.stage == 'privacy' or full:
        for name in available_generators():
            if name in ['CTGAN_no_log', 'TVAE_900']:
                continue
            for seed in PROTOCOL['seeds']:
                if (PUBLIC / 'generators' / (name + '_seed' + str(seed) + '.json')).exists():
                    run('evaluate', ['--generator', name, '--seed', seed, '--phase', 'privacy'], seed, interpreter('anonymeter'))
    if args.stage == 'anonymeter' or full:
        run('anonymeter_worker', ['--all-cells'], 0, interpreter('anonymeter'))
    if full:
        run('constraint_proposal', [], 0, sys.executable)
    write_json(PUBLIC / 'RUN_STATUS.json', {'state': 'awaiting_structural_constraint_approval' if full else 'stage_complete',
        'stage': args.stage, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    from .report_progress import main as refresh_report
    refresh_report()


if __name__ == '__main__':
    main()
