"""Create isolated Linux runtimes; package operations never access clinical data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--sdv-torch-index', help='Override the PyTorch 2.8 wheel index for your driver')
    parser.add_argument('--dp-only', action='store_true', help='Repair/create only .venv-dp; reuse the other environments')
    args = parser.parse_args()
    if os.name == 'nt' or sys.version_info[:2] != (3, 10):
        parser.error('Use Linux with Python 3.10. This setup is separate from the existing Windows runtimes.')
    os.umask(0o077)
    logs = ROOT / 'results/private/revision/logs'
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / ('server-setup-' + time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '.log')
    env = dict(os.environ, PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='4')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    base_index = args.sdv_torch_index or ('https://download.pytorch.org/whl/cu126'
        if args.device == 'cuda' else 'https://download.pytorch.org/whl/cpu')
    ddpm_index = 'https://download.pytorch.org/whl/' + ('cu121' if args.device == 'cuda' else 'cpu')
    with log_path.open('w', encoding='utf-8') as log:
        def run(command, timeout=1800):
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)

        def create(name):
            path = ROOT / ('.venv-' + name)
            if path.exists():
                raise RuntimeError(str(path) + ' already exists. Reuse a completed setup or choose a fresh clone; no environment was replaced.')
            venv.EnvBuilder(with_pip=True).create(path)
            python = path / 'bin/python'
            run([python, '-m', 'pip', 'install', '--upgrade', 'pip==25.3', 'wheel==0.45.1', 'setuptools==80.9.0'])
            return python

        def overlay(python, kind):
            run([python, '-m', 'pip', 'install', '--no-deps', '-r', ROOT / 'server/requirements' / (kind + '.txt')])

        def prepare_dp(reuse=False):
            path = ROOT / '.venv-dp'
            dp = path / 'bin/python' if reuse and path.exists() else create('dp')
            run([dp, '-m', 'revision.setup_dp'], timeout=1200)

        if args.dp_only:
            try:
                prepare_dp(reuse=True)
            except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print('DP-only repair stopped: ' + type(exc).__name__ + '. Inspect ' + str(log_path.relative_to(ROOT)), file=sys.stderr)
                return 1
            print('DP-only repair and checks passed. Existing experiment/attempt records were preserved.', flush=True)
            return 0

        print('Creating runtimes. Setup log: ' + str(log_path.relative_to(ROOT)), flush=True)
        try:
            sdv = create('sdv')
            run([sdv, '-m', 'pip', 'install', 'numpy==1.25.2', 'torch==2.8.0', '--extra-index-url', base_index])
            overlay(sdv, 'base')
            run([sdv, '-m', 'revision.server_smoke', '--kind', 'sdv', '--device', args.device])
            run([sdv, '-m', 'unittest', 'revision.test_audit', 'revision.test_preprocessing',
                 'revision.test_metrics', 'revision.test_runtime_paths', 'revision.test_report_progress',
                 'revision.test_source_files', 'revision.test_constraint_proposal',
                 'revision.test_finalization', 'revision.test_aim'])
            print('SDV, evaluation and metric checks passed.', flush=True)

            synthcity = create('synthcity')
            run([synthcity, '-m', 'pip', 'install', 'numpy==1.26.4', 'torch==2.2.2',
                 'torchvision==0.17.2', '--extra-index-url', ddpm_index])
            overlay(synthcity, 'synthcity')
            run([synthcity, '-m', 'revision.patch_synthcity'])
            run([synthcity, '-m', 'revision.server_smoke', '--kind', 'synthcity', '--device', args.device])
            print('Synthcity DDPM checks passed.', flush=True)

            anonymeter = create('anonymeter')
            overlay(anonymeter, 'anonymeter')
            # The Linux XGBoost wheel declares NCCL even for the CPU attack worker.
            run([anonymeter, '-m', 'pip', 'install', 'nvidia-nccl-cu12==2.19.3'])
            run([anonymeter, '-m', 'revision.inspect_environment', 'anonymeter'])
            run([anonymeter, '-m', 'revision.smoke_anonymeter'])
            print('Anonymeter checks passed.', flush=True)

            try:
                prepare_dp()
                dp_state = 'installed_and_verified'
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                dp_state = 'unavailable'
                output = ROOT / 'results/revision'
                output.mkdir(parents=True, exist_ok=True)
                (output / 'dp_attempt.json').write_text(json.dumps({'state': 'unavailable',
                    'phase': 'server_dependency_setup', 'exception': type(exc).__name__,
                    'private_log': str(log_path.relative_to(ROOT))}, indent=2) + '\n')
                print('Optional AIM setup stopped; details are in the private setup log.', flush=True)
            (ROOT / '.venv-sdv/SERVER_READY.json').write_text(json.dumps({
                'device': args.device, 'python': sys.version, 'dp': dp_state,
                'sdv_torch_index': base_index, 'ddpm_torch_index': ddpm_index}, indent=2) + '\n')
        except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print('Setup stopped: ' + type(exc).__name__ + '. Inspect ' + str(log_path.relative_to(ROOT)), file=sys.stderr)
            return 1
    print('Setup checks passed. Run: bash server/run.sh ' + args.device, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
