"""A bounded, logged dependency attempt for the optional AIM generator."""
import json
import subprocess
import time
from .safe_reporting import ROOT, PUBLIC, private_path, write_json
from .runtime_paths import require_environment


def main():
    started = time.perf_counter()
    executable = require_environment('dp')
    log = private_path('logs/aim_setup.log')
    commands = [[str(executable), '-m', 'pip', 'install', 'smartnoise-synth', 'numpy==1.25.2',
                 'pandas==2.2.2', 'scikit-learn==1.3.2', 'scipy==1.13.1'],
                [str(executable), '-m', 'pip', 'install',
                 'git+https://github.com/ryan112358/private-pgm.git@01f02f17eba440f4e76c1d06fa5ee9eed0bd2bca']]
    try:
        with log.open('w', encoding='utf-8') as stream:
            for command in commands:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
                if result.returncode:
                    raise RuntimeError('Dependency resolution or installation failed')
        write_json(PUBLIC / 'dp_setup.json', {'state': 'installed_pending_smoke', 'wall_seconds': time.perf_counter() - started})
        print('AIM dependency install finished; smoke check still required.', flush=True)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        # Setup has not accessed private data; these package errors are safe to quote exactly.
        errors = [line for line in log.read_text(encoding='utf-8', errors='replace').splitlines() if line.startswith('ERROR:')]
        write_json(PUBLIC / 'dp_setup.json', {'state': 'unavailable', 'exception': type(exc).__name__,
            'package_errors': errors, 'wall_seconds': time.perf_counter() - started, 'setup_limit_seconds': 600})
        print('AIM setup stopped within its dependency-attempt budget.', flush=True)


if __name__ == '__main__':
    main()
