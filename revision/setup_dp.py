"""A bounded, logged dependency attempt for the optional AIM generator."""
import subprocess
import json
import time
from .safe_reporting import ROOT, PUBLIC, private_path, write_json
from .runtime_paths import require_environment


def main():
    started = time.perf_counter()
    executable = require_environment('dp')
    log = private_path('logs/aim_setup.log')
    commands = [([str(executable), '-m', 'pip', 'install', '-r', str(ROOT / 'server/requirements/dp.txt')], 600),
                ([str(executable), '-m', 'revision.patch_aim'], 120),
                ([str(executable), '-m', 'revision.smoke_dp'], 300)]
    try:
        with log.open('w', encoding='utf-8') as stream:
            for command, limit in commands:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=limit)
                if result.returncode:
                    raise RuntimeError('Dependency resolution or installation failed')
        smoke = json.loads((PUBLIC / 'aim_software_smoke.json').read_text())
        write_json(PUBLIC / 'dp_setup.json', {'state': 'installed_and_verified',
            'wall_seconds': time.perf_counter() - started, 'compatibility': smoke['compatibility'],
            'checks': ['pip check', 'cdp_rho(1.0, 1e-6)', 'generated-data fit_sample']})
        print('AIM dependencies, compatibility patch and generated-data checks passed.', flush=True)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        # Setup has not accessed private data; these package errors are safe to quote exactly.
        errors = [line for line in log.read_text(encoding='utf-8', errors='replace').splitlines() if line.startswith('ERROR:')]
        write_json(PUBLIC / 'dp_setup.json', {'state': 'unavailable', 'exception': type(exc).__name__,
            'package_errors': errors, 'wall_seconds': time.perf_counter() - started, 'setup_limit_seconds': 600})
        print('AIM setup stopped within its dependency-attempt budget.', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
