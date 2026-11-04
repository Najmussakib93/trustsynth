"""Observe the long-run supervisor without mixing optional-pilot status messages."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv-synthcity' / 'Lib' / 'site-packages'))
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    if 'revision.pipeline' not in process.cmdline():
        raise RuntimeError('The selected process is not the revision supervisor')
    target = ROOT / 'results' / 'revision' / 'MAIN_RUN_STATUS.json'
    last = {}
    print('Watching the revision supervisor; status is aggregate process metadata.', flush=True)
    while process.is_running():
        try:
            candidates = [p for p in process.children() if '-m' in p.cmdline()]
            if candidates:
                child = candidates[0]
                command = child.cmdline()
                position = command.index('-m')
                last = {'state': 'running', 'module': command[position + 1], 'arguments': command[position + 2:],
                    'supervisor_pid': process.pid, 'worker_pid': child.pid,
                    'worker_started_utc': datetime.fromtimestamp(child.create_time(), timezone.utc).isoformat(),
                    'observed_utc': datetime.now(timezone.utc).isoformat()}
                target.write_text(json.dumps(last, indent=2) + '\n', encoding='utf-8')
            time.sleep(5)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            time.sleep(1)
    shared = ROOT / 'results' / 'revision' / 'RUN_STATUS.json'
    status = json.loads(shared.read_text()) if shared.exists() else {}
    if status.get('state') in ['awaiting_structural_constraint_approval', 'stage_complete', 'failed', 'worker_timeout']:
        target.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    else:
        target.write_text(json.dumps({'state': 'supervisor_exited', 'last_observed_step': last}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
