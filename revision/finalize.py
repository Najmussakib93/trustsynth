"""Resume after the recorded structural approval; never fit a generator again."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
import zipfile

import pandas as pd

from .integrity import APPROVAL, validate_approval
from .protocol import COHORT, PROTOCOL
from .safe_reporting import ROOT, PUBLIC, PRIVATE, count, private_path, write_json

MAIN = ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']
MECHANISMS = {'CTGAN_no_log', 'TVAE_900'}


def sanitize_software_line(line):
    line = re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@', r'\1<credentials-redacted>@', line)
    line = re.sub(r'(?i)(password|token|api[_-]?key)([=: ]+)[^\s]+', r'\1\2<redacted>', line)
    line = re.sub(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}', '<redacted>', line)
    return line.strip()[:1000]


def dp_failure_details():
    path = PUBLIC / 'dp_attempt.json'
    details = json.loads(path.read_text()) if path.exists() else {'state': 'not_recorded'}
    # This exact setup phase ran package operations/invented smoke checks only.
    # Never promote text from fit/sample or data-processing logs to public output.
    if details.get('phase') == 'server_dependency_setup':
        relative = Path(details.get('private_log', ''))
        candidate = (ROOT / relative).resolve()
        logs = (PRIVATE / 'logs').resolve()
        if logs in candidate.parents and candidate.name.startswith('server-setup-') and candidate.suffix == '.log' and candidate.is_file():
            lines = candidate.read_text(encoding='utf-8', errors='replace').splitlines()
            details['software_errors'] = [sanitize_software_line(line) for line in lines
                if re.search(r'(?i)(?:^|\s)(?:ERROR:|error:|fatal:)', line)]
            details['log_scope_verified'] = 'dependency setup and invented-data checks; no eICU records loaded'
    write_json(PUBLIC / 'DP_FAILURE_DETAILS.json', details)
    return details


def input_audit():
    if APPROVAL['status'] != 'approved_by_author':
        raise RuntimeError('The recorded Task 8(c) approval is required')
    validate_approval()
    manifest = json.loads((PUBLIC / 'preprocessing/manifest.json').read_text())
    if manifest.get('repeat_identical') is not True:
        raise RuntimeError('The cohort did not pass its independent extraction check')
    if hashlib.sha256(COHORT.read_bytes()).hexdigest() != manifest['cohort_sha256']:
        raise RuntimeError('The local cohort no longer matches its preprocessing receipt')
    cells, main_keys, attack_warnings = [], set(), []
    for path in sorted((PUBLIC / 'generators').glob('*_seed*.json')):
        if '_pilot' in path.name:
            continue
        metadata = json.loads(path.read_text())
        name, seed = metadata['settings']['generator'], metadata['settings']['seed']
        if metadata['settings']['protocol'] != PROTOCOL:
            raise RuntimeError('A generator receipt uses a different protocol')
        if seed == 0 and name in MAIN and metadata.get('same_seed_independent_fit_identical') is not True:
            raise RuntimeError('A full-length main generator seed-0 repeat is unverified')
        location = (ROOT / 'data/revision' / metadata['private_artifact']).resolve()
        if (ROOT / 'data/revision').resolve() not in location.parents or not location.is_file():
            raise RuntimeError('A required private synthetic cache is absent or outside its permitted directory')
        directory = PUBLIC / 'cells' / (name + '-' + str(seed))
        required = ['quality_status.json', 'quality_metrics.csv']
        if name not in MECHANISMS:
            required += ['privacy_status.json', 'privacy_metrics.csv']
        for filename in required:
            if not (directory / filename).is_file():
                raise RuntimeError('A prerequisite aggregate evaluation is missing: ' + name + '-' + str(seed) + '/' + filename)
        if name in MAIN:
            main_keys.add((name, seed))
            kinds = ['singling_out'] + (['linkability', 'inference_ethnicity_grouped', 'inference_hospital_mortality'] if seed < 5 else [])
            for kind in kinds:
                attack_path = PUBLIC / 'anonymeter' / (name + '-' + str(seed) + '-' + kind + '.json')
                if not attack_path.exists():
                    raise RuntimeError('A required Anonymeter aggregate receipt is missing')
                receipt = json.loads(attack_path.read_text())
                if seed == 0 and receipt.get('independent_repeat_identical') is not True:
                    raise RuntimeError('An Anonymeter seeded-repeat check is unverified')
                if receipt.get('query_generation_complete') is False:
                    attack_warnings.append({'generator': name, 'seed': seed, 'attack': kind, 'status': 'query_generation_underfilled'})
        cells.append({'generator': name, 'seed': seed, 'synthetic_sha256': metadata['synthetic_sha256'],
                      'device': metadata['settings'].get('device'), 'repeat_identical': metadata.get('same_seed_independent_fit_identical')})
    if main_keys != {(name, seed) for name in MAIN for seed in PROTOCOL['seeds']}:
        raise RuntimeError('All four main generators require ten completed seed cells before finalization')
    expected_mechanisms = {(name, seed) for name in MECHANISMS for seed in PROTOCOL['mechanism_seeds']}
    if not expected_mechanisms.issubset({(row['generator'], row['seed']) for row in cells}):
        raise RuntimeError('The prescribed three-seed mechanism fits are missing')
    # Identifiers are processed locally only to emit a suppressed aggregate count.
    identifiers = pd.read_pickle(ROOT / 'data/revision/selected_ids.pkl')
    if len(identifiers) != manifest['flow']['cohort_sample']:
        raise RuntimeError('Cohort identifier metadata and cohort size disagree')
    write_json(PUBLIC / 'cohort_context.json', {'distinct_hospitals': count(identifiers.hospitalid.nunique()),
        'cohort_sha256': manifest['cohort_sha256'], 'scope': 'fixed-cohort aggregate only'})
    audit = {'main_generators': MAIN, 'main_seed_cells': len(main_keys), 'cells': cells,
        'anonymeter_warnings': attack_warnings, 'approval': APPROVAL,
        'host': {'platform': platform.platform(), 'python': sys.version},
        'scope': 'Existing generator/evaluation receipts; finalization never trains generators',
        'dp': dp_failure_details()}
    write_json(PUBLIC / 'FINALIZATION_INPUT_AUDIT.json', audit)
    write_json(PUBLIC / 'structural_approval.json', APPROVAL)
    return cells


def plan(cells):
    steps = [('integrity', ['--benchmark'])]
    steps += [('integrity', ['--generator', row['generator'], '--seed', str(row['seed'])]) for row in cells]
    steps += [('final_analysis', [])]
    return steps


def export_review():
    target = PUBLIC / 'trustsynth_aggregate_review.zip'
    allowed_top = {
        'REVISION_REPORT.md', 'supplementary.md', 'ALGORITHM1.md', 'per_seed_results.csv',
        'dimension_scores_per_seed.csv', 'metric_summary.csv', 'paired_comparisons.csv',
        'score_components_per_seed.csv',
        'representation_seed_counts.csv', 'mechanism_comparisons.csv', 'weight_winner_shares.csv',
        'score_eligibility.csv', 'normalization_sensitivity.csv', 'threshold_sensitivity.csv',
        'old_vs_new_manuscript.csv', 'original_findings.csv', 'normalization.json',
        'table_I_dimensions.csv', 'table_I_raw_metrics.csv', 'table_II_representation.csv',
        'table_I_dimensions.tex', 'table_I_raw_metrics.tex', 'table_II_representation.tex',
        'analysis_manifest.json', 'figure_manifest.json', 'FINALIZATION_INPUT_AUDIT.json',
        'DP_FAILURE_DETAILS.json', 'dp_attempt.json', 'structural_approval.json',
        'finalization_runtime_estimate.json', 'cohort_context.json', 'protocol.json'}
    candidates = [PUBLIC / name for name in sorted(allowed_top)]
    candidates += [PUBLIC / 'preprocessing/manifest.json']
    candidates += sorted((PUBLIC / 'figures').glob('*.png'))  # TIFF originals remain on server.
    # Only exact aggregate schemas emitted by the evaluator are included.
    cell_names = {'subgroup_utility.csv', 'representation.csv', 'parity.csv', 'near_duplicates.csv',
                  'population_consistency.csv', 'integrity_constraints.csv'}
    candidates += [path for path in (PUBLIC / 'cells').glob('*/*.csv') if path.name in cell_names]
    candidates += sorted((PUBLIC / 'anonymeter').glob('*.json'))
    candidates += sorted((PUBLIC / 'environments').glob('*_imported_versions.json'))
    candidates += sorted((PUBLIC / 'integrity').glob('sdv_inherited_rules_seed*.csv'))
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in candidates:
            if path.is_file() and PUBLIC.resolve() in path.resolve().parents:
                archive.write(path, arcname=str(Path('revision') / path.relative_to(PUBLIC)))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run approved post-training finalization with private records kept local')
    args = parser.parse_args()
    if not args.run:
        parser.error('Pass --run after pulling the approved finalization code; no setup or training rerun is needed')
    os.umask(0o077)
    log_path = private_path('logs/finalization-' + time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '.log')
    lock_path = private_path('finalization.lock')
    env = dict(os.environ, PYTHONHASHSEED='0', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_NUM_THREADS=str(PROTOCOL['model_jobs']), CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    print('Approved finalization only; no generator retraining. Private log: ' + str(log_path.relative_to(ROOT)), flush=True)
    started = time.perf_counter()
    try:
        with lock_path.open('a') as lock:
            if os.name != 'nt':
                import fcntl
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print('Another finalization job already holds the project lock.', flush=True)
                    return 1
            with log_path.open('w', encoding='utf-8') as stream:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    cells = input_audit()
                previous = PUBLIC / 'REVISION_REPORT.md'
                backup = PUBLIC / 'INTERIM_REPORT_BEFORE_APPROVED_CHECKS.md'
                if previous.exists() and not backup.exists():
                    backup.write_bytes(previous.read_bytes())
                for module, arguments in plan(cells):
                    write_json(PUBLIC / 'RUN_STATUS.json', {'state': 'finalizing', 'module': module,
                        'arguments': arguments, 'supervisor_pid': os.getpid(),
                        'started_utc': datetime.now(timezone.utc).isoformat(),
                        'private_log': str(log_path.relative_to(ROOT))})
                    result = subprocess.run([sys.executable, '-m', 'revision.' + module, *arguments],
                        cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError('Finalization step failed: ' + module)
            archive = export_review()
            status = {'state': 'complete', 'scope': 'approved structural checks and aggregate publication analysis',
                'wall_seconds': time.perf_counter() - started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'review_archive': str(archive.relative_to(ROOT)), 'generator_retraining': False}
            write_json(PUBLIC / 'FINALIZATION_STATUS.json', status)
            write_json(PUBLIC / 'RUN_STATUS.json', dict(status, state='finalized'))
            print('Finalization complete. Review results/revision/REVISION_REPORT.md', flush=True)
            print('Aggregate-only review archive: ' + str(archive.relative_to(ROOT)), flush=True)
    except Exception as exc:
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(traceback.format_exc())
        failed = {'state': 'failed', 'exception': type(exc).__name__,
            'private_log': str(log_path.relative_to(ROOT)), 'generator_retraining': False}
        write_json(PUBLIC / 'FINALIZATION_STATUS.json', failed)
        write_json(PUBLIC / 'RUN_STATUS.json', failed)
        print('Finalization stopped: ' + type(exc).__name__ + '; inspect the private finalization log with revision.debug_failure.', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
