"""AIM compatibility and scoped retry regressions; no credentialed data."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from . import pipeline, dp_generator
from .patch_aim import corrected_source, FUNCTIONS
from .protocol import PROTOCOL
import pandas as pd


def utility_source(domain='dp.atom_domain(T=float)'):
    return '\n'.join('def ' + name + '():\n    return ' + domain + '\n' for name in FUNCTIONS)


class AimCompatibilityTests(unittest.TestCase):
    def test_partial_aim_enters_aggregate_analysis_without_inventing_missing_seeds(self):
        from .testing_aggregate_fixture import create
        from .final_analysis import main
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary)
            create(public)
            source = public / 'cells/CTGAN-0'
            target = public / 'cells/AIM_eps1-0'
            target.mkdir()
            for path in source.glob('*.csv'):
                frame = pd.read_csv(path)
                frame['generator'] = 'AIM_eps1'
                frame.to_csv(target / path.name, index=False)
            (target / 'integrity_status.json').write_text('{"test_fixture":true}')
            (public / 'dp_attempt.json').write_text('{"state":"budget_exhausted"}')
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(public, make_figures=False)
            aim = result['summary'].query("generator == 'AIM_eps1'")
            self.assertFalse(aim.empty)
            self.assertTrue(aim.n_seeds.eq(1).all())
            eligibility = pd.read_csv(public / 'score_eligibility.csv').set_index('generator')
            self.assertFalse(eligibility.loc['AIM_eps1', 'eligible'])
            report = (public / 'REVISION_REPORT.md').read_text(encoding='utf-8')
            self.assertIn('time-limited partial attempt', report)
            self.assertNotIn('This failure must be disclosed', report)

    def test_all_cell_attacks_skip_unfinished_optional_aim_seeds(self):
        from . import anonymeter_worker
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary)
            (public / 'generators').mkdir()
            (public / 'generators/AIM_eps1_seed0.json').write_text('{}')
            with patch.object(anonymeter_worker, 'PUBLIC', public), patch.object(anonymeter_worker, 'run') as run, patch.object(sys, 'argv', ['anonymeter_worker', '--all-cells']):
                anonymeter_worker.main()
            self.assertEqual(run.call_count, 4)
            self.assertTrue(all(call.args[:2] == ('AIM_eps1', 0) for call in run.call_args_list))

    def test_opendp_repeat_is_diagnostic_and_compatibility_is_in_receipt(self):
        from . import environment_lock
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, public = root / 'data', root / 'public'
            real = pd.DataFrame({'gender': ['FEMALE', 'MALE'] * 20})
            other = real.iloc[::-1].reset_index(drop=True)
            details = {'software': {'opacus': '0.14.0', 'opendp': '0.14.2'}, 'change': 'test-fix'}
            samples = iter([(real, {'fit_seconds': 1.}), (other, {'fit_seconds': 1.})])
            def fake_fit(*args):
                return next(samples)
            def private(name):
                target = root / 'private' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                return target
            with patch.object(dp_generator, 'DATA', data), patch.object(dp_generator, 'PUBLIC', public), patch.object(dp_generator, 'private_path', private), patch.object(dp_generator, 'split_cohort', return_value=(real, real)), patch('revision.patch_aim.provenance', return_value=details), patch.object(dp_generator, 'fit_sample', fake_fit), patch.object(environment_lock, 'save_lock'):
                with contextlib.redirect_stdout(io.StringIO()):
                    dp_generator.run(0, 1.)
            receipt = json.loads((public / 'generators/AIM_eps1_seed0.json').read_text())
            self.assertFalse(receipt['same_seed_independent_fit_identical'])
            self.assertEqual(receipt['settings']['compatibility'], details)
            self.assertEqual(receipt['settings']['software']['opacus'], '0.14.0')
            self.assertEqual(receipt['settings']['aim']['delta'], 1e-6)
            self.assertEqual(receipt['settings']['protocol']['dp_attempt_limit_seconds'], 1200)

    def test_dp_only_setup_reuses_dp_and_leaves_other_environments_and_attempt_untouched(self):
        from server import setup
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for env in ['sdv', 'synthcity', 'anonymeter', 'dp']:
                (root / ('.venv-' + env) / 'bin').mkdir(parents=True)
                (root / ('.venv-' + env) / 'sentinel').write_text('keep')
            attempt = root / 'results/revision/dp_attempt.json'
            attempt.parent.mkdir(parents=True)
            attempt.write_text('{"state":"failed"}')
            with patch.object(setup, 'ROOT', root), patch.object(setup, 'os', SimpleNamespace(name='posix', environ=os.environ, umask=lambda value: None)), patch.object(setup, 'sys', SimpleNamespace(version_info=(3, 10), version='fixture', stderr=sys.stderr)), patch.object(sys, 'argv', ['setup', '--dp-only']), patch.object(setup.subprocess, 'run') as run, patch.object(setup.venv, 'EnvBuilder') as builder:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(setup.main(), 0)
            builder.assert_not_called()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], [str(root / '.venv-dp/bin/python'), '-m', 'revision.setup_dp'])
            self.assertEqual(attempt.read_text(), '{"state":"failed"}')
            self.assertTrue(all((root / ('.venv-' + env) / 'sentinel').read_text() == 'keep' for env in ['sdv', 'synthcity', 'anonymeter', 'dp']))

    def test_only_three_domain_declarations_change_and_patch_is_idempotent(self):
        source = utility_source() + '\ndef unrelated():\n    return dp.atom_domain(T=float)\n'
        result = corrected_source(source)
        self.assertEqual(result.count('nan=False'), 3)
        self.assertIn('def unrelated():\n    return dp.atom_domain(T=float)', result)
        self.assertEqual(result, corrected_source(result))
        self.assertEqual(result.replace(', nan=False', ''), source)

    def test_unexpected_source_is_rejected(self):
        for source in [utility_source('dp.atom_domain(T=int)'), utility_source('dp.atom_domain(T=float, nan=True)'), '']:
            with self.assertRaises(RuntimeError):
                corrected_source(source)

    def test_terminal_attempt_is_preserved_without_explicit_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary)
            attempt = public / 'dp_attempt.json'
            attempt.write_text('{"state": "failed", "original": true}')
            original = attempt.read_bytes()
            with patch.object(dp_generator, 'PUBLIC', public), patch.object(sys, 'argv', ['dp_generator', '--all-seeds']), patch.object(dp_generator, 'run') as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    dp_generator.main()
            self.assertEqual(attempt.read_bytes(), original)
            run.assert_not_called()

    def test_aim_generation_has_hard_original_timeout_and_preserves_main_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary)
            main = public / 'RUN_STATUS.json'
            main.write_text('{"state":"awaiting_structural_constraint_approval"}')
            original = main.read_bytes()
            calls = []

            def fake_run(module, arguments, seed, executable, **kwargs):
                calls.append((module, arguments, kwargs))
                (public / 'dp_attempt.json').write_text(json.dumps({'state': 'running', 'compatibility': {'change': 'test'}}))
                raise subprocess.TimeoutExpired('generated-test-worker', kwargs['timeout'])

            with patch.object(pipeline, 'PUBLIC', public), patch.object(pipeline, 'run', fake_run), patch.object(pipeline, 'interpreter', return_value='test-python'):
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.aim_only('aim')
            self.assertEqual(main.read_bytes(), original)
            self.assertEqual(calls[0][0:2], ('dp_generator', ['--all-seeds']))
            self.assertEqual(calls[0][2]['timeout'], PROTOCOL['dp_attempt_limit_seconds'])
            self.assertEqual(calls[0][2]['status_name'], 'AIM_RUN_STATUS.json')
            attempt = json.loads((public / 'dp_attempt.json').read_text())
            self.assertEqual(attempt['state'], 'budget_exhausted')
            self.assertEqual(attempt['compatibility']['change'], 'test')

    def test_evaluate_only_completed_aim_cells_without_structural_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary)
            (public / 'generators').mkdir()
            for name in ['AIM_eps1_seed0', 'AIM_eps10_seed7', 'CTGAN_seed0']:
                (public / 'generators' / (name + '.json')).write_text('{}')
            with patch.object(pipeline, 'PUBLIC', public), patch.object(pipeline, 'run') as run, patch.object(pipeline, 'interpreter', side_effect=lambda env: env), patch('revision.report_progress.main'):
                pipeline.aim_only('aim-evaluate')
            calls = run.call_args_list
            self.assertTrue(calls)
            self.assertEqual({call.args[0] for call in calls}, {'evaluate', 'anonymeter_worker'})
            self.assertEqual({call.args[1][1] for call in calls}, {'AIM_eps1', 'AIM_eps10'})
            self.assertTrue(all(call.kwargs['refresh'] is False for call in calls))
            self.assertFalse((public / 'RUN_STATUS.json').exists())
            self.assertEqual(sum(call.args[0] == 'anonymeter_worker' for call in calls), 5)


if __name__ == '__main__':
    unittest.main()
