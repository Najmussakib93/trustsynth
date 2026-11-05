"""Scientific edge cases and orchestration checks using invented data only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .integrity import hard_rules, parameters, population_profile, compare_profiles, APPROVAL, inherited_rule_audit
from .scores import score_cell, component_values, REFERENCES, weight_sensitivity, calculate
from .statistics import exact_wilcoxon, paired, holm
from .finalize import plan, sanitize_software_line


def score_inputs():
    return {'fidelity.ks_observed.mean': .1, 'fidelity.js_divergence.mean': .02,
        'fidelity.association_abs_error.mean': .05, 'privacy.mia_accuracy': .52,
        'privacy.dcr_ratio': 1.1, 'fairness.representation.ethnicity_grouped.A': 1.,
        'fairness.representation.ethnicity_grouped.B': 1., 'fairness.dpd_difference.ethnicity_grouped': .01,
        'fairness.dpd_difference.gender': -.01, 'utility.auroc_gap_absolute': .03,
        'utility.importance_spearman': .8, 'integrity.csr_record_all': 1.}


def invented_records(n=800):
    rng = np.random.RandomState(17)
    return pd.DataFrame({'age': rng.uniform(18, 100, n), 'heart_rate_mean': rng.uniform(60, 100, n),
        'sbp_mean': rng.uniform(110, 140, n), 'dbp_mean': rng.uniform(65, 85, n),
        'spo2_mean': rng.uniform(85, 100, n), 'resp_rate_mean': rng.uniform(10, 30, n),
        'temperature_mean': rng.uniform(36, 38, n), 'los_days': rng.uniform(1, 10, n),
        'hospital_mortality': rng.binomial(1, .4, n), 'apache_score': rng.uniform(10, 100, n),
        'creatinine_mean': rng.uniform(.5, 2, n), 'hemoglobin_mean': rng.uniform(9, 15, n),
        'admission_type_grouped': np.resize(['CARDIAC', 'MEDICAL', 'OTHER', 'SURGICAL'], n)})


class FinalizationTests(unittest.TestCase):
    def test_undefined_dimension_is_not_zero_or_renormalized(self):
        metrics = score_inputs()
        self.assertTrue(np.isfinite(score_cell(metrics)['TrustScore']))
        metrics['fairness.dpd_difference.ethnicity_grouped'] = np.nan
        result = score_cell(metrics)
        self.assertTrue(np.isnan(result['fairness']))
        self.assertTrue(np.isnan(result['TrustScore']))

    def test_js_scale_conversion_and_clipping(self):
        metrics = score_inputs()
        metrics['fidelity.js_divergence.mean'] = .05 / np.log(2)
        self.assertAlmostEqual(component_values(metrics)['fidelity.jsd'], .8)
        metrics['privacy.mia_accuracy'] = .4
        metrics['utility.importance_spearman'] = -.5
        values = component_values(metrics)
        self.assertEqual(values['privacy.mia'], 1.)
        self.assertEqual(values['utility.importance'], 0.)
        self.assertTrue(all(0 <= value <= 1 for value in values.values()))

    def test_exact_wilcoxon_minimum_and_ties(self):
        differences = np.arange(1., 11.)
        statistic, probability, n = exact_wilcoxon(differences)
        self.assertEqual(probability, 2 / 1024)
        self.assertAlmostEqual(probability, wilcoxon(differences, method='exact').pvalue)
        self.assertEqual(exact_wilcoxon([1., 1., 1., 0.])[1], .25)
        self.assertEqual(exact_wilcoxon(np.zeros(10))[1], 1.)

    def test_pair_alignment_and_holm(self):
        a = pd.Series([1., 2., np.nan], index=[0, 1, 2])
        b = pd.Series([1., .5, 3.], index=[1, 0, 3])
        result = paired(a, b)
        self.assertEqual(result['n_seeds'], 2)
        self.assertEqual(result['mean'], .75)
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_record_csr_is_not_mean_per_constraint(self):
        frame = invented_records(200)
        frame.loc[:19, 'apache_score'] = 400
        rows, metrics = hard_rules(frame, set(APPROVAL['excluded_primary_rules']))
        self.assertAlmostEqual(metrics['integrity.csr_record_all'], .9)
        self.assertGreater(metrics['integrity.csr_constraint_mean'], .9)
        frame.loc[20:59, 'temperature_mean'] = np.nan
        _, metrics = hard_rules(frame, set(APPROVAL['excluded_primary_rules']))
        self.assertAlmostEqual(metrics['integrity.complete_rule_coverage'], .8)
        self.assertAlmostEqual(metrics['integrity.csr_record_all'], 140 / 160)

    def test_small_violation_rate_is_not_back_calculable(self):
        frame = invented_records(200)
        frame.loc[0, 'apache_score'] = 400
        rows, metrics = hard_rules(frame, set(APPROVAL['excluded_primary_rules']))
        self.assertTrue(np.isnan(metrics['integrity.csr_record_all']))
        row = next(row for row in rows if row['constraint'] == 'apache_upper')
        self.assertEqual(row['violations_n'], '<11')
        self.assertTrue(np.isnan(row['csr']))

    def test_population_shift_and_independent_repeat(self):
        real = invented_records()
        synth = real.copy()
        synth['los_days'] += 2
        cutoffs = parameters(real)
        a = population_profile(real, cutoffs, 5, 40)
        b = population_profile(synth, cutoffs, 10005, 40)
        rows = compare_profiles(a, b)
        pd.testing.assert_frame_equal(pd.DataFrame(rows), pd.DataFrame(compare_profiles(a, population_profile(synth, cutoffs, 10005, 40))))
        los = [row for row in rows if row['family'] == 'los_by_unit']
        np.testing.assert_allclose([row['difference'] for row in los], 2.)
        self.assertEqual({row['family'] for row in rows}, set(APPROVAL['families']))
        self.assertFalse(any('pass' in row for row in rows))

    def test_weight_ties_split_and_repeat(self):
        raw = pd.DataFrame([{'generator': name, 'seed': seed, 'metric': metric, 'value': value}
            for name in ['A', 'B'] for seed in range(10) for metric, value in score_inputs().items()])
        scores = calculate(raw)
        first, _ = weight_sensitivity(scores, samples=200)
        second, _ = weight_sensitivity(calculate(raw), samples=200)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_allclose(first.winner_share, [.5, .5])

    def test_finalizer_never_schedules_training(self):
        steps = plan([{'generator': 'CTGAN', 'seed': 0}])
        self.assertEqual({module for module, arguments in steps}, {'integrity', 'final_analysis'})
        self.assertNotIn('generators', repr(steps))

    def test_bound_inheritance_does_not_claim_cross_column_guarantee(self):
        frame = invented_records()
        rows = {row['constraint']: row for row in inherited_rule_audit(frame)}
        self.assertTrue(rows['hr_upper']['inherited_when_observed'])
        self.assertFalse(rows['sbp_mean_greater_than_dbp_mean']['inherited_when_observed'])
        frame.loc[0, 'temperature_mean'] = 30
        rows = {row['constraint']: row for row in inherited_rule_audit(frame)}
        self.assertFalse(rows['temp_lower']['inherited_when_observed'])

    def test_aggregate_publication_pipeline_keeps_erased_profile_unrankable(self):
        from .testing_aggregate_fixture import create
        from .final_analysis import main
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            create(directory)
            result = main(directory, make_figures=False)
            eligibility = pd.read_csv(directory / 'score_eligibility.csv').set_index('generator')
            self.assertFalse(eligibility.loc['TVAE', 'eligible'])
            report = (directory / 'REVISION_REPORT.md').read_text(encoding='utf-8')
            self.assertIn('INVENTED AGGREGATE TEST FIXTURE', report)
            self.assertEqual(set(result['findings'].finding), set(pd.read_csv(directory / 'original_findings.csv').finding))

    def test_software_credentials_redacted(self):
        value = sanitize_software_line('ERROR: https://someone:secret@example.org token=hidden')
        self.assertNotIn('secret', value)
        self.assertNotIn('hidden', value)
