"""Invented-data checks for metric definitions and undefined subgroup behavior."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from .utility import bootstrap_auc_ap
from .fairness import dpd
from .privacy import nearest_gower
from .fidelity import association, evaluate_fidelity
from .protocol import FEATURES, CONTINUOUS_FEATURES


class MetricTests(unittest.TestCase):
    def test_bootstrap_matches_explicit_rows_with_ties(self):
        y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
        p = np.array([.9, .9, .7, .7, .4, .4, .1, .1])
        actual = bootstrap_auc_ap(y, p, 3, 80)
        rng = np.random.RandomState(3)
        weights = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=80)
        auc, ap = [], []
        for w in weights:
            rows = np.repeat(np.arange(len(y)), w)
            if len(np.unique(y[rows])) == 2:
                auc.append(roc_auc_score(y[rows], p[rows]))
                ap.append(average_precision_score(y[rows], p[rows]))
        np.testing.assert_allclose([actual['auroc_ci_low'], actual['auroc_ci_high']], np.quantile(auc, [.025, .975]))
        np.testing.assert_allclose([actual['auprc_ci_low'], actual['auprc_ci_high']], np.quantile(ap, [.025, .975]))

    def test_empty_group_is_undefined_not_parity(self):
        frame = pd.DataFrame({'ethnicity_grouped': ['A'] * 20, 'hospital_mortality': [0, 1] * 10})
        self.assertTrue(np.isnan(dpd(frame, 'ethnicity_grouped', ['A', 'B'])))

    def test_missing_aware_gower_and_leave_self_out(self):
        a = np.array([[0., 0.], [1., 0.], [np.nan, 1.]])
        np.testing.assert_allclose(nearest_gower(a, a, np.array([False, True]), True), [.5, .5, 1.])
        q = np.array([[np.nan, 1.], [.5, 0.]])
        np.testing.assert_allclose(nearest_gower(q, a, np.array([False, True])), [0., .25])

    def test_mixed_association_detects_perfect_group_separation(self):
        frame = pd.DataFrame({'gender': ['A'] * 12 + ['B'] * 12, 'age': [20.] * 12 + [80.] * 12})
        self.assertAlmostEqual(association(frame, 'gender', 'age'), 1.)

    def test_erased_observed_feature_does_not_get_perfect_mean_ks(self):
        rng = np.random.RandomState(5)
        real = pd.DataFrame({c: rng.normal(size=80) if c in CONTINUOUS_FEATURES else rng.choice(['A', 'B'], 80) for c in FEATURES})
        synth = real.copy()
        synth['temperature_mean'] = np.nan
        metrics = evaluate_fidelity(real, synth)
        self.assertTrue(np.isnan(metrics['fidelity.ks_observed.mean']))
        self.assertLess(metrics['fidelity.ks_observed.supported_fraction'], 1.)
        self.assertEqual(metrics['fidelity.ks_observed.available_mean'], 0.)


if __name__ == '__main__':
    unittest.main()
