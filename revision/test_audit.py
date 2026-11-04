"""Small tests use invented data only, never credentialed records."""
import unittest

import numpy as np
import pandas as pd

from trustsynth_gower import gower_min_distance
from .safe_reporting import category_counts, count, event_summary, fraction, private_path


class DisclosureTests(unittest.TestCase):
    def test_small_counts_and_complements(self):
        self.assertEqual(count(0), "<11")
        self.assertEqual(count(10), "<11")
        self.assertEqual(count(11), 11)
        self.assertEqual(fraction(10, 100), "<11")
        self.assertEqual(fraction(99, 100), "<11")
        self.assertEqual(event_summary(99, 100)["count"], "suppressed")
        self.assertEqual(event_summary(11, 100)["count"], 11)

    def test_secondary_suppression_is_not_mislabeled_small(self):
        summary = category_counts(pd.Series(["A"] * 80 + ["B"] * 18 + ["C"] * 2))
        self.assertEqual(summary, {"A": 80, "B": "suppressed", "C": "<11"})

    def test_private_path_cannot_escape(self):
        with self.assertRaises(ValueError):
            private_path("../../../exposed.csv")


class GowerTests(unittest.TestCase):
    def test_package_equivalence_and_constant_column(self):
        import gower
        rng = np.random.RandomState(11)
        frame = pd.DataFrame({"a": rng.uniform(1, 50, 67), "b": rng.choice(["X", "Y"], 67),
                              "constant": np.ones(67), "binary": rng.choice([0, 1], 67)})
        X, Y = frame.iloc[:23], frame.iloc[23:]
        flags = [False, True, False, False]
        expected = gower.gower_matrix(X, Y, cat_features=flags).min(axis=1)
        actual = gower_min_distance(X, Y, flags, n_jobs=1)
        np.testing.assert_array_equal(expected, actual)

    def test_recomputing_ranges_per_chunk_changes_metric(self):
        # Deliberately place a numeric extreme in only one query block.
        X = pd.DataFrame({"a": [2., 3., 4., 100.], "b": ["X", "X", "Y", "Y"]})
        Y = pd.DataFrame({"a": [0., 1., 5.], "b": ["X", "Y", "X"]})
        whole = gower_min_distance(X, Y, [False, True], n_jobs=1)
        chunks = np.concatenate([gower_min_distance(X.iloc[:2], Y, [False, True], n_jobs=1),
                                 gower_min_distance(X.iloc[2:], Y, [False, True], n_jobs=1)])
        self.assertFalse(np.array_equal(whole, chunks))


if __name__ == "__main__":
    unittest.main()
