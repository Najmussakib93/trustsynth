"""Regression tests with invented rows: intact stays, labels, missingness and screening."""
import unittest
import numpy as np
import pandas as pd
from .preprocess import selected_patients, clean_values
from .encoding import MissingnessCodec
from .protocol import FEATURES, CONTINUOUS_FEATURES


class PreprocessingTests(unittest.TestCase):
    def test_stay_selection_does_not_borrow_later_outcome(self):
        frame = pd.DataFrame({'uniquepid': ['a', 'a', 'b'], 'patientunitstayid': [1, 2, 3],
            'age': [50, 50, 60], 'gender': ['Male'] * 3, 'hospitaldischargestatus': [None, 'Expired', 'Alive'],
            'unitdischargeoffset': [1440] * 3, 'ethnicity': ['Caucasian'] * 3, 'unittype': ['MICU'] * 3})
        out, _ = selected_patients(frame)
        self.assertEqual(out.uniquepid.tolist(), ['b'])
        self.assertEqual(out.hospital_mortality.tolist(), [0])

    def test_screening_drops_instead_of_clips_and_includes_endpoints(self):
        actual = clean_values(pd.Series([19., 20., 300., 301., np.inf, np.nan]), [20, 300])
        np.testing.assert_allclose(actual, [np.nan, 20, 300, np.nan, np.nan, np.nan], equal_nan=True)

    def test_codec_round_trip_and_train_only_fill(self):
        train = pd.DataFrame({c: [1., 3., np.nan] if c in CONTINUOUS_FEATURES else ['A', 'B', 'A'] for c in FEATURES})
        codec = MissingnessCodec().fit(train)
        pd.testing.assert_frame_equal(train, codec.inverse_transform(codec.transform(train)))
        test = train.copy()
        test['age'] = [1000., np.nan, 2000.]
        self.assertEqual(codec.transform(test).age.iloc[1], 2.)
        self.assertEqual(codec.fills['age'], 2.)


if __name__ == '__main__':
    unittest.main()
