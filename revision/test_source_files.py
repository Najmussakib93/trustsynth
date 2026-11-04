"""Compression must not change the cohort; every table below is invented."""
import contextlib
import gzip
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import preprocess
from .source_files import source_file


class SourceFileTests(unittest.TestCase):
    def test_gzip_wins_over_partial_csv_and_optional_files_can_be_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'patient.csv').write_text('partial')
            with gzip.open(directory / 'patient.csv.gz', 'wt') as stream:
                stream.write('example\ncomplete\n')
            self.assertEqual(source_file(directory, 'patient.csv').name, 'patient.csv.gz')
            self.assertIsNone(source_file(directory, 'missing.csv', required=False))
            with self.assertRaises(FileNotFoundError):
                source_file(directory, 'missing.csv')

    def test_corrupt_gzip_does_not_fall_back_to_plain_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contents = b'example\ncomplete\n'
            (directory / 'patient.csv').write_bytes(contents)
            (directory / 'patient.csv.gz').write_bytes(gzip.compress(contents)[:-8])
            with self.assertRaises(EOFError):
                pd.read_csv(source_file(directory, 'patient.csv'))

    def test_full_extraction_matches_for_plain_gzip_and_mixed_sources(self):
        n = 24
        ids = np.arange(100, 100 + n)
        tables = {
            'patient.csv': pd.DataFrame({
                'uniquepid': ['invented-' + str(i) for i in range(n)], 'patientunitstayid': ids,
                'age': 45 + np.arange(n), 'gender': ['Male', 'Female'] * (n // 2),
                'ethnicity': 'Caucasian', 'unittype': 'MICU', 'hospitalid': 1,
                'hospitaldischargestatus': ['Expired' if i % 4 == 0 else 'Alive' for i in range(n)],
                'unitdischargeoffset': 1440 + np.arange(n)}),
            'apachePatientResult.csv': pd.DataFrame({'patientunitstayid': ids, 'apacheversion': 'IV', 'apachescore': 30}),
            'hospital.csv': pd.DataFrame({'hospitalid': [1], 'region': ['West']}),
            'vitalPeriodic.csv': pd.DataFrame({'patientunitstayid': ids, 'heartrate': 75,
                'sao2': 97, 'respiration': 18, 'systemicsystolic': 120, 'systemicdiastolic': 75, 'temperature': 37}),
            'lab.csv': pd.DataFrame({'patientunitstayid': np.repeat(ids, 3),
                'labname': ['glucose', 'creatinine', 'Hgb'] * n, 'labresult': [110., 1., 13.] * n})}
        frames = []
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ['plain', 'gzip', 'mixed']:
                root = Path(temporary) / mode
                raw = root / 'data'
                raw.mkdir(parents=True)
                for index, (name, frame) in enumerate(tables.items()):
                    compressed = mode == 'gzip' or mode == 'mixed' and index % 2 == 0
                    frame.to_csv(raw / (name + '.gz' if compressed else name), index=False)
                data, public = raw / 'revision', root / 'results/revision'
                cohort = data / 'cohort.csv'
                with patch.multiple(preprocess, ROOT=root, DATA=data, COHORT=cohort, PUBLIC=public):
                    with patch('sys.argv', ['01_extract_cohort.py']), contextlib.redirect_stdout(io.StringIO()):
                        preprocess.main()
                frames.append(pd.read_csv(cohort))
        pd.testing.assert_frame_equal(frames[0], frames[1])
        pd.testing.assert_frame_equal(frames[0], frames[2])
