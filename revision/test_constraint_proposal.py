"""Proposal-only refresh must not access records or run additional checks."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import constraint_proposal


class ProposalOnlyTests(unittest.TestCase):
    def test_refresh_uses_aggregate_baseline_without_loading_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary)
            (public / 'integrity').mkdir()
            with (public / 'integrity/original_real_train.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['constraint', 'all_applicable_records_valid'])
                writer.writeheader()
                writer.writerows([{'constraint': 'sbp_upper', 'all_applicable_records_valid': False},
                                  {'constraint': 'age_valid', 'all_applicable_records_valid': True}])
            with patch.object(constraint_proposal, 'PUBLIC', public):
                with patch.object(constraint_proposal, 'split_cohort', side_effect=AssertionError('Records must not be loaded')):
                    with patch('sys.argv', ['constraint_proposal', '--proposal-only']), contextlib.redirect_stdout(io.StringIO()):
                        constraint_proposal.main()
            self.assertTrue((public / 'STRUCTURAL_CONSTRAINT_PROPOSAL.md').is_file())
