"""A fresh server checkout must not depend on a workstation's ignored audit."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import report_progress


class FreshServerReportTests(unittest.TestCase):
    def test_multiple_refreshes_without_historical_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / 'results/revision'
            (public / 'preprocessing').mkdir(parents=True)
            (public / 'preprocessing/manifest.json').write_text(json.dumps({
                'flow': {'cohort_sample': 100}, 'mortality': {'fraction': .2},
                'repeat_identical': True, 'optional_sources_available': {}}))
            with patch.object(report_progress, 'ROOT', root), patch.object(report_progress, 'PUBLIC', public):
                with contextlib.redirect_stdout(io.StringIO()):
                    report_progress.main()
                    report_progress.main()
            self.assertTrue((public / 'protocol.json').is_file())
            self.assertTrue((public / 'REVISION_REPORT.md').is_file())
            self.assertFalse((public / 'audit/INITIAL_AUDIT_REPORT.md').exists())


if __name__ == '__main__':
    unittest.main()
