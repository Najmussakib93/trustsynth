"""Worker isolation must survive a Windows-to-Linux repository transfer."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .runtime_paths import environment_python, require_environment


class RuntimePathTests(unittest.TestCase):
    def test_platform_specific_worker_paths(self):
        root = Path('project')
        self.assertEqual(environment_python('synthcity', root, 'posix'),
                         root / '.venv-synthcity/bin/python')
        self.assertEqual(environment_python('synthcity', root, 'nt'),
                         root / '.venv-synthcity/Scripts/python.exe')

    def test_missing_worker_never_falls_back_to_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('revision.runtime_paths.ROOT', Path(temporary)):
                with self.assertRaisesRegex(RuntimeError, 'Missing isolated environment'):
                    require_environment('anonymeter')


if __name__ == '__main__':
    unittest.main()
