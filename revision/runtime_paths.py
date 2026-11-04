"""Resolve isolated workers on Windows and Unix without silently mixing environments."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def environment_python(name, root=None, platform=None):
    directory = Path(root or ROOT) / ('.venv-' + name)
    windows = (os.name if platform is None else platform) == 'nt'
    return directory / ('Scripts/python.exe' if windows else 'bin/python')


def require_environment(name):
    executable = environment_python(name)
    if not executable.is_file():
        raise RuntimeError('Missing isolated environment: ' + str(executable) +
                           '. See server/README.md for setup.')
    return str(executable)
