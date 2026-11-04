"""Resolve source encodings without expanding large archives on disk."""
from pathlib import Path


def source_file(directory, name, required=True):
    directory = Path(directory)
    # A failed manual decompression can leave a partial CSV next to its archive.
    # Prefer gzip and never silently fall back if its contents are corrupt.
    for candidate in [directory / (name + '.gz'), directory / name]:
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError('Required source missing: ' + str(directory / name) + ' (or its .gz archive)')
    return None
