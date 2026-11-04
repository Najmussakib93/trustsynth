"""Activate reviewed revision commands after preserving their earlier sources."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    '02_run_evaluation.py': '''"""Approved revision runner: see revision/README.md for --stage and --device."""
from revision.pipeline import main
if __name__ == '__main__':
    main()
''',
    '03_generate_outputs.py': '''"""Refresh completed aggregate cells and the interim report."""
from revision.report_progress import main
if __name__ == '__main__':
    main()
''',
    'config.py': '''"""Approved revision configuration. Legacy values live in revision/legacy/config.py.
Fidelity means distributional similarity; utility means downstream prediction.
"""
from revision.protocol import (PROTOCOL, FEATURES as ALL_FEATURES,
    CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES, DATA, COHORT, TARGET)
from revision.safe_reporting import ROOT, PUBLIC
EICU_DIR = ROOT / 'data'
DATA_DIR = DATA
RESULTS_DIR = PUBLIC
FIGURES_DIR = PUBLIC / 'figures'
SEEDS = PROTOCOL['seeds']
RANDOM_SEED = PROTOCOL['cohort_seed']
TEST_SPLIT = PROTOCOL['test_fraction']
MIN_AGE = PROTOCOL['minimum_age']
MAX_COHORT_SIZE = PROTOCOL['cohort_size']
ALLOWED_GENDERS = ['MALE', 'FEMALE']
CTGAN_EPOCHS = PROTOCOL['ctgan_epochs']
TVAE_EPOCHS = PROTOCOL['tvae_epochs']
TSTR_TARGET = TARGET
'''}
for name, text in FILES.items():
    archived = ROOT / 'revision' / 'legacy' / {'config.py': 'config.py',
        '02_run_evaluation.py': 'run_evaluation.py', '03_generate_outputs.py': 'generate_outputs.py'}[name]
    if not archived.exists():
        raise RuntimeError('Required legacy source archive is absent')
    (ROOT / name).write_text(text, encoding='utf-8')
print('Revision entry points activated; legacy sources preserved.')
