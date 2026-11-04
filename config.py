"""Approved revision configuration. Legacy values live in revision/legacy/config.py.
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
