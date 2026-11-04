"""Versioned, explicit protocol for the approved revision (not the legacy audit)."""
from .legacy.config import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
from .safe_reporting import ROOT

FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET = 'hospital_mortality'
DATA = ROOT / 'data' / 'revision'
COHORT = DATA / 'cohort.csv'
PROTOCOL = {
    'version': 'approved-preprocessing-v1',
    'cohort_seed': 42, 'cohort_size': 20000, 'seeds': list(range(10)),
    'test_fraction': 0.2, 'minimum_age': 18, 'apache_version': 'IV',
    'minimum_observed_features': (len(FEATURES) + 1) // 2,
    'aggregation': 'arithmetic mean of valid observations across entire selected ICU stay',
    'stay_selection': 'intact adult stay with minimum patientunitstayid per uniquepid',
    'screening_bounds_inclusive': {
        'heartrate': [20, 300], 'sao2': [50, 100], 'respiration': [4, 60],
        'systemicsystolic': [40, 300], 'systemicdiastolic': [20, 200],
        'temperature': [25, 45]},
    'ctgan_epochs': 300, 'tvae_epochs': 300, 'mechanism_seeds': [0, 1, 2],
    'tvae_mechanism_epochs': 900, 'bootstrap_resamples': 1000,
    'permutation_resamples': 1000, 'permutation_importance_repeats': 10,
    'torch_threads': 4, 'model_jobs': 4,
    'sdv_enforce_min_max_values': True, 'sdv_enforce_rounding': True,
    'dp_epsilons': [1.0, 10.0], 'dp_attempt_limit_seconds': 1200,
    'anonymeter_attacks': 500, 'anonymeter_secondary_seeds': [0, 1, 2, 3, 4],
    'bar_min_real_dpd': 0.01,
}
