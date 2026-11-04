"""
TrustSynth Experimental Configuration — eICU-CRD v2.0
Edit the paths below to match your local setup.
"""
from pathlib import Path

# ============================================================================
# PATHS — Edit these to match your setup
# ============================================================================

# Path to eICU-CRD data directory (flat folder with patient.csv, lab.csv, etc.)
EICU_DIR = Path("./data")

# Output directories (created automatically)
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
FIGURES_DIR = Path("./figures")

# ============================================================================
# COHORT PARAMETERS
# ============================================================================
MIN_AGE = 18
MAX_COHORT_SIZE = 20000  # Subsample for tractability on a laptop
RANDOM_SEED = 42
TEST_SPLIT = 0.2  # 20% held out for TSTR and privacy baselines

# Seeds for the multi-seed evaluation (Task 2). Every seed re-draws the
# stratified 80/20 split on hospital mortality, re-trains all generators from
# scratch, and recomputes every metric. RANDOM_SEED above is retained only
# for cohort subsampling in 01_extract_cohort.py so that the cohort itself is
# held fixed across seeds -- seed variation is meant to capture split and
# training variability, not cohort-sampling variability.
SEEDS = [0, 1, 2, 3, 4]

# Reuse synthetic datasets already written to data/synthetic/seed{S}_{name}.csv.
# Set False to force retraining.
USE_SYNTH_CACHE = True

# Restrict the cohort to recorded binary gender. eICU `gender` also contains
# "UNKNOWN" and "OTHER" at counts below 10 in 20,000 stays; these are too few
# to evaluate subgroup representation on, and the manuscript states a
# binary-gender cohort. Enforced in 01_extract_cohort.py BEFORE any split or
# generator training. Set ALLOWED_GENDERS = None to disable the filter.
ALLOWED_GENDERS = ["MALE", "FEMALE"]

# ----------------------------------------------------------------------------
# Vital-sign source resolution (Task 3)
# ----------------------------------------------------------------------------
# vitalPeriodic.systemicsystolic/systemicdiastolic are charted only for
# patients with an indwelling arterial line (74% missing in this cohort), and
# vitalPeriodic.temperature is charted intermittently (89% missing). Median
# imputation of that much mass turned the real marginals into point masses and
# produced spurious KS statistics of 0.38-0.51 for every generator.
#
# Blood pressure is therefore resolved across two tables and temperature across
# two tables, in the priority order below. The first source with an observed
# per-stay mean wins; remaining gaps fall back to median imputation and the
# residual fill mass is reported by 01_extract_cohort.py.
#
#   "aperiodic" -> vitalAperiodic.noninvasivesystolic / noninvasivediastolic
#   "periodic"  -> vitalPeriodic.systemicsystolic / systemicdiastolic
#   "nurse"     -> nurseCharting, cell type label "Temperature"
#
# Non-invasive cuff pressure is preferred for BP because it is the modality the
# large majority of ICU patients receive; using the arterial line first would
# make the measurement modality itself a proxy for illness severity (in this
# cohort, arterial-line patients have 15.6% mortality vs 7.2% without).
BP_SOURCE_PRIORITY = ["aperiodic", "periodic"]
TEMP_SOURCE_PRIORITY = ["nurse", "periodic"]

# Warn if any continuous feature still carries more than this fraction of
# median-imputed rows after source resolution.
MAX_ACCEPTABLE_FILL_MASS = 0.20

# ============================================================================
# GENERATOR PARAMETERS
# ============================================================================
CTGAN_EPOCHS = 300
TVAE_EPOCHS = 300
DPGAN_EPSILON = 1.0  # Privacy budget for DPGAN
SYNTHETIC_SIZE_MULTIPLIER = 1.0  # Generate same number of rows as training set

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Dimension 1: Statistical Utility
# *_THRESHOLD values are the pass/fail lines drawn on figures and used to
# count violations. *_NORM_REF values are the denominators that map a raw
# metric onto the [0, 1] dimension score: score = max(0, 1 - value / ref),
# so a metric equal to its reference scores 0 and a perfect metric scores 1.
KS_THRESHOLD = 0.10
JSD_THRESHOLD = 0.05
CORR_THRESHOLD = 0.10
KS_NORM_REF = 0.50
JSD_NORM_REF = 0.25
CORR_NORM_REF = 0.50

# Dimension 2: Privacy
MIA_SHADOW_MODELS = 5
MIA_THRESHOLD = 0.55
DCR_PERCENTILE = 5  # Compare median DCR against this percentile of holdout-to-real
DCR_SAMPLE_N = 2000  # Rows sampled from each dataset for the Gower DCR

# Dimension 3: Fairness
PROTECTED_ATTRIBUTES = ["gender", "ethnicity_grouped"]
REPRESENTATION_THRESHOLD_LOW = 0.8
REPRESENTATION_THRESHOLD_HIGH = 1.2
BAR_THRESHOLD = 1.1

# Dimension 4: Downstream Fidelity
TSTR_TARGET = "hospital_mortality"  # Binary outcome to predict
TSTR_AUC_THRESHOLD = 0.05
FI_CORRELATION_THRESHOLD = 0.8
TSTR_GAP_NORM_REF = 0.20  # AUC gap that maps to a fidelity sub-score of 0

# Task 4: permutation importance is computed on the REAL held-out test set
# for both the TRTR model and every TSTR model, scored by ROC-AUC.
PERM_IMPORTANCE_REPEATS = 10

# Dimension 5: Structural Integrity
#
# NOTE: every entry below is a single-column min/max bound. SDV satisfies
# these automatically -- enforce_min_max_values defaults to True on
# GaussianCopulaSynthesizer, CTGANSynthesizer and TVAESynthesizer in SDV
# 1.17.2 and is never overridden here -- so they cannot discriminate between
# generators. They are retained for continuity with the original submission.
# The constraints that actually test learned structure live in
# trustsynth_constraints.py (Task 6).
DOMAIN_CONSTRAINTS = {
    "age_valid": ("age", ">=", 18),
    "age_upper": ("age", "<=", 120),
    "hr_lower": ("heart_rate_mean", ">", 0),
    "hr_upper": ("heart_rate_mean", "<", 300),
    "sbp_lower": ("sbp_mean", ">", 0),
    "sbp_upper": ("sbp_mean", "<", 300),
    "dbp_lower": ("dbp_mean", ">", 0),
    "spo2_range": ("spo2_mean", "<=", 100),
    "temp_lower": ("temperature_mean", ">", 30),
    "temp_upper": ("temperature_mean", "<", 45),
    "los_nonneg": ("los_days", ">=", 0),
    "mortality_binary": ("hospital_mortality", "in", [0, 1]),
    "apache_nonneg": ("apache_score", ">=", 0),
    "apache_upper": ("apache_score", "<=", 300),
}

# Cross-column constraint: systolic > diastolic
CROSS_CONSTRAINTS = [
    ("sbp_mean", ">", "dbp_mean"),  # SBP should exceed DBP
]

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================
CONTINUOUS_FEATURES = [
    "age", "heart_rate_mean", "sbp_mean", "dbp_mean",
    "temperature_mean", "spo2_mean", "resp_rate_mean",
    "glucose_mean", "creatinine_mean", "hemoglobin_mean",
    "los_days", "apache_score"
]

CATEGORICAL_FEATURES = [
    "gender", "ethnicity_grouped", "admission_type_grouped",
    "hospital_region"
]

BINARY_FEATURES = [
    "hospital_mortality"
]

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
