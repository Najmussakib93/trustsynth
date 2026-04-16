"""
TrustSynth Experimental Configuration
Edit the paths below to match your local setup.
"""
from pathlib import Path

# ============================================================================
# PATHS — Edit these to match your setup
# ============================================================================

# Path to your MIMIC-IV data directory (the folder containing hosp/ and icu/)
# For MIMIC-IV demo: the folder containing mimic-iv-demo/2.2/hosp/ and icu/
MIMIC_DIR = Path("./data")  # Flat directory — all CSVs in one folder

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
KS_THRESHOLD = 0.10
JSD_THRESHOLD = 0.05
CORR_THRESHOLD = 0.10

# Dimension 2: Privacy
MIA_SHADOW_MODELS = 5
MIA_THRESHOLD = 0.55
DCR_PERCENTILE = 5  # Compare median DCR against this percentile of holdout-to-real

# Dimension 3: Fairness
PROTECTED_ATTRIBUTES = ["gender", "ethnicity_grouped"]
REPRESENTATION_THRESHOLD_LOW = 0.8
REPRESENTATION_THRESHOLD_HIGH = 1.2
BAR_THRESHOLD = 1.1

# Dimension 4: Downstream Fidelity
TSTR_TARGET = "hospital_mortality"  # Binary outcome to predict
TSTR_AUC_THRESHOLD = 0.05
FI_CORRELATION_THRESHOLD = 0.8

# Dimension 5: Structural Integrity
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
    "los_days"
]

CATEGORICAL_FEATURES = [
    "gender", "ethnicity_grouped", "admission_type_grouped",
    "insurance"
]

BINARY_FEATURES = [
    "hospital_mortality"
]

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
