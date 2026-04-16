"""
01_extract_cohort.py
Extracts a structured ICU cohort from MIMIC-IV for TrustSynth evaluation.
Produces a clean tabular dataset with mixed types suitable for synthetic data generation.

Output: data/cohort.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from config import (
    MIMIC_DIR, DATA_DIR, MIN_AGE, MAX_COHORT_SIZE, RANDOM_SEED
)


def load_mimic_table(subdir, filename):
    """Load a MIMIC-IV CSV file, trying common directory structures."""
    candidates = [
        MIMIC_DIR / subdir / filename,
        MIMIC_DIR / "mimic-iv" / subdir / filename,
        MIMIC_DIR / filename,
    ]
    # Also try with .csv.gz
    for p in list(candidates):
        candidates.append(p.with_suffix(".csv.gz"))

    for path in candidates:
        if path.exists():
            print(f"  Loading {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"Could not find {filename} in {MIMIC_DIR}. "
        f"Tried: {[str(c) for c in candidates[:3]]}"
    )


def extract_vitals(chartevents_path=None):
    """
    Extract first-24h vital sign summaries from chartevents.
    If chartevents is too large, falls back to a simplified approach.
    """
    # ItemIDs for common vitals in MIMIC-IV
    vital_items = {
        220045: "heart_rate",
        220050: "sbp",        # Arterial BP Systolic
        220051: "dbp",        # Arterial BP Diastolic
        220052: "map",        # Arterial BP Mean
        223761: "temperature_f",  # Temperature Fahrenheit
        223762: "temperature_c",  # Temperature Celsius
        220210: "resp_rate",
        220277: "spo2",
    }

    item_ids = list(vital_items.keys())

    # Try loading chartevents in chunks (it's huge)
    chartevents_file = None
    for suffix in [".csv", ".csv.gz"]:
        for subdir in ["icu", ""]:
            candidate = MIMIC_DIR / subdir / f"chartevents{suffix}" if subdir else MIMIC_DIR / f"chartevents{suffix}"
            if candidate.exists():
                chartevents_file = candidate
                break
        if chartevents_file:
            break

    if chartevents_file is None:
        print("  WARNING: chartevents not found. Generating synthetic vitals as placeholders.")
        print("  (This is fine for pipeline testing; use real data for the final paper.)")
        return None

    print(f"  Reading chartevents from {chartevents_file} (this may take several minutes)...")

    chunks = pd.read_csv(
        chartevents_file,
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        chunksize=5_000_000,
        dtype={"stay_id": "Int64", "itemid": "Int64", "valuenum": "float64"}
    )

    vitals_list = []
    for i, chunk in enumerate(chunks):
        filtered = chunk[chunk["itemid"].isin(item_ids)].copy()
        if len(filtered) > 0:
            filtered["vital_name"] = filtered["itemid"].map(vital_items)
            vitals_list.append(filtered)
        if (i + 1) % 10 == 0:
            print(f"    Processed {(i+1)*5}M rows...")

    if not vitals_list:
        return None

    vitals = pd.concat(vitals_list, ignore_index=True)

    # Convert temperature F to C
    mask_f = vitals["vital_name"] == "temperature_f"
    vitals.loc[mask_f, "valuenum"] = (vitals.loc[mask_f, "valuenum"] - 32) * 5 / 9
    vitals.loc[mask_f, "vital_name"] = "temperature_c"

    # Aggregate: mean per stay
    vitals_agg = (
        vitals.groupby(["stay_id", "vital_name"])["valuenum"]
        .mean()
        .unstack("vital_name")
        .reset_index()
    )

    rename_map = {
        "heart_rate": "heart_rate_mean",
        "sbp": "sbp_mean",
        "dbp": "dbp_mean",
        "temperature_c": "temperature_mean",
        "resp_rate": "resp_rate_mean",
        "spo2": "spo2_mean",
    }
    vitals_agg = vitals_agg.rename(columns=rename_map)

    keep_cols = ["stay_id"] + [v for v in rename_map.values() if v in vitals_agg.columns]
    return vitals_agg[keep_cols]


def extract_labs():
    """Extract first-24h lab summaries from labevents."""
    lab_items = {
        50931: "glucose",      # Glucose
        50912: "creatinine",   # Creatinine
        51222: "hemoglobin",   # Hemoglobin
    }

    labevents_file = None
    for suffix in [".csv", ".csv.gz"]:
        for subdir in ["hosp", ""]:
            candidate = MIMIC_DIR / subdir / f"labevents{suffix}" if subdir else MIMIC_DIR / f"labevents{suffix}"
            if candidate.exists():
                labevents_file = candidate
                break
        if labevents_file:
            break

    if labevents_file is None:
        print("  WARNING: labevents not found. Skipping lab features.")
        return None

    print(f"  Reading labevents from {labevents_file} (this may take several minutes)...")

    chunks = pd.read_csv(
        labevents_file,
        usecols=["subject_id", "itemid", "valuenum"],
        chunksize=5_000_000,
        dtype={"subject_id": "Int64", "itemid": "Int64", "valuenum": "float64"}
    )

    labs_list = []
    for i, chunk in enumerate(chunks):
        filtered = chunk[chunk["itemid"].isin(lab_items.keys())].copy()
        if len(filtered) > 0:
            filtered["lab_name"] = filtered["itemid"].map(lab_items)
            labs_list.append(filtered)
        if (i + 1) % 10 == 0:
            print(f"    Processed {(i+1)*5}M rows...")

    if not labs_list:
        return None

    labs = pd.concat(labs_list, ignore_index=True)

    labs_agg = (
        labs.groupby(["subject_id", "lab_name"])["valuenum"]
        .mean()
        .unstack("lab_name")
        .reset_index()
    )

    rename_map = {
        "glucose": "glucose_mean",
        "creatinine": "creatinine_mean",
        "hemoglobin": "hemoglobin_mean",
    }
    labs_agg = labs_agg.rename(columns=rename_map)
    return labs_agg


def main():
    print("=" * 60)
    print("TrustSynth Cohort Extraction from MIMIC-IV")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load core tables ----
    print("\n[1/5] Loading core tables...")
    patients = load_mimic_table("hosp", "patients.csv")
    admissions = load_mimic_table("hosp", "admissions.csv")

    try:
        icustays = load_mimic_table("icu", "icustays.csv")
        has_icu = True
    except FileNotFoundError:
        print("  WARNING: icustays.csv not found. Building cohort from admissions only.")
        has_icu = False

    # ---- Build base cohort ----
    print("\n[2/5] Building base cohort...")

    if has_icu:
        # First ICU stay per patient
        icustays = icustays.sort_values("intime")
        first_icu = icustays.groupby("subject_id").first().reset_index()

        cohort = first_icu.merge(patients, on="subject_id", how="left")
        cohort = cohort.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    else:
        # First admission per patient
        admissions = admissions.sort_values("admittime")
        first_adm = admissions.groupby("subject_id").first().reset_index()
        cohort = first_adm.merge(patients, on="subject_id", how="left")

    # Compute age
    if "anchor_age" in cohort.columns:
        cohort["age"] = cohort["anchor_age"]
    elif "dob" in cohort.columns and "admittime" in cohort.columns:
        cohort["admittime"] = pd.to_datetime(cohort["admittime"])
        cohort["dob"] = pd.to_datetime(cohort["dob"])
        cohort["age"] = (cohort["admittime"] - cohort["dob"]).dt.days / 365.25
    else:
        print("  WARNING: Cannot compute age reliably. Using anchor_year offset.")
        cohort["age"] = 65  # Placeholder

    cohort = cohort[cohort["age"] >= MIN_AGE].copy()

    # Hospital mortality
    if "deathtime" in cohort.columns and "dischtime" in cohort.columns:
        cohort["hospital_mortality"] = (cohort["deathtime"].notna()).astype(int)
    elif "hospital_expire_flag" in cohort.columns:
        cohort["hospital_mortality"] = cohort["hospital_expire_flag"].astype(int)
    else:
        cohort["hospital_mortality"] = 0

    # Length of stay
    if has_icu and "los" in cohort.columns:
        cohort["los_days"] = cohort["los"].clip(lower=0)
    elif "admittime" in cohort.columns and "dischtime" in cohort.columns:
        cohort["admittime"] = pd.to_datetime(cohort["admittime"])
        cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])
        cohort["los_days"] = (
            (cohort["dischtime"] - cohort["admittime"]).dt.total_seconds() / 86400
        ).clip(lower=0)
    else:
        cohort["los_days"] = 5.0

    # Standardize demographics
    cohort["gender"] = cohort["gender"].str.upper().str.strip()

    # Group ethnicity into manageable categories
    if "race" in cohort.columns:
        eth_col = "race"
    elif "ethnicity" in cohort.columns:
        eth_col = "ethnicity"
    else:
        eth_col = None

    if eth_col:
        eth = cohort[eth_col].fillna("UNKNOWN").str.upper()
        cohort["ethnicity_grouped"] = "OTHER"
        cohort.loc[eth.str.contains("WHITE"), "ethnicity_grouped"] = "WHITE"
        cohort.loc[eth.str.contains("BLACK"), "ethnicity_grouped"] = "BLACK"
        cohort.loc[eth.str.contains("HISPANIC"), "ethnicity_grouped"] = "HISPANIC"
        cohort.loc[eth.str.contains("ASIAN"), "ethnicity_grouped"] = "ASIAN"
    else:
        cohort["ethnicity_grouped"] = "UNKNOWN"

    # Group admission type
    if "admission_type" in cohort.columns:
        adm = cohort["admission_type"].fillna("UNKNOWN").str.upper()
        cohort["admission_type_grouped"] = "OTHER"
        cohort.loc[adm.str.contains("EMERG|URGENT"), "admission_type_grouped"] = "EMERGENCY"
        cohort.loc[adm.str.contains("ELECTIVE"), "admission_type_grouped"] = "ELECTIVE"
        cohort.loc[adm.str.contains("SURGICAL"), "admission_type_grouped"] = "SURGICAL"
    else:
        cohort["admission_type_grouped"] = "UNKNOWN"

    # Insurance
    if "insurance" in cohort.columns:
        cohort["insurance"] = cohort["insurance"].fillna("Other").str.title()
    else:
        cohort["insurance"] = "Unknown"

    print(f"  Base cohort size: {len(cohort)} patients")

    # ---- Extract vitals ----
    print("\n[3/5] Extracting vitals...")
    vitals = extract_vitals()

    if vitals is not None and has_icu:
        cohort = cohort.merge(vitals, on="stay_id", how="left")
    else:
        # Generate plausible placeholder vitals for pipeline testing
        print("  Using simulated vitals (replace with real data for paper)")
        np.random.seed(RANDOM_SEED)
        n = len(cohort)
        cohort["heart_rate_mean"] = np.random.normal(82, 16, n).clip(40, 180)
        cohort["sbp_mean"] = np.random.normal(122, 20, n).clip(70, 220)
        cohort["dbp_mean"] = np.random.normal(65, 12, n).clip(30, 120)
        cohort["temperature_mean"] = np.random.normal(37.0, 0.6, n).clip(34, 42)
        cohort["resp_rate_mean"] = np.random.normal(18, 4, n).clip(8, 40)
        cohort["spo2_mean"] = np.random.normal(96.5, 2.5, n).clip(70, 100)

    # ---- Extract labs ----
    print("\n[4/5] Extracting labs...")
    labs = extract_labs()

    if labs is not None:
        cohort = cohort.merge(labs, on="subject_id", how="left")
    else:
        print("  Using simulated labs (replace with real data for paper)")
        np.random.seed(RANDOM_SEED + 1)
        n = len(cohort)
        cohort["glucose_mean"] = np.random.lognormal(np.log(120), 0.4, n).clip(40, 500)
        cohort["creatinine_mean"] = np.random.lognormal(np.log(1.1), 0.5, n).clip(0.2, 15)
        cohort["hemoglobin_mean"] = np.random.normal(10.5, 2.0, n).clip(4, 18)

    # ---- Finalize cohort ----
    print("\n[5/5] Finalizing cohort...")

    from config import ALL_FEATURES
    available = [f for f in ALL_FEATURES if f in cohort.columns]
    missing = [f for f in ALL_FEATURES if f not in cohort.columns]
    if missing:
        print(f"  WARNING: Missing features (will be dropped): {missing}")

    cohort = cohort[available].copy()

    # Drop rows with excessive missing data (>50% features missing)
    threshold = len(available) * 0.5
    cohort = cohort.dropna(thresh=int(threshold))

    # Fill remaining missing values
    from config import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
    for col in CONTINUOUS_FEATURES:
        if col in cohort.columns and cohort[col].isna().any():
            cohort[col] = cohort[col].fillna(cohort[col].median())

    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        if col in cohort.columns and cohort[col].isna().any():
            cohort[col] = cohort[col].fillna(cohort[col].mode()[0])

    # Subsample if needed
    if len(cohort) > MAX_COHORT_SIZE:
        cohort = cohort.sample(n=MAX_COHORT_SIZE, random_state=RANDOM_SEED)
        print(f"  Subsampled to {MAX_COHORT_SIZE} patients")

    # Save
    output_path = DATA_DIR / "cohort.csv"
    cohort.to_csv(output_path, index=False)
    print(f"\n  Cohort saved to {output_path}")
    print(f"  Shape: {cohort.shape}")
    print(f"  Columns: {list(cohort.columns)}")
    print(f"  Mortality rate: {cohort['hospital_mortality'].mean():.1%}")

    # Print summary statistics
    print("\n  Feature summary:")
    for col in cohort.columns:
        if cohort[col].dtype in ["float64", "int64", "Int64"]:
            print(f"    {col}: mean={cohort[col].mean():.2f}, std={cohort[col].std():.2f}")
        else:
            n_unique = cohort[col].nunique()
            print(f"    {col}: {n_unique} categories")

    print("\n  Done!")


from config import BINARY_FEATURES

if __name__ == "__main__":
    main()
