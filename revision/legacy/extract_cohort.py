"""
01_extract_cohort.py
Extracts a structured ICU cohort from eICU-CRD v2.0 for TrustSynth evaluation.
Uses patientunitstayid as the primary join key.

Output: data/cohort.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from revision.legacy.config import (
    EICU_DIR, DATA_DIR, MIN_AGE, MAX_COHORT_SIZE, RANDOM_SEED,
    ALLOWED_GENDERS, BP_SOURCE_PRIORITY, TEMP_SOURCE_PRIORITY,
    MAX_ACCEPTABLE_FILL_MASS,
    ALL_FEATURES, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
)


def parse_age(val):
    """Parse eICU age field; "> 89" (censored) is returned as 90."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.startswith(">"):
        return 90.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_vitals(cohort_ids):
    """Load mean vitals per stay from vitalPeriodic.csv.

    Clips and aggregates per-chunk to stay within laptop RAM limits.
    vitalPeriodic is ~140M rows; concatenating before aggregation OOMs.
    """
    vital_file = EICU_DIR / "vitalPeriodic.csv"
    if not vital_file.exists():
        print("  vitalPeriodic.csv not found — simulating vitals.")
        return None

    print(f"  Reading vitalPeriodic.csv (10-15 min on large files)...")
    use_cols = [
        "patientunitstayid", "temperature", "sao2",
        "heartrate", "respiration", "systemicsystolic", "systemicdiastolic",
    ]
    clip_bounds = {
        "heartrate":        (20, 300),
        "sao2":             (50, 100),
        "respiration":      (4,  60),
        "systemicsystolic": (40, 300),
        "systemicdiastolic":(20, 200),
        "temperature":      (25, 45),
    }
    id_set = set(cohort_ids)

    # Accumulate running sum and count per stay — never hold full dataset
    sum_acc = {}   # {stay_id: {col: sum}}
    cnt_acc = {}   # {stay_id: {col: count}}

    for i, chunk in enumerate(pd.read_csv(
        vital_file, usecols=use_cols, chunksize=500_000,
        dtype={"patientunitstayid": "Int64"}
    )):
        chunk = chunk[chunk["patientunitstayid"].isin(id_set)].copy()
        if chunk.empty:
            if (i + 1) % 20 == 0:
                print(f"    {(i+1)/2:.0f}M rows processed...")
            continue

        # Clip within this chunk (small, fast)
        for col, (lo, hi) in clip_bounds.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].clip(lo, hi)

        for col in clip_bounds:
            if col not in chunk.columns:
                continue
            grp = chunk.groupby("patientunitstayid")[col]
            for stay_id, s in grp.sum().items():
                sum_acc.setdefault(stay_id, {})[col] = sum_acc.get(stay_id, {}).get(col, 0.0) + s
            for stay_id, c in grp.count().items():
                cnt_acc.setdefault(stay_id, {})[col] = cnt_acc.get(stay_id, {}).get(col, 0) + c

        if (i + 1) % 20 == 0:
            print(f"    {(i+1)/2:.0f}M rows processed...")

    if not sum_acc:
        return None

    # Build mean dataframe from accumulators
    records = []
    for stay_id in sum_acc:
        row = {"patientunitstayid": stay_id}
        for col in clip_bounds:
            s = sum_acc[stay_id].get(col, np.nan)
            c = cnt_acc[stay_id].get(col, 0)
            row[col] = s / c if c > 0 else np.nan
        records.append(row)

    agg = pd.DataFrame(records)
    return agg.rename(columns={
        "heartrate": "heart_rate_mean",
        "sao2": "spo2_mean",
        "respiration": "resp_rate_mean",
        "systemicsystolic": "sbp_mean",
        "systemicdiastolic": "dbp_mean",
        "temperature": "temperature_mean",
    })


def load_vitals_aperiodic(cohort_ids):
    """Per-stay mean NON-INVASIVE blood pressure from vitalAperiodic.csv.

    vitalAperiodic holds cuff pressures, which most ICU patients receive;
    vitalPeriodic.systemicsystolic is arterial-line only. ~25M rows, so the
    same running-accumulator approach as load_vitals() is used.
    """
    f = EICU_DIR / "vitalAperiodic.csv"
    if not f.exists():
        print("  vitalAperiodic.csv not found — skipping non-invasive BP.")
        return None

    print("  Reading vitalAperiodic.csv (3-6 min)...")
    use_cols = ["patientunitstayid", "noninvasivesystolic", "noninvasivediastolic"]
    clip_bounds = {"noninvasivesystolic": (40, 300), "noninvasivediastolic": (20, 200)}
    id_set = set(cohort_ids)

    sum_acc, cnt_acc = {}, {}
    for i, chunk in enumerate(pd.read_csv(
        f, usecols=use_cols, chunksize=500_000,
        dtype={"patientunitstayid": "Int64"}
    )):
        chunk = chunk[chunk["patientunitstayid"].isin(id_set)]
        if chunk.empty:
            continue
        for col, (lo, hi) in clip_bounds.items():
            c = chunk[col].clip(lo, hi)
            grp = c.groupby(chunk["patientunitstayid"])
            for stay_id, v in grp.sum().items():
                sum_acc.setdefault(stay_id, {})[col] = sum_acc.get(stay_id, {}).get(col, 0.0) + v
            for stay_id, v in grp.count().items():
                cnt_acc.setdefault(stay_id, {})[col] = cnt_acc.get(stay_id, {}).get(col, 0) + v
        if (i + 1) % 20 == 0:
            print(f"    {(i+1)/2:.0f}M rows processed...")

    if not sum_acc:
        return None

    records = []
    for stay_id in sum_acc:
        row = {"patientunitstayid": stay_id}
        for col in clip_bounds:
            tot, n = sum_acc[stay_id].get(col, np.nan), cnt_acc[stay_id].get(col, 0)
            row[col] = tot / n if n > 0 else np.nan
        records.append(row)

    return pd.DataFrame(records).rename(columns={
        "noninvasivesystolic": "sbp_aperiodic",
        "noninvasivediastolic": "dbp_aperiodic",
    })


def load_nursecharting_temp(cohort_ids):
    """Per-stay mean temperature (Celsius) from nurseCharting.csv.

    nurseCharting records temperature far more completely than vitalPeriodic.
    Values are stored as free text under both "(C)" and "(F)" value names;
    Fahrenheit entries are converted to Celsius before aggregation. ~150M rows.
    """
    f = EICU_DIR / "nurseCharting.csv"
    if not f.exists():
        print("  nurseCharting.csv not found — skipping nurse-charted temperature.")
        return None

    print("  Reading nurseCharting.csv (15-30 min, largest scan)...")
    use_cols = ["patientunitstayid", "nursingchartcelltypevallabel",
                "nursingchartcelltypevalname", "nursingchartvalue"]
    id_set = set(cohort_ids)

    sum_acc, cnt_acc = {}, {}
    for i, chunk in enumerate(pd.read_csv(
        f, usecols=use_cols, chunksize=1_000_000,
        dtype={"patientunitstayid": "Int64", "nursingchartvalue": "object"}
    )):
        chunk = chunk[
            chunk["patientunitstayid"].isin(id_set)
            & chunk["nursingchartcelltypevallabel"].eq("Temperature")
            & chunk["nursingchartcelltypevalname"].isin(
                ["Temperature (C)", "Temperature (F)"])
        ]
        if chunk.empty:
            if (i + 1) % 20 == 0:
                print(f"    {i+1}M rows processed...")
            continue

        val = pd.to_numeric(chunk["nursingchartvalue"], errors="coerce")
        is_f = chunk["nursingchartcelltypevalname"].eq("Temperature (F)")
        val = val.where(~is_f, (val - 32.0) * 5.0 / 9.0)
        # Drop physiologically impossible readings outright rather than
        # clipping them onto the bound (clipping created a 0.24% spike at 45C
        # in the vitalPeriodic extraction).
        val = val.where(val.between(25.0, 45.0))

        grp = val.groupby(chunk["patientunitstayid"])
        for stay_id, v in grp.sum().items():
            sum_acc[stay_id] = sum_acc.get(stay_id, 0.0) + v
        for stay_id, v in grp.count().items():
            cnt_acc[stay_id] = cnt_acc.get(stay_id, 0) + v

        if (i + 1) % 20 == 0:
            print(f"    {i+1}M rows processed...")

    if not sum_acc:
        return None

    return pd.DataFrame([
        {"patientunitstayid": k, "temperature_nurse": sum_acc[k] / cnt_acc[k]}
        for k in sum_acc if cnt_acc.get(k, 0) > 0
    ])


def resolve_vital_sources(patient):
    """Coalesce multi-source vitals according to the configured priority.

    Returns the patient frame with final sbp_mean / dbp_mean /
    temperature_mean columns and prints the coverage contributed by each
    source. See config.BP_SOURCE_PRIORITY / TEMP_SOURCE_PRIORITY.
    """
    col_map = {
        "sbp_mean":         {"aperiodic": "sbp_aperiodic",       "periodic": "sbp_periodic"},
        "dbp_mean":         {"aperiodic": "dbp_aperiodic",       "periodic": "dbp_periodic"},
        "temperature_mean": {"nurse":     "temperature_nurse",   "periodic": "temperature_periodic"},
    }
    priority = {
        "sbp_mean": BP_SOURCE_PRIORITY,
        "dbp_mean": BP_SOURCE_PRIORITY,
        "temperature_mean": TEMP_SOURCE_PRIORITY,
    }

    print("\n  Resolving multi-source vitals (priority order per config.py):")
    for final_col, sources in col_map.items():
        resolved = pd.Series(np.nan, index=patient.index, dtype="float64")
        provenance = pd.Series("missing", index=patient.index, dtype="object")
        for src in priority[final_col]:
            src_col = sources.get(src)
            if src_col is None or src_col not in patient.columns:
                continue
            take = resolved.isna() & patient[src_col].notna()
            resolved.loc[take] = patient.loc[take, src_col]
            provenance.loc[take] = src
        patient[final_col] = resolved
        counts = provenance.value_counts()
        detail = ", ".join(f"{k}={v:,} ({v/len(patient):.1%})" for k, v in counts.items())
        print(f"    {final_col:18s} {detail}")
    return patient


def load_labs(cohort_ids):
    """Load mean lab values per stay from lab.csv."""
    lab_file = EICU_DIR / "lab.csv"
    if not lab_file.exists():
        print("  lab.csv not found — simulating labs.")
        return None

    target_labs = {"glucose", "bedside glucose", "creatinine", "Hgb"}
    id_set = set(cohort_ids)
    print("  Reading lab.csv (5-10 min)...")
    chunks_list = []

    for i, chunk in enumerate(pd.read_csv(
        lab_file,
        usecols=["patientunitstayid", "labname", "labresult"],
        chunksize=1_000_000,
        dtype={"patientunitstayid": "Int64", "labresult": "float64"}
    )):
        filtered = chunk[
            chunk["patientunitstayid"].isin(id_set) &
            chunk["labname"].isin(target_labs)
        ]
        if len(filtered):
            chunks_list.append(filtered)
        if (i + 1) % 10 == 0:
            print(f"    {(i+1)}M rows processed...")

    if not chunks_list:
        return None

    labs = pd.concat(chunks_list, ignore_index=True)
    labs["labname"] = labs["labname"].replace("bedside glucose", "glucose")

    agg = (
        labs.groupby(["patientunitstayid", "labname"])["labresult"]
        .mean()
        .unstack("labname")
        .reset_index()
    )
    return agg.rename(columns={
        "glucose": "glucose_mean",
        "creatinine": "creatinine_mean",
        "Hgb": "hemoglobin_mean",
    })


def main():
    print("=" * 60)
    print("TrustSynth Cohort Extraction — eICU-CRD v2.0")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- [1/5] Patient table ----
    print("\n[1/5] Loading patient.csv...")
    patient = pd.read_csv(EICU_DIR / "patient.csv")
    print(f"  Loaded {len(patient):,} unit stays")

    patient["age"] = patient["age"].apply(parse_age)
    patient = patient[patient["age"] >= MIN_AGE].copy()
    print(f"  After age >= {MIN_AGE} filter: {len(patient):,} stays")

    # First ICU unit stay per unique patient
    patient = patient.sort_values("patientunitstayid")
    patient = patient.groupby("uniquepid").first().reset_index()
    print(f"  First stay per patient: {len(patient):,}")

    # Outcome: hospital mortality
    patient["hospital_mortality"] = (
        patient["hospitaldischargestatus"].str.lower().str.strip() == "expired"
    ).astype(int)

    # LOS: unitdischargeoffset is in minutes
    patient["los_days"] = (patient["unitdischargeoffset"] / 1440.0).clip(lower=0)

    # Gender — normalise, then restrict to the binary-gender cohort described
    # in the manuscript. This runs before every downstream join, before the
    # train/test split, and therefore before any generator sees the data.
    patient["gender"] = patient["gender"].str.upper().str.strip()
    print(f"\n  Gender before filter:")
    print(patient["gender"].value_counts(dropna=False).to_string())
    if ALLOWED_GENDERS:
        n_before = len(patient)
        patient = patient[patient["gender"].isin(ALLOWED_GENDERS)].copy()
        print(f"  Gender filter {ALLOWED_GENDERS}: {n_before:,} -> {len(patient):,} stays "
              f"({n_before - len(patient):,} dropped)")
        print(patient["gender"].value_counts(dropna=False).to_string())

    # Ethnicity grouping (eICU labels differ from MIMIC)
    eth = patient["ethnicity"].fillna("Unknown").str.strip()
    patient["ethnicity_grouped"] = "OTHER"
    patient.loc[eth.str.contains("Caucasian", case=False, na=False), "ethnicity_grouped"] = "WHITE"
    patient.loc[eth.str.contains("African American", case=False, na=False), "ethnicity_grouped"] = "BLACK"
    patient.loc[eth.str.contains("Hispanic", case=False, na=False), "ethnicity_grouped"] = "HISPANIC"
    # Use word boundary so "Caucasian" (contains "asian") is not misclassified
    patient.loc[eth.str.contains(r"\bAsian\b", case=False, na=False), "ethnicity_grouped"] = "ASIAN"

    # Admission type from ICU unit type
    unit = patient["unittype"].fillna("").str.strip()
    patient["admission_type_grouped"] = "OTHER"
    patient.loc[unit.str.contains("MICU|Med-Surg|medical", case=False, na=False), "admission_type_grouped"] = "MEDICAL"
    patient.loc[unit.str.contains("SICU|surgical", case=False, na=False), "admission_type_grouped"] = "SURGICAL"
    patient.loc[unit.str.contains("CCU|Cardiac|CTICU|CSICU", case=False, na=False), "admission_type_grouped"] = "CARDIAC"

    # ---- [2/5] APACHE scores ----
    print("\n[2/5] Loading apachePatientResult.csv...")
    apache = pd.read_csv(EICU_DIR / "apachePatientResult.csv")
    apache_agg = (
        apache.groupby("patientunitstayid")["apachescore"]
        .mean()
        .reset_index()
        .rename(columns={"apachescore": "apache_score"})
    )
    patient = patient.merge(apache_agg, on="patientunitstayid", how="left")
    n_apache = patient["apache_score"].notna().sum()
    print(f"  APACHE score available: {n_apache:,} / {len(patient):,} stays")

    # ---- [3/5] Hospital region ----
    print("\n[3/5] Loading hospital.csv...")
    hospital = pd.read_csv(EICU_DIR / "hospital.csv")
    patient = patient.merge(hospital[["hospitalid", "region"]], on="hospitalid", how="left")
    patient = patient.rename(columns={"region": "hospital_region"})
    patient["hospital_region"] = patient["hospital_region"].fillna("Unknown")
    print(f"  Region distribution:\n{patient['hospital_region'].value_counts().to_string()}")

    cohort_ids = patient["patientunitstayid"].tolist()

    # ---- [4/5] Vitals ----
    print("\n[4/5] Extracting vitals...")

    def _cached(cache_name, loader):
        """Load a per-stay aggregate from cache, else build it and cache it."""
        cache = DATA_DIR / cache_name
        if cache.exists():
            print(f"  Loading cached {cache_name}...")
            return pd.read_csv(cache, dtype={"patientunitstayid": "Int64"})
        agg = loader(cohort_ids)
        if agg is not None:
            agg.to_csv(cache, index=False)
            print(f"  Cached -> {cache}")
        return agg

    # 4a. vitalPeriodic: HR, SpO2, respiration (well populated) plus
    #     arterial-line BP and intermittent temperature (both sparse).
    vitals = _cached("vitals_agg.csv", load_vitals)
    if vitals is None:
        raise RuntimeError(
            "vitalPeriodic.csv is missing and no cached vitals_agg.csv exists. "
            "Refusing to fabricate vital signs for a results pipeline."
        )
    # Mark the sparse columns with their provenance; the final sbp_mean /
    # dbp_mean / temperature_mean are resolved from all sources below.
    vitals = vitals.rename(columns={
        "sbp_mean": "sbp_periodic",
        "dbp_mean": "dbp_periodic",
        "temperature_mean": "temperature_periodic",
    })
    patient = patient.merge(vitals, on="patientunitstayid", how="left")
    print(f"  vitalPeriodic merged for {len(vitals):,} stays")

    # 4b. vitalAperiodic: non-invasive cuff blood pressure (Task 3 fix).
    aperiodic = _cached("vitals_aperiodic_agg.csv", load_vitals_aperiodic)
    if aperiodic is not None:
        patient = patient.merge(aperiodic, on="patientunitstayid", how="left")
        print(f"  vitalAperiodic merged for {len(aperiodic):,} stays")

    # 4c. nurseCharting: nurse-charted temperature (Task 3 fix).
    nurse_temp = _cached("nursecharting_temp_agg.csv", load_nursecharting_temp)
    if nurse_temp is not None:
        patient = patient.merge(nurse_temp, on="patientunitstayid", how="left")
        print(f"  nurseCharting temperature merged for {len(nurse_temp):,} stays")

    # 4d. Coalesce the multi-source vitals into the final feature columns.
    patient = resolve_vital_sources(patient)

    # ---- [5/5] Labs ----
    print("\n[5/5] Extracting labs...")
    labs_cache = DATA_DIR / "labs_agg.csv"
    if labs_cache.exists():
        print("  Loading cached labs aggregate...")
        labs = pd.read_csv(labs_cache, dtype={"patientunitstayid": "Int64"})
    else:
        labs = load_labs(cohort_ids)
        if labs is not None:
            labs.to_csv(labs_cache, index=False)
            print(f"  Cached labs to {labs_cache}")

    if labs is None:
        raise RuntimeError(
            "lab.csv is missing and no cached labs_agg.csv exists. "
            "Refusing to fabricate laboratory values for a results pipeline."
        )
    patient = patient.merge(labs, on="patientunitstayid", how="left")
    print("  Labs merged")

    # ---- Finalize cohort ----
    available = [f for f in ALL_FEATURES if f in patient.columns]
    missing = [f for f in ALL_FEATURES if f not in patient.columns]
    if missing:
        print(f"\n  WARNING: Missing features (dropped): {missing}")

    cohort = patient[available].copy()

    # Drop rows missing > 50% of features
    thresh = int(len(available) * 0.5)
    cohort = cohort.dropna(thresh=thresh)

    # Impute remaining NaNs.
    # Every median-filled row becomes an exact duplicate of the column median,
    # so a large fill fraction turns the real marginal into a point mass and
    # inflates the KS statistic against any generator (Task 3). The share
    # filled per feature is recorded here and reported below.
    fill_mass = {}
    for col in CONTINUOUS_FEATURES:
        if col in cohort.columns:
            fill_mass[col] = float(cohort[col].isna().mean())
            if cohort[col].isna().any():
                cohort[col] = cohort[col].fillna(cohort[col].median())
    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        if col in cohort.columns and cohort[col].isna().any():
            cohort[col] = cohort[col].fillna(cohort[col].mode()[0])

    if len(cohort) > MAX_COHORT_SIZE:
        cohort = cohort.sample(n=MAX_COHORT_SIZE, random_state=RANDOM_SEED)
        print(f"  Subsampled to {MAX_COHORT_SIZE:,}")

    # Hard guarantee: the saved cohort contains only the allowed genders.
    if ALLOWED_GENDERS:
        leaked = sorted(set(cohort["gender"].unique()) - set(ALLOWED_GENDERS))
        assert not leaked, f"Gender filter leaked categories into cohort: {leaked}"

    output_path = DATA_DIR / "cohort.csv"
    cohort.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Cohort saved: {output_path}")
    print(f"  Shape: {cohort.shape[0]:,} rows × {cohort.shape[1]} columns")
    print(f"  Mortality rate: {cohort['hospital_mortality'].mean():.1%}")
    print(f"\n  Gender distribution (final cohort):")
    print(cohort["gender"].value_counts(dropna=False).to_string())
    print(f"\n  Median-fill mass per continuous feature "
          f"(share of rows imputed; inflates KS when large):")
    for col, frac in sorted(fill_mass.items(), key=lambda kv: -kv[1]):
        flag = "  <-- EXCEEDS MAX_ACCEPTABLE_FILL_MASS" if frac > MAX_ACCEPTABLE_FILL_MASS else ""
        print(f"    {col:20s} {frac:6.1%}{flag}")
    worst = [c for c, f in fill_mass.items() if f > MAX_ACCEPTABLE_FILL_MASS]
    if worst:
        print(f"    WARNING: {len(worst)} feature(s) above the "
              f"{MAX_ACCEPTABLE_FILL_MASS:.0%} fill threshold: {worst}")

    print(f"\n  Ethnicity distribution:")
    print(cohort["ethnicity_grouped"].value_counts().to_string())
    print(f"\n  Feature summary:")
    for col in cohort.columns:
        if cohort[col].dtype in ["float64", "int64", "Int64"]:
            print(f"    {col}: mean={cohort[col].mean():.2f}, std={cohort[col].std():.2f}")
        else:
            print(f"    {col}: {cohort[col].nunique()} categories — {cohort[col].value_counts().to_dict()}")
    print("\n  Done!")


if __name__ == "__main__":
    main()
