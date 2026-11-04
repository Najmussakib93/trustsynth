"""Reconstruct extraction variants for audit without changing preprocessing.

Only aggregates leave results/private/. The unfiltered variant is a candidate
paper cohort, not an assertion of provenance. It must match archived metrics.
The binary variant must reproduce data/cohort.csv before its masks are trusted.
"""
import importlib.util
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .legacy import config
from .safe_reporting import (ROOT, PUBLIC, category_counts, count, fraction,
                             event_summary, private_path, write_csv, write_json)


def extractor():
    spec = importlib.util.spec_from_file_location("cohort_source", ROOT / "revision" / "legacy" / "extract_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct(binary_gender, raw, apache, hospital, vitals, labs):
    source = extractor()
    patient = raw.copy()
    flow = [{"stage": "all_unit_stays", "count": count(len(patient))}]
    patient["age"] = patient["age"].apply(source.parse_age)
    patient = patient[patient.age >= config.MIN_AGE].copy()
    flow.append({"stage": "adult_stays", "count": count(len(patient))})
    patient = patient.sort_values("patientunitstayid")
    physical = patient.drop_duplicates("uniquepid", keep="first").set_index("uniquepid")
    patient = patient.groupby("uniquepid").first()
    diagnostics = {}
    for col in ["gender", "ethnicity", "unittype", "hospitaldischargestatus", "unitdischargeoffset", "hospitalid"]:
        left, right = patient[col], physical[col].reindex(patient.index)
        changed = ~((left == right) | (left.isna() & right.isna()))
        diagnostics[col] = event_summary(changed.sum(), len(patient))
    patient = patient.reset_index()
    flow.append({"stage": "groupby_first_per_patient", "count": count(len(patient))})
    patient["hospital_mortality"] = patient.hospitaldischargestatus.str.lower().str.strip().eq("expired").astype(int)
    patient["los_days"] = (patient.unitdischargeoffset / 1440.0).clip(lower=0)
    patient["gender"] = patient.gender.str.upper().str.strip()
    if binary_gender:
        patient = patient[patient.gender.isin(config.ALLOWED_GENDERS)].copy()
    flow.append({"stage": "gender_eligibility", "count": count(len(patient))})
    eth = patient.ethnicity.fillna("Unknown").str.strip()
    patient["ethnicity_grouped"] = "OTHER"
    for regex, label in [("Caucasian", "WHITE"), ("African American", "BLACK"),
                         ("Hispanic", "HISPANIC"), (r"\bAsian\b", "ASIAN")]:
        patient.loc[eth.str.contains(regex, case=False, na=False), "ethnicity_grouped"] = label
    unit = patient.unittype.fillna("").str.strip()
    patient["admission_type_grouped"] = "OTHER"
    for regex, label in [("MICU|Med-Surg|medical", "MEDICAL"), ("SICU|surgical", "SURGICAL"),
                         ("CCU|Cardiac|CTICU|CSICU", "CARDIAC")]:
        patient.loc[unit.str.contains(regex, case=False, na=False), "admission_type_grouped"] = label
    apache_mean = apache.groupby("patientunitstayid").apachescore.mean().rename("apache_score")
    patient = patient.merge(apache_mean, on="patientunitstayid", how="left", validate="one_to_one")
    patient = patient.merge(hospital[["hospitalid", "region"]], on="hospitalid", how="left", validate="many_to_one")
    patient["hospital_region"] = patient.region.fillna("Unknown")
    patient = patient.merge(vitals, on="patientunitstayid", how="left", validate="one_to_one")
    patient = patient.merge(labs, on="patientunitstayid", how="left", validate="one_to_one")
    # These sources are absent in this audit inventory. Do not quietly ignore
    # them if supplied later: their resolution would require a new audit.
    extra = ["vitals_aperiodic_agg.csv", "nursecharting_temp_agg.csv"]
    if any((ROOT / "data" / name).exists() for name in extra):
        raise RuntimeError("Additional vital-source caches require an updated audit reconstruction")
    features = [f for f in config.ALL_FEATURES if f in patient]
    if features != config.ALL_FEATURES:
        raise RuntimeError("Feature schema differs from the audited configuration")
    cohort_raw = patient[features].copy()
    cohort_raw = cohort_raw.dropna(thresh=int(len(features) * 0.5))
    flow.append({"stage": "missingness_eligibility", "count": count(len(cohort_raw))})
    cohort = cohort_raw.copy()
    imputation = {}
    for col in config.CONTINUOUS_FEATURES:
        fill = float(cohort[col].median())
        imputation[col] = fill
        cohort[col] = cohort[col].fillna(fill)
    for col in config.CATEGORICAL_FEATURES + config.BINARY_FEATURES:
        if cohort[col].isna().any():
            cohort[col] = cohort[col].fillna(cohort[col].mode().iloc[0])
    # Reproduce pandas.sample's index selection without displaying records.
    if len(cohort) > config.MAX_COHORT_SIZE:
        positions = np.random.RandomState(config.RANDOM_SEED).choice(len(cohort), config.MAX_COHORT_SIZE, replace=False)
        cohort = cohort.iloc[positions]
    ids = patient.loc[cohort.index, "patientunitstayid"].to_numpy()
    before_fill = cohort_raw.loc[cohort.index].reset_index(drop=True)
    cohort = cohort.reset_index(drop=True)
    flow.append({"stage": "random_sample_seed_42", "count": count(len(cohort))})
    summary = {
        "flow": flow, "gender": category_counts(cohort.gender),
        "ethnicity": category_counts(cohort.ethnicity_grouped),
        "mortality": event_summary(cohort.hospital_mortality.sum(), len(cohort)),
        "groupby_first_changed_from_physical_row": diagnostics,
        "missing_outcome_treated_as_alive": count(patient.hospitaldischargestatus.isna().sum()),
        "negative_apache": event_summary((cohort.apache_score < 0).sum(), len(cohort)),
        "missingness": {c: event_summary(before_fill[c].isna().sum(), len(cohort)) for c in features},
        "feature_columns": features,
        "dtypes": {c: str(cohort[c].dtype) for c in features},
    }
    return {"frame": cohort, "before_fill": before_fill, "stay_ids": ids,
            "imputation_values": imputation, "summary": summary}


def main():
    started = time.perf_counter()
    data = ROOT / "data"
    patient = pd.read_csv(data / "patient.csv", usecols=["uniquepid", "patientunitstayid", "age", "gender",
        "ethnicity", "unittype", "hospitaldischargestatus", "unitdischargeoffset", "hospitalid"])
    apache = pd.read_csv(data / "apachePatientResult.csv", usecols=["patientunitstayid", "apachescore"])
    hospital = pd.read_csv(data / "hospital.csv", usecols=["hospitalid", "region"])
    vitals = pd.read_csv(data / "vitals_agg.csv")
    labs = pd.read_csv(data / "labs_agg.csv")
    variants = {name: reconstruct(flag, patient, apache, hospital, vitals, labs)
                for name, flag in [("unfiltered_candidate", False), ("binary_current", True)]}
    current = pd.read_csv(data / "cohort.csv")
    rebuilt = variants["binary_current"]["frame"]
    checks = {"same_shape": current.shape == rebuilt.shape,
              "same_columns": list(current) == list(rebuilt)}
    if checks["same_shape"] and checks["same_columns"]:
        numeric = config.CONTINUOUS_FEATURES + config.BINARY_FEATURES
        checks["numeric_equal_at_1e_minus_10"] = bool(np.allclose(current[numeric], rebuilt[numeric], rtol=0, atol=1e-10, equal_nan=True))
        checks["categorical_equal"] = bool(current[config.CATEGORICAL_FEATURES].equals(rebuilt[config.CATEGORICAL_FEATURES]))
    if not all(checks.values()):
        write_json(PUBLIC / "audit" / "reconstruction_check.json", checks)
        raise RuntimeError("Current cohort reconstruction mismatch; masks must not be used")
    rows = []
    for name, bundle in variants.items():
        df = bundle["frame"]
        train, test = train_test_split(df, test_size=0.2, stratify=df.hospital_mortality, random_state=42)
        bundle["train"], bundle["test"] = train.reset_index(drop=True), test.reset_index(drop=True)
        for label, frame in [("cohort", df), ("train", train), ("test", test)]:
            for gender, n in category_counts(frame.gender, ["MALE", "FEMALE", "UNKNOWN", "OTHER"]).items():
                rows.append({"variant": name, "dataset": label, "gender": gender, "count": n})
        with private_path(name + ".pkl").open("wb") as stream:
            pickle.dump(bundle, stream, protocol=4)
    for generator in ["GaussianCopula", "CTGAN", "TVAE"]:
        synthetic = pd.read_csv(data / ("synthetic_" + generator + ".csv"))
        for gender, n in category_counts(synthetic.gender, ["MALE", "FEMALE", "UNKNOWN", "OTHER"]).items():
            rows.append({"variant": "archived_synthetic_provenance_unverified", "dataset": generator,
                         "gender": gender, "count": n})
    report = {"current_reconstruction": checks,
              "candidate_status": "Unfiltered cohort reconstructed from current extraction rules with gender filtering disabled; provenance requires metric comparisons.",
              "variants": {k: v["summary"] for k, v in variants.items()},
              "wall_seconds": time.perf_counter() - started}
    write_json(PUBLIC / "audit" / "cohort_audit.json", report)
    write_csv(PUBLIC / "audit" / "gender_counts.csv", rows)
    print("Current cohort reconstruction:", checks)
    for name, bundle in variants.items():
        print(name, "gender", bundle["summary"]["gender"], "mortality", bundle["summary"]["mortality"])
    print("Aggregate cohort audit and suppressed gender counts written; patient intermediates are private.")


if __name__ == "__main__":
    main()
