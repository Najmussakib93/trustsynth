"""Further aggregate cohort-method checks; does not alter selection or fields."""
import inspect
import json
import pickle

import numpy as np
import pandas as pd

from .legacy import config
from .audit_baseline import legacy_module
from .safe_reporting import (ROOT, PUBLIC, category_counts, count, event_summary,
                             private_path, write_csv, write_json)


def main():
    data = ROOT / "data"
    patient = pd.read_csv(data / "patient.csv", usecols=["patientunitstayid", "patienthealthsystemstayid", "uniquepid",
        "unitvisitnumber", "hospitaladmitoffset", "hospitaldischargeyear", "hospitaldischargestatus", "hospitalid"])
    apache = pd.read_csv(data / "apachePatientResult.csv", usecols=["patientunitstayid", "apachescore", "apacheversion"])
    methods = {"apache_versions": category_counts(apache.apacheversion),
               "apache_scores_negative": event_summary((apache.apachescore < 0).sum(), len(apache))}
    multiple = apache.groupby("patientunitstayid").apachescore.agg(["count", "nunique"])
    methods["apache_stays_with_disagreeing_scores_across_rows"] = event_summary((multiple["nunique"] > 1).sum(), len(multiple))
    methods["variants"] = {}
    checks = []
    module, _ = legacy_module()
    for name in ["unfiltered_candidate", "binary_current"]:
        with private_path(name + ".pkl").open("rb") as stream:
            bundle = pickle.load(stream)
        selected = patient.set_index("patientunitstayid").loc[bundle["stay_ids"]]
        positive_visit = selected.unitvisitnumber > 1
        # Chronology is only definable within a hospitalization. Across admissions
        # in the same year eICU does not provide a full chronological ordering.
        earliest = patient.groupby("patienthealthsystemstayid").hospitaladmitoffset.max()
        later = selected.hospitaladmitoffset < selected.patienthealthsystemstayid.map(earliest)
        methods["variants"][name] = {
            "distinct_hospitals": count(selected.hospitalid.nunique()),
            "visit_number_greater_than_one": event_summary(positive_visit.sum(), len(selected)),
            "later_than_another_stay_in_same_hospitalization": event_summary(later.sum(), len(selected)),
            "physical_selected_row_missing_hospital_outcome": event_summary(selected.hospitaldischargestatus.isna().sum(), len(selected)),
            "discharge_years": category_counts(selected.hospitaldischargeyear),
        }
        train = bundle["train"]
        integrity = module.evaluate_integrity(train, "real_train")
        for constraint, csr in integrity["constraint_results"].items():
            checks.append({"variant": name, "dataset": "real_train_seed42", "constraint": constraint,
                           "csr": csr, "violated": bool(csr < 1)})
    # Read installed constructor defaults; no synthesizer is fitted.
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
    methods["installed_sdv_defaults"] = {}
    for cls in [GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer]:
        signature = inspect.signature(cls)
        methods["installed_sdv_defaults"][cls.__name__] = {
            key: signature.parameters[key].default
            for key in ["enforce_min_max_values", "enforce_rounding"] if key in signature.parameters}
    write_json(PUBLIC / "audit" / "method_checks.json", methods)
    write_csv(PUBLIC / "audit" / "original_constraints_real_baseline.csv", checks)
    print(json.dumps(methods, indent=2))
    print("Original constraints with real-training violations:")
    for row in checks:
        if row["violated"]:
            print(row)


if __name__ == "__main__":
    main()
