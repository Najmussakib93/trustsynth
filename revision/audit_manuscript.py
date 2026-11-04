"""Extract manuscript table values from DOCX and compare with computed results.

The manuscript is read-only. No reported numeric cell is transcribed by hand.
Experimental/check counts are suppressed in printed tables when below 11.
"""
import argparse
import hashlib
import json
import pickle
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from docx import Document
import numpy as np

from .legacy import config
from .safe_reporting import ROOT, PUBLIC, count, private_path, write_csv, write_json

GENERATORS = ["GaussianCopula", "CTGAN", "TVAE"]
TABLE1_METRICS = [("utility", "utility_score"), ("privacy", "privacy_score"),
    ("fairness", "fairness_score"), ("fidelity", "fidelity_score"), ("integrity", "integrity_score"),
    ("utility", "ks_mean"), ("privacy", "mia_accuracy"), ("fidelity", "tstr_gap"), ("integrity", "csr")]


def main(path):
    source = Path(path)
    document = Document(source)
    paragraphs = [p.text.replace("TV AE", "TVAE") for p in document.paragraphs]
    with private_path("legacy_metrics.pkl").open("rb") as stream:
        results = pickle.load(stream)["unfiltered_candidate"]
    with private_path("unfiltered_candidate.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    methods = json.loads((PUBLIC / "audit" / "method_checks.json").read_text())
    records = []

    def add(location, generator, metric, stated, calculated, is_count=False, note=""):
        precision = len(stated.split(".")[1]) if "." in stated else 0
        displayed = Decimal(str(calculated)).quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP)
        half_unit = Decimal(5).scaleb(-precision - 1)
        difference = abs(Decimal(str(calculated)) - Decimal(stated))
        boundary = not is_count and difference == half_unit
        same = difference < half_unit or difference == 0
        records.append({"location": location, "generator": generator, "metric": metric,
                        "paper_value": count(float(stated)) if is_count else stated,
                        "recomputed_legacy": count(calculated) if is_count else calculated,
                        "at_paper_precision": count(calculated) if is_count else str(displayed),
                        "status": "rounding_boundary" if boundary else "match" if same else "mismatch",
                        "note": (note + "; archived value is exactly at the manuscript rounding midpoint; presentation convention requires consistency") if boundary else note})

    for row in document.tables[0].rows:
        cells = [c.text.strip().replace("TV AE", "TVAE") for c in row.cells]
        if cells[0] not in GENERATORS:
            continue
        if len(cells) != len(TABLE1_METRICS) + 1:
            raise RuntimeError("Unexpected Table I schema")
        for stated, (dimension, metric) in zip(cells[1:], TABLE1_METRICS):
            add("Table I", cells[0], dimension + "." + metric, stated, results[cells[0]][dimension][metric])
    for row in document.tables[1].rows:
        cells = [c.text.strip().replace("TV AE", "TVAE") for c in row.cells]
        group = cells[0]
        if group in ["White", "Black", "Hispanic", "Asian", "Other"]:
            metric = "ethnicity_grouped_" + group.upper()
            for generator, stated in zip(GENERATORS, cells[1:]):
                add("Table II", generator, "representation_ratio." + group, stated,
                    results[generator]["fairness"]["representation_ratios"][metric])
        elif group.startswith("Violations"):
            for generator, stated in zip(GENERATORS, cells[1:]):
                ratios = {k: v for k, v in results[generator]["fairness"]["representation_ratios"].items()
                          if k.startswith("ethnicity_grouped_")}
                failed = [k.replace("ethnicity_grouped_", "") for k, value in ratios.items()
                          if value < config.REPRESENTATION_THRESHOLD_LOW or value > config.REPRESENTATION_THRESHOLD_HIGH]
                add("Table II", generator, "ethnicity_violations", stated, len(failed), is_count=True,
                    note="Violated groups: " + ", ".join(failed))
        elif group == "Ethnicity BAR":
            for generator, stated in zip(GENERATORS, cells[1:]):
                add("Table II", generator, "BAR_ethnicity_grouped", stated,
                    results[generator]["fairness"]["BAR_ethnicity_grouped"],
                    note="Legacy estimator reproduced; its empty-group handling remains invalid")
    cohort_text = next(p for p in paragraphs if "The mortality rate was" in p)
    frame = bundle["frame"]
    stated = re.search(r"mortality rate was\s+(\d+(?:\.\d+)?)%", cohort_text).group(1)
    add("Cohort methods", "real", "mortality_percent", stated, 100 * frame.hospital_mortality.mean())
    for group in ["White", "Black", "Other", "Hispanic", "Asian"]:
        stated = re.search(group + r"\s+(\d+(?:\.\d+)?)%", cohort_text).group(1)
        add("Cohort methods", "real", group + "_percent", stated, 100 * frame.ethnicity_grouped.eq(group.upper()).mean())
    stated = re.search(r"final cohort\s+contained\s+([\d,]+)", cohort_text).group(1).replace(",", "")
    add("Cohort methods", "real", "patients", stated, len(frame), is_count=True)
    stated = re.search(r"comprised\s+(\d+)\s+features", cohort_text).group(1)
    add("Cohort methods", "real", "features", stated, len(frame.columns), is_count=True)
    caption = next(p for p in paragraphs if p.startswith("TABLE I."))
    stated = re.search(r"\((\d+)\s*Hospitals\)", caption, re.I).group(1)
    add("Table I caption and conclusions", "real", "sample_hospitals", stated,
        methods["variants"]["unfiltered_candidate"]["distinct_hospitals"], is_count=True,
        note="Manuscript uses full database hospital count as though it were the sampled cohort count")
    utility_text = next(p for p in paragraphs if "We trained XGBoost" in p)
    stated = re.search(r"baseline\s+AUC\s+of\s+(\d+\.\d+)", utility_text).group(1)
    add("Utility prose", "TRTR", "AUROC", stated, results["GaussianCopula"]["fidelity"]["auc_trtr"])
    for generator in GENERATORS:
        if generator == "TVAE":
            expression = r"TVAE achieved.*?AUC of (\d+\.\d+).*?gap of (\d+\.\d+)"
        else:
            expression = generator + r" at (\d+\.\d+).*?gap of\s+(\d+\.\d+)"
        match = re.search(expression, utility_text)
        if not match:
            raise RuntimeError("Unable to map a TSTR prose claim")
        add("Utility prose", generator, "AUROC", match.group(1), results[generator]["fidelity"]["auc_tstr"])
        add("Utility prose", generator, "TSTR_gap", match.group(2), results[generator]["fidelity"]["tstr_gap"])
    fidelity_text = next(p for p in paragraphs if "Its correlation deviation" in p)
    stated = re.search(r"lowest\s*\((\d+\.\d+)\)", fidelity_text).group(1)
    add("Fidelity prose", "GaussianCopula", "correlation_deviation", stated, results["GaussianCopula"]["utility"]["corr_deviation"])
    fairness_text = next(p for p in paragraphs if "Fairness evaluation produced" in p)
    # Match values by the associated subgroup rather than by their numeric value.
    for group in ["Asian", "Hispanic", "Other"]:
        quoted = r"[“\"]?" + group + r"[”\"]?"
        match = re.search(quoted + r"(?: patients)?(?: were)?(?: inflated)?\s+to\s+(\d+\.\d+)×", fairness_text)
        if match:
            add("Fairness prose", "CTGAN", "representation_ratio." + group, match.group(1),
                results["CTGAN"]["fairness"]["representation_ratios"]["ethnicity_grouped_" + group.upper()])
    # Source values for BAR prose are extracted by generator-specific sentences.
    for generator, expression in [("TVAE", r"ethnicity-based bias amplification ratio reached\s+(\d+\.\d+)×"),
                                  ("CTGAN", r"CTGAN[’']s ethnicity-based bias amplification ratio reached\s+(\d+\.\d+)×")]:
        match = re.search(expression, fairness_text)
        if match:
            add("Fairness prose", generator, "BAR_ethnicity_grouped", match.group(1), results[generator]["fairness"]["BAR_ethnicity_grouped"])
    write_csv(PUBLIC / "audit" / "manuscript_comparison.csv", records)
    consistency = {
        "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "working_source_basis": "Most recent located editable manuscript; same empirical tables and binary-gender contradiction as Downloads/Trustsynth (3).pdf",
        "mapped_empirical_cells_and_claims": len(records),
        "mismatches": [r for r in records if r["status"] == "mismatch"],
        "additional_contradictions": [
            "Binary-gender cohort claimed, but the reproduced evaluation used the unfiltered cohort.",
            "First chronological stay claimed, but identifier order does not establish chronology.",
            "The text calls TVAE the worst-fairness generator although CTGAN has the lower composite score.",
            "The metric code averages per-constraint satisfaction, rather than requiring every constraint simultaneously per row.",
            "Mechanistic and subgroup-harm claims are not tested by the single-run representation ratios.",
        ],
        "scope": "Main empirical tables and mapped cohort/results claims. Bibliographic identifiers, figure tick marks and general framework constants are not experimental findings; final complete figure and normalization inventory remains Task 11.",
        "source_search": "No vitalAperiodic or nurseCharting filenames found by recursive search of accessible C/D/E/F paths, including compressed filenames with these prefixes; inaccessible directories and archives with unrelated names are not ruled out.",
    }
    write_json(PUBLIC / "audit" / "manuscript_audit.json", consistency)
    print("Manuscript numerical comparisons saved. Mismatched claims:")
    for row in consistency["mismatches"]:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    main(parser.parse_args().path)
