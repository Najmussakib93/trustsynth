"""Generate the staged revision report exclusively from aggregate audit outputs.

This is a Tasks 0/1 report, not the completed multi-seed revision. Narrative
interpretation is explicit; every numeric table cell comes from audit artifacts.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .legacy import config
from .audit_baseline import GENERATORS, legacy_module, repeat_prediction_check
from .diagnose_vitals import bounds_from_code
from .safe_reporting import ROOT, PUBLIC, private_path, write_csv, write_json


def read_json(relative):
    return json.loads((PUBLIC / relative).read_text(encoding="utf-8"))


def number(value, digits=4):
    if value is None or pd.isna(value):
        return "pending"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def percent(value, digits=2):
    return value if isinstance(value, str) else f"{100 * value:.{digits}f}%"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def repeat_all():
    module, _ = legacy_module()
    results = {}
    for variant in ["unfiltered_candidate", "binary_current"]:
        with private_path(variant + ".pkl").open("rb") as stream:
            bundle = pickle.load(stream)
        results[variant] = {}
        for generator in GENERATORS:
            synthetic = pd.read_csv(ROOT / "data" / ("synthetic_" + generator + ".csv"))
            results[variant][generator] = repeat_prediction_check(module, bundle["train"], bundle["test"], synthetic)
    write_json(PUBLIC / "audit" / "prediction_repeat_checks.json", results)
    return results


def main():
    comparison = pd.read_csv(PUBLIC / "audit" / "baseline_comparison.csv")
    matched = comparison[comparison.variant == "unfiltered_candidate"]
    if not matched.status.eq("match_at_archived_precision").all():
        raise RuntimeError("Audit report requires an explicit accounting of unmatched legacy results")
    cohort = read_json("audit/cohort_audit.json")
    methods = read_json("audit/method_checks.json")
    raw = read_json("diagnostics/raw_vitals_audit.json")
    vital_summary = read_json("diagnostics/vitals_summary.json")
    ks = pd.read_csv(PUBLIC / "diagnostics/vitals_ks_decomposition.csv")
    inventory = read_json("audit/inventory.json")
    repeat_path = PUBLIC / "audit" / "prediction_repeat_checks.json"
    repeats = read_json("audit/prediction_repeat_checks.json") if repeat_path.exists() else repeat_all()
    manuscript = read_json("audit/manuscript_audit.json")
    manuscript_rows = pd.read_csv(PUBLIC / "audit" / "manuscript_comparison.csv")
    manuscript_table = table(["Location", "Generator", "Quantity", "Paper", "Recomputed", "Note"],
        [[r.location, r.generator, r.metric, r.paper_value, r.at_paper_precision, r.note]
         for r in manuscript_rows[manuscript_rows.status == "mismatch"].itertuples()])
    with private_path("legacy_metrics.pkl").open("rb") as stream:
        legacy = pickle.load(stream)["unfiltered_candidate"]
    old = json.loads((ROOT / "results" / "detailed_metrics.json").read_text(encoding="utf-8"))
    baseline_rows, metrics_csv = [], []
    for generator in GENERATORS:
        previous, new = old[generator], legacy[generator]
        row = [generator]
        for dimension, metric in [("fidelity", "auc_trtr"), ("fidelity", "auc_tstr"), ("privacy", "dcr_ratio")]:
            row.extend([number(previous[dimension][metric]), number(new[dimension][metric])])
        baseline_rows.append(row)
        for dimension, metric in [("fidelity", "auc_trtr"), ("fidelity", "auc_tstr"), ("privacy", "dcr_ratio"),
                                  ("privacy", "mia_accuracy"), ("utility", "ks_mean")]:
            metrics_csv.append({"generator": generator, "metric": dimension + "." + metric,
                                "archived": previous[dimension][metric], "recomputed": new[dimension][metric]})
    write_csv(PUBLIC / "audit" / "headline_baseline.csv", metrics_csv)
    baseline_table = table(["Generator", "Old TRTR", "Recomputed TRTR", "Old TSTR", "Recomputed TSTR", "Old DCR", "Recomputed DCR"], baseline_rows)
    flow = table(["Extraction stage", "Legacy reconstruction", "Current binary cohort"],
                 [[a["stage"], a["count"], b["count"]] for a, b in zip(
                     cohort["variants"]["unfiltered_candidate"]["flow"], cohort["variants"]["binary_current"]["flow"])])
    clip_table = table(["Raw field", "Lower clipping bound", "Upper clipping bound"],
                       [[name, *limits] for name, limits in bounds_from_code().items()])
    diagnostics = []
    for feature in ["sbp_mean", "dbp_mean", "temperature_mean"]:
        for generator in GENERATORS:
            row = ks[(ks.variant == "unfiltered_candidate") & (ks.feature == feature) & (ks.generator == generator)].iloc[0]
            diagnostics.append([feature, generator, percent(row.imputed_fraction), number(row.ks_all),
                                number(row.ks_observed_only_diagnostic), number(row.atom_KS_lower_bound)])
    diagnostic_table = table(["Feature", "Generator", "Imputed share", "KS, all real values", "KS, observed real only", "Imputation-atom KS lower bound"], diagnostics)
    integer_table = table(["Dataset", "SBP integer share", "DBP integer share", "Temperature integer share"],
        [[dataset] + [percent(next(s for s in vital_summary if s["variant"] == "unfiltered_candidate" and
            s["dataset"] == dataset and s["feature"] == feature)["integer_fraction"])
          for feature in ["sbp_mean", "dbp_mean", "temperature_mean"]]
         for dataset in ["real_all", "real_observed"] + GENERATORS])
    legacy_raw = raw["unfiltered_candidate"]
    unit_stats = legacy_raw["temperature_Fahrenheit_like_77_to_113"]
    negative_apache = cohort["variants"]["binary_current"]["negative_apache"]
    first = methods["variants"]["unfiltered_candidate"]
    group_first = cohort["variants"]["binary_current"]["groupby_first_changed_from_physical_row"]
    # Legacy fairness components are recomputed from the very same rounded
    # metrics as the historical composite; no hypothetical corrected score.
    ct = legacy["CTGAN"]["fairness"]
    rep_unclipped = 1 - np.mean([abs(r - 1) for r in ct["representation_ratios"].values()])
    bar_unclipped = 1 - np.mean([abs(v - 1) for k, v in ct.items() if k.startswith("BAR_")])
    write_json(PUBLIC / "audit" / "legacy_ctgan_fairness_components.json", {
        "representation_component_before_floor": float(rep_unclipped),
        "BAR_component_before_floor": float(bar_unclipped),
        "representation_component_after_floor": float(max(0, rep_unclipped)),
        "BAR_component_after_floor": float(max(0, bar_unclipped)),
        "saved_score": ct["fairness_score"],
        "rule": "arithmetic mean of each component floored at zero; legacy BAR uses first two observed groups",
    })
    all_repeats = all(v for variant in repeats.values() for generator in variant.values() for v in generator.values())
    versions_table = table(["Package", "Version during audit"], list(inventory["versions"].items()))
    report = f"""# TrustSynth revision report — Tasks 0 and 1

Status: baseline metric reproduction and vital-sign diagnosis completed for the available local artifacts. Preprocessing changes await the author's approval. Tasks 2–11, generator retraining, and final multi-seed conclusions remain pending. The latest located editable manuscript is `{manuscript['source']}`; it contains the empirical tables and binary-gender contradiction also found in `C:/Users/Acer/Downloads/Trustsynth (3).pdf`. The manuscript was inspected read-only.

## Reproduction scope and old versus recomputed values

Every scalar metric in the archived `results/detailed_metrics.json` agrees with the reconstruction at its stored precision. The comparison file covers {len(matched)} scalar entries for the unfiltered reconstruction; the full old-versus-recomputed listing is [baseline_comparison.csv](audit/baseline_comparison.csv). This reproduces evaluation of the saved synthetic sets, not their original stochastic training. The original eICU extraction snapshot and training logs were not available; Git HEAD contains a MIMIC implementation. Matching all archived metrics provides strong consistency evidence for the reconstructed unfiltered cohort but is not a substitute for the missing provenance.

{baseline_table}

### Manuscript tables and prose

Values were extracted programmatically from the located DOCX tables and empirical prose, then compared against computed results in [manuscript_comparison.csv](audit/manuscript_comparison.csv). Table I and the representation-ratio/BAR values in Table II are consistent within rounding precision. Some archived values lie exactly at rounding midpoints, and the manuscript does not use a uniform midpoint display convention; these cases are explicitly flagged as `rounding_boundary` with the archived values preserved. The substantive table discrepancies are below; the small violation tally is suppressed but the affected ethnicity groups are listed.

{manuscript_table}

The sampled legacy cohort contains {methods['variants']['unfiltered_candidate']['distinct_hospitals']} hospitals; the current binary cohort contains {methods['variants']['binary_current']['distinct_hospitals']}. Database-wide coverage must not be substituted for sampled-cohort coverage. The prose also calls TVAE the worst-fairness generator, although the table's CTGAN composite score is lower. Mechanism and subgroup-harm claims remain hypotheses until their requested experiments. This checkpoint maps the principal empirical tables and cohort/results quantities; the complete final figure/normalization inventory remains part of Task 11.

The reconstructed TRTR AUROC rounds to {legacy['GaussianCopula']['fidelity']['auc_trtr']:.3f}, matching the value named in the request. The current binary cohort's TRTR and its comparisons against archived synthetics are also in the comparison file, explicitly labeled `binary_current`. Those TSTR comparisons are compatibility diagnostics: archived generators were not retrained on the new split. They must not be reported as corrected generator performance.

Independent same-seed XGBoost fits produced bitwise-identical prediction arrays for TRTR and every saved-set TSTR model in both reconstruction variants: **{all_repeats}**. The seed was {config.RANDOM_SEED}. Generator training reproducibility for SDV/synthcity is still pending; no generator was trained in this audit. Predictor checks are in [prediction_repeat_checks.json](audit/prediction_repeat_checks.json).

## Task 0: cohort and methods audit

The current extractor applies the binary-gender filter before subsampling and training, and the current evaluator rejects incompatible real cohorts. The archived synthetic sets and reconstructed legacy cohort retain categories excluded by that filter, so those artifacts cannot be treated as outputs of the corrected protocol. Gender value counts for each cohort, train/test split and archived synthetic set are in [gender_counts.csv](audit/gender_counts.csv), with small cells suppressed. Corrected synthetic sets will be generated after preprocessing is approved.

{flow}

Adults are selected using age >= {config.MIN_AGE}; the censored age value above 89 is encoded as 90. The pipeline sorts by `patientunitstayid`, calls `groupby('uniquepid').first()`, derives mortality from `hospitaldischargestatus == 'expired'`, and sets ICU LOS to discharge offset divided by minutes per day, floored at zero. Ethnicity is grouped as White, Black, Hispanic, Asian, or Other; missing ethnicity is included in Other. Unit type is mapped to Medical, Surgical, Cardiac, or Other, and missing hospital region becomes Unknown. These are unit-type groups, not directly observed admission types.

The final cohort is a seeded random sample of eligible records, not the first records in file order. Continuous variables are median-filled and categorical variables mode-filled over the eligible pool **before** sampling and splitting, so held-out information influences preprocessing. The missingness filter uses `int(number_of_features * 0.5)` nonmissing fields, which is not identical to rejecting all records with more than half their fields missing when the feature count is odd. Neither vital nor laboratory extraction filters measurement offsets: both aggregate across the available stay, including measurements outside the first day. Labs combine glucose and bedside glucose and take arithmetic means; the current lab loader applies no clipping bounds. APACHE scores are averaged across available result rows, without selecting an APACHE version or excluding negative values.

The raw-vitals audit found {percent(legacy_raw['after_24h']['fraction'])} of selected vital rows after the first day and {percent(legacy_raw['before_admission']['fraction'], 3)} before the ICU-admission offset. Cached per-stay vital means and missingness matched the full raw scan within the prespecified numerical tolerance. Vital readings are clipped individually before the per-stay arithmetic mean, using the bounds below. These facts describe the actual implementation; an early-prediction interpretation is unsupported while discharge LOS and all-stay measurements remain predictors.

{clip_table}

The monitor source records five-minute median summaries and identifies the blood-pressure fields as invasive measurements; its temperature field is documented as Celsius. Non-invasive cuff pressure is recorded separately. See the official [vitalPeriodic documentation](https://eicu.mit.edu/eicutables/vitalperiodic/) and [vitalAperiodic documentation](https://eicu.mit.edu/eicutables/vitalaperiodic/).

### Distance and attack definitions

The reproduced DCR is Gower distance over the current feature schema. Categorical fields contribute a mismatch indicator; numeric differences are scaled using extrema of the combined query sample and reference set, then all feature contributions are averaged. The installed package computes extrema in float32 and uses a two-stage normalization, which reduces algebraically to range scaling for positive maxima. The audit uses the existing fixed-constant implementation after verifying exact agreement with `gower=={inventory['versions']['gower']}` on invented mixed-type data. Its per-query arrays remain private.

The legacy ratio divides the rounded median synthetic-to-train distance by the rounded median heldout-to-train distance, then rounds the ratio again. It does not divide by a fifth percentile, despite a misleading configuration comment. The archived numbers match this Gower route exactly. The original evaluator also permits a Euclidean fallback; absent historical logs we cannot prove which branch originally ran. Current evaluation's naive parallel Gower calls are not generally equivalent because each chunk can infer a different range; a constructed-data regression test demonstrates this. The audit bypasses that defect without changing the production evaluator.

The membership attack is separate: a logistic regression uses the minimum, mean, tenth percentile and standard deviation of standardized Euclidean distances to a subset of synthetic records, and reports cross-validated accuracy. It is not a maximum-ROC-gap attack, so the requested analytic KS null does not apply to this legacy statistic. Permutation null calibration remains Task 7.

### Additional problems confirmed

- Selecting the lowest stay identifier does not establish temporal order. In the reconstructed legacy sample, {first['later_than_another_stay_in_same_hospitalization']['count']} selected stays ({percent(first['later_than_another_stay_in_same_hospitalization']['fraction'])}) have an earlier stay within the same hospitalization. The official [patient documentation](https://github.com/MIT-LCP/eicu-code/blob/main/website/content/eicutables/patient.md) explicitly warns that stay identifiers do not encode chronology and that admissions within a year cannot always be ordered.
- `groupby().first()` can borrow nonmissing fields from later rows: in the eligible adult patient pool, hospital discharge status differs from the first physical row for {group_first['hospitaldischargestatus']['count']} patients and ethnicity for {group_first['ethnicity']['count']}. This is a data-selection defect, separate from the temporal-order problem.
- The current binary cohort contains {negative_apache['count']} negative APACHE values ({percent(negative_apache['fraction'])}). These conflict with the nonnegative constraint and need explicit invalid/missing-value handling. The APACHE result table contains both IV and IVa rows; the audit does not establish that their scores differ.
- Some original integrity checks fail on real training data. Their aggregate CSRs are in [original_constraints_real_baseline.csv](audit/original_constraints_real_baseline.csv). A strict temperature upper bound also conflicts with inclusive clipping at that boundary. Installed SDV constructors default to min/max enforcement, but that does not guarantee every strict clinical bound if the real training data violate it.
- Both legacy CTGAN fairness components are floored at zero: the pre-floor representation component is {number(rep_unclipped)} and the BAR component is {number(bar_unclipped)}. Thus its exact zero score is a clipping result, not a direct measurement of complete unfairness. The binary-gender correction can materially change that finding; no revised score is claimed before retraining.
- `README.md` still describes MIMIC. The configured remote is `{inventory['git_remote']}`, which differs from the URL in the request. This checks local repository configuration, not remote ownership or accessibility.

## Task 1: high KS on vital signs

Median imputation is a major cause of the high marginal KS values. It replaces most absent blood-pressure and temperature measurements with one exact value, creating large jumps in the real empirical CDF. GaussianCopula's discrepancy is much smaller when only genuinely observed values are used as the diagnostic reference. CTGAN and TVAE retain substantial observed-value discrepancies, so the results do not support attributing every generator's error entirely to imputation.

{diagnostic_table}

The observed-only comparison uses the saved pre-imputation mask, verified through reconstruction; it does not assume every value equal to the median was imputed. It changes the reference population and is therefore a diagnostic, not a silently substituted headline metric. The atom lower bound is half the positive difference between real and synthetic point mass at the imputation value. Quantiles, means, standard deviations, integer shares, and boundary shares are generated in [vitals_summary.json](diagnostics/vitals_summary.json).

{integer_table}

The per-stay means are generally non-integer, so integer raw monitor recordings alone do not explain the dominant KS effect. Fractions that would reveal small event or complement counts are suppressed. Histograms use common bins across datasets and suppress small bin counts, including secondary suppression where needed.

The raw temperature audit detected {unit_stats['count']} readings in the Fahrenheit-like interval ({percent(unit_stats['fraction'])} of observed temperature readings). In total, {percent(legacy_raw['features']['temperature_mean']['outside_extraction_bounds']['fraction'])} of temperature readings fall outside the extractor's bounds. The current clip operation turns high values into the upper Celsius bound before averaging. Since the source is documented as Celsius and has no per-reading unit field, this is evidence consistent with unit contamination or other invalid measurements, not proof that each high value is Fahrenheit; automatic conversion would be unjustified. Clipping affects a much smaller share of BP readings than the missingness rate and cannot explain its large imputation mass.

Neither `vitalAperiodic.csv` nor `nurseCharting.csv` is available in the project. A recursive filename search of accessible C, D, E and F drive paths found no matching CSV or compressed filenames; inaccessible folders and archives with unrelated names are not ruled out. The current code contains optional loaders for them, but their absence means the advertised multi-source fix did not contribute to the current cohort. The existing diagnostic also pointed to a vanished scratch directory and inferred missingness from median equality; the new audit avoids both assumptions.

## Proposed preprocessing decision — approval required

The concrete proposal is in [PREPROCESSING_PROPOSAL.md](PREPROCESSING_PROPOSAL.md). No production preprocessing, original cohort, cached source aggregates, saved synthetics, or archived result files were changed. Approving this proposal would authorize a corrected extraction and split-specific preprocessing before the multi-seed experiments.

## Original findings: status at this checkpoint

The archived evidence for minority erasure, demographic inflation, demographic preservation, dimension trade-offs, and the low TVAE DCR has been reproduced under the legacy evaluation. Their persistence after correcting the cohort and preprocessing remains untested. No mechanism reruns have been performed. The uniformly-high-integrity interpretation is already qualified by real-data violations and automatic range enforcement; the later structural analysis must establish which checks are valid and discriminative.

## Remaining tasks

Tasks 2–11 remain pending in the requested order. This includes terminology migration; ten-seed SDV, DDPM and DP generation; mixed-type associations; AUPRC, ECE and subgroup utility; fairness bootstraps and mechanism checks; calibrated attacks and Anonymeter; approved structural constraints; TrustScore sensitivity; paired inference; and the final supplementary material and pseudocode. The current audit files are not a completed journal revision. The located manuscript has been mapped for the baseline comparison; every final table, figure and empirical claim will be reconciled again after the revised experiments.

## Artifacts, commands and verification

- [Complete scalar comparison](audit/baseline_comparison.csv)
- [Manuscript table and prose comparison](audit/manuscript_comparison.csv)
- [Suppressed gender counts](audit/gender_counts.csv)
- [Cohort flow and missingness](audit/cohort_audit.json)
- [Source and environment inventory](audit/inventory.json)
- [Cohort-method checks and installed SDV defaults](audit/method_checks.json)
- [Privacy definition and runtime](audit/privacy_audit.json)
- [Per-feature KS diagnosis](diagnostics/vitals_ks_decomposition.csv)
- [Full raw-vitals audit](diagnostics/raw_vitals_audit.json)
- [Legacy histogram PNG](figures/vitals_unfiltered_candidate.png) and [TIFF](figures/vitals_unfiltered_candidate.tiff)
- [Current-cohort histogram PNG](figures/vitals_binary_current.png) and [TIFF](figures/vitals_binary_current.tiff)

All histograms are exported at the requested resolution, recorded in the figure manifest. Row-level reconstructions, masks, identifiers and raw-distance intermediates are restricted to ignored `results/private/revision/`. Small counts are shown as `<11`; `suppressed` marks a secondary suppressed cell and must not be interpreted as a small count. Safe-reporting and distance tests use invented records only. The audit command sequence and installed dependency snapshot are in [audit/README.md](audit/README.md).

{versions_table}
"""
    (PUBLIC / "REVISION_REPORT.md").write_text(report, encoding="utf-8")
    proposal = """# Preprocessing proposal for author approval

The audit reproduced the saved metrics, then identified problems in the underlying cohort. The following changes are proposed; none has been applied to production extraction or training.

1. **Use one intact stay record per patient.** Replace `groupby().first()` with selection of an actual row, preserving the existing deterministic minimum-stay-ID rule for continuity. Describe it accurately as a selected stay, not a chronological first stay. If chronological first-stay selection is essential, it requires a separate cohort definition: eICU supports ordering within hospitalization but not always between hospitalizations within a year.
2. **Retain the binary-gender restriction and require an observed mortality label.** Exclude records with unavailable or unrecognized hospital discharge status instead of treating them as survivors. Apply the exact more-than-half-missing rule rather than its current integer-floor approximation. Convert negative APACHE scores to missing and select the specified APACHE version explicitly; report eligibility and missingness changes.
3. **Retain missingness through synthesis.** Store unfilled features and per-feature masks; split before fitting imputation or encoding. When a generator needs filled inputs, use a reversible value-plus-missingness-indicator representation learned from its training split and restore missing values before evaluation. Fit downstream model imputation on that model's training data only. Report missingness fidelity and observed-value fidelity separately; retain the legacy median-filled analysis as an explicit comparison. Do not add random jitter to make KS smaller.
4. **Handle implausible readings as missing before aggregation.** Use the documented bounds initially as screening bounds, with explicit inclusive/exclusive rules, instead of moving invalid readings onto the endpoints. Convert Fahrenheit only when the source explicitly identifies the unit; do not reinterpret high values in a Celsius field by guesswork. Log aggregate exclusions and the resulting coverage.
5. **Use the existing additional-source loaders when credentialed source files are available.** Add non-invasive cuff BP from `vitalAperiodic` and unit-labelled temperature from `nurseCharting`, preserving source provenance and reporting how much each source contributes. Both files are absent locally. If unavailable, use the available periodic data with explicit missingness and report that limitation; do not claim source coverage has improved.
6. **Keep the primary evaluation a retrospective all-stay benchmark.** Document the all-stay arithmetic means and the inclusion of discharge LOS. Do not describe the mortality model as an admission-time or first-day prediction model. A first-day prediction analysis would require an explicit prediction time, eligibility definition, offset restriction and removal of future features; it should be a separately agreed protocol, not an implicit change.

The source bounds above are provisional preprocessing rules, not validated clinical constraints. Constraint changes remain subject to the separate Task 8 proposal and approval. All downstream generators must be refitted; the existing synthetic sets are baseline artifacts and cannot be reused as corrected results. Tables and conclusions may change, including the CTGAN zero fairness score, subgroup representation claims and TVAE privacy signal.

Please approve this proposal or identify changes to it. The manuscript has been located and the optional source filenames were not found on accessible local drives. The proposed missingness-preserving fallback can run on the available periodic source without those optional files.
"""
    (PUBLIC / "PREPROCESSING_PROPOSAL.md").write_text(proposal, encoding="utf-8")
    commands = """# Tasks 0/1 audit commands

Run from the repository root in PowerShell. These scripts read credentialed files locally and emit aggregates only. They do not execute the original pipeline entry points or retrain generators.

```powershell
$env:OMP_NUM_THREADS = '4'
python -m revision.inventory
python -m revision.audit_cohort
python -m revision.audit_baseline quick
python -m revision.audit_baseline benchmark
# Review the runtime estimate before the longer calculation.
python -m revision.audit_baseline privacy
python -m revision.diagnose_vitals cached
python -m revision.diagnose_vitals pilot
# Review the runtime estimate before the full source scan.
python -m revision.diagnose_vitals raw
python -m revision.audit_methods
python -m revision.audit_manuscript 'F:\\NIW\\TrustSynth_editable.docx'
python -m unittest revision.test_audit -v
python -m revision.build_audit_report
```

`audit_cohort` requires the existing `vitals_agg.csv` and `labs_agg.csv` plus patient, hospital and APACHE sources under `data/`. It validates its reconstruction against the current cohort before trusting imputation masks. `diagnose_vitals raw` independently verifies the vital cache against the full raw vitalPeriodic source. No source-file previews are permitted.

The environment snapshot records what was installed during reproduction; it is not yet a clean dependency lock for the future generator environments. Generator RNG and multi-seed reproducibility checks remain pending. Public counts below the disclosure threshold are suppressed; secondary suppression and suppression of revealing rates are applied where applicable. This audit does not grant permission to publish raw data or fitted models.

The manuscript comparison uses the latest located editable version and records its path and hash. Main empirical table cells and mapped prose are extracted from the document; no table values are transcribed by hand. The full final figure and normalization inventory remains part of the later revision tasks.
"""
    (PUBLIC / "audit" / "README.md").write_text(commands, encoding="utf-8")
    (PUBLIC / "audit" / "requirements-observed.txt").write_text(
        "# Observed package versions; not a tested independent environment lock.\n" +
        "\n".join(f"{k}=={v}" for k, v in inventory["versions"].items() if v != "not installed") + "\n", encoding="utf-8")
    log = ROOT / "CHANGES.md"
    previous = log.read_text(encoding="utf-8-sig")
    archived = private_path("CHANGES_before_audit.md")
    if not archived.exists():
        archived.write_text(previous, encoding="utf-8")
    # Replace a historical paragraph that printed small credentialed cells.
    start, end = previous.find("### 1.1 Binary-gender filter added"), previous.find("\n## 2. Task 3")
    if start >= 0 and end > start:
        replacement = """### 1.1 Binary-gender filter added (historical entry; disclosure-safe summary)
The earlier implementation omitted the binary-gender restriction. The current
extractor filters before sampling and the evaluator rejects disallowed real
categories. The saved synthetic sets predate that correction. Historical
small-cell counts in this entry have been replaced by the code-generated,
suppressed audit in `results/revision/audit/gender_counts.csv`; the original
log is retained only under ignored `results/private/revision/`.
The impact on the revised generator results remains untested until retraining.

---
"""
        previous = previous[:start] + replacement + previous[end:]
    marker = "\n## 2026-11-02 — Tasks 0/1 reproducibility and diagnostic audit"
    if marker in previous:
        previous = previous.split(marker)[0]
    entry = """
## 2026-11-02 — Tasks 0/1 reproducibility and diagnostic audit

- Added aggregate-only reconstruction, baseline evaluation, raw-vitals diagnostics,
  disclosure controls and report-generation code under `revision/`.
- Added `results/private/` and separate-environment patterns to `.gitignore`.
- Reproduced all archived scalar metrics using reconstructed legacy inputs and
  saved synthetic sets; this is evaluation reproduction, not generator retraining.
- Verified independent predictor reruns before metric rounding and verified the
  fixed-constant Gower implementation against its installed package using invented
  mixed-type data. Added targeted disclosure and distance tests.
- Independently checked the vital cache against the raw source and diagnosed
  median-imputation mass, possible temperature contamination, whole-stay aggregation,
  record-selection defects and invalid APACHE values. All findings and numbers are
  generated in `results/revision/REVISION_REPORT.md` and linked aggregate artifacts.
- Kept the manuscript and production preprocessing unchanged. The proposed changes
  in `results/revision/PREPROCESSING_PROPOSAL.md` await explicit author approval.
- Current new audit prose uses fidelity for distributions and utility for predictive
  performance. Legacy metric keys remain unchanged for exact comparison; the full
  Task 2 migration is pending, with an explicit mapping to be logged then.
- Replaced a historical unsuppressed small-cell paragraph with a suppressed audit
  reference; preserved its original only in ignored private storage.

No patient-level file was printed, no credentialed data was transmitted, and no
commit or push was made. No values were manually adjusted to preserve findings.
"""
    log.write_text(previous.rstrip() + "\n" + entry, encoding="utf-8")
    print("Generated staged REVISION_REPORT.md, PREPROCESSING_PROPOSAL.md and audit commands.")


if __name__ == "__main__":
    main()
