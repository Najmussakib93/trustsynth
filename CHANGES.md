# CHANGES.md — TrustSynth revision log

Revision pass for *Methods of Information in Medicine* submission.
Every entry records **what** changed, **why**, and **what it affects**.

No result number in any table or figure has been hand-edited. All numbers are
produced by the code in this repository.

---

## 0. Repository hygiene

### 0.1 `.gitignore` hardened (PhysioNet DUA)
**What:** Added patterns for `*cohort*.csv`, `*synthetic_*.csv`, `seed_*/`,
`runs/`, `scratch/`, model checkpoints (`*.pkl *.pt *.pth *.ckpt *.joblib *.h5`,
`models/`, `checkpoints/`) and `venv_dp/`.
**Why:** `data/` was already excluded, but the multi-seed refactor writes
per-seed synthetic datasets, and a fitted synthesizer is itself a
patient-derived artifact. The new patterns guard anything written outside
`data/`.
**Verified:** `git log --all --name-only` confirms no patient-level or derived
CSV has ever been committed. Only `results/dimension_scores.csv` (aggregate
metric values, no patient rows) is tracked.
**Affects:** nothing in the results.

### 0.2 `CHANGES.md` added
This file.

---

## 1. Task 1 — gender-filter contradiction

### 1.1 Binary-gender filter added (historical entry; disclosure-safe summary)
The earlier implementation omitted the binary-gender restriction. The current
extractor filters before sampling and the evaluator rejects disallowed real
categories. The saved synthetic sets predate that correction. Historical
small-cell counts in this entry have been replaced by the code-generated,
suppressed audit in `results/revision/audit/gender_counts.csv`; the original
log is retained only under ignored `results/private/revision/`.
The impact on the revised generator results remains untested until retraining.

---

## 2. Task 3 — high KS on SBP / DBP / temperature (DIAGNOSIS ONLY, no fix yet)

### 2.1 Diagnostic script added: `diagnostics/task3_vitals_diagnostic.py`
Read-only. Produces `figures/diagnostic_vitals_hist.{png,pdf}` and
`diagnostics/task3_ks_decomposition.csv`. Findings recorded in the final
report. **No fix applied — awaiting approval.**

---

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

## 2026-11-02 — approved preprocessing and revision runner

- Applied approved preprocessing in `revision/preprocess.py`; prior production scripts/configuration are frozen in `revision/legacy/`.
- Preserved the original cohort and outputs. Corrected records/masks go to ignored `data/revision/`; public aggregates go to `results/revision/`.
- Terminology: statistical utility → fidelity; downstream fidelity → utility. Historical artifacts keep original names.
- Added train-only missingness encoding, seeds 0–9, deterministic workers, content-keyed caches, runtime pilots and resumable aggregate evaluation.
- Added mixed associations, mortality AUPRC/ECE, subgroup CIs, DPD differences/BAR rules, membership permutation calibration and fixed-range missing-aware Gower.
- Added DDPM, bounded AIM and declared Anonymeter workers. These require smoke checks and actual execution; code is not completion evidence.
- New structural tests retain the separate Task 8 approval gate. The manuscript remains untouched.

## 2026-11-03 — GitHub publication and Linux server support

- Prepared the revision code for the author's requested GitHub transfer. New experiment outputs, live status, patient data, synthetic records, models and logs remain excluded. Existing historical result files are not replaced in this publication.
- Added platform-aware isolated-worker paths with an explicit error for missing environments. Adapted the import-only Synthcity patch and bounded AIM setup to Linux paths without changing generator algorithms or the scientific protocol.
- Added `server/setup.py`, Linux package overlays, invented-data smoke checks, a private-log run wrapper and a Slurm template. Linux setup targets Python 3.10 and separate CPU/CUDA Torch versions for SDV and DDPM.
- Added documentation for private eICU provisioning, server benchmarks, checkpoint reuse within a host and the existing structural-constraint approval gate. Linux/CUDA execution still requires the server-side setup checks; Windows results do not establish cross-platform equality.

## 2026-11-04 — compressed source files for server storage limits

- Added transparent `.csv.gz` source resolution for the five required inputs and optional sources. Mixed CSV/gzip inputs are supported; gzip takes priority over a possibly partial CSV from interrupted manual decompression.
- Kept the cohort rules, screening, missingness handling and generator protocol unchanged. Source identities and aggregate cache keys record the file actually used.
- Added invented-data checks proving complete cohort extraction agrees for plain, gzip and mixed sources. Documented that compressed inputs do not eliminate generated-output storage requirements.
