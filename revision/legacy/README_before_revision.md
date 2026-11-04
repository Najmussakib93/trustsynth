# TrustSynth Experimental Evaluation Pipeline

## Overview
This pipeline runs the TrustSynth five-dimensional evaluation framework on a
MIMIC-IV ICU cohort using four synthetic data generators:

1. **GaussianCopula** (statistical baseline)
2. **CTGAN** (GAN-based)
3. **TVAE** (VAE-based)
4. **DPGAN** (differentially private GAN, via synthcity)

All generators are CPU-friendly and should complete on a standard laptop in under 1 hour total.

## Prerequisites

1. **MIMIC-IV access**: You need credentialed access to MIMIC-IV on PhysioNet.
   Download the following CSV files from `mimic-iv/3.1/hosp/` and `mimic-iv/3.1/icu/`:
   - `hosp/patients.csv`
   - `hosp/admissions.csv`
   - `icu/icustays.csv`
   - `icu/chartevents.csv` (large file, ~30GB — see alternative below)
   - `hosp/labevents.csv` (large file — see alternative below)

   **RECOMMENDED ALTERNATIVE**: Use the MIMIC-IV demo dataset (freely available)
   to test the pipeline first, then run on the full dataset:
   https://physionet.org/content/mimic-iv-demo/2.2/

2. **Python 3.9+**

## Setup

```bash
pip install -r requirements.txt
```

## Running

### Step 1: Extract the cohort
Edit `config.py` to set `MIMIC_DIR` to your MIMIC-IV data directory.

```bash
python 01_extract_cohort.py
```
This produces `data/cohort.csv` (~15-20 features, ~20K patients).

### Step 2: Run the full TrustSynth evaluation
```bash
python 02_run_evaluation.py
```
This trains all generators, computes all 5 dimension scores, and saves results.

### Step 3: Generate paper figures and tables
```bash
python 03_generate_outputs.py
```
This produces LaTeX tables and PDF figures ready for the paper.

## Output Files
- `results/dimension_scores.csv` — raw scores per method per dimension
- `results/summary_table.tex` — LaTeX table for the paper
- `figures/radar_chart.pdf` — spider chart comparing methods
- `figures/distribution_comparison.pdf` — marginal distribution plots
- `figures/dcr_distributions.pdf` — privacy DCR analysis
- `figures/tstr_comparison.pdf` — downstream fidelity comparison

## Expected Runtime
- Cohort extraction: 2-5 minutes
- Generator training (all 4): 15-30 minutes
- Evaluation (all 5 dimensions): 10-20 minutes
- Total: ~30-60 minutes on a modern laptop
