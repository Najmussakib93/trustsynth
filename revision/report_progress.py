"""Refresh an interim report and combine only aggregate per-seed metrics."""
import csv
import json
from datetime import datetime, timezone
from .protocol import PROTOCOL, FEATURES
from .safe_reporting import ROOT, PUBLIC, write_json, write_csv


def main():
    rows = []
    for path in sorted((PUBLIC / 'cells').glob('*/*_metrics.csv')):
        with path.open(newline='', encoding='utf-8') as stream:
            rows.extend(csv.DictReader(stream))
    if rows:
        write_csv(PUBLIC / 'per_seed_results.csv', rows, columns=['seed', 'generator', 'metric', 'value'])
    write_json(PUBLIC / 'protocol.json', dict(PROTOCOL, feature_columns=FEATURES))
    status_path = PUBLIC / 'RUN_STATUS.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'not_started'}
    preprocessing = json.loads((PUBLIC / 'preprocessing' / 'manifest.json').read_text())
    historical = PUBLIC / 'audit' / 'INITIAL_AUDIT_REPORT.md'
    report_path = PUBLIC / 'REVISION_REPORT.md'
    # A fresh server checkout has no workstation audit. Never relabel an interim
    # progress report as that historical audit or require its directory to exist.
    audit_text = ('The archived results were reproduced at stored precision on the source workstation using the reconstructed unfiltered cohort. The paper\'s binary-gender statement contradicts those artifacts; the audit also found mixed-stay field selection and unknown outcomes labelled as survival. Comparisons are linked in [the initial audit report](audit/INITIAL_AUDIT_REPORT.md), [manuscript comparison](audit/manuscript_comparison.csv) and [baseline comparison](audit/baseline_comparison.csv).'
        if historical.exists() else
        'The legacy reproduction audit was performed on the source workstation. Its outputs are not included in this code checkout; this report does not claim that the legacy manuscript results have been reproduced on this host.')
    optional = preprocessing.get('optional_sources_available', {})
    source_text = '; '.join(name + ': ' + ('available' if present else 'unavailable') for name, present in optional.items()) or 'See the preprocessing manifest for source availability.'
    completed = {}
    for path in sorted((PUBLIC / 'generators').glob('*_seed*.json')):
        if '_pilot' not in path.name:
            data = json.loads(path.read_text())
            completed.setdefault(data['settings']['generator'], []).append(data['settings']['seed'])
    inventory = '\n'.join('- ' + name + ': seeds ' + ', '.join(map(str, seeds)) for name, seeds in completed.items())
    report = f'''# TrustSynth revision report — interim

Generated {datetime.now(timezone.utc).isoformat()}. Last runner state: **{status['state']}**. This is a staged report; incomplete experiments are not final paper findings.

## Tasks 0–1: audit and approved preprocessing

{audit_text}

The author approved the preprocessing proposal on 2026-11-02; its source-workstation document is not required to run this code. The corrected extractor selects intact stays, requires observed outcomes, chooses APACHE IV explicitly, screens readings before averaging and preserves missingness. Its generated cohort has {preprocessing['flow']['cohort_sample']} patients and mortality prevalence {preprocessing['mortality']['fraction']:.5%}; repeat-identical extraction is **{preprocessing['repeat_identical']}**. Optional source status: {source_text}. The primary benchmark remains retrospective with all-stay means and discharge LOS.

## Task 2: terminology

Active revision code uses **fidelity** for distributions and **utility** for prediction. Metric names use `fidelity.*` and `utility.*`. Frozen legacy code and original artifacts retain historical names for reproducibility. Final figures and tables remain pending the experiments and structural-constraint approval.

## Task 3: generators and seeds

The protocol fixes seeds 0–9 and redraws a mortality-stratified 80/20 split for each seed. Generators fit only real train and sample its size, with training-only reversible missingness encoding for non-private generators. Artifact metadata record fit time, software, settings, hashes and independent repeat checks. Currently available non-pilot artifacts are:

{inventory or '- None completed.'}

Artifact presence does not establish completion of all ten seeds. Each receipt records its device, software and settings. Environment checks and independent repeat tests are required on each host; unavailable generators are identified explicitly. Cross-host bit-for-bit equality is not assumed.

## Tasks 4–5: fidelity and utility

Fidelity code covers observed-value KS, categorical Jensen–Shannon divergence in bits, missingness errors and all pairwise Pearson/Cramér's V/correlation-ratio deviations. A train-median-filled KS analysis is retained as a separate sensitivity comparison. Utility fits imputation/encoding on each model's own training data and reports TRTR/TSTR AUROC, AUPRC, gaps, ten-bin ECE and permutation-importance Spearman correlation. Subgroup tables cover ethnicity, sex and age with 1,000 bootstrap resamples, a flag below 50 positives, and an additional pooled non-White estimate that preserves individual ethnicity rows.

## Task 6: fairness

Representation ratios and erasure/inflation flags are generated per cell with suppressed small counts. The primary parity measure is synthetic minus real DPD, with record-bootstrap CIs conditional on the trained generator. BAR is undefined if a required group has fewer than 11 records or real DPD is below 0.01. CTGAN without log-frequency sampling and TVAE at 900 epochs are configured on seeds 0–2; mechanism conclusions await completed runs.

## Task 7: privacy

The legacy attack is five-fold logistic classification of membership from four standardized Euclidean distance summaries. Its null refits the attack after each of 1,000 label permutations; the max-ROC-gap analytic formula does not apply. Revised Gower uses fixed real-train ranges and explicit missing-value contributions, with a leave-self-out real–real fifth-percentile near-duplicate threshold. Anonymeter attacker knowledge is declared in `revision/attacker_config.json`; unrun attacks are not results.

## Tasks 8–11 and scientific conclusions

New structural tests require the separate approval explicitly requested in Task 8(c), and none has been implemented. Score sensitivity, full paired statistics, final figures and supplementary methods follow that gate. Minority erasure, inflation, dominance, memorization and uniformly high structural integrity remain **unresolved under the corrected protocol** until the required experiments finish. Original findings will not be preserved by assumption.

## Files

- `preprocessing/manifest.json`: flow, missingness, source screening, provenance and repeat extraction hash.
- `generators/*.json`: completed generator metadata; pilot filenames explicitly include `pilot`.
- `cells/*/`: available per-seed aggregate tables.
- `per_seed_results.csv`: combined aggregate metrics, when cells exist.
- `audit/`, `diagnostics/`, `figures/`: earlier reproduction and vital-sign diagnosis.
- `RUN_STATUS.json`: last runner step; an interrupted worker can leave a stale running status.

Records, identifiers, predictions, nearest distances and raw library logs remain in ignored `data/` or `results/private/`. Code publication does not publish these artifacts. The manuscript remains unchanged.
'''
    report_path.write_text(report, encoding='utf-8')
    changes = ROOT / 'CHANGES.md'
    marker = '## 2026-11-02 — approved preprocessing and revision runner'
    existing = changes.read_text(encoding='utf-8') if changes.exists() else '# Changes\n'
    if marker not in existing:
        changes.write_text(existing + '\n' + marker + '''

- Applied approved preprocessing in `revision/preprocess.py`; prior production scripts/configuration are frozen in `revision/legacy/`.
- Preserved the original cohort and outputs. Corrected records/masks go to ignored `data/revision/`; public aggregates go to `results/revision/`.
- Terminology: statistical utility → fidelity; downstream fidelity → utility. Historical artifacts keep original names.
- Added train-only missingness encoding, seeds 0–9, deterministic workers, content-keyed caches, runtime pilots and resumable aggregate evaluation.
- Added mixed associations, mortality AUPRC/ECE, subgroup CIs, DPD differences/BAR rules, membership permutation calibration and fixed-range missing-aware Gower.
- Added DDPM, bounded AIM and declared Anonymeter workers. These require smoke checks and actual execution; code is not completion evidence.
- New structural tests retain the separate Task 8 approval gate. The manuscript remains untouched.
''', encoding='utf-8')
    print('Interim report and aggregate metric index refreshed.', flush=True)


if __name__ == '__main__':
    main()
