"""Generate manuscript-support prose from executable settings and aggregate outputs."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import PROTOCOL, FEATURES, CONTINUOUS_FEATURES
from .integrity import APPROVAL
from .scores import REFERENCES, FORMULAS, DIMENSIONS
from .legacy.config import DOMAIN_CONSTRAINTS, CROSS_CONSTRAINTS


def generate(public, summary, paired, dimensions, raw, ethnicity, changes, findings, weights,
             eligibility, counts, mechanisms, figures):
    from .final_analysis import markdown, display, source_reference
    manifest = json.loads((public / 'preprocessing/manifest.json').read_text())
    fixture_notice = ('> INVENTED AGGREGATE TEST FIXTURE — NOT PAPER RESULTS.\n\n'
                      if manifest.get('test_fixture') else '')
    reference = source_reference()
    clipping = reference['aggregates']['legacy_fairness_clipping']
    dp_path = public / 'DP_FAILURE_DETAILS.json'
    dp = json.loads(dp_path.read_text()) if dp_path.exists() else json.loads((public / 'dp_attempt.json').read_text()) if (public / 'dp_attempt.json').exists() else {'state': 'not_recorded'}
    validation_path = public / 'FINALIZATION_INPUT_AUDIT.json'
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {'scope': 'aggregate-only regeneration; full input audit unavailable'}
    genotype = ', '.join(dimensions.Generator)
    rules = []
    for name, definition in DOMAIN_CONSTRAINTS.items():
        rules.append({'rule': name, 'definition': str(definition), 'primary': name not in APPROVAL['excluded_primary_rules']})
    for a, op, b in CROSS_CONSTRAINTS:
        name = a + '_greater_than_' + b
        rules.append({'rule': name, 'definition': a + ' ' + op + ' ' + b, 'primary': name not in APPROVAL['excluded_primary_rules']})
    references = pd.DataFrame([{'reference': name, 'value': format(value, '.12g')} for name, value in REFERENCES.items()])
    attacker = json.loads(Path(__file__).with_name('attacker_config.json').read_text())
    versions = {path.name: json.loads(path.read_text()) for path in sorted((public / 'environments').glob('*_imported_versions.json'))}
    source_map = {name: inspect.getsource(function) for name, function in _formula_sources().items()}
    formula_sections = '\n\n'.join('### ' + name + '\n\n```python\n' + code + '\n```' for name, code in source_map.items())
    normalization_text = '\n'.join('- **' + name + '**: `' + formula + '`.' for name, formula in FORMULAS.items())
    supplementary = f'''# TrustSynth supplementary methods

This document is generated from executable configuration, source definitions and aggregate results. It describes the corrected retrospective eICU protocol. It does not edit the manuscript or certify a synthetic release as anonymous or differentially private.

## Cohort, observations and randomness

The fixed cohort sample size is {PROTOCOL['cohort_size']}; cohort sampling seed is {PROTOCOL['cohort_seed']}. Adult eligibility starts at {PROTOCOL['minimum_age']}, with observed hospital outcomes and recorded binary gender. Intact selected stays use the minimum adult patientunitstayid per uniquepid, not an asserted chronological first admission. APACHE version is `{PROTOCOL['apache_version']}`; negative scores are missing. Readings pass inclusive screening before means over all recorded observations linked to the selected stay; there is no first-24-hour restriction. Labs use all-stay means. Discharge LOS is retrospective and negative values are floored at zero.

At least {PROTOCOL['minimum_observed_features']} of {len(FEATURES)} fields must be observed before fixed-size random sampling. Missingness is retained. Each model fits numeric fills/category handling only to its own training data; generator missing indicators are reversed after synthesis. Sources may be CSV or gzip. Optional sources and all preprocessing counts are in `preprocessing/manifest.json`.

Seeds are `{PROTOCOL['seeds']}` with mortality-stratified test fraction {PROTOCOL['test_fraction']}. CTGAN/TVAE have {PROTOCOL['ctgan_epochs']} epochs; mechanism seeds are `{PROTOCOL['mechanism_seeds']}`, using CTGAN log_frequency=False or TVAE {PROTOCOL['tvae_mechanism_epochs']} epochs. Full independent seed-0 fits and seeded attack repeats must match before accepting results. Python, NumPy, Torch/SDV and XGBoost randomness is controlled; serial generator transforms avoid unsafe shared RNG contexts. Exact devices, library versions and source hashes remain in each receipt. CPU/GPU or operating-system bitwise equality is not assumed.

## Metric definitions

- KS: `sup_x |F_real(x)-F_syn(x)|` using observed numeric values, with at least 11 observations in each dataset. The median-filled sensitivity uses the real training median on both sides. The primary mean is undefined if any required feature is unsupported; an explicitly labelled available-only mean and supported fraction remain diagnostics.
- Jensen–Shannon divergence: `0.5*KL_2(p||m)+0.5*KL_2(q||m)`, `m=(p+q)/2`, over the union of category support, with missing as its own category. Values are divergence in bits, not the square-root distance; zero-probability terms are handled by SciPy. Mortality is categorical for this metric.
- Associations: Pearson correlation for numeric pairs; `V=sqrt(chi2/(n*(min(r,c)-1)))` without Yates correction for categorical pairs; `eta=sqrt(sum_g n_g*(mean_g-mean)^2 / sum_i(x_i-mean)^2)` for mixed pairs. Report absolute synthetic-minus-real deviation for every pair; unsupported constant/empty cases remain undefined. There are `{len(FEATURES)*(len(FEATURES)-1)//2}` pairs under the active feature list.
- Utility: XGBoost mortality AUROC, average precision (AUPRC), and ten-equal-width-bin ECE `sum_b (n_b/n)*|mean(y_b)-mean(p_b)|`. Report signed and absolute AUROC gaps and signed AUPRC gap. Permutation importance is measured on the identical real heldout test set using AUROC, with {PROTOCOL['permutation_importance_repeats']} repeats; compare original-feature importance rankings by Spearman correlation. Raw signed gaps remain interpretable even if a synthetic model outperforms TRTR on one split.
- Subgroup utility: ethnicity, sex and age bins `[18,45),[45,65),[65,80),[80,infinity)` plus a supplementary non-White pool. {PROTOCOL['bootstrap_resamples']} ordinary paired-record bootstrap replicates of labels/predictions yield percentile intervals; weighted multinomial calculations exactly reproduce tied-score sklearn AUROC/AP. Flag fewer than 50 test positives; undefined one-class replicates are counted, never given perfect scores.
- Representation: `(n_syn,g/N_syn)/(n_real,g/N_real)`; exact-zero erasure is explicit, inflation is ratio >1.2, and the historical tolerance is 0.8–1.2. Patient counts below 11 and small nonzero synthetic ratios are censored.
- Parity: use all required real-train groups, `DPD=max_g P(Y=1|g)-min_g P(Y=1|g)`. The primary comparison is `DPD_syn-DPD_real`. Missing or fewer-than-11 groups make the full-support contrast undefined; no absent group receives mortality zero. BAR is reported only if full support is evaluable and `DPD_real >= {PROTOCOL['bar_min_real_dpd']}`. Independent ordinary-record multinomial group-by-outcome bootstrap samples produce difference and BAR intervals conditional on the trained generator. Bootstrap BAR denominators must also meet that floor. Undefined BAR never silently changes the required group universe.
- Membership: four Euclidean-distance features (minimum, mean, tenth percentile, population SD) to the first 5,000 standardized synthetic records; each of at most 2,000 real-train and 2,000 real-test points is sampled with the seed. Continuous missingness uses train medians; categorical encodings use train support. The scaler is fitted on synthetic records. Logistic regression with max_iter=1000 and five-fold stratified CV yields accuracy. For each of {PROTOCOL['permutation_resamples']} label permutations, refit the whole CV attack; `p=(1+sum(null>=observed))/(B+1)`, with null mean, 95th percentile and observed-minus-null mean. This is not max-|TPR-FPR|; the 0.8687 analytic formula therefore does not apply.
- Gower: equal original-column weights and fixed real-train numeric ranges, numeric differences capped at one, categorical mismatch one, both missing zero and exactly one missing one. A constant train numeric range contributes zero for two observed values. Nearest real–real distances exclude the same record. DCR is the median synthetic-to-train distance divided by the median heldout-to-train distance on seeded samples of at most 2,000. Near-duplicate rate uses distance strictly below the fifth percentile of real leave-self-out nearest distances, overall and by ethnicity. Small events/complements are suppressed. Distances and calibrated attacks are warnings, not standalone proof of copying or safety.
- Anonymeter: native control-adjusted risk and native confidence intervals, separately for singling-out, linkability and inference. The exact configuration below was declared before execution; query-generation shortfalls remain explicit. Across-seed t-CIs in summary figures differ from per-attack native intervals in the cell JSON/CSV.

## Hard rules and approved population comparisons

{markdown(pd.DataFrame(rules))}

Per-seed `integrity/sdv_inherited_rules_seed*.csv` identifies which original rules
follow mathematically from observed training bounds or categorical support.
Cross-column ordering is not enforced by univariate bounds. These are properties
of the configured SDV transform, not evidence of learned clinical reasoning.

Any original rule violated in any approved real-train partition is excluded from the primary rule set. This run's approved exclusions are `{APPROVAL['excluded_primary_rules']}`. Primary record-level CSR is the fraction satisfying **all retained rules among records where every retained rule is applicable**; complete-rule coverage must accompany it. Per-constraint CSR and coverage, original excluded-rule diagnostics and mean-per-constraint CSR remain separate. SDV min/max and categorical support can enforce many retained rules automatically; these must not be presented as learned clinical structure. Missing observations are not counted as satisfied rules.

The author approved seven separate population families: `{APPROVAL['families']}`. APACHE uses fixed real-train quintile edges, a full mortality-risk curve and top-minus-bottom risk difference. Joint respiratory consistency is the joint lower-SpO2/upper-respiratory-rate quintile event among pairs observed in both fields. LOS by unit reports median and upper quartile; LOS by outcome reports deceased-minus-surviving median and upper-quartile contrasts. Age-conditioned creatinine uses median/upper quartile, age-conditioned hemoglobin median/lower quartile, and unit-conditioned APACHE median/IQR. Lab deviations are in recorded source units; no unverified universal clinical cutoff or unit-harmonization claim is introduced.

Population bootstrap samples are ordinary independent real/synthetic record samples, conditional on fixed train cutoffs and trained generators, using {APPROVAL['bootstrap_resamples']} replicates and percentile limits `{APPROVAL['interval']}`. The conditional numeric implementation first draws group size from its full-record binomial distribution, then samples values within that empirical group; outcome contrasts draw group sizes jointly. Applicable groups need at least 11 observations; binary-rate disclosure additionally requires at least 11 events and complements. Pointwise intervals containing zero do **not** establish preservation/equivalence. No population preservation pass rate or substitution into hard-rule CSR is made; real-versus-itself is a numerical reference only.

## Normalization, clipping and TrustScore

{markdown(references)}

{normalization_text}

All required components must be finite. A missing component makes its dimension undefined; a missing dimension makes primary TrustScore undefined, with no hidden renormalization. The JSD reference is converted from the legacy natural-log scale to bits by dividing by log(2). Fidelity now averages numeric, categorical and mixed-pair association deviations; it is not identical to the old Pearson-only score. The corrected fairness score uses representation and the mathematically bounded positive DPD difference, replacing unstable BAR as a primary score input. Negative DPD differences are reported but receive no extra score above one. Population tests, native privacy risks, AUPRC and ECE remain essential uncompressed outputs rather than being silently folded into arbitrarily chosen normalization ranges.

The archived CTGAN fairness score was exactly `{clipping['saved_score']}` because its representation component before flooring was `{clipping['representation_component_before_floor']}` and its BAR component was `{clipping['BAR_component_before_floor']}`; each was floored to zero. Those are historical computed values, not evidence that revised CTGAN fairness is zero. The original BAR's group-order and missing-group handling differ from the corrected estimator.

Weight sensitivity samples 10,000 vectors uniformly from the five-dimensional simplex with Dirichlet(1,...,1), fixed seed 42. Share of weight space won is evaluated on mean complete ten-seed profiles; ties within 1e-12 split credit. Monte Carlo SE/intervals describe sampled weight vectors, not patient uncertainty. Generators without complete five-dimension profiles stay unrankable and are listed in `score_eligibility.csv`; conditional winners cannot establish superiority over excluded profiles. Each declared normalization reference is independently perturbed by -25%/+25%; raw metric values never change. Threshold curves retain their defined-seed denominator and are descriptive, not clinical decision thresholds.

## Across-seed statistics

Every exported numeric quantity has mean, sample SD and `mean +/- t_(0.975,n-1)*SD/sqrt(n)`, without clipping statistical intervals to [0,1]. Every generator pair is aligned on the same seed for difference estimates, paired t-tests and Wilcoxon signed ranks. Wilcoxon uses exact enumeration of signs on average absolute ranks, discarding zero differences; it handles ties rather than using a no-ties lookup table. With ten nonzero differences, its minimum two-sided p is `{2/2**len(PROTOCOL['seeds']):.8f}` (about 0.002), and three nonzero mechanism pairs cannot attain a two-sided p below `{2/2**len(PROTOCOL['mechanism_seeds']):.2f}`. Holm-adjusted p-values are provided within each metric's generator-pair family; they do not control error across every metric in this exploratory report. Overlapping splits of one fixed cohort are not independent clinical samples, so intervals and tests describe protocol variability and require cautious generalization.

## Attacker configuration

```json
{json.dumps(attacker, indent=2)}
```

## Software actually imported in recorded environments

```json
{json.dumps(versions, indent=2)}
```

Synthcity's isolated lazy-import compatibility patch leaves its DDPM algorithm/plugin sources unchanged; the receipts and patch are retained in revision outputs. AIM status is `{dp.get('state')}`; no unavailable DP result is imputed. Public seeds, non-private cohort construction and evaluation outputs are not an end-to-end certified DP release. Source access remains PhysioNet credentialed; no patient-level export belongs in GitHub or the aggregate review archive.

## Executable formula excerpts

{formula_sections}
'''
    (public / 'supplementary.md').write_text(supplementary, encoding='utf-8')
    algorithm = f'''# Algorithm 1 — TrustSynth corrected revision

1. Access credentialed eICU-CRD v2.0 privately. Resolve each required CSV/gzip source; select intact adult stays, recorded binary gender and observed outcomes; explicitly select APACHE {PROTOCOL['apache_version']}.
2. Screen valid observations using `protocol.py`; form all-stay vital/lab means, retain missingness and require at least {PROTOCOL['minimum_observed_features']} observed features. Select up to {PROTOCOL['cohort_size']} eligible patients with cohort seed {PROTOCOL['cohort_seed']}; verify repeated extraction hash.
3. For each seed in `{PROTOCOL['seeds']}`, redraw the mortality-stratified {1-PROTOCOL['test_fraction']:.0%}/{PROTOCOL['test_fraction']:.0%} split. Fit every generator on train only and sample train size; use reversible train-only missingness encoding except the DP generator's fixed public-domain transform. Verify full independent seed-0 generator repeats before accepting caches.
4. Evaluate observed-value KS, categorical JS divergence in bits, missingness and every mixed-type association pair. Fit real/synthetic XGBoost pipelines separately and score both on untouched real test; retain AUROC/AP/ECE and real-test permutation importance. Compute subgroup estimates and {PROTOCOL['bootstrap_resamples']}-replicate bootstrap intervals with disclosure and instability flags.
5. Evaluate representation, full-support DPD difference and guarded BAR; preserve erasure/undefined states. Run three-seed CTGAN no-log and TVAE {PROTOCOL['tvae_mechanism_epochs']}-epoch mechanism checks under identical seed splits.
6. Evaluate the seeded logistic membership attack and {PROTOCOL['permutation_resamples']} refitted-label-permutation null, fixed-range Gower nearest distances and near-duplicate rates. Run declared Anonymeter attacks with real test controls and verify seeded repeats; retain native risk intervals and query-completion flags. Stop a failed optional DP attempt within its bounded budget.
7. Audit every original rule on every real-train split; use the author-approved exclusions for hard CSR. After the recorded Task 8(c) approval, compute per-rule rates, complete-record CSR and applicability; separately evaluate all seven population families with fixed real-train cutoffs and {APPROVAL['bootstrap_resamples']} independent record-bootstrap resamples. Retain deviations/CIs without equivalence pass/fail claims.
8. Normalize using `scores.REFERENCES` and `scores.FORMULAS`; do not impute or drop undefined required components. Compute equal-weight TrustScore only when every dimension is defined; retain unrankable profiles.
9. Aggregate every quantity across matching seeds with mean/SD/t-CI and paired tests. Sample 10,000 seeded simplex weights, perturb each reference by +/-25%, and sweep thresholds; document denominators and conditional ranking scope.
10. Generate all aggregate CSV/LaTeX tables, 500-dpi PNG/TIFF figures, numerical old/new manuscript mapping, evidence-based finding assessments and this supplementary material. Keep records/models/predictions/distances/logs private. Export only the explicit aggregate review archive. Do not edit the manuscript or tune values to retain its story.
'''
    (public / 'ALGORITHM1.md').write_text(algorithm, encoding='utf-8')

    mechanism_counts = counts[counts.generator.isin(['CTGAN_no_log', 'TVAE_900'])] if not counts.empty else counts
    finding_rows = markdown(findings)
    dp_errors = dp.get('software_errors', [])
    dp_detail = '\n'.join('- `' + str(error).replace('`', '') + '`' for error in dp_errors) or 'The recorded status identifies the phase and exception; consult the private software-setup log for any additional detail.'
    figures_table = pd.DataFrame([{'path': row['path'], 'dpi': row['dpi'], 'description': row['description']} for row in figures])
    query_shortfalls = pd.DataFrame(validation.get('anonymeter_warnings', []))
    n_table_rows = len(changes)
    report = f'''# TrustSynth revision report

This report is generated from aggregate outputs of the corrected protocol. It does not edit the manuscript. Interpret undefined components, coverage and the optional DP failure before using any composite ranking.

## Task 0 — reproduction and cohort audit

The source-workstation legacy audit supplies {n_table_rows} mapped empirical table/prose cells and their archived/recomputed values; `old_vs_new_manuscript.csv` attaches the corrected seed means, SDs and CIs where the definitions permit comparison. Source audit references are versioned in `revision/reference/legacy_audit.json`, explicitly labelled as source-workstation evaluation reproduction rather than server retraining. The corrected fixed cohort contains {manifest['flow']['cohort_sample']} patients, with mortality summary `{manifest['mortality']}` and repeated extraction `{manifest['repeat_identical']}`. Changes to cohort, missingness, parity support, mixed associations and CSR definitions mean old and new cells are not all estimates of an identical quantity.

## Task 1 — high vital-sign KS

The legacy diagnostic compared filled and observed-value distributions and found large imputation masses for pressure and temperature; source-workstation aggregate diagnostic values remain in the reference bundle. Corrected preprocessing screens measurements before all-stay means and retains missingness rather than imposing a global pre-split median. Current feature-level KS and missingness results are in `metric_summary.csv` and `figures/ks_comparison.png`, with supported fractions explicitly retained. Optional-source availability is `{manifest.get('optional_sources_available', {})}`; missing cuff/nurse files must not be described as resolved by those sources.

## Task 2 — terminology

Distributional similarity is labelled fidelity, and train-on-synthetic prediction is labelled utility. Active metric prefixes and new table/figure labels follow this mapping; frozen legacy source and mapped old values retain their original labels. Definitions and score changes are documented in `supplementary.md` and CHANGES.md, rather than silently relabelling old estimates as corrected results.

## Task 3 — generators and seeds

Available main-generator summary rows cover {genotype}; exact per-cell completeness and independent repeat checks are in `FINALIZATION_INPUT_AUDIT.json`. The prescribed splits use seeds `{PROTOCOL['seeds']}`, and each accepted generator trains only on real train and emits train size. Fit-time means, SDs and CIs are included under `runtime.*` in `metric_summary.csv`, with pilot and repeat times distinguished. The optional DP state is `{dp.get('state')}` at phase `{dp.get('phase', 'see receipt')}`; unavailable models receive no fabricated epsilon result or score.

## Task 4 — fidelity

Every numeric feature has observed-value KS, every categorical feature including mortality has JS divergence in bits, and all {len(FEATURES)*(len(FEATURES)-1)//2} mixed-type pairs have the declared association deviation. Missingness errors, median-filled sensitivity values and supported fractions remain separate. The strict mean becomes undefined if a required feature/pair is unsupported; the available-only mean is explicitly a sensitivity diagnostic. Per-seed values, across-seed intervals and paired comparisons are in `per_seed_results.csv`, `metric_summary.csv` and `paired_comparisons.csv`.

## Task 5 — utility and subgroup prediction

The raw-metric table below reports actual TRTR/TSTR AUROC, AUPRC, gaps and CSR; ECE and feature-importance Spearman estimates are in the metric summary. Every model's preprocessing is fitted on its own training data and performance is measured on the same real heldout split. Subgroup rows include within-seed bootstrap intervals, positive-count instability flags and an additional non-White pool without replacing the individual ethnicities. Across-seed subgroup statistics retain undefined/suppressed values and explicit effective seed counts, so a small-group point estimate is not treated as stable evidence of harm.

## Task 6 — fairness and mechanisms

`representation_seed_counts.csv` gives erasure and >1.2 inflation counts across seeds for each group. Primary parity estimates are synthetic-minus-real full-support DPD, with bootstrap intervals; BAR remains conditional on full group support and the real-DPD floor. Mechanism effects are paired on seeds `{PROTOCOL['mechanism_seeds']}` in `mechanism_comparisons.csv`, and the mechanism erasure/inflation counts are shown below. Only three mechanism pairs are available, so their exact two-sided Wilcoxon minimum is {2/2**len(PROTOCOL['mechanism_seeds']):.2f}; these experiments do not justify strong causal explanations from significance alone.

## Task 7 — privacy

Membership statistics retain the original logistic distance-feature attack and its {PROTOCOL['permutation_resamples']}-permutation refitted null, rather than substituting the analytic max-ROC-gap formula. Gower distances use fixed real-train ranges, and subgroup near-duplicate rates are compared alongside the DCR ratio in their own tables. Anonymeter native risk/CI and query-completion receipts remain distinct from across-seed uncertainty; incomplete query generation is a stated limitation. A low empirical attack score is not a general anonymity guarantee, and a distance warning alone is not proof of memorization.

## Task 8 — hard rules and approved population consistency

The author approved the corrected seven-family plan on {APPROVAL['date']}; population comparisons are separate from record-level hard CSR. The real baseline requires excluding `{APPROVAL['excluded_primary_rules']}` from primary CSR while retaining their diagnostics. Per-constraint results, complete-record applicability and population deviations with conditional bootstrap CIs are exported for every accepted cell; the heatmap and seven population figures must be read with coverage. An interval containing zero is not called preservation, and high inherited-bound CSR is not claimed as validation of clinical structure.

## Task 9 — normalization and sensitivity

`normalization.json` and `supplementary.md` state every reference, conversion and clipping rule. Historical CTGAN fairness was zero because both legacy subcomponents floored to zero, with pre-floor values `{clipping['representation_component_before_floor']}` and `{clipping['BAR_component_before_floor']}`; corrected scores are recomputed without preserving that result. Weight-space shares, reference perturbations and threshold curves are in their named CSVs and figures, with complete-profile eligibility shown below. A generator lacking a required dimension remains unrankable, so a winner among the eligible subset does not dominate excluded models by assumption.

## Task 10 — seed statistics

Every numeric aggregate quantity has its defined-seed count, mean, sample SD and t-based 95% CI, and every available generator pair has seed-aligned differences, paired t and exact sign-enumerated Wilcoxon tests. Ten nonzero pairs permit a minimum two-sided Wilcoxon p of {2/2**len(PROTOCOL['seeds']):.8f}, approximately 0.002. Zero differences, ties, zero variance and missing pairs are handled explicitly, and Holm sensitivity controls the pair family within a metric. Since seed splits overlap within a fixed cohort, these results describe training/split variability and are not independent patient-cohort replications.

## Task 11 — outputs and reproducibility

All new estimates are generated as CSV/LaTeX tables and all listed figures are exported as 500-dpi PNG and TIFF; image hashes and descriptions are in `figure_manifest.json`. `supplementary.md` and `ALGORITHM1.md` are generated from the active configuration, formulas, constraints and recorded software environments. The actual repository is `https://github.com/Najmussakib93/trustsynth`, differing from the manuscript's `github.com/Najmus-Sakib/TrustSynth`; the manuscript remains unchanged. Source data, synthetic rows, fitted models, predictions, nearest distances and private logs remain excluded from the aggregate review archive and GitHub.

## Dimension scores — means, SDs and across-seed 95% CIs

{markdown(dimensions)}

## Raw metrics — means, SDs and across-seed 95% CIs

{markdown(raw)}

## Original findings under the corrected protocol

{finding_rows}

## Mechanism erasure/inflation seed counts

{markdown(mechanism_counts)}

## Composite-score eligibility

{markdown(eligibility)}

## Weight-space winners (conditional on that eligible set)

{markdown(weights)}

## Differentially private generator limitation

```json
{json.dumps(dp, indent=2)}
```

{dp_detail}

No additional unbounded dependency retry or generator training is triggered by finalization. This failure must be disclosed, rather than replaced with an invented DP comparison.

## Numerical comparisons and table files

- `old_vs_new_manuscript.csv`: every mapped manuscript empirical cell with old value, legacy reproduction and corrected estimate/status.
- `table_I_dimensions.csv/.tex`, `table_I_raw_metrics.csv/.tex`, `table_II_representation.csv/.tex`: generated replacement tables.
- `per_seed_results.csv`, `metric_summary.csv`, `paired_comparisons.csv`: all numeric results and effective sample sizes.
- `cells/*/subgroup_utility.csv`, `representation.csv`, `parity.csv`, `near_duplicates.csv`, `population_consistency.csv`, `integrity_constraints.csv`: detailed aggregates and within-seed intervals.
- `representation_seed_counts.csv`, `mechanism_comparisons.csv`, `score_eligibility.csv`: fairness mechanism and missing-score evidence.
- `weight_winner_shares.csv`, `normalization_sensitivity.csv`, `threshold_sensitivity.csv`: score/threshold sensitivity.
- `FINALIZATION_INPUT_AUDIT.json`, `analysis_manifest.json`, `normalization.json`, `structural_approval.json`: protocol and computation audit.

## Figures

{markdown(figures_table)}

## Additional limitations

All-stay features and discharge LOS make this a retrospective benchmark, not an admission-time mortality deployment study. Missing optional sources, per-group disclosure censoring, constant/erased features, potentially unrankable composite profiles and conditional uncertainty must remain visible. Population-pattern agreement is not a universal physiological law, no independent clinical validation was performed, and finite attacks do not prove privacy. Cross-host equality and DP guarantees cannot be inferred from a successful server execution.

Recorded Anonymeter query-generation shortfalls (if any):

{markdown(query_shortfalls)}
'''
    (public / 'REVISION_REPORT.md').write_text(fixture_notice + report, encoding='utf-8')


def _formula_sources():
    from .fidelity import association, evaluate_fidelity
    from .utility import metrics, bootstrap_auc_ap
    from .fairness import dpd, bootstrap_dpd
    from .privacy import evaluate_membership, _nearest_loop
    from .scores import score_cell, component_values
    from .statistics import summarize, exact_wilcoxon
    return {'mixed association': association, 'fidelity': evaluate_fidelity,
            'utility and ECE': metrics, 'subgroup bootstrap': bootstrap_auc_ap,
            'DPD': dpd, 'DPD bootstrap': bootstrap_dpd, 'membership null': evaluate_membership,
            'fixed-range missing-aware Gower': _nearest_loop, 'normalization components': component_values, 'normalization': score_cell,
            'across-seed t interval': summarize, 'exact tied-rank Wilcoxon': exact_wilcoxon}
