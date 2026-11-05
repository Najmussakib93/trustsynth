"""Analyze aggregate seed exports, generate tables/figures and the revision report.

This module never loads real/synthetic records or private prediction caches.
"""
import hashlib
import json
from itertools import combinations
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .aggregate_results import collect, representation_seed_counts
from .protocol import PROTOCOL
from .scores import (DIMENSIONS, REFERENCES, FORMULAS, calculate, eligible_profiles,
                     normalization_sensitivity, threshold_sensitivity, weight_sensitivity)
from .statistics import summarize, paired, tables, holm
from .safe_reporting import PUBLIC, ROOT, write_json


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def markdown(frame, columns=None):
    if frame.empty:
        return '_No defined results._'
    frame = frame[columns] if columns else frame
    escape = lambda value: str(value).replace('|', r'\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(escape, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(map(escape, row)) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def display(summary, generator, metric):
    part = summary[summary.generator.eq(generator) & summary.metric.eq(metric)]
    if part.empty or not np.isfinite(part.iloc[0]['mean']):
        return 'undefined'
    row = part.iloc[0]
    return f"{row['mean']:.4f} (SD {row.sd:.4f}); 95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}]; n={int(row.n_seeds)}"


def source_reference():
    path = Path(__file__).with_name('reference') / 'legacy_audit.json'
    return json.loads(path.read_text(encoding='utf-8'))


def old_new(summary, public):
    reference = source_reference()
    manifest = json.loads((public / 'preprocessing/manifest.json').read_text())
    context_path = public / 'cohort_context.json'
    context = json.loads(context_path.read_text()) if context_path.exists() else {}
    rows = []
    direct = {'utility.utility_score': 'score.fidelity', 'privacy.privacy_score': 'score.privacy',
        'fairness.fairness_score': 'score.fairness', 'fidelity.fidelity_score': 'score.utility',
        'integrity.integrity_score': 'score.integrity', 'utility.ks_mean': 'fidelity.ks_observed.mean',
        'privacy.mia_accuracy': 'privacy.mia_accuracy', 'privacy.dcr_ratio': 'privacy.dcr_ratio',
        'fidelity.tstr_gap': 'utility.auroc_gap_absolute', 'TSTR_gap': 'utility.auroc_gap_absolute',
        'fidelity.auc_tstr': 'utility.tstr.auroc', 'fidelity.auc_trtr': 'utility.trtr.auroc',
        'utility.corr_deviation': 'fidelity.association_abs_error.mean',
        'correlation_deviation': 'fidelity.association_abs_error.mean',
        'integrity.csr': 'integrity.csr_record_all',
        'BAR_ethnicity_grouped': 'fairness.bar.ethnicity_grouped',
        'ethnicity_violations': 'fairness.ethnicity_violation_groups'}
    for row in reference['aggregates']['manuscript_comparison']:
        record = dict(row)
        metric, generator = row['metric'], row['generator']
        new_metric = direct.get(metric)
        if metric.startswith('representation_ratio.'):
            new_metric = 'fairness.representation.ethnicity_grouped.' + metric.split('.')[-1].upper()
        if metric == 'AUROC':
            new_metric = 'utility.trtr.auroc' if generator == 'TRTR' else 'utility.tstr.auroc'
        if new_metric:
            generator = 'GaussianCopula' if generator == 'TRTR' else generator
            part = summary[summary.generator.eq(generator) & summary.metric.eq(new_metric)]
            numeric = part.iloc[0].to_dict() if len(part) else {}
            record.update(new_metric=new_metric, new_mean=numeric.get('mean', np.nan), new_sd=numeric.get('sd', np.nan),
                new_ci_low=numeric.get('ci_low', np.nan), new_ci_high=numeric.get('ci_high', np.nan), new_n_seeds=numeric.get('n_seeds', 0))
            record['comparison_note'] = 'Corrected cohort and ten-seed protocol; definitions/normalization may differ. Undefined is not zero.'
            if 'BAR' in metric:
                record['comparison_note'] += ' Full required-group support and real DPD >= 0.01 are now required.'
            if metric in ['integrity.csr', 'integrity.integrity_score']:
                record['comparison_note'] += ' Primary CSR now requires all retained rules among fully applicable records, with coverage.'
        else:
            value = np.nan
            if metric == 'patients':
                value = manifest['flow']['cohort_sample']
            elif metric == 'features':
                value = len(manifest['features'])
            elif metric == 'mortality_percent':
                fraction = manifest['mortality']['fraction']
                value = 100 * fraction if isinstance(fraction, (float, int)) else np.nan
            elif metric.endswith('_percent'):
                group = metric.removesuffix('_percent').upper()
                n = manifest['ethnicity'].get(group)
                total = manifest['flow']['cohort_sample']
                value = 100 * n / total if isinstance(n, int) and isinstance(total, int) else np.nan
            elif metric == 'sample_hospitals':
                value = context.get('distinct_hospitals', np.nan)
            record.update(new_metric='cohort.' + metric, new_mean=value, new_sd=np.nan, new_ci_low=np.nan,
                          new_ci_high=np.nan, new_n_seeds=0, comparison_note='Fixed-cohort aggregate, not a random-seed estimate.')
        rows.append(record)
    return pd.DataFrame(rows)


def add_group_violation_metric(tidy):
    additions = []
    part = tidy[tidy.metric.str.startswith('fairness.representation.ethnicity_grouped.')]
    for (generator, seed), group in part.groupby(['generator', 'seed']):
        values = pd.to_numeric(group.value, errors='coerce').to_numpy()
        value = int(((values < .8) | (values > 1.2)).sum()) if len(values) == 5 and np.isfinite(values).all() else np.nan
        additions.append({'generator': generator, 'seed': seed, 'metric': 'fairness.ethnicity_violation_groups', 'value': value})
    return pd.concat([tidy, pd.DataFrame(additions)], ignore_index=True)


def mechanism_comparisons(tidy):
    numeric = tidy.copy()
    numeric['numeric'] = pd.to_numeric(numeric.value, errors='coerce')
    rows = []
    for alternative, baseline in [('CTGAN_no_log', 'CTGAN'), ('TVAE_900', 'TVAE')]:
        subset = numeric[numeric.generator.isin([alternative, baseline]) & numeric.seed.isin(PROTOCOL['mechanism_seeds'])]
        for metric, part in subset.groupby('metric'):
            pivot = part.pivot(index='seed', columns='generator', values='numeric')
            if alternative in pivot and baseline in pivot:
                rows.append({'comparison': alternative + '_minus_' + baseline, 'metric': metric,
                             **paired(pivot[alternative], pivot[baseline])})
    return pd.DataFrame(rows)


def findings(tidy, summary, scored, seed_counts):
    numeric = tidy.copy()
    numeric['numeric'] = pd.to_numeric(numeric.value, errors='coerce')
    result = []
    def event_counts(generator, ending, seeds=None):
        data = numeric[numeric.generator.eq(generator) & numeric.metric.str.startswith('representation.ethnicity_grouped.') & numeric.metric.str.endswith(ending)]
        if seeds is not None:
            data = data[data.seed.isin(seeds)]
        return [(metric.split('.')[2], int(part.numeric.sum()), int(part.numeric.notna().sum())) for metric, part in data.groupby('metric')]
    erased = event_counts('TVAE', '.erased')
    minorities = [row for row in erased if row[0] in ['ASIAN', 'HISPANIC', 'OTHER']]
    detail = '; '.join(f'{group}: {events}/{n} seeds erased' for group, events, n in minorities)
    result.append({'finding': 'TVAE erases minority groups', 'assessment': 'observed' if any(events for _, events, _ in minorities) else 'not observed in available cells', 'evidence': detail or 'No defined erasure results.'})
    inflated = [row for row in event_counts('CTGAN', '.inflated_gt1_2') if row[0] in ['ASIAN', 'HISPANIC', 'OTHER']]
    result.append({'finding': 'CTGAN inflates minority groups', 'assessment': 'observed' if any(events for _, events, _ in inflated) else 'not observed in available cells',
                   'evidence': '; '.join(f'{group}: {events}/{n} seeds above 1.2' for group, events, n in inflated)})
    ratios = numeric[numeric.generator.eq('GaussianCopula') & numeric.metric.str.startswith('fairness.representation.')]
    complete = len(ratios) == 7 * len(PROTOCOL['seeds']) and ratios.numeric.notna().all()
    within = ratios.numeric.between(.8, 1.2)
    result.append({'finding': 'GaussianCopula preserves demographics',
        'assessment': 'all evaluated ratios within prespecified 0.8–1.2 band' if complete and within.all() else 'not uniformly demonstrated',
        'evidence': f'{int(within.sum())}/{len(ratios)} defined-or-suppressed group/seed ratios lie within the band; this is not an equivalence test.'})
    profiles, eligibility = eligible_profiles(scored)
    dominance = []
    for a, b in combinations(sorted(profiles), 2):
        if np.all(profiles[a] >= profiles[b]) and np.any(profiles[a] > profiles[b]):
            dominance.append(a + ' dominates ' + b)
        elif np.all(profiles[b] >= profiles[a]) and np.any(profiles[b] > profiles[a]):
            dominance.append(b + ' dominates ' + a)
    all_main = all(name in profiles for name in ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'])
    result.append({'finding': 'No generator dominates',
        'assessment': ('no point-estimate Pareto dominance among complete main profiles' if not dominance else 'point-estimate dominance observed') if all_main else 'not assessable across all main generators',
        'evidence': '; '.join(dominance) or 'Conditional eligible set: ' + ', '.join(sorted(profiles)) + '; incomplete profiles are never assigned zero.'})
    inversions = []
    for a, b in combinations(sorted(set(summary.generator) - {'CTGAN_no_log', 'TVAE_900'}), 2):
        def mean(name, metric):
            part = summary[summary.generator.eq(name) & summary.metric.eq(metric)]
            return part.iloc[0]['mean'] if len(part) else np.nan
        ks_a, ks_b = mean(a, 'fidelity.ks_observed.mean'), mean(b, 'fidelity.ks_observed.mean')
        au_a, au_b = mean(a, 'utility.tstr.auroc'), mean(b, 'utility.tstr.auroc')
        if np.isfinite([ks_a, ks_b, au_a, au_b]).all() and (ks_a - ks_b) * (au_a - au_b) > 0:
            inversions.append(a + ' vs ' + b)
    result.append({'finding': 'Fidelity does not guarantee utility', 'assessment': 'ranking tradeoff observed' if inversions else 'not demonstrated by these defined mean KS/AUROC comparisons',
                   'evidence': '; '.join(inversions) or 'Missing metrics and finite-sample means cannot establish a general implication.'})
    dcr = summary[summary.generator.eq('TVAE') & summary.metric.eq('privacy.dcr_ratio')]
    if len(dcr) and np.isfinite(dcr.iloc[0]['mean']):
        row = dcr.iloc[0]
        assessment = 'distance-based warning persists' if np.isfinite(row.ci_high) and row.ci_high < 1 else 'not consistently below the holdout-distance reference'
        evidence = display(summary, 'TVAE', 'privacy.dcr_ratio') + '; inspect near-duplicate subgroup rates and calibrated attacks separately; distance alone does not prove memorization.'
    else:
        assessment, evidence = 'unresolved', 'DCR is unavailable.'
    result.append({'finding': 'TVAE memorization signal', 'assessment': assessment, 'evidence': evidence})
    csr = summary[summary.metric.eq('integrity.csr_record_all') & ~summary.generator.isin(['CTGAN_no_log', 'TVAE_900'])]
    result.append({'finding': 'Structural integrity uniformly high',
        'assessment': 'only a restricted hard-rule result; not a general clinical-validity conclusion',
        'evidence': '; '.join(name + ': ' + display(summary, name, 'integrity.csr_record_all') for name in csr.generator) +
            '. Applicability and separate population deviations must accompany CSR; three real-violated rules were excluded.'})
    return pd.DataFrame(result)


def latex(frame, path, caption):
    substitutions = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                     '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    escape = lambda text: ''.join(substitutions.get(character, character) for character in str(text))
    lines = [r'\begin{table}[ht]', r'\centering', r'\scriptsize', r'\caption{' + escape(caption) + '}',
             r'\begin{tabular}{' + 'l' * len(frame.columns) + '}', r'\hline',
             ' & '.join(map(escape, frame.columns)) + r' \\', r'\hline']
    lines += [' & '.join(map(escape, row)) + r' \\' for row in frame.itertuples(index=False, name=None)]
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(public=None, make_figures=True):
    public = Path(public or PUBLIC)
    for directory in (public / 'cells').glob('*'):
        if directory.is_dir() and not (directory / 'integrity_status.json').exists():
            raise RuntimeError('Approved integrity results are missing; run revision.finalize --run before publication analysis')
    tidy = add_group_violation_metric(collect(public))
    scored = calculate(tidy)
    components = calculate(tidy, components=True)
    # Repeat the seeded score/sensitivity calculation independently, not just the formatting.
    pd.testing.assert_frame_equal(scored, calculate(tidy))
    weights, eligibility = weight_sensitivity(scored)
    repeated_weights, _ = weight_sensitivity(calculate(tidy))
    pd.testing.assert_frame_equal(weights, repeated_weights)
    normalization = normalization_sensitivity(tidy)
    thresholds = threshold_sensitivity(tidy, scored)
    all_tidy = pd.concat([tidy, scored, components], ignore_index=True).sort_values(['generator', 'seed', 'metric'], kind='stable')
    summary, comparisons = tables(all_tidy)
    counts = representation_seed_counts(tidy)
    mechanisms = mechanism_comparisons(tidy)
    original_findings = findings(tidy, summary, scored, counts)
    changes = old_new(summary, public)
    outputs = {'per_seed_results.csv': all_tidy, 'dimension_scores_per_seed.csv': scored, 'score_components_per_seed.csv': components,
        'metric_summary.csv': summary, 'paired_comparisons.csv': comparisons, 'representation_seed_counts.csv': counts,
        'mechanism_comparisons.csv': mechanisms, 'weight_winner_shares.csv': weights, 'score_eligibility.csv': eligibility,
        'normalization_sensitivity.csv': normalization, 'threshold_sensitivity.csv': thresholds,
        'old_vs_new_manuscript.csv': changes, 'original_findings.csv': original_findings}
    for filename, frame in outputs.items():
        frame.to_csv(public / filename, index=False)
    names = sorted(set(summary.generator) - {'CTGAN_no_log', 'TVAE_900'})
    dimension_table = pd.DataFrame([dict(Generator=name, **{dimension.capitalize(): display(summary, name, 'score.' + dimension) for dimension in DIMENSIONS},
        TrustScore=display(summary, name, 'score.TrustScore')) for name in names])
    raw_table = pd.DataFrame([{'Generator': name, **{title: display(summary, name, metric) for title, metric in [
        ('KS', 'fidelity.ks_observed.mean'), ('MIA', 'privacy.mia_accuracy'), ('DCR', 'privacy.dcr_ratio'),
        ('TRTR AUROC', 'utility.trtr.auroc'), ('TSTR AUROC', 'utility.tstr.auroc'), ('TSTR AUPRC', 'utility.tstr.auprc'),
        ('AUROC gap', 'utility.auroc_gap_absolute'), ('CSR', 'integrity.csr_record_all'), ('CSR coverage', 'integrity.complete_rule_coverage')]}} for name in names])
    dimension_table.to_csv(public / 'table_I_dimensions.csv', index=False)
    raw_table.to_csv(public / 'table_I_raw_metrics.csv', index=False)
    ethnicity = pd.DataFrame([{'Ethnicity': group, 'Generator': name,
        'Representation ratio': display(summary, name, 'fairness.representation.ethnicity_grouped.' + group)}
        for group in ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER'] for name in names])
    ethnicity.to_csv(public / 'table_II_representation.csv', index=False)
    latex(dimension_table, public / 'table_I_dimensions.tex', 'Corrected protocol: normalized dimensions; undefined components remain undefined.')
    latex(raw_table, public / 'table_I_raw_metrics.tex', 'Raw metric means, SDs and across-seed t-based 95% confidence intervals; see coverage and sample counts.')
    latex(ethnicity, public / 'table_II_representation.tex', 'Ethnicity representation ratios under the corrected cohort and ten-seed protocol.')
    write_json(public / 'normalization.json', clean({'references': REFERENCES, 'formulas': FORMULAS,
        'dimensions': DIMENSIONS, 'weights': {dimension: 1 / len(DIMENSIONS) for dimension in DIMENSIONS},
        'missing_rule': 'No dropping of required components or dimensions; undefined stays undefined.',
        'legacy_js_reference_nats': REFERENCES['js_divergence_bits'] * np.log(2),
        'fairness_change': 'Primary bounded positive DPD difference replaces unstable BAR in the score; BAR remains a separately reported sensitivity metric.',
        'score_repeat_identical': True, 'weight_sensitivity_repeat_identical': True}))
    if make_figures:
        from .publication_figures import generate
        figures = generate(summary, scored, weights, normalization, thresholds, public)
    else:
        figures = []
    from .publication_docs import generate
    generate(public, summary, comparisons, dimension_table, raw_table, ethnicity, changes, original_findings, weights, eligibility, counts, mechanisms, figures)
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in public.glob('*.csv')}
    write_json(public / 'analysis_manifest.json', {'aggregate_csv_sha256': files, 'input_kind': 'aggregate_cell_exports_only',
        'figure_files': len(figures), 'code_revision': 'approved-structural-finalization-v1',
        'completion': 'analysis_outputs_generated; review original_findings.csv and all undefined/coverage flags'})
    print('Aggregate summaries, paired statistics, sensitivity, tables and publication outputs generated.', flush=True)
    return {'summary': summary, 'comparisons': comparisons, 'findings': original_findings, 'files': files}


if __name__ == '__main__':
    main()
