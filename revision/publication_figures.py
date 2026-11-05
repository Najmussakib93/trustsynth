"""Publication figures from aggregate seed statistics only (500 dpi PNG/TIFF)."""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .integrity import APPROVAL
from .scores import DIMENSIONS, eligible_profiles
from .safe_reporting import PUBLIC, write_json

COLORS = ['#286b8e', '#b04a64', '#bc7a14', '#3c8561', '#7165a3', '#825b4e']


def generate(summary, scored, weight_table, normalization, thresholds, public=None):
    public = Path(public or PUBLIC)
    directory = public / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(set(summary.generator) - {'CTGAN_no_log', 'TVAE_900'})
    colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(names)}
    manifest = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})

    def save(fig, name, description):
        fig.tight_layout()
        for extension in ['png', 'tiff']:
            path = directory / (name + '.' + extension)
            kwargs = {'pil_kwargs': {'compression': 'tiff_lzw'}} if extension == 'tiff' else {}
            fig.savefig(path, dpi=500, **kwargs)
            manifest.append({'path': str(path.relative_to(public)), 'dpi': 500,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'description': description})
        plt.close(fig)

    def points(ax, metric, title, reference=None):
        found = False
        for index, name in enumerate(names):
            part = summary[summary.generator.eq(name) & summary.metric.eq(metric)]
            if len(part) and np.isfinite(part.iloc[0]['mean']):
                row = part.iloc[0]
                error = np.array([[max(0., row['mean'] - row.ci_low)], [max(0., row.ci_high - row['mean'])]]) if np.isfinite(row.ci_low) else None
                ax.errorbar(index, row['mean'], yerr=error, fmt='o', color=colors[name], capsize=3)
                found = True
            else:
                ax.text(index, .04, 'undefined', rotation=90, ha='center', va='bottom', transform=ax.get_xaxis_transform(), fontsize=7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right')
        ax.set_title(title)
        if reference is not None:
            ax.axhline(reference, color='.45', linestyle='--', linewidth=.8)
        ax.grid(axis='y', alpha=.18)
        if not found:
            ax.set_ylim(0, 1)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, dimension in zip(axes, DIMENSIONS):
        points(ax, 'score.' + dimension, dimension.capitalize())
    save(fig, 'dimension_profiles', 'Normalized dimension means and pointwise across-seed t-based 95% CIs; undefined components are labelled.')

    profiles, _ = eligible_profiles(scored)
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    theta = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={'polar': True})
    for name, values in profiles.items():
        lower, upper = [], []
        for dimension in DIMENSIONS:
            row = summary[summary.generator.eq(name) & summary.metric.eq('score.' + dimension)].iloc[0]
            lower.append(row.ci_low)
            upper.append(row.ci_high)
        ax.plot(theta, np.r_[values, values[0]], label=name, color=colors[name])
        ax.fill_between(theta, np.clip(np.r_[lower, lower[0]], 0, 1), np.clip(np.r_[upper, upper[0]], 0, 1), color=colors[name], alpha=.08)
    ax.set_xticks(angles)
    ax.set_xticklabels([name.capitalize() for name in DIMENSIONS])
    ax.set_ylim(0, 1)
    if profiles:
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15))
    else:
        ax.text(.5, .5, 'No complete five-dimension profiles', transform=ax.transAxes, ha='center')
    save(fig, 'radar_chart', 'Complete profiles only; marginal t-CI bands clipped to plotting bounds only, never in statistical tables.')

    for filename, definitions in [
        ('tstr_comparison', [('utility.trtr.auroc', 'TRTR AUROC'), ('utility.tstr.auroc', 'TSTR AUROC'),
                              ('utility.tstr.auprc', 'TSTR AUPRC'), ('utility.tstr.ece', 'TSTR ECE')]),
        ('dcr_comparison', [('privacy.dcr_ratio', 'Gower DCR ratio'), ('privacy.near_duplicate_rate', 'Near-duplicate rate')]),
        ('membership_calibration', [('privacy.mia_accuracy', 'Membership accuracy'),
            ('privacy.mia_excess_over_null', 'Accuracy minus permutation-null mean'),
            ('privacy.mia_permutation_p', 'Permutation p-value')]),
        ('parity_difference', [('fairness.dpd_difference.ethnicity_grouped', 'Ethnicity: synthetic minus real DPD'),
                               ('fairness.dpd_difference.gender', 'Sex: synthetic minus real DPD')])]:
        fig, axes = plt.subplots(1, len(definitions), figsize=(4 * len(definitions), 4), squeeze=False)
        for ax, (metric, title) in zip(axes[0], definitions):
            points(ax, metric, title, 1 if metric == 'privacy.dcr_ratio' else 0 if 'difference' in metric or 'excess' in metric else None)
        save(fig, filename, 'Point estimates and across-seed t-based 95% CIs; subgroup/native attack CIs remain in the cell tables.')

    for prefix, filename, reference in [('fidelity.ks_observed.', 'ks_comparison', .1),
                                        ('fairness.representation.ethnicity_grouped.', 'ethnicity_representation', 1),
                                        ('privacy.near_duplicate_rate.ethnicity_grouped.', 'near_duplicates_by_ethnicity', None)]:
        metrics = sorted(set(summary.loc[summary.metric.str.startswith(prefix), 'metric']))
        metrics = [metric for metric in metrics if not metric.endswith(('.mean', '.available_mean', '.supported_fraction'))]
        if not metrics:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(metrics)), 5))
        for i, name in enumerate(names):
            for j, metric in enumerate(metrics):
                part = summary[summary.generator.eq(name) & summary.metric.eq(metric)]
                if len(part) and np.isfinite(part.iloc[0]['mean']):
                    row = part.iloc[0]
                    error = np.array([[max(0., row['mean'] - row.ci_low)], [max(0., row.ci_high - row['mean'])]]) if np.isfinite(row.ci_low) else None
                    ax.errorbar(j + (i - (len(names) - 1) / 2) * .16, row['mean'], yerr=error,
                        fmt='o', capsize=2, color=colors[name], label=name if j == 0 else None)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([metric[len(prefix):].replace('_mean', '').replace('_', ' ') for metric in metrics], rotation=35, ha='right')
        if reference is not None:
            ax.axhline(reference, color='.5', linestyle='--', linewidth=.8)
        ax.legend()
        ax.set_ylabel('Mean and across-seed 95% CI')
        save(fig, filename, 'Per-feature/subgroup means with across-seed t-based CIs. Suppressed or undefined values are omitted, never drawn as zero.')

    constraints = sorted(set(summary.loc[summary.metric.str.startswith('integrity.constraint.') & summary.metric.str.endswith('.csr'), 'metric']))
    if constraints:
        matrix = np.full((len(names), len(constraints)), np.nan)
        annotations = {}
        for i, name in enumerate(names):
            for j, metric in enumerate(constraints):
                part = summary[summary.generator.eq(name) & summary.metric.eq(metric)]
                if len(part) and np.isfinite(part.iloc[0]['mean']):
                    row = part.iloc[0]
                    matrix[i, j] = row['mean']
                    annotations[(i, j)] = f"{row['mean']:.3f}\n[{row.ci_low:.3f}, {row.ci_high:.3f}]"
        fig, ax = plt.subplots(figsize=(max(12, len(constraints)), max(3, len(names))))
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        for (i, j), label in annotations.items():
            ax.text(j, i, label, ha='center', va='center', fontsize=6, color='white' if matrix[i, j] > .65 else 'black')
        labels = [metric.removeprefix('integrity.constraint.').removesuffix('.csr') for metric in constraints]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([label + (' *' if label in APPROVAL['excluded_primary_rules'] else '') for label in labels], rotation=45, ha='right')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        fig.colorbar(image, ax=ax, label='Conditional rule CSR; * excluded from primary score')
        save(fig, 'integrity_heatmap', 'All original rules as diagnostics, annotated mean and across-seed CI; applicability coverage is exported separately.')

    for family in APPROVAL['families']:
        metrics = sorted(set(summary.loc[summary.metric.str.startswith('population.' + family + '.') & summary.metric.str.endswith('.difference'), 'metric']))
        if not metrics:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, .65 * len(metrics))))
        for i, name in enumerate(names):
            for j, metric in enumerate(metrics):
                part = summary[summary.generator.eq(name) & summary.metric.eq(metric)]
                if len(part) and np.isfinite(part.iloc[0]['mean']):
                    row = part.iloc[0]
                    error = np.array([[max(0., row['mean'] - row.ci_low)], [max(0., row.ci_high - row['mean'])]]) if np.isfinite(row.ci_low) else None
                    ax.errorbar(row['mean'], j + (i - (len(names) - 1) / 2) * .17, xerr=error,
                        fmt='o', capsize=2, color=colors[name], label=name if j == 0 else None)
        prefix = 'population.' + family + '.'
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([metric[len(prefix):].removesuffix('.difference') for metric in metrics])
        ax.axvline(0, color='.5', linestyle='--', linewidth=.8)
        ax.set_xlabel('Synthetic minus real estimate (native units); across-seed t-based 95% CI')
        ax.set_title(family.replace('_', ' ') + ': descriptive deviations, no preservation pass/fail')
        ax.legend()
        save(fig, 'population_' + family, 'No equivalence claim follows from an interval containing zero; within-seed bootstrap CIs are retained in population_consistency.csv.')

    attacks = ['singling_out', 'linkability', 'inference_ethnicity_grouped', 'inference_hospital_mortality']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, attack in zip(axes, attacks):
        points(ax, 'privacy.anonymeter.' + attack + '.risk', attack.replace('_', ' '), 0)
    save(fig, 'anonymeter_risks', 'Control-adjusted native Anonymeter risks; plotted intervals describe across-seed variation, native attack intervals are in cell exports.')

    if not weight_table.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(weight_table.generator, weight_table.winner_share, color=[colors[name] for name in weight_table.generator])
        ax.errorbar(range(len(weight_table)), weight_table.winner_share.to_numpy(),
            yerr=np.vstack([weight_table.winner_share - weight_table.mc_ci_low, weight_table.mc_ci_high - weight_table.winner_share]), fmt='none', ecolor='black', capsize=3)
        ax.set_ylabel('Share of sampled weight space won')
        ax.set_title('Conditional on complete profiles; error bars are Monte Carlo uncertainty')
        save(fig, 'weight_sensitivity', 'Uniform Dirichlet simplex sampling with fixed seed; ties split equally; unrankable generators are disclosed separately.')
    if not normalization.empty:
        scenarios = normalization.scenario.drop_duplicates().tolist()
        matrix = normalization.pivot(index='generator', columns='scenario', values='rank').reindex(index=names, columns=scenarios)
        fig, ax = plt.subplots(figsize=(15, max(3, len(names))))
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), vmin=1, vmax=max(2, len(names)), cmap='viridis_r', aspect='auto')
        for i in range(len(matrix)):
            for j in range(len(scenarios)):
                value = matrix.iloc[i, j]
                ax.text(j, i, str(int(value)) if np.isfinite(value) else 'NA', ha='center', va='center', fontsize=7)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=60, ha='right', fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        fig.colorbar(image, ax=ax, label='Conditional equal-weight rank among eligible profiles')
        save(fig, 'normalization_sensitivity', 'Each normalization reference is perturbed individually by -25% and +25%; undefined profiles remain unranked.')
    if not thresholds.empty:
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        for ax, dimension in zip(axes, DIMENSIONS):
            for name in names:
                part = thresholds[thresholds.generator.eq(name) & thresholds.metric.eq('score.' + dimension)].sort_values('threshold')
                if len(part):
                    ax.plot(part.threshold.to_numpy(), part.flagged_fraction.to_numpy(), color=colors[name], label=name)
                    ax.fill_between(part.threshold.to_numpy(), np.clip(part.flagged_ci_low.to_numpy(), 0, 1),
                                    np.clip(part.flagged_ci_high.to_numpy(), 0, 1), color=colors[name], alpha=.08)
            ax.set_title(dimension.capitalize())
            ax.set_xlabel('Flag if score < threshold')
            ax.set_ylim(0, 1)
        axes[0].set_ylabel('Fraction of defined seeds flagged')
        axes[-1].legend(fontsize=6)
        save(fig, 'threshold_sensitivity', 'Shading is descriptive across-seed t-CI, clipped for plotting; missing seed denominators remain explicit in CSV.')
    write_json(public / 'figure_manifest.json', {'dpi': 500, 'figures': manifest})
    return manifest
