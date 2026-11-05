"""Read only aggregate cell exports and retain disclosure/undefined markers."""
from pathlib import Path
import numpy as np
import pandas as pd

from .safe_reporting import PUBLIC


def collect(public=None):
    public = Path(public or PUBLIC)
    blocks = []
    for path in sorted((public / 'cells').glob('*/*_metrics.csv')):
        table = pd.read_csv(path, keep_default_na=False)
        if not {'seed', 'generator', 'metric', 'value'}.issubset(table):
            raise ValueError('Invalid aggregate metric schema: ' + path.name)
        blocks.append(table[['seed', 'generator', 'metric', 'value']])
    additions = []
    for directory in sorted((public / 'cells').glob('*')):
        if not directory.is_dir():
            continue
        for filename, dimensions, prefix, quantities in [
            ('subgroup_utility.csv', ['model', 'attribute', 'group'], 'subgroup',
             ['auroc', 'auprc', 'ece', 'auroc_ci_low', 'auroc_ci_high', 'auprc_ci_low', 'auprc_ci_high', 'n', 'positive_count', 'unstable_lt50_positive']),
            ('representation.csv', ['attribute', 'group'], 'representation',
             ['ratio', 'erased', 'inflated_gt1_2', 'real_n', 'synthetic_n']),
            ('parity.csv', ['attribute'], 'parity',
             ['dpd_real', 'dpd_synthetic', 'dpd_difference', 'difference_ci_low', 'difference_ci_high', 'bar', 'bar_ci_low', 'bar_ci_high']),
            ('near_duplicates.csv', ['ethnicity'], 'near_duplicates',
             ['synthetic_n', 'near_duplicate_n', 'near_duplicate_rate'])]:
            path = directory / filename
            if not path.exists():
                continue
            for row in pd.read_csv(path, keep_default_na=False).to_dict('records'):
                context = '.'.join(str(row[name]) for name in dimensions)
                for metric in quantities:
                    if metric in row:
                        value = row[metric]
                        if value in [True, 'True']:
                            value = 1.
                        elif value in [False, 'False']:
                            value = 0.
                        additions.append({'seed': int(row['seed']), 'generator': row['generator'],
                            'metric': prefix + '.' + context + '.' + metric, 'value': value})
                        if filename == 'near_duplicates.csv' and metric == 'near_duplicate_rate':
                            additions.append({'seed': int(row['seed']), 'generator': row['generator'],
                                'metric': 'privacy.near_duplicate_rate.ethnicity_grouped.' + str(row['ethnicity']), 'value': value})
    if additions:
        blocks.append(pd.DataFrame(additions))
    if not blocks:
        raise RuntimeError('No aggregate cell metrics are available')
    result = pd.concat(blocks, ignore_index=True)
    result['seed'] = result.seed.astype(int)
    keys = ['seed', 'generator', 'metric']
    if result.duplicated(keys).any():
        raise RuntimeError('Duplicate aggregate metric keys; do not silently combine conflicting cells')
    return result.sort_values(['generator', 'seed', 'metric'], kind='stable').reset_index(drop=True)


def representation_seed_counts(tidy):
    rows = []
    for (generator, metric), part in tidy[tidy.metric.str.startswith('representation.') &
        tidy.metric.str.endswith(('.erased', '.inflated_gt1_2'))].groupby(['generator', 'metric']):
        values = pd.to_numeric(part.value, errors='coerce').dropna()
        rows.append({'generator': generator, 'metric': metric, 'defined_seeds': len(values),
                     'event_seeds': int(values.sum()), 'fraction_of_defined_seeds': float(values.mean()) if len(values) else np.nan})
    return pd.DataFrame(rows)
