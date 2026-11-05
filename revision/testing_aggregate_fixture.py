"""Invented aggregate-only fixtures for publication pipeline verification."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .integrity import APPROVAL
from .protocol import FEATURES


def create(public):
    public = Path(public)
    (public / 'preprocessing').mkdir(parents=True, exist_ok=True)
    (public / 'preprocessing/manifest.json').write_text(json.dumps({'test_fixture': True,
        'flow': {'cohort_sample': 20000}, 'features': FEATURES, 'repeat_identical': True,
        'mortality': {'count': 1800, 'fraction': .09},
        'ethnicity': {'WHITE': 15000, 'BLACK': 2200, 'HISPANIC': 1000, 'ASIAN': 400, 'OTHER': 1400},
        'optional_sources_available': {'vitalAperiodic.csv': False, 'nurseCharting.csv': False}}))
    (public / 'dp_attempt.json').write_text(json.dumps({'state': 'unavailable', 'phase': 'invented_software_fixture'}))
    for g, name in enumerate(['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']):
        for seed in range(10):
            directory = public / 'cells' / (name + '-' + str(seed))
            directory.mkdir(parents=True, exist_ok=True)
            metrics = {'fidelity.ks_observed.mean': .08 + g * .03 + seed * .001,
                'fidelity.js_divergence.mean': .02 + g * .01, 'fidelity.association_abs_error.mean': .04 + g * .01,
                'privacy.mia_accuracy': .51 + g * .002 + seed * .0001,
                'privacy.mia_null_mean': .5, 'privacy.mia_null_p95': .515, 'privacy.mia_permutation_p': .3,
                'privacy.mia_excess_over_null': .01 + g * .002, 'privacy.dcr_ratio': 1.2 - g * .1,
                'privacy.near_duplicate_rate': .06 + g * .02,
                'utility.trtr.auroc': .89 + seed * .001, 'utility.tstr.auroc': .75 + g * .02 + seed * .001,
                'utility.tstr.auprc': .2 + g * .02, 'utility.tstr.ece': .04,
                'utility.auroc_gap_absolute': .14 - g * .02, 'utility.importance_spearman': .6 + g * .05,
                'fairness.dpd_difference.ethnicity_grouped': .01 * g if name != 'TVAE' else np.nan,
                'fairness.dpd_difference.gender': .005, 'fairness.bar.ethnicity_grouped': 1.1 if name != 'TVAE' else np.nan,
                'integrity.csr_record_all': 1., 'integrity.complete_rule_coverage': .3 + g * .05}
            for feature in ['age', 'heart_rate_mean', 'temperature_mean']:
                metrics['fidelity.ks_observed.' + feature] = .08 + .02 * g + .001 * seed
            for rule in ['age_valid', 'apache_upper', 'sbp_upper', 'temp_lower']:
                metrics['integrity.constraint.' + rule + '.csr'] = 1.
                metrics['integrity.constraint.' + rule + '.coverage'] = .5
            for family in APPROVAL['families']:
                for component in ['invented_component_A', 'invented_component_B']:
                    metrics['population.' + family + '.' + component + '.difference'] = (g - 1) * .1 + seed * .001
            representations = []
            for attr, groups in [('ethnicity_grouped', ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER']), ('gender', ['MALE', 'FEMALE'])]:
                for group in groups:
                    ratio = 0. if name == 'TVAE' and group == 'ASIAN' else 1. + g * .03 + .001 * seed
                    metrics['fairness.representation.' + attr + '.' + group] = ratio
                    representations.append({'seed': seed, 'generator': name, 'attribute': attr, 'group': group,
                        'real_n': 500, 'synthetic_n': '<11' if ratio == 0 else 500, 'ratio': ratio,
                        'erased': ratio == 0, 'inflated_gt1_2': ratio > 1.2})
            for kind in ['singling_out', 'linkability', 'inference_ethnicity_grouped', 'inference_hospital_mortality']:
                if kind == 'singling_out' or seed < 5:
                    metrics['privacy.anonymeter.' + kind + '.risk'] = .01 * g + .001 * seed
            pd.DataFrame([{'seed': seed, 'generator': name, 'metric': key, 'value': value} for key, value in metrics.items()]).to_csv(directory / 'quality_metrics.csv', index=False)
            pd.DataFrame(representations).to_csv(directory / 'representation.csv', index=False)
            (directory / 'integrity_status.json').write_text(json.dumps({'test_fixture': True}))
    return public
