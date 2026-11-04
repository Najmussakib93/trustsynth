"""Audit original rules and write the separate Task 8 approval proposal.

No proposed new structural test is evaluated here. Only existing 15 rules run.
"""
import operator
import numpy as np
from .generators import split_cohort
from .legacy.config import DOMAIN_CONSTRAINTS, CROSS_CONSTRAINTS
from .protocol import PROTOCOL
from .safe_reporting import PUBLIC, write_csv, count


def original_checks(frame):
    for name, (column, operation, bound) in DOMAIN_CONSTRAINTS.items():
        applicable = frame[column].notna()
        valid = frame[column].isin(bound) if operation == 'in' else {
            '>': operator.gt, '>=': operator.ge, '<': operator.lt, '<=': operator.le}[operation](frame[column], bound)
        yield name, applicable, valid
    for a, operation, b in CROSS_CONSTRAINTS:
        applicable = frame[a].notna() & frame[b].notna()
        yield a + '_greater_than_' + b, applicable, frame[a] > frame[b]


def main():
    rows, violated = [], set()
    for seed in PROTOCOL['seeds']:
        train, _ = split_cohort(seed)
        for name, applicable, valid in original_checks(train):
            invalid = int((applicable & ~valid).sum())
            if invalid:
                violated.add(name)
            rows.append({'seed': seed, 'dataset': 'REAL_TRAIN', 'constraint': name,
                'applicable_n': count(applicable.sum()), 'violation_n': count(invalid),
                'all_applicable_records_valid': invalid == 0,
                'csr_observed': float(valid[applicable].mean()) if applicable.any() else '',
                'coverage': float(applicable.mean())})
    write_csv(PUBLIC / 'integrity' / 'original_real_train.csv', rows)
    rules = '\n'.join('- `' + name + '`: `' + str(rule) + '`.' for name, rule in DOMAIN_CONSTRAINTS.items())
    rules += '\n- `sbp_mean > dbp_mean` when both are observed.'
    failed = ', '.join('`' + name + '`' for name in sorted(violated)) or 'None'
    proposal = f'''# Structural-integrity proposal — awaiting author approval

The preprocessing approval does not cover these additions. Task 8(c) of the original request explicitly requires approval before implementation. No proposed new test below has been implemented or run.

## Existing rules and real-data baseline

The original rules are:

{rules}

Rules violated in at least one corrected real training partition: {failed}. The full baseline is generated in `integrity/original_real_train.csv`. Missing measurements are excluded from each rule's applicability denominator and coverage is reported separately. A high CSR with low coverage must not be described as universally valid data.

SDV's `enforce_min_max_values=True` and rounding enforcement apply to observed training bounds, not to every clinical rule. Bounds weaker than the observed training extrema are inherited automatically. Strict inequalities at an observed endpoint, cross-column rules, and restored missingness are not guaranteed by that setting. Rules the real data violate will be dropped from the primary hard-constraint CSR, with their original results retained as diagnostics; no record will be altered to manufacture perfect validity.

## Proposed additions

The available features are all-stay averages, discharge LOS and a severity score. They do not support five defensible universal clinical rules: observed associations and conditional tails are not hard physiological laws. I propose the following **population consistency tests**, reported separately from record-level hard-constraint CSR. This distinction avoids calling legitimate real-patient variation a clinical violation.

| Test | Prespecified definition | Clinical rationale and limitation |
|---|---|---|
| APACHE–mortality ordering | Use real-train APACHE quintile boundaries; compare the synthetic top-minus-bottom mortality risk difference with the real reference. Report the full risk curve and a bootstrap CI for the difference. | Severity should carry the mortality association present in this cohort. No claim of perfect individual prediction or universal monotonicity is made. |
| Joint respiratory pattern | Use the real-train SpO2 lower quintile and respiratory-rate upper quintile as fixed cutoffs; compare their joint probability and its bootstrap CI. | Tests whether a learned joint respiratory pattern survives synthesis. Aggregate means cannot establish simultaneous respiratory distress. |
| LOS by unit group | Compare each unit group's median and upper-quartile LOS with real train, using bootstrap uncertainty. | ICU types have different care pathways. These are descriptive conditional distributions, not maximum allowable stays. |
| LOS by mortality | Compare deceased-versus-surviving differences in median and upper-quartile LOS with real train. | Tests whether the generator preserves outcome-related stay patterns without asserting that every death must have a short or long stay. |
| Age–creatinine relationship | Fix age groups at 18–44, 45–64, 65–79 and 80+; compare conditional creatinine medians and upper quartiles with real train. | Age can affect renal physiology and case mix; sex, muscle mass and treatment are unavailable confounders. No universal age-specific lab range is claimed. |
| Age–hemoglobin relationship | Apply the same age groups and compare conditional hemoglobin medians and lower quartiles with real train. | Tests an age-conditioned lab relationship; chronic disease, sex and treatment preclude a hard clinical cutoff. |
| Unit–severity consistency | Compare each unit group's APACHE median and interquartile range with real train. | Tests whether unit mix and illness severity remain coherent in the generated cohort. Unit group is not admission urgency. |

For each comparison, resample real and synthetic records independently 1,000 times with the experimental seed, keeping the trained generator fixed. The primary output is the estimated deviation and 95% CI. A proposed population test passes when its two-sided bootstrap CI includes zero; family-level pass rates and all component estimates will be reported, with a multiplicity-adjusted sensitivity analysis. If a required group has fewer than 11 applicable records, the comparison is undefined and coverage is reduced, rather than counted as a pass. The real-versus-itself baseline is reported explicitly and is a calibration check, not independent clinical validation.

The resulting report would contain **hard-rule CSR** and **population consistency** as separate outputs. Existing per-constraint results remain visible. Please approve this list and separation, or specify which tests to revise, before implementation.
'''
    (PUBLIC / 'STRUCTURAL_CONSTRAINT_PROPOSAL.md').write_text(proposal, encoding='utf-8')
    print('Original-rule baseline and structural proposal written; new constraints await author approval.', flush=True)


if __name__ == '__main__':
    main()
