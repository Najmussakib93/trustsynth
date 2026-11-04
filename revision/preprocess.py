"""Approved extraction. Patient-level files stay in data/revision; stdout is aggregate-only."""
import argparse
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd

from .protocol import DATA, COHORT, FEATURES, PROTOCOL
from .legacy.extract_cohort import parse_age
from .safe_reporting import ROOT, PUBLIC, category_counts, count, event_summary, write_json
from .source_files import source_file

RENAME = dict(heartrate='heart_rate_mean', sao2='spo2_mean', respiration='resp_rate_mean',
              systemicsystolic='sbp_mean', systemicdiastolic='dbp_mean', temperature='temperature_mean')


def file_identity(path):
    stat = path.stat()
    return {'name': path.name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def selected_patients(patient):
    """Return actual stay rows; never fill a field from a different stay."""
    p = patient.copy()
    p['age'] = p.age.map(parse_age)
    p = p.loc[p.age.ge(PROTOCOL['minimum_age'])].sort_values('patientunitstayid', kind='stable')
    p = p.drop_duplicates('uniquepid', keep='first').copy()
    selected_n = len(p)
    p['gender'] = p.gender.str.strip().str.upper()
    p = p.loc[p.gender.isin(['MALE', 'FEMALE'])].copy()
    gender_n = len(p)
    status = p.hospitaldischargestatus.str.strip().str.lower()
    p = p.loc[status.isin(['alive', 'expired'])].copy()
    p['hospital_mortality'] = status.loc[p.index].eq('expired').astype(int)
    # Preserve approved legacy LOS definition, recording clipping separately.
    negative_los = int(p.unitdischargeoffset.lt(0).sum())
    p['los_days'] = (p.unitdischargeoffset / 1440).clip(lower=0)
    p['ethnicity_grouped'] = 'OTHER'
    for pattern, label in [('Caucasian', 'WHITE'), ('African American', 'BLACK'),
                           ('Hispanic', 'HISPANIC'), (r'\bAsian\b', 'ASIAN')]:
        p.loc[p.ethnicity.str.contains(pattern, case=False, na=False), 'ethnicity_grouped'] = label
    p['admission_type_grouped'] = 'OTHER'
    for pattern, label in [('MICU|Med-Surg|medical', 'MEDICAL'), ('SICU|surgical', 'SURGICAL'),
                           ('CCU|Cardiac|CTICU|CSICU', 'CARDIAC')]:
        p.loc[p.unittype.str.contains(pattern, case=False, na=False), 'admission_type_grouped'] = label
    return p, {'all_stays': count(len(patient)), 'selected_adult_stays': count(selected_n),
               'binary_gender_eligible': count(gender_n), 'observed_outcome_eligible': count(len(p)),
               'negative_los_floored': event_summary(negative_los, len(p))}


def clean_values(values, bounds):
    values = pd.to_numeric(values, errors='coerce')
    finite = np.isfinite(values)
    if bounds is not None:
        finite &= values.between(*bounds, inclusive='both')
    return values.where(finite)


def stream_means(path, ids, fields, rename, kind='wide', force=False):
    """Bounded-memory sums/counts. Cache includes source identity, IDs and rules."""
    identity = {'source': file_identity(path), 'id_hash': hashlib.sha256(ids.to_numpy().tobytes()).hexdigest(),
                'fields': fields, 'kind': kind, 'version': PROTOCOL['version']}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = DATA / ('aggregate_' + key + '.pkl')
    if cache.exists() and not force:
        with cache.open('rb') as stream:
            return pickle.load(stream)
    sums = {c: np.zeros(len(ids)) for c in fields}
    nums = {c: np.zeros(len(ids), dtype=np.int64) for c in fields}
    observed = {c: 0 for c in fields}
    rejected = {c: 0 for c in fields}
    columns = ['patientunitstayid'] + list(fields)
    if kind == 'lab':
        columns = ['patientunitstayid', 'labname', 'labresult']
    elif kind == 'nurse':
        columns = ['patientunitstayid', 'nursingchartcelltypevallabel',
                   'nursingchartcelltypevalname', 'nursingchartvalue']
    for chunk in pd.read_csv(path, usecols=columns, chunksize=400000):
        pos = ids.get_indexer(chunk.patientunitstayid)
        take = pos >= 0
        chunk = chunk.loc[take].copy()
        pos = pos[take]
        if kind == 'lab':
            chunk['labname'] = chunk.labname.replace({'bedside glucose': 'glucose'})
        for c, bounds in fields.items():
            positions = pos
            if kind == 'lab':
                mask = chunk.labname.eq(c).to_numpy()
                values, positions = chunk.loc[mask, 'labresult'], pos[mask]
            elif kind == 'nurse':
                mask = (chunk.nursingchartcelltypevallabel.eq('Temperature') &
                        chunk.nursingchartcelltypevalname.isin(['Temperature (C)', 'Temperature (F)'])).to_numpy()
                values = pd.to_numeric(chunk.loc[mask, 'nursingchartvalue'], errors='coerce')
                fahrenheit = chunk.loc[mask, 'nursingchartcelltypevalname'].eq('Temperature (F)')
                values = values.where(~fahrenheit, (values - 32) * 5 / 9)
                positions = pos[mask]
            else:
                values = chunk[c]
            numeric = pd.to_numeric(values, errors='coerce')
            valid_values = clean_values(numeric, bounds)
            good = valid_values.notna().to_numpy()
            observed[c] += int(numeric.notna().sum())
            rejected[c] += int((numeric.notna() & valid_values.isna()).sum())
            sums[c] += np.bincount(positions[good], weights=valid_values.to_numpy()[good], minlength=len(ids))
            nums[c] += np.bincount(positions[good], minlength=len(ids))
    out = pd.DataFrame({'patientunitstayid': ids.to_numpy()})
    report = {}
    for c in fields:
        out[rename[c]] = np.divide(sums[c], nums[c], out=np.full(len(ids), np.nan), where=nums[c] > 0)
        report[c] = {'observed_readings': count(observed[c]),
                     'excluded_readings': event_summary(rejected[c], observed[c]),
                     'stays_with_valid_readings': event_summary(np.count_nonzero(nums[c]), len(ids))}
    result = (out, report)
    with cache.open('wb') as stream:
        pickle.dump(result, stream, protocol=4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-source-scan', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    DATA.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data'
    required_names = ['patient.csv', 'apachePatientResult.csv', 'hospital.csv', 'vitalPeriodic.csv', 'lab.csv']
    sources = {name: source_file(source, name) for name in required_names}
    patient = pd.read_csv(sources['patient.csv'], usecols=['uniquepid', 'patientunitstayid', 'age',
        'gender', 'ethnicity', 'unittype', 'hospitaldischargestatus', 'unitdischargeoffset', 'hospitalid'])
    p, flow = selected_patients(patient)
    ids = pd.Index(p.patientunitstayid.to_numpy())
    apache = pd.read_csv(sources['apachePatientResult.csv'], usecols=['patientunitstayid', 'apacheversion', 'apachescore'])
    apache = apache.loc[apache.apacheversion.eq(PROTOCOL['apache_version'])].copy()
    if apache.empty:
        raise RuntimeError('Required APACHE version unavailable')
    apache['apachescore'] = apache.apachescore.where(apache.apachescore.ge(0))
    apache = apache.groupby('patientunitstayid').apachescore.mean().rename('apache_score')
    p = p.merge(apache, on='patientunitstayid', how='left', validate='one_to_one')
    hospitals = pd.read_csv(sources['hospital.csv'], usecols=['hospitalid', 'region'])
    p = p.merge(hospitals, on='hospitalid', how='left', validate='many_to_one')
    p['hospital_region'] = p.region.fillna('Unknown')
    print('Building approved all-stay source aggregates; output contains no patient rows.', flush=True)
    vitals, vital_report = stream_means(sources['vitalPeriodic.csv'], ids,
        PROTOCOL['screening_bounds_inclusive'], RENAME, force=args.force_source_scan)
    p = p.merge(vitals, on='patientunitstayid', how='left', validate='one_to_one')
    provenance = pd.DataFrame(index=p.index)
    for c in ['sbp_mean', 'dbp_mean', 'temperature_mean']:
        provenance[c] = np.where(p[c].notna(), 'periodic', 'missing')
    optional = {}
    optional_reports = {}
    for name, fields, rename, kind in [
        ('vitalAperiodic.csv', {'noninvasivesystolic': [40, 300], 'noninvasivediastolic': [20, 200]},
         {'noninvasivesystolic': 'sbp_mean', 'noninvasivediastolic': 'dbp_mean'}, 'wide'),
        ('nurseCharting.csv', {'temperature': [25, 45]}, {'temperature': 'temperature_mean'}, 'nurse')]:
        optional_path = source_file(source, name, required=False)
        optional[name] = optional_path is not None
        if optional[name]:
            extra, optional_reports[name] = stream_means(optional_path, ids, fields, rename, kind, args.force_source_scan)
            extra = extra.set_index('patientunitstayid').reindex(p.patientunitstayid)
            for c in rename.values():
                values = extra[c].to_numpy()
                take = pd.notna(values)
                p.loc[take, c] = values[take]
                provenance.loc[take, c] = name.removesuffix('.csv')
    print('Vital aggregates ready; building all-stay lab means.', flush=True)
    labs, lab_report = stream_means(sources['lab.csv'], ids,
        {'glucose': None, 'creatinine': None, 'Hgb': None},
        {'glucose': 'glucose_mean', 'creatinine': 'creatinine_mean', 'Hgb': 'hemoglobin_mean'},
        'lab', args.force_source_scan)
    p = p.merge(labs, on='patientunitstayid', how='left', validate='one_to_one')
    p = p.loc[p[FEATURES].notna().sum(axis=1).ge(PROTOCOL['minimum_observed_features'])]
    flow['missingness_eligible'] = count(len(p))
    if len(p) > PROTOCOL['cohort_size']:
        positions = np.random.RandomState(PROTOCOL['cohort_seed']).choice(len(p), PROTOCOL['cohort_size'], replace=False)
        p = p.iloc[positions]
    frame = p[FEATURES].reset_index(drop=True)
    assert frame.gender.isin(['MALE', 'FEMALE']).all()
    assert frame.hospital_mortality.notna().all()
    frame.to_csv(COHORT, index=False)
    frame.isna().to_pickle(DATA / 'missingness_masks.pkl')
    p[['uniquepid', 'patientunitstayid', 'hospitalid']].reset_index(drop=True).to_pickle(DATA / 'selected_ids.pkl')
    provenance.loc[p.index].reset_index(drop=True).to_pickle(DATA / 'vital_provenance.pkl')
    digest = hashlib.sha256(COHORT.read_bytes()).hexdigest()
    manifest_path = PUBLIC / 'preprocessing' / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    flow['cohort_sample'] = count(len(frame))
    report = {'protocol': PROTOCOL, 'features': FEATURES, 'flow': flow,
        'cohort_sha256': digest, 'repeat_identical': previous['cohort_sha256'] == digest if previous else None,
        'gender': category_counts(frame.gender), 'ethnicity': category_counts(frame.ethnicity_grouped),
        'mortality': event_summary(frame.hospital_mortality.sum(), len(frame)),
        'missingness': {c: event_summary(frame[c].isna().sum(), len(frame)) for c in FEATURES},
        'optional_sources_available': optional, 'vital_screening': vital_report,
        'optional_screening': optional_reports, 'lab_aggregation': lab_report,
        'source_provenance': {c: category_counts(provenance.loc[p.index, c]) for c in provenance},
        'sources': [file_identity(sources[name]) for name in required_names],
        'wall_seconds': time.perf_counter() - started}
    write_json(manifest_path, report)
    print('Approved cohort:', flow['cohort_sample'], 'gender:', report['gender'],
          'mortality:', report['mortality'], 'identical repeat:', report['repeat_identical'], flush=True)


if __name__ == '__main__':
    main()
