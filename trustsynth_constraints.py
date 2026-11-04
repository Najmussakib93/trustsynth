"""
trustsynth_constraints.py
Structural-integrity constraints for Dimension 5 (Task 6).

Two families:

  RANGE constraints (`config.DOMAIN_CONSTRAINTS`, `config.CROSS_CONSTRAINTS`)
      Simple per-column min/max and one cross-column comparison. SDV enforces
      min/max for free: `enforce_min_max_values` defaults to True on
      GaussianCopulaSynthesizer, CTGANSynthesizer and TVAESynthesizer in
      SDV 1.17.2, and the pipeline never overrides it. These constraints
      therefore cannot discriminate between generators and are retained only
      for continuity with the original submission.

  STRUCTURAL constraints (this module)
      Joint, conditional and derived rules that no SDV synthesizer enforces.
      A generator only satisfies them by learning the dependence structure
      between columns.

Every threshold is CALIBRATED FROM THE REAL TRAINING SPLIT of the current
seed (`fit_constraints`), never hand-chosen, so the rules travel with the
data rather than encoding the author's expectations. Calibrated values are
returned alongside the results so they can be reported in the supplement.

Each constraint returns a boolean Series over the synthetic rows. Rows for
which the rule does not apply (e.g. an outcome-conditional rule evaluated on
a survivor) count as satisfied, so the per-constraint satisfaction rate is
always "share of rows not violating".
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

AGE_BANDS = [(18, 40), (40, 65), (65, 80), (80, 200)]


def fit_constraints(real_df, target="hospital_mortality"):
    """Derive every structural-constraint threshold from the real training split.

    Returns a dict of calibrated parameters, serialisable to JSON for the
    supplementary material.
    """
    p = {}

    # --- joint vital-sign envelopes -------------------------------------
    pp = real_df["sbp_mean"] - real_df["dbp_mean"]
    p["pulse_pressure_lo"] = float(np.percentile(pp, 0.5))
    p["pulse_pressure_hi"] = float(np.percentile(pp, 99.5))

    mapv = real_df["dbp_mean"] + (real_df["sbp_mean"] - real_df["dbp_mean"]) / 3.0
    p["map_lo"] = float(np.percentile(mapv, 0.5))
    p["map_hi"] = float(np.percentile(mapv, 99.5))

    # SpO2 / respiratory-rate extremes: the real cohort essentially never
    # pairs a high respiratory rate with a normal saturation, or profound
    # hypoxaemia with an unremarkable respiratory rate.
    p["spo2_high"] = float(np.percentile(real_df["spo2_mean"], 90))
    p["resp_high"] = float(np.percentile(real_df["resp_rate_mean"], 99))
    p["spo2_low"] = float(np.percentile(real_df["spo2_mean"], 1))
    p["resp_low"] = float(np.percentile(real_df["resp_rate_mean"], 10))

    # --- outcome-conditional rules ---------------------------------------
    dec = real_df[real_df[target] == 1]
    sur = real_df[real_df[target] == 0]
    p["apache_deceased_p05"] = float(np.percentile(dec["apache_score"], 5)) if len(dec) else np.nan
    p["apache_survivor_p95"] = float(np.percentile(sur["apache_score"], 95)) if len(sur) else np.nan
    p["los_deceased_p01"] = float(np.percentile(dec["los_days"], 1)) if len(dec) else np.nan

    # direction check reference: real APACHE gap between outcome groups
    p["apache_gap_real"] = (
        float(dec["apache_score"].mean() - sur["apache_score"].mean()) if len(dec) and len(sur) else np.nan
    )

    # --- unit-type-conditional LOS ---------------------------------------
    p["los_p995_by_unit"] = {
        str(k): float(np.percentile(v, 99.5))
        for k, v in real_df.groupby("admission_type_grouped")["los_days"]
        if len(v) >= 30
    }
    p["los_p995_global"] = float(np.percentile(real_df["los_days"], 99.5))

    # --- age-dependent laboratory ranges ----------------------------------
    cr_by_band = {}
    for lo, hi in AGE_BANDS:
        sl = real_df[(real_df["age"] >= lo) & (real_df["age"] < hi)]["creatinine_mean"]
        if len(sl) >= 30:
            cr_by_band[f"{lo}-{hi}"] = [float(np.percentile(sl, 0.5)),
                                        float(np.percentile(sl, 99.5))]
    p["creatinine_by_age_band"] = cr_by_band

    hb_by_band = {}
    for lo, hi in AGE_BANDS:
        sl = real_df[(real_df["age"] >= lo) & (real_df["age"] < hi)]["hemoglobin_mean"]
        if len(sl) >= 30:
            hb_by_band[f"{lo}-{hi}"] = [float(np.percentile(sl, 0.5)),
                                        float(np.percentile(sl, 99.5))]
    p["hemoglobin_by_age_band"] = hb_by_band

    return p


# ---------------------------------------------------------------------------
# Individual constraints — each returns a boolean mask over synth_df rows
# ---------------------------------------------------------------------------

def _c_pulse_pressure(df, p):
    """Pulse pressure (SBP - DBP) inside the real 0.5-99.5 percentile envelope.

    Not a range check: both endpoints are free, only the difference is
    constrained, so a generator that models SBP and DBP independently fails.
    """
    pp = df["sbp_mean"] - df["dbp_mean"]
    return (pp >= p["pulse_pressure_lo"]) & (pp <= p["pulse_pressure_hi"])


def _c_map_plausible(df, p):
    """Derived mean arterial pressure DBP + (SBP - DBP)/3 within real range."""
    mapv = df["dbp_mean"] + (df["sbp_mean"] - df["dbp_mean"]) / 3.0
    return (mapv >= p["map_lo"]) & (mapv <= p["map_hi"])


def _c_spo2_resp(df, p):
    """No implausible SpO2 / respiratory-rate pairing.

    Violations are (a) near-normal saturation with a respiratory rate in the
    top real percentile, and (b) profound hypoxaemia with a low-normal
    respiratory rate. Both require the joint distribution to be learned.
    """
    bad_hi = (df["spo2_mean"] >= p["spo2_high"]) & (df["resp_rate_mean"] >= p["resp_high"])
    bad_lo = (df["spo2_mean"] <= p["spo2_low"]) & (df["resp_rate_mean"] <= p["resp_low"])
    return ~(bad_hi | bad_lo)


def _c_apache_mortality(df, p, target="hospital_mortality"):
    """APACHE score consistent with the direction of mortality risk.

    A synthetic in-hospital death carrying an APACHE score below the 5th
    percentile of real deaths, or a survivor carrying a score above the 95th
    percentile of real survivors, inverts the severity-outcome relationship.
    Rows are only tested against the rule for their own outcome.
    """
    ok = pd.Series(True, index=df.index)
    dec = df[target] == 1
    sur = df[target] == 0
    if not np.isnan(p["apache_deceased_p05"]):
        ok[dec] = df.loc[dec, "apache_score"] >= p["apache_deceased_p05"]
    if not np.isnan(p["apache_survivor_p95"]):
        ok[sur] = df.loc[sur, "apache_score"] <= p["apache_survivor_p95"]
    return ok


def _c_deceased_los(df, p, target="hospital_mortality"):
    """Deceased patients have a plausible (non-degenerate) unit LOS.

    Conditional lower bound taken from the 1st percentile of LOS among real
    non-survivors; survivors are unconstrained by this rule.
    """
    ok = pd.Series(True, index=df.index)
    dec = df[target] == 1
    if not np.isnan(p["los_deceased_p01"]):
        ok[dec] = df.loc[dec, "los_days"] >= p["los_deceased_p01"]
    return ok


def _c_los_by_unit(df, p):
    """LOS consistent with the admitting unit type.

    Upper bound is the real 99.5th percentile of LOS WITHIN each unit type,
    so a generator must condition LOS on `admission_type_grouped`. A global
    min/max bound (which SDV supplies free) does not satisfy this.
    """
    bounds = df["admission_type_grouped"].astype(str).map(p["los_p995_by_unit"])
    bounds = bounds.fillna(p["los_p995_global"])
    return df["los_days"] <= bounds


def _band_of(age):
    for lo, hi in AGE_BANDS:
        if lo <= age < hi:
            return f"{lo}-{hi}"
    return None


def _c_lab_by_age(df, p, lab_col, key):
    """Laboratory value inside the real age-band-specific reference range."""
    table = p[key]
    if not table:
        return pd.Series(True, index=df.index)
    bands = df["age"].map(_band_of)
    lo = bands.map(lambda b: table.get(b, [-np.inf, np.inf])[0])
    hi = bands.map(lambda b: table.get(b, [-np.inf, np.inf])[1])
    return (df[lab_col] >= lo) & (df[lab_col] <= hi)


def _c_creatinine_by_age(df, p):
    """Creatinine inside the real age-band reference range (renal function
    declines with age, so a single global range is too permissive)."""
    return _c_lab_by_age(df, p, "creatinine_mean", "creatinine_by_age_band")


def _c_hemoglobin_by_age(df, p):
    """Haemoglobin inside the real age-band reference range."""
    return _c_lab_by_age(df, p, "hemoglobin_mean", "hemoglobin_by_age_band")


STRUCTURAL_CONSTRAINTS = {
    "pulse_pressure_plausible":   _c_pulse_pressure,
    "map_plausible":              _c_map_plausible,
    "spo2_resp_consistent":       _c_spo2_resp,
    "apache_mortality_consistent": _c_apache_mortality,
    "deceased_los_plausible":     _c_deceased_los,
    "los_by_unit_type_plausible": _c_los_by_unit,
    "creatinine_by_age_plausible": _c_creatinine_by_age,
    "hemoglobin_by_age_plausible": _c_hemoglobin_by_age,
}

# Human-readable definitions, reused verbatim by the supplementary material.
CONSTRAINT_DOCS = {
    "pulse_pressure_plausible":
        "SBP - DBP lies within the [0.5, 99.5] percentile interval of pulse "
        "pressure in the real training split.",
    "map_plausible":
        "Derived mean arterial pressure, DBP + (SBP - DBP)/3, lies within the "
        "[0.5, 99.5] percentile interval of the real training split.",
    "spo2_resp_consistent":
        "The record does not pair SpO2 at or above the real 90th percentile "
        "with a respiratory rate at or above the real 99th percentile, nor "
        "SpO2 at or below the real 1st percentile with a respiratory rate at "
        "or below the real 10th percentile.",
    "apache_mortality_consistent":
        "A record with hospital_mortality = 1 has an APACHE score at or above "
        "the 5th percentile of APACHE among real non-survivors; a record with "
        "hospital_mortality = 0 has an APACHE score at or below the 95th "
        "percentile of APACHE among real survivors.",
    "deceased_los_plausible":
        "A record with hospital_mortality = 1 has los_days at or above the "
        "1st percentile of LOS among real non-survivors.",
    "los_by_unit_type_plausible":
        "los_days does not exceed the 99.5th percentile of real LOS within "
        "the record's own admission_type_grouped category (global 99.5th "
        "percentile for categories with fewer than 30 real records).",
    "creatinine_by_age_plausible":
        "creatinine_mean lies within the [0.5, 99.5] percentile interval of "
        "real creatinine for the record's age band (18-40, 40-65, 65-80, 80+).",
    "hemoglobin_by_age_plausible":
        "hemoglobin_mean lies within the [0.5, 99.5] percentile interval of "
        "real haemoglobin for the record's age band (18-40, 40-65, 65-80, 80+).",
}

# Which constraints SDV enforces for free, for reporting in the supplement.
FREE_UNDER_SDV = (
    "All config.DOMAIN_CONSTRAINTS entries are single-column min/max bounds "
    "and are satisfied automatically because enforce_min_max_values defaults "
    "to True on every SDV 1.17.2 synthesizer used here. The eight structural "
    "constraints above, and the sbp_mean > dbp_mean cross-column check, are "
    "not enforced by SDV."
)


def evaluate_structural(synth_df, params, target="hospital_mortality"):
    """Per-constraint satisfaction rates for the structural family."""
    out = {}
    n = len(synth_df)
    if n == 0:
        return {k: np.nan for k in STRUCTURAL_CONSTRAINTS}
    for name, fn in STRUCTURAL_CONSTRAINTS.items():
        try:
            if name in ("apache_mortality_consistent", "deceased_los_plausible"):
                mask = fn(synth_df, params, target)
            else:
                mask = fn(synth_df, params)
            out[name] = float(np.asarray(mask, dtype=bool).mean())
        except Exception as e:  # pragma: no cover - defensive
            print(f"      constraint {name} failed: {e}")
            out[name] = np.nan
    return out


def direction_checks(synth_df, params, target="hospital_mortality"):
    """Dataset-level sanity checks reported alongside the per-row CSR.

    These are not part of CSR (they are properties of the whole table, not of
    a row) but they capture the severity-outcome direction the reviewer asked
    about.
    """
    dec = synth_df[synth_df[target] == 1]
    sur = synth_df[synth_df[target] == 0]
    gap = (float(dec["apache_score"].mean() - sur["apache_score"].mean())
           if len(dec) and len(sur) else np.nan)
    return {
        "apache_gap_synth": gap,
        "apache_gap_real": params.get("apache_gap_real"),
        "apache_direction_preserved": bool(gap > 0) if not np.isnan(gap) else False,
    }
