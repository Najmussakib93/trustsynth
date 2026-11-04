"""
trustsynth_fairness.py
Dimension 3 fairness metrics, including the Task 5 rule for Bias
Amplification Ratio (BAR) under empty synthetic subgroups.

THE PROBLEM
-----------
BAR is the ratio of demographic parity difference (DPD) measured on the
synthetic data to DPD measured on the real data. DPD is a contrast between
subgroup outcome rates. When a generator emits zero records for a protected
subgroup -- which TVAE does for HISPANIC, ASIAN and OTHER ethnicity -- that
subgroup has no outcome rate, so the synthetic contrast is not defined.

The original implementation substituted a rate of 0.0 for any subgroup absent
from the synthetic data:

    synth_rate_g0 = synth_df[synth_df[attr] == g0][target].mean() \
                    if g0 in synth_df[attr].values else 0

This silently converts "undefined" into "this subgroup has a 0% mortality
rate", which produces a finite BAR that looks like a measurement. A generator
that erases a subgroup entirely could score better than one that represents it
imperfectly. The original code also compared only the first two groups in
`Series.unique()` order, which is the order of first appearance in the file
and is therefore arbitrary.

THE RULE IMPLEMENTED HERE
-------------------------
See `BAR_RULE_TEXT` at the bottom of this module for the paste-ready
paragraph. In short: DPD is the max-minus-min outcome-rate spread over the
*common support* -- the subgroups represented in both the real and synthetic
data at a minimum count -- and the real DPD in the denominator is recomputed
over that same restricted subgroup set so numerator and denominator are
commensurable. If fewer than two subgroups survive, BAR is reported as
undefined (NaN), never imputed. Subgroup dropout is reported separately as
an explicit coverage metric so that erasure is penalised rather than hidden.
"""

import numpy as np
import pandas as pd

# A subgroup must have at least this many records in a dataset to be treated
# as present in it. Guards against a single stray synthetic row making a
# subgroup look represented.
MIN_SUBGROUP_N = 10


def _rates(df, attr, target, groups):
    return {g: float(df.loc[df[attr] == g, target].mean()) for g in groups}


def bias_amplification(real_df, synth_df, attr, target,
                       min_n=MIN_SUBGROUP_N):
    """BAR for one protected attribute under the common-support rule.

    Returns a dict with the BAR value (NaN when undefined), the flags that
    explain why, and the subgroup-coverage diagnostics.
    """
    real_counts = real_df[attr].value_counts()
    synth_counts = synth_df[attr].value_counts()

    real_groups = [g for g, c in real_counts.items() if c >= min_n]
    present = [g for g in real_groups if synth_counts.get(g, 0) >= min_n]
    missing = [g for g in real_groups if synth_counts.get(g, 0) < min_n]
    empty = [g for g in real_groups if synth_counts.get(g, 0) == 0]

    # Share of real records whose subgroup survives into the synthetic data.
    covered = float(real_counts.reindex(present).sum() / real_counts.reindex(real_groups).sum()) \
        if real_groups else np.nan

    out = {
        f"BAR_{attr}": np.nan,
        f"BAR_{attr}_defined": False,
        f"BAR_{attr}_status": "undefined_insufficient_common_support",
        f"DPD_real_{attr}": np.nan,
        f"DPD_synth_{attr}": np.nan,
        f"groups_real_{attr}": len(real_groups),
        f"groups_common_{attr}": len(present),
        f"groups_missing_{attr}": len(missing),
        f"groups_empty_{attr}": len(empty),
        f"missing_group_names_{attr}": ";".join(map(str, sorted(map(str, missing)))),
        f"empty_group_names_{attr}": ";".join(map(str, sorted(map(str, empty)))),
        f"subgroup_coverage_{attr}": covered,
    }

    if len(present) < 2:
        return out

    r_real = _rates(real_df, attr, target, present)
    r_synth = _rates(synth_df, attr, target, present)
    dpd_real = max(r_real.values()) - min(r_real.values())
    dpd_synth = max(r_synth.values()) - min(r_synth.values())

    out[f"DPD_real_{attr}"] = round(dpd_real, 6)
    out[f"DPD_synth_{attr}"] = round(dpd_synth, 6)

    # A real DPD of (numerically) zero leaves the ratio undefined too --
    # report it rather than dividing by an epsilon.
    if dpd_real < 1e-6:
        out[f"BAR_{attr}_status"] = "undefined_zero_real_dpd"
        return out

    out[f"BAR_{attr}"] = round(dpd_synth / dpd_real, 6)
    out[f"BAR_{attr}_defined"] = True
    out[f"BAR_{attr}_status"] = (
        "defined_full_support" if not missing else "defined_restricted_support"
    )
    return out


def representation_ratios(real_df, synth_df, attrs):
    """Synthetic-to-real subgroup proportion ratio for every subgroup."""
    out = {}
    for attr in attrs:
        if attr not in real_df.columns or attr not in synth_df.columns:
            continue
        real_props = real_df[attr].value_counts(normalize=True)
        synth_props = synth_df[attr].value_counts(normalize=True)
        for group in real_props.index:
            out[f"{attr}_{group}"] = round(
                float(synth_props.get(group, 0.0) / max(float(real_props[group]), 1e-10)), 6
            )
    return out


def evaluate_fairness(real_df, synth_df, protected_attributes, target,
                      rep_low, rep_high):
    """Dimension 3 composite.

    The composite keeps the original two-term form -- a representation term
    and a BAR term, each 1 - mean|value - 1| floored at zero -- so that
    changes relative to the original submission are attributable to the BAR
    rule rather than to a new scoring formula. The only change is that
    undefined BAR values are now dropped from the BAR term instead of being
    imputed as zero-rate subgroups, and `subgroup_coverage` is reported
    alongside (it is NOT folded into the score; the representation term
    already penalises a dropped subgroup with |0 - 1| = 1).
    """
    results = {}

    rep = representation_ratios(real_df, synth_df, protected_attributes)
    results["representation_ratios"] = rep
    results["representation_violations"] = sum(
        1 for r in rep.values() if r < rep_low or r > rep_high
    )
    results["representation_total_groups"] = len(rep)

    bar_defined, coverages = [], []
    for attr in protected_attributes:
        if attr not in real_df.columns or real_df[attr].nunique() < 2:
            continue
        info = bias_amplification(real_df, synth_df, attr, target)
        results.update(info)
        if info[f"BAR_{attr}_defined"]:
            bar_defined.append(info[f"BAR_{attr}"])
        cov = info[f"subgroup_coverage_{attr}"]
        if not np.isnan(cov):
            coverages.append(cov)

    results["n_bar_defined"] = len(bar_defined)
    results["subgroup_coverage_mean"] = (
        round(float(np.mean(coverages)), 6) if coverages else np.nan
    )

    scores = []
    if rep:
        scores.append(max(0.0, 1 - float(np.mean([abs(r - 1.0) for r in rep.values()]))))
    if bar_defined:
        scores.append(max(0.0, 1 - float(np.mean([abs(b - 1.0) for b in bar_defined]))))
        results["fairness_score_bar_term_available"] = True
    else:
        # No BAR term is available for this generator. The score rests on the
        # representation term alone and the generator is flagged, so the
        # table can mark it rather than presenting a silently different score.
        results["fairness_score_bar_term_available"] = False

    results["fairness_score"] = round(float(np.mean(scores)), 6) if scores else np.nan
    return results


BAR_RULE_TEXT = """\
Bias amplification ratio under incomplete subgroup support. For a protected
attribute A with subgroups g in G, we define the demographic parity
difference as the spread of the outcome rate across subgroups,
DPD(D, S) = max_{g in S} P_D(Y = 1 | A = g) - min_{g in S} P_D(Y = 1 | A = g),
evaluated on dataset D over a subgroup set S. The bias amplification ratio is
BAR = DPD(D_synth, S*) / DPD(D_real, S*), where S* is the common support: the
set of subgroups represented by at least n_min = 10 records in both the real
training split and the synthetic dataset. Restricting both the numerator and
the denominator to S* keeps the two contrasts commensurable. Because a
synthetic dataset that omits a subgroup provides no estimate of that
subgroup's outcome rate, we do not impute a rate for it; if fewer than two
subgroups remain in S*, or if DPD(D_real, S*) is zero, BAR is reported as
undefined and the generator is flagged rather than assigned a numeric value.
Subgroup erasure is reported separately and is not absorbed into BAR: for
each generator and protected attribute we report the number of real
subgroups absent from the synthetic data, their names, and the subgroup
coverage, defined as the proportion of real records belonging to subgroups
that survive into the synthetic data. In the composite fairness score,
undefined BAR values are excluded from the BAR term; omitted subgroups are
already penalised through the representation term, which contributes
|0 - 1| = 1 for every subgroup with zero synthetic representation.
"""
