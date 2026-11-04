"""Aggregate-only vital diagnostics; never changes cohort preprocessing.

Cached phase uses reconstructed pre-imputation masks, not equality to median.
Raw phase streams the entire vitalPeriodic source for the union of both cohort
variants. It measures units/clipping/time windows before clipping, and verifies
the cached means against the existing extraction algorithm.
"""
import argparse
import ast
import json
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .safe_reporting import (ROOT, PUBLIC, MIN_COUNT, count, fraction, event_summary,
                             private_path, write_csv, write_json)

FEATURES = {"systemicsystolic": "sbp_mean", "systemicdiastolic": "dbp_mean",
            "temperature": "temperature_mean", "heartrate": "heart_rate_mean",
            "sao2": "spo2_mean", "respiration": "resp_rate_mean"}
GENERATORS = ["GaussianCopula", "CTGAN", "TVAE"]


def bounds_from_code():
    tree = ast.parse((ROOT / "revision" / "legacy" / "extract_cohort.py").read_text(encoding="utf-8-sig"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_vitals")
    assignment = next(node for node in fn.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "clip_bounds" for t in node.targets))
    return ast.literal_eval(assignment.value)


def load_variants():
    result = {}
    for name in ["unfiltered_candidate", "binary_current"]:
        with private_path(name + ".pkl").open("rb") as stream:
            result[name] = pickle.load(stream)
    return result


def safe_numeric_summary(values):
    series = pd.Series(values).dropna()
    if len(series) < MIN_COUNT:
        return {"observed_n": count(len(series)), "status": "insufficient_support"}
    result = {"observed_n": count(len(series)), "mean": float(series.mean()), "sd": float(series.std()),
              "integer_fraction": fraction(np.isclose(series, np.round(series), rtol=0, atol=1e-9).sum(), len(series))}
    for label, q in [("p01", .01), ("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95), ("p99", .99)]:
        result[label] = float(series.quantile(q))
    return result


def cached():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = load_variants()
    synth = {g: pd.read_csv(ROOT / "data" / ("synthetic_" + g + ".csv")) for g in GENERATORS}
    summary, comparisons, histogram_rows = [], [], []
    clips = bounds_from_code()
    figure_paths = []
    for variant, bundle in variants.items():
        frame, raw = bundle["frame"], bundle["before_fill"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, (source_col, feature) in zip(axes.flat, FEATURES.items()):
            fill = bundle["imputation_values"][feature]
            missing = raw[feature].isna()
            observed = raw.loc[~missing, feature]
            sets = {"real_all": frame[feature], "real_observed": observed,
                    **{g: df[feature] for g, df in synth.items()}}
            for dataset, values in sets.items():
                s = safe_numeric_summary(values)
                lo, hi = clips[source_col]
                n = len(values)
                s.update({"variant": variant, "dataset": dataset, "feature": feature,
                          "at_clip_lower_fraction": fraction(np.isclose(values, lo, rtol=0, atol=1e-9).sum(), n),
                          "at_clip_upper_fraction": fraction(np.isclose(values, hi, rtol=0, atol=1e-9).sum(), n)})
                if dataset == "real_all":
                    s.update({"missing_before_fill_count": count(missing.sum()),
                              "imputed_fraction": fraction(missing.sum(), n), "fill_value": float(fill),
                              "observed_equal_fill_fraction": fraction(np.isclose(observed, fill, rtol=0, atol=1e-9).sum(), len(observed))})
                summary.append(s)
            for generator, synthetic in synth.items():
                rv, sv = frame[feature].to_numpy(), synthetic[feature].dropna().to_numpy()
                # ECDF discontinuity at the documented fill value bounds KS.
                real_atom = float(np.mean(rv == fill))
                synth_atom = float(np.mean(sv == fill))
                at_left = abs(float(np.mean(rv < fill) - np.mean(sv < fill)))
                at_right = abs(float(np.mean(rv <= fill) - np.mean(sv <= fill)))
                comparisons.append({"variant": variant, "generator": generator, "feature": feature,
                    "ks_all": float(ks_2samp(rv, sv).statistic),
                    "ks_observed_only_diagnostic": float(ks_2samp(observed, sv).statistic),
                    "imputed_fraction": fraction(missing.sum(), len(rv)),
                    "ks_at_imputation_jump": max(at_left, at_right),
                    "atom_KS_lower_bound": max(0.0, (real_atom - synth_atom) / 2),
                    "note": "observed-only comparison changes the reference population; diagnostic, not a replacement metric"})
            # Same bins for all datasets. Do not expose small bins or a lone
            # suppressed bin recoverable from total and the other bins.
            low = min(float(np.quantile(v, .001)) for v in sets.values())
            high = max(float(np.quantile(v, .999)) for v in sets.values())
            edges = np.linspace(low, high, 61)
            for dataset, values in sets.items():
                # Outer bins include tails; finite plotting endpoints represent
                # the display window only, not exclusion of tail observations.
                calculation_edges = edges.copy()
                calculation_edges[0], calculation_edges[-1] = -np.inf, np.inf
                bins, _ = np.histogram(values, bins=calculation_edges)
                hidden = bins < MIN_COUNT
                secondary = np.zeros_like(hidden)
                if hidden.sum() == 1 and (~hidden).any():
                    eligible = np.flatnonzero(~hidden)
                    secondary[eligible[np.argmin(bins[eligible])]] = True
                visible = ~(hidden | secondary)
                density = bins.astype(float) / (len(values) * np.diff(edges))
                density[~visible] = np.nan
                centers = (edges[:-1] + edges[1:]) / 2
                ax.step(centers, density, where="mid", label=dataset, alpha=.85)
                for i, n in enumerate(bins):
                    histogram_rows.append({"variant": variant, "feature": feature, "dataset": dataset,
                        "bin_lower": "-inf" if i == 0 else edges[i],
                        "bin_upper": "inf" if i == len(bins)-1 else edges[i+1],
                        "count": "<11" if hidden[i] else "suppressed" if secondary[i] else int(n)})
            ax.axvline(fill, color="gray", linestyle=":")
            ax.set_title(feature.replace("_mean", "").replace("_", " "))
            ax.set_ylabel("Density (small bins suppressed)")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(variant + ": saved synthetics versus reconstructed real marginals")
        fig.tight_layout()
        folder = PUBLIC / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        for extension in ["png", "tiff"]:
            path = folder / ("vitals_" + variant + "." + extension)
            kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
            fig.savefig(path, dpi=500, **kwargs)
            figure_paths.append(path.relative_to(ROOT).as_posix())
        plt.close(fig)
    write_json(PUBLIC / "diagnostics" / "vitals_summary.json", summary)
    write_csv(PUBLIC / "diagnostics" / "vitals_ks_decomposition.csv", comparisons)
    write_csv(PUBLIC / "diagnostics" / "vitals_histogram_bins.csv", histogram_rows)
    write_json(PUBLIC / "diagnostics" / "vitals_figure_manifest.json", {"paths": figure_paths, "dpi": 500})
    for row in comparisons:
        if row["variant"] == "unfiltered_candidate" and row["feature"] in ["sbp_mean", "dbp_mean", "temperature_mean"]:
            print(row["feature"], row["generator"], "KS", round(row["ks_all"], 4),
                  "observed-only", round(row["ks_observed_only_diagnostic"], 4),
                  "imputed", row["imputed_fraction"], flush=True)


def raw_pilot():
    source = ROOT / "data" / "vitalPeriodic.csv"
    start = time.perf_counter()
    with source.open("rb") as stream:
        reader = pd.read_csv(stream, usecols=["patientunitstayid", "observationoffset"] + list(FEATURES), chunksize=500_000)
        first_chunk = next(reader)
        consumed = stream.tell()
        del first_chunk
        reader.close()
    elapsed = time.perf_counter() - start
    result = {"source_bytes": source.stat().st_size, "pilot_bytes_consumed": consumed,
              "pilot_seconds": elapsed, "estimated_read_minutes": source.stat().st_size / consumed * elapsed / 60,
              "estimate_caveat": "full diagnostic grouping adds CPU work; allow 2-3 times the streaming estimate"}
    write_json(PUBLIC / "diagnostics" / "raw_scan_runtime_pilot.json", result)
    print(json.dumps(result, indent=2))


def raw_scan():
    start = time.perf_counter()
    variants = load_variants()
    ids = np.unique(np.concatenate([v["stay_ids"] for v in variants.values()]))
    lookup = pd.Series(np.arange(len(ids)), index=ids)
    clips = bounds_from_code()
    accumulated = {feature: {"sum": np.zeros(len(ids)), "n": np.zeros(len(ids), dtype=np.int64),
                             "out_of_range": np.zeros(len(ids), dtype=np.int64),
                             "integer": np.zeros(len(ids), dtype=np.int64)} for feature in FEATURES}
    total_readings = np.zeros(len(ids), dtype=np.int64)
    after24 = np.zeros(len(ids), dtype=np.int64)
    before0 = np.zeros(len(ids), dtype=np.int64)
    temp_f_like = np.zeros(len(ids), dtype=np.int64)
    source = ROOT / "data" / "vitalPeriodic.csv"
    last_update = time.perf_counter()
    with source.open("rb") as stream:
        for chunk in pd.read_csv(stream, usecols=["patientunitstayid", "observationoffset"] + list(FEATURES), chunksize=500_000):
            selected = chunk[chunk.patientunitstayid.isin(ids)]
            index = selected.patientunitstayid.map(lookup).to_numpy(dtype=int)
            total_readings += np.bincount(index, minlength=len(ids))
            after24 += np.bincount(index[selected.observationoffset.to_numpy() > 1440], minlength=len(ids))
            before0 += np.bincount(index[selected.observationoffset.to_numpy() < 0], minlength=len(ids))
            for raw, (lo, hi) in clips.items():
                values = selected[raw].to_numpy(dtype=float)
                valid = np.isfinite(values)
                loc, val = index[valid], values[valid]
                acc = accumulated[raw]
                acc["sum"] += np.bincount(loc, weights=np.clip(val, lo, hi), minlength=len(ids))
                acc["n"] += np.bincount(loc, minlength=len(ids))
                acc["out_of_range"] += np.bincount(loc[(val < lo) | (val > hi)], minlength=len(ids))
                acc["integer"] += np.bincount(loc[np.isclose(val, np.round(val), rtol=0, atol=1e-9)], minlength=len(ids))
                if raw == "temperature":
                    temp_f_like += np.bincount(loc[(val >= 77) & (val <= 113)], minlength=len(ids))
            if time.perf_counter() - last_update > 30:
                print("Raw vital scan", format(stream.tell() / source.stat().st_size, ".0%"), "complete", flush=True)
                last_update = time.perf_counter()
    report = {}
    for variant, bundle in variants.items():
        positions = lookup.loc[bundle["stay_ids"]].to_numpy(dtype=int)
        n_total = total_readings[positions].sum()
        stats = {"vital_rows": count(n_total),
                 "after_24h": event_summary(after24[positions].sum(), n_total),
                 "before_admission": event_summary(before0[positions].sum(), n_total),
                 "features": {}}
        for raw, feature in FEATURES.items():
            acc = accumulated[raw]
            n = acc["n"][positions]
            means = np.divide(acc["sum"][positions], n, out=np.full(len(n), np.nan), where=n > 0)
            cached_values = bundle["before_fill"][feature].to_numpy(dtype=float)
            valid = np.isfinite(means) & np.isfinite(cached_values)
            stats["features"][feature] = {
                "observed_readings": count(n.sum()),
                "outside_extraction_bounds": event_summary(acc["out_of_range"][positions].sum(), n.sum()),
                "integer_readings_fraction": fraction(acc["integer"][positions].sum(), n.sum()),
                "stays_without_readings": event_summary((n == 0).sum(), len(n)),
                "cache_missingness_matches": bool(np.array_equal(np.isnan(means), np.isnan(cached_values))),
                "cache_values_match_at_1e_minus_8": bool(np.allclose(means, cached_values, rtol=0, atol=1e-8, equal_nan=True)),
                "cache_max_absolute_difference": float(np.abs(means[valid] - cached_values[valid]).max()) if valid.any() else None,
                "extraction_bounds": list(clips[raw]),
            }
        stats["temperature_Fahrenheit_like_77_to_113"] = event_summary(temp_f_like[positions].sum(), accumulated["temperature"]["n"][positions].sum())
        stats["temperature_unit_caution"] = "Value interval is only a screening heuristic; no explicit unit tag is available in vitalPeriodic."
        report[variant] = stats
    report["wall_seconds"] = time.perf_counter() - start
    write_json(PUBLIC / "diagnostics" / "raw_vitals_audit.json", report)
    # These arrays are per-stay intermediates, so they are private.
    with private_path("raw_vitals_accumulators.pkl").open("wb") as output:
        pickle.dump({"stay_ids": ids, "accumulated": accumulated}, output, protocol=4)
    print("Full raw-vitals aggregate audit saved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["cached", "pilot", "raw"])
    args = parser.parse_args()
    {"cached": cached, "pilot": raw_pilot, "raw": raw_scan}[args.phase]()
