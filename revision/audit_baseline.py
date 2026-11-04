"""Re-evaluate saved synthetic artifacts using identifiable legacy metric code.

The original Git evaluator is loaded as source, never its main entry point.
Its feature configuration is the current eICU schema, so this is explicitly a
reconstruction, not a claim that HEAD generated the eICU paper. No generators
are retrained here. All scalar results retain the legacy rounding convention.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import pickle
import random
import subprocess
import time
import types

import numpy as np
import pandas as pd

from .legacy import config
from trustsynth_gower import gower_min_distance
from .safe_reporting import ROOT, PUBLIC, count, private_path, write_csv, write_json

GENERATORS = ["GaussianCopula", "CTGAN", "TVAE"]


def legacy_module():
    source = subprocess.check_output(["git", "show", "HEAD:02_run_evaluation.py"], cwd=ROOT).decode("utf-8")
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "config":
            node.module = "revision.legacy.config"
    module = types.ModuleType("legacy_audit")
    exec(compile(tree, "git_HEAD_02_run_evaluation.py", "exec"), module.__dict__)
    return module, hashlib.sha256(source.encode("utf-8")).hexdigest()


def flatten(tree, prefix=""):
    for key, value in tree.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            yield from flatten(value, name + ".")
        elif isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            yield name, float(value)


def compare(old, fresh, variant):
    rows = []
    for generator in GENERATORS:
        expected = dict(flatten(old[generator]))
        actual = dict(flatten(fresh.get(generator, {})))
        for metric, prior in expected.items():
            value = actual.get(metric)
            finite = value is not None and np.isfinite(value)
            # These counts concern checks, not individual patients, but suppress
            # small printed counts conservatively under the user's broad rule.
            is_count = metric.endswith(("representation_violations", "representation_total_groups",
                                         "constraints_below_95", "total_constraints"))
            rows.append({"variant": variant, "generator": generator, "legacy_metric": metric,
                         "archived": count(prior) if is_count else prior,
                         "recomputed": (count(value) if is_count else value) if finite else "",
                         "difference": "" if not finite or is_count else value - prior,
                         "status": "pending" if value is None else "undefined" if not finite
                         else "match_at_archived_precision" if abs(value - prior) < 5e-8 else "changed"})
    return rows


def repeat_prediction_check(module, train, test, synthetic):
    """Independent fits compared before metric rounding; no predictions exported."""
    from xgboost import XGBClassifier
    columns = [c for c in config.ALL_FEATURES if c != config.TSTR_TARGET]
    encoded_train, encoders = module.encode_dataframe(train)
    encoded_test, _ = module.encode_dataframe(test, encoders)
    encoded_synthetic, _ = module.encode_dataframe(synthetic, encoders)
    Xtest = np.nan_to_num(encoded_test[columns].to_numpy(dtype=float))
    answers = {}
    for label, frame in [("TRTR", encoded_train), ("TSTR", encoded_synthetic)]:
        predictions = []
        for repeat in range(2):
            random.seed(42)
            np.random.seed(42)
            model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                                  eval_metric="logloss", verbosity=0, n_jobs=4)
            model.fit(np.nan_to_num(frame[columns].to_numpy(dtype=float)), frame.hospital_mortality.to_numpy(dtype=int))
            predictions.append(model.predict_proba(Xtest))
        answers[label + "_predictions_bitwise_identical"] = bool(np.array_equal(*predictions))
    return answers


def run_quick():
    started = time.perf_counter()
    module, source_hash = legacy_module()
    old = json.loads((ROOT / "results" / "detailed_metrics.json").read_text(encoding="utf-8"))
    synth = {g: pd.read_csv(ROOT / "data" / ("synthetic_" + g + ".csv")) for g in GENERATORS}
    comparisons, summaries = [], {}
    private_result = {}
    for variant in ["unfiltered_candidate", "binary_current"]:
        with private_path(variant + ".pkl").open("rb") as stream:
            bundle = pickle.load(stream)
        train, test = bundle["train"], bundle["test"]
        results = {}
        for generator, synthetic in synth.items():
            random.seed(42)
            np.random.seed(42)
            # Historical code may print unsuppressed exceptions. Capture them,
            # and export only known scalar metric fields.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                results[generator] = {
                    "utility": module.evaluate_utility(train, synthetic, generator),
                    "fairness": module.evaluate_fairness(train, synthetic, test, generator),
                    "fidelity": module.evaluate_fidelity(train, synthetic, test, generator),
                    "integrity": module.evaluate_integrity(synthetic, generator),
                }
            print(variant, generator, "TRTR", results[generator]["fidelity"].get("auc_trtr"),
                  "TSTR", results[generator]["fidelity"].get("auc_tstr"),
                  "KS", results[generator]["utility"].get("ks_mean"), flush=True)
        private_result[variant] = results
        comparisons.extend(compare(old, results, variant))
        summaries[variant] = repeat_prediction_check(module, train, test, synth["GaussianCopula"])
    with private_path("legacy_metrics.pkl").open("wb") as stream:
        pickle.dump(private_result, stream, protocol=4)
    write_csv(PUBLIC / "audit" / "baseline_comparison.csv", comparisons)
    write_json(PUBLIC / "audit" / "baseline_run.json", {
        "status": "recomputed_saved_synthetic_sets_not_generator_retraining",
        "legacy_evaluator_source_sha256": source_hash,
        "configuration": "current eICU feature schema and original 15 constraints",
        "seed": 42, "same_seed_repeat_checks": summaries,
        "wall_seconds": time.perf_counter() - started,
        "caution": "binary_current uses archived synthetics with an unverified training cohort; it is a compatibility diagnostic, not a corrected TSTR experiment",
        "missing": ["manuscript numerical inventory", "original eICU extraction snapshot", "original generator fit provenance"],
    })
    print("Scalar baseline comparison saved; raw prediction repeat checks:", summaries)


def benchmark():
    """Verify fixed-constant implementation on fabricated data, then time real aggregate calculation."""
    import gower
    rng = np.random.RandomState(42)
    toy = pd.DataFrame({"number": rng.normal(5, 1, 77), "category": rng.choice(["A", "B"], 77),
                        "constant": np.ones(77)})
    X, Y = toy.iloc[:29], toy.iloc[29:]
    reference = gower.gower_matrix(X, Y, cat_features=[False, True, False]).min(axis=1)
    candidate = gower_min_distance(X, Y, [False, True, False], n_jobs=1)
    exact = bool(np.array_equal(reference, candidate))
    if not exact:
        raise RuntimeError("Gower compatibility test failed")
    with private_path("unfiltered_candidate.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    train = bundle["train"]
    synthetic = pd.read_csv(ROOT / "data" / "synthetic_GaussianCopula.csv")
    idx = rng.choice(len(synthetic), 32, replace=False)
    t0 = time.perf_counter()
    # Arrays remain in memory; only time and implementation agreement leave it.
    gower_min_distance(synthetic.iloc[idx], train,
                       [c in config.CATEGORICAL_FEATURES for c in train], n_jobs=1)
    elapsed = time.perf_counter() - t0
    result = {"fabricated_data_bitwise_agreement_with_gower_0_1_2": exact,
              "pilot_query_rows": 32, "reference_rows": count(len(train)),
              "pilot_seconds": elapsed,
              "estimated_four_2000_query_evaluations_minutes": elapsed * (2000 / 32) * 4 / 60,
              "memory_strategy": "one distance row against reference at a time; no full matrix"}
    write_json(PUBLIC / "audit" / "privacy_runtime_pilot.json", result)
    print(json.dumps(result, indent=2))


def run_privacy():
    started = time.perf_counter()
    module, _ = legacy_module()
    old = json.loads((ROOT / "results" / "detailed_metrics.json").read_text(encoding="utf-8"))
    with private_path("unfiltered_candidate.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    with private_path("legacy_metrics.pkl").open("rb") as stream:
        results = pickle.load(stream)
    train, test = bundle["train"], bundle["test"]
    holdout = None
    def fixed_distance(df1, df2, n_sample=1000, seed=42):
        nonlocal holdout
        if df1 is test and holdout is not None:
            return holdout
        idx = np.random.RandomState(seed).choice(len(df1), min(n_sample, len(df1)), replace=False)
        distances = gower_min_distance(df1.iloc[idx], df2,
            [c in config.CATEGORICAL_FEATURES for c in df1], n_jobs=1)
        if df1 is test:
            holdout = distances
        return distances
    module.compute_gower_distance_sample = fixed_distance
    for generator in GENERATORS:
        synthetic = pd.read_csv(ROOT / "data" / ("synthetic_" + generator + ".csv"))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            metrics = module.evaluate_privacy(train, test, synthetic, generator)
        if metrics.get("dcr_ratio") is None or metrics.get("mia_accuracy") is None:
            raise RuntimeError("Legacy privacy metric failed; no silent distance fallback allowed")
        results["unfiltered_candidate"][generator]["privacy"] = metrics
        print(generator, "MIA", metrics["mia_accuracy"], "DCR ratio", metrics["dcr_ratio"], flush=True)
        with private_path("legacy_metrics.pkl").open("wb") as stream:
            pickle.dump(results, stream, protocol=4)
    rows = []
    for variant, values in results.items():
        rows.extend(compare(old, values, variant))
    write_csv(PUBLIC / "audit" / "baseline_comparison.csv", rows)
    write_json(PUBLIC / "audit" / "privacy_audit.json", {
        "distance": "Gower 0.1.2 equivalent fixed constants, no Euclidean fallback",
        "numeric_normalization": "combined query-sample/reference range via package float32 extrema and two-stage arithmetic",
        "categorical_columns": config.CATEGORICAL_FEATURES,
        "binary_target_distance": "numeric (0/1 range)",
        "ratio": "round(median(synthetic_to_train),4) / round(median(test_to_train),4), then rounded to 4 decimals",
        "queries_per_set": 2000,
        "membership_attack": "5-fold cross-validated logistic regression accuracy on min, mean, 10th percentile and SD of standardized Euclidean distances to first 5000 synthetic rows",
        "analytic_KS_null_applicable": False,
        "wall_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["quick", "benchmark", "privacy"])
    args = parser.parse_args()
    {"quick": run_quick, "benchmark": benchmark, "privacy": run_privacy}[args.phase]()
