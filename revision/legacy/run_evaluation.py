"""
02_run_evaluation.py
Main TrustSynth evaluation pipeline — multi-seed (Task 2).

For every seed in config.SEEDS:
  * re-draw the stratified 80/20 split on hospital mortality
  * re-train every generator from scratch
  * recompute all five dimensions

Input:  data/cohort.csv (from 01_extract_cohort.py)
Output: results/tidy_results.csv       one row per (seed, generator, metric)
        results/summary_mean_sd.csv    mean +/- SD per generator per metric
        results/dimension_scores.csv   mean scores, for 03_generate_outputs.py
        results/detailed_metrics.json  per-seed nested detail
        results/constraint_params.json calibrated constraint thresholds
        results/gender_audit.csv       Task 1 verification
"""

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from config import *
from trustsynth_constraints import (
    CONSTRAINT_DOCS, STRUCTURAL_CONSTRAINTS, direction_checks,
    evaluate_structural, fit_constraints,
)
from trustsynth_fairness import evaluate_fairness as fairness_dimension

SYNTH_DIR = DATA_DIR / "synthetic"


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_all_seeds(seed):
    """Seed every RNG the pipeline touches.

    SDV 1.17 exposes no `random_state` constructor argument; its synthesizers
    draw from the global numpy and torch generators during both fit and
    sample, so seeding globally before `fit` is what makes a run
    reproducible. `_set_random_state` is additionally called on each fitted
    synthesizer before sampling (see `_fit_and_sample`). XGBoost and
    scikit-learn estimators receive `random_state=seed` explicitly at every
    construction site.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def software_versions():
    """Versions of everything that can move a number, for the supplement."""
    import platform
    import sklearn
    import scipy
    v = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for mod in ["sdv", "ctgan", "rdt", "copulas", "torch", "xgboost", "gower"]:
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "not installed"
    return v


# ============================================================================
# SDV METADATA HELPER (handles SDV 1.x API differences)
# ============================================================================

def make_sdv_metadata(df):
    """Create SDV single-table metadata, compatible with SDV >= 1.0."""
    try:
        from sdv.metadata import Metadata
        return Metadata.detect_from_dataframe(data=df)
    except (AttributeError, TypeError):
        pass
    from sdv.metadata import SingleTableMetadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def encode_dataframe(df, label_encoders=None):
    """Encode categorical features for ML models. Returns encoded df and encoders."""
    df_enc = df.copy()
    if label_encoders is None:
        label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df_enc.columns:
            if col not in label_encoders:
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders[col]
                known = set(le.classes_)
                df_enc[col] = df_enc[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df_enc[col] = le.transform(df_enc[col])
    return df_enc, label_encoders


def _gower_chunk(sample_chunk, reference, cat_features):
    import gower
    return gower.gower_matrix(sample_chunk, reference,
                              cat_features=cat_features).min(axis=1)


def compute_gower_distance_sample(df1, df2, n_sample=1000, seed=42, n_jobs=-1):
    """Minimum Gower distance from a sample of df1 to every row of df2.

    Parallelised across row-chunks of the sample. The distance values are
    identical to the single-threaded computation -- Gower distance is
    computed per (row of df1, row of df2) pair and the row-wise minimum does
    not depend on how the rows are grouped -- only the wall-clock time
    changes.
    """
    try:
        import gower
    except ImportError:
        print("  gower package not found; using Euclidean distance on encoded features.")
        return _fallback_distance(df1, df2, n_sample, seed)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df1), size=min(n_sample, len(df1)), replace=False)
    sample = df1.iloc[idx].reset_index(drop=True)
    cat_features = [col in CATEGORICAL_FEATURES for col in sample.columns]

    try:
        from joblib import Parallel, delayed
        n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
        n_workers = max(1, min(n_workers, len(sample)))
        chunks = np.array_split(np.arange(len(sample)), n_workers)
        parts = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_gower_chunk)(sample.iloc[c].reset_index(drop=True), df2, cat_features)
            for c in chunks if len(c)
        )
        return np.concatenate(parts)
    except Exception as e:
        print(f"  parallel gower failed ({e}); falling back to single-threaded.")
        try:
            return gower.gower_matrix(sample, df2, cat_features=cat_features).min(axis=1)
        except Exception as e2:
            print(f"  gower.gower_matrix failed ({e2}); using Euclidean distance.")
            return _fallback_distance(df1, df2, n_sample, seed)


def _fallback_distance(df1, df2, n_sample=1000, seed=42):
    """Euclidean distance fallback if gower is unavailable."""
    enc1, les = encode_dataframe(df1)
    enc2, _ = encode_dataframe(df2, les)
    scaler = MinMaxScaler()
    cols = [c for c in enc1.columns if c in enc2.columns]
    X1 = scaler.fit_transform(enc1[cols].values.astype(float))
    X2 = scaler.transform(enc2[cols].values.astype(float))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X1), size=min(n_sample, len(X1)), replace=False)
    return cdist(X1[idx], X2, metric="euclidean").min(axis=1)


# ============================================================================
# STEP 1: DATA LOADING AND SPLITTING
# ============================================================================

def load_cohort():
    df = pd.read_csv(DATA_DIR / "cohort.csv")
    if ALLOWED_GENDERS:
        bad = sorted(set(df["gender"].unique()) - set(ALLOWED_GENDERS))
        if bad:
            raise RuntimeError(
                f"cohort.csv contains disallowed gender categories {bad}. "
                "Re-run 01_extract_cohort.py."
            )
    return df


def split_for_seed(df, seed):
    """Stratified 80/20 split on the mortality outcome, re-drawn per seed."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, random_state=seed,
        stratify=df[TSTR_TARGET] if TSTR_TARGET in df.columns else None,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================================
# STEP 2: SYNTHETIC DATA GENERATION
# ============================================================================

def _fit_and_sample(synth, train_df, n_synth, seed):
    t0 = time.time()
    synth.fit(train_df)
    try:
        synth._set_random_state(seed)
    except Exception:
        pass
    out = synth.sample(num_rows=n_synth)
    return out, time.time() - t0


def generate_synthetic(train_df, seed):
    """Train generators for one seed. Per-seed cache keyed by seed number.

    The original implementation returned early as soon as ANY cached
    `synthetic_*.csv` existed, which both skipped untrained generators and
    made a multi-seed run impossible. Caching is now keyed on the seed and
    checked per generator.
    """
    print(f"\n[Step 2] Generators for seed {seed}...")
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_datasets, generation_times = {}, {}
    n_synth = int(len(train_df) * SYNTHETIC_SIZE_MULTIPLIER)

    builders = {
        "GaussianCopula": lambda md: __import__(
            "sdv.single_table", fromlist=["GaussianCopulaSynthesizer"]
        ).GaussianCopulaSynthesizer(md),
        "CTGAN": lambda md: __import__(
            "sdv.single_table", fromlist=["CTGANSynthesizer"]
        ).CTGANSynthesizer(md, epochs=CTGAN_EPOCHS, verbose=False, cuda=False),
        "TVAE": lambda md: __import__(
            "sdv.single_table", fromlist=["TVAESynthesizer"]
        ).TVAESynthesizer(md, epochs=TVAE_EPOCHS, cuda=False),
    }

    for name, build in builders.items():
        cache = SYNTH_DIR / f"seed{seed}_{name}.csv"
        if cache.exists() and USE_SYNTH_CACHE:
            print(f"  [{name}] cached -> {cache.name}")
            synthetic_datasets[name] = pd.read_csv(cache)
            generation_times[name] = np.nan
            continue
        print(f"  [{name}] training...", flush=True)
        try:
            set_all_seeds(seed)
            metadata = make_sdv_metadata(train_df)
            sdf, elapsed = _fit_and_sample(build(metadata), train_df, n_synth, seed)
            synthetic_datasets[name] = sdf
            generation_times[name] = elapsed
            sdf.to_csv(cache, index=False)
            print(f"  [{name}] done in {elapsed:.1f}s, {len(sdf)} rows")
        except Exception as e:
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")

    return synthetic_datasets, generation_times


# ============================================================================
# DIMENSION 1: STATISTICAL UTILITY
# ============================================================================

def evaluate_utility(real_df, synth_df):
    results = {}

    ks_scores = {}
    for col in CONTINUOUS_FEATURES:
        if col in real_df.columns and col in synth_df.columns:
            rv, sv = real_df[col].dropna().values, synth_df[col].dropna().values
            if len(rv) and len(sv):
                ks_scores[col] = round(float(stats.ks_2samp(rv, sv).statistic), 6)
    results["ks_per_feature"] = ks_scores
    results["ks_mean"] = round(float(np.mean(list(ks_scores.values()))), 6) if ks_scores else None

    jsd_scores = {}
    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        if col in real_df.columns and col in synth_df.columns:
            rc = real_df[col].value_counts(normalize=True)
            sc = synth_df[col].value_counts(normalize=True)
            cats = sorted(set(rc.index) | set(sc.index), key=str)
            p = np.array([rc.get(c, 0) for c in cats]) + 1e-10
            q = np.array([sc.get(c, 0) for c in cats]) + 1e-10
            p, q = p / p.sum(), q / q.sum()
            from scipy.spatial.distance import jensenshannon
            jsd_scores[col] = round(float(jensenshannon(p, q) ** 2), 6)
    results["jsd_per_feature"] = jsd_scores
    results["jsd_mean"] = round(float(np.mean(list(jsd_scores.values()))), 6) if jsd_scores else None

    cont = [c for c in CONTINUOUS_FEATURES if c in real_df.columns and c in synth_df.columns]
    if len(cont) >= 2:
        cr, cs = real_df[cont].corr().values, synth_df[cont].corr().values
        mask = np.triu(np.ones_like(cr, dtype=bool), k=1)
        results["corr_deviation"] = round(float(np.nanmean(np.abs(cr[mask] - cs[mask]))), 6)
    else:
        results["corr_deviation"] = None

    scores = []
    if results["ks_mean"] is not None:
        scores.append(max(0.0, 1 - results["ks_mean"] / KS_NORM_REF))
    if results["jsd_mean"] is not None:
        scores.append(max(0.0, 1 - results["jsd_mean"] / JSD_NORM_REF))
    if results["corr_deviation"] is not None:
        scores.append(max(0.0, 1 - results["corr_deviation"] / CORR_NORM_REF))
    results["utility_score"] = round(float(np.mean(scores)), 6) if scores else None
    return results


# ============================================================================
# DIMENSION 2: PRIVACY PRESERVATION
# ============================================================================

def evaluate_privacy(train_df, test_df, synth_df, seed, dcr_holdout_cache=None):
    results = {}
    enc_train, les = encode_dataframe(train_df)
    enc_test, _ = encode_dataframe(test_df, les)
    enc_synth, _ = encode_dataframe(synth_df, les)
    all_cols = [c for c in enc_train.columns if c in enc_synth.columns and c in enc_test.columns]

    # --- 2a: membership inference -------------------------------------
    try:
        n_sample = min(2000, len(enc_train), len(enc_test))
        members = enc_train[all_cols].sample(n=n_sample, random_state=seed)
        non_members = enc_test[all_cols].sample(n=n_sample, random_state=seed)

        scaler = StandardScaler()
        X_synth = scaler.fit_transform(enc_synth[all_cols].values.astype(float))
        X_mem = scaler.transform(members.values.astype(float))
        X_non = scaler.transform(non_members.values.astype(float))
        synth_sample = X_synth[:min(5000, len(X_synth))]

        def distance_features(X):
            d = cdist(X, synth_sample, metric="euclidean")
            return np.column_stack([d.min(axis=1), d.mean(axis=1),
                                    np.percentile(d, 10, axis=1), np.std(d, axis=1)])

        X_attack = np.vstack([distance_features(X_mem), distance_features(X_non)])
        y_attack = np.array([1] * n_sample + [0] * n_sample)
        attack = LogisticRegression(max_iter=1000, random_state=seed)
        mia = cross_val_score(attack, X_attack, y_attack, cv=5, scoring="accuracy")
        results["mia_accuracy"] = round(float(np.mean(mia)), 6)
        results["mia_std"] = round(float(np.std(mia)), 6)
    except Exception as e:
        print(f"    MIA failed: {e}")
        results["mia_accuracy"] = None

    # --- 2b: distance to closest record -------------------------------
    try:
        dcr_s = compute_gower_distance_sample(synth_df, train_df,
                                              n_sample=DCR_SAMPLE_N, seed=seed)
        if dcr_holdout_cache is None:
            dcr_h = compute_gower_distance_sample(test_df, train_df,
                                                  n_sample=DCR_SAMPLE_N, seed=seed)
        else:
            dcr_h = dcr_holdout_cache
        results["dcr_synth_median"] = round(float(np.median(dcr_s)), 6)
        results["dcr_synth_5pct"] = round(float(np.percentile(dcr_s, DCR_PERCENTILE)), 6)
        results["dcr_holdout_median"] = round(float(np.median(dcr_h)), 6)
        results["dcr_holdout_5pct"] = round(float(np.percentile(dcr_h, DCR_PERCENTILE)), 6)
        results["dcr_ratio"] = round(
            results["dcr_synth_median"] / max(results["dcr_holdout_median"], 1e-10), 6)
        results["_dcr_holdout_raw"] = dcr_h
    except Exception as e:
        print(f"    DCR failed: {e}")
        results["dcr_synth_median"] = None

    scores = []
    if results.get("mia_accuracy") is not None:
        scores.append(max(0.0, 1 - 2 * (results["mia_accuracy"] - 0.5)))
    if results.get("dcr_ratio") is not None:
        scores.append(min(1.0, results["dcr_ratio"]))
    results["privacy_score"] = round(min(1.0, float(np.mean(scores))), 6) if scores else None
    return results


# ============================================================================
# DIMENSION 4: DOWNSTREAM TASK FIDELITY (incl. Task 4)
# ============================================================================

def _permutation_importance_on_real_test(model, X_test, y_test, seed):
    """Permutation importance measured on the REAL held-out test set.

    Using the same evaluation data for the TRTR model and every TSTR model is
    what makes the importance vectors comparable: any difference is then a
    property of the model, not of the data it is scored on. The original
    implementation instead compared XGBoost gain importances, which are
    computed from each model's own training data (real for TRTR, synthetic
    for TSTR) and so confound the two.
    """
    r = permutation_importance(
        model, X_test, y_test,
        n_repeats=PERM_IMPORTANCE_REPEATS,
        random_state=seed,
        scoring="roc_auc",
        n_jobs=1,
    )
    return r.importances_mean


def evaluate_fidelity(train_df, synth_df, test_df, seed):
    results = {}
    target = TSTR_TARGET
    if target not in train_df.columns:
        return {"fidelity_score": None}

    feature_cols = [c for c in ALL_FEATURES if c != target and c in train_df.columns]
    enc_train, les = encode_dataframe(train_df)
    enc_synth, _ = encode_dataframe(synth_df, les)
    enc_test, _ = encode_dataframe(test_df, les)

    X_train = np.nan_to_num(enc_train[feature_cols].values.astype(float))
    y_train = enc_train[target].values.astype(int)
    X_synth = np.nan_to_num(enc_synth[feature_cols].values.astype(float))
    y_synth = enc_synth[target].values.astype(int)
    X_test = np.nan_to_num(enc_test[feature_cols].values.astype(float))
    y_test = enc_test[target].values.astype(int)

    try:
        from xgboost import XGBClassifier

        def mk():
            return XGBClassifier(n_estimators=100, max_depth=4, random_state=seed,
                                 eval_metric="logloss", verbosity=0, n_jobs=4,
                                 tree_method="hist")

        model_real = mk().fit(X_train, y_train)
        auc_trtr = roc_auc_score(y_test, model_real.predict_proba(X_test)[:, 1])

        if len(np.unique(y_synth)) < 2:
            # A generator that emitted only one outcome class cannot be
            # trained against; record it rather than crashing.
            results.update(auc_trtr=round(float(auc_trtr), 6), auc_tstr=None,
                           tstr_gap=None, fi_rank_correlation=None,
                           tstr_single_class=True)
            return results

        model_synth = mk().fit(X_synth, y_synth)
        auc_tstr = roc_auc_score(y_test, model_synth.predict_proba(X_test)[:, 1])

        results["auc_trtr"] = round(float(auc_trtr), 6)
        results["auc_tstr"] = round(float(auc_tstr), 6)
        results["tstr_gap"] = round(float(abs(auc_trtr - auc_tstr)), 6)
        results["tstr_single_class"] = False

        # --- Task 4: permutation-importance rank correlation -----------
        fi_real = _permutation_importance_on_real_test(model_real, X_test, y_test, seed)
        fi_synth = _permutation_importance_on_real_test(model_synth, X_test, y_test, seed)
        rho, pval = spearmanr(fi_real, fi_synth)
        results["fi_rank_correlation"] = round(float(rho), 6) if np.isfinite(rho) else None
        results["fi_rank_correlation_p"] = round(float(pval), 6) if np.isfinite(pval) else None
        results["fi_importances_trtr"] = {f: round(float(v), 6)
                                          for f, v in zip(feature_cols, fi_real)}
        results["fi_importances_tstr"] = {f: round(float(v), 6)
                                          for f, v in zip(feature_cols, fi_synth)}

        # Gain-based correlation retained so the change from the original
        # submission is auditable.
        rho_gain, _ = spearmanr(model_real.feature_importances_,
                                model_synth.feature_importances_)
        results["fi_rank_correlation_gain_legacy"] = (
            round(float(rho_gain), 6) if np.isfinite(rho_gain) else None)

        results["brier_score_tstr"] = round(
            float(brier_score_loss(y_test, model_synth.predict_proba(X_test)[:, 1])), 6)
        results["brier_score_trtr"] = round(
            float(brier_score_loss(y_test, model_real.predict_proba(X_test)[:, 1])), 6)
    except Exception as e:
        print(f"    Fidelity evaluation failed: {type(e).__name__}: {e}")
        results["tstr_gap"] = None

    scores = []
    if results.get("tstr_gap") is not None:
        scores.append(max(0.0, 1 - results["tstr_gap"] / TSTR_GAP_NORM_REF))
    if results.get("fi_rank_correlation") is not None:
        scores.append(max(0.0, results["fi_rank_correlation"]))
    results["fidelity_score"] = round(float(np.mean(scores)), 6) if scores else None
    return results


# ============================================================================
# DIMENSION 5: STRUCTURAL INTEGRITY (incl. Task 6)
# ============================================================================

def evaluate_integrity(synth_df, constraint_params):
    results = {}
    n = len(synth_df)
    range_results, structural_results = {}, {}

    for name, (col, op, val) in DOMAIN_CONSTRAINTS.items():
        if col not in synth_df.columns:
            continue
        s = synth_df[col]
        sat = {">=": lambda: (s >= val).sum(), "<=": lambda: (s <= val).sum(),
               ">": lambda: (s > val).sum(), "<": lambda: (s < val).sum(),
               "in": lambda: s.isin(val).sum()}.get(op, lambda: n)()
        range_results[name] = round(float(sat / n), 6) if n else 0.0

    for col1, op, col2 in CROSS_CONSTRAINTS:
        if col1 in synth_df.columns and col2 in synth_df.columns:
            sat = (synth_df[col1] > synth_df[col2]).sum() if op == ">" \
                else (synth_df[col1] < synth_df[col2]).sum() if op == "<" else n
            range_results[f"{col1}_{op}_{col2}"] = round(float(sat / n), 6) if n else 0.0

    structural_results = {k: round(float(v), 6)
                          for k, v in evaluate_structural(synth_df, constraint_params).items()}

    all_results = {**range_results, **structural_results}
    results["constraint_results"] = all_results
    results["constraint_results_range"] = range_results
    results["constraint_results_structural"] = structural_results
    results["csr"] = round(float(np.nanmean(list(all_results.values()))), 6) if all_results else None
    results["csr_range_only"] = round(float(np.nanmean(list(range_results.values()))), 6) if range_results else None
    results["csr_structural_only"] = round(float(np.nanmean(list(structural_results.values()))), 6) if structural_results else None
    results["constraints_below_95"] = int(sum(1 for v in all_results.values() if v < 0.95))
    results["total_constraints"] = len(all_results)
    results.update(direction_checks(synth_df, constraint_params))
    results["integrity_score"] = results["csr"]
    return results


# ============================================================================
# TIDY RESULT ASSEMBLY
# ============================================================================

def flatten_to_tidy(seed, generator, dims, gen_time):
    """Flatten the nested per-dimension dicts into (metric, value) rows."""
    rows = []

    def put(metric, value):
        if isinstance(value, (bool, np.bool_)):
            value = float(value)
        if value is None or isinstance(value, (str, list, dict)):
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        rows.append({"seed": seed, "generator": generator,
                     "metric": metric, "value": v})

    put("gen_time_s", gen_time)
    for dim, res in dims.items():
        for k, v in res.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                for sub, sv in v.items():
                    put(f"{dim}.{k}.{sub}", sv)
            else:
                put(f"{dim}.{k}", v)
    return rows


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 72)
    print("  TrustSynth Multi-Seed Evaluation Pipeline")
    print("=" * 72)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    versions = software_versions()
    print("\nSoftware:")
    for k, v in versions.items():
        print(f"  {k:16s} {v}")
    print(f"\nSeeds: {SEEDS}")
    print(f"CTGAN epochs: {CTGAN_EPOCHS}, TVAE epochs: {TVAE_EPOCHS}")

    cohort = load_cohort()
    print(f"\nCohort: {len(cohort):,} rows x {len(cohort.columns)} columns")
    print(f"  gender value counts: {cohort['gender'].value_counts().to_dict()}")

    tidy_rows, detailed, gender_audit, constraint_params_by_seed = [], {}, [], {}
    t_start = time.time()

    for si, seed in enumerate(SEEDS, 1):
        print("\n" + "#" * 72)
        print(f"#  SEED {seed}   ({si}/{len(SEEDS)})   elapsed {(time.time()-t_start)/60:.1f} min")
        print("#" * 72)

        set_all_seeds(seed)
        train_df, test_df = split_for_seed(cohort, seed)
        print(f"  Train {len(train_df):,} / Test {len(test_df):,} "
              f"(mortality {train_df[TSTR_TARGET].mean():.3%} / {test_df[TSTR_TARGET].mean():.3%})")

        # --- Task 1 audit: gender counts in train and test -------------
        for split_name, d in [("train", train_df), ("test", test_df)]:
            vc = d["gender"].value_counts().to_dict()
            print(f"  gender[{split_name}]: {vc}")
            for g, c in vc.items():
                gender_audit.append({"seed": seed, "dataset": split_name,
                                     "gender": g, "count": int(c)})

        # Constraint thresholds calibrated on this seed's real training split
        cparams = fit_constraints(train_df, target=TSTR_TARGET)
        constraint_params_by_seed[str(seed)] = cparams

        synthetic_datasets, gen_times = generate_synthetic(train_df, seed)
        if not synthetic_datasets:
            print("  ERROR: no generator succeeded for this seed.")
            continue

        detailed[str(seed)] = {}
        dcr_holdout_cache = None

        for name, synth_df in synthetic_datasets.items():
            print(f"\n  --- {name} (seed {seed}) ---")
            common = [c for c in train_df.columns if c in synth_df.columns]
            synth_aligned = synth_df[common].copy()

            vc = synth_aligned["gender"].value_counts().to_dict()
            print(f"    gender[{name}]: {vc}")
            for g, c in vc.items():
                gender_audit.append({"seed": seed, "dataset": f"synthetic_{name}",
                                     "gender": g, "count": int(c)})

            d1 = evaluate_utility(train_df[common], synth_aligned)
            print(f"    D1 KS={d1.get('ks_mean')} JSD={d1.get('jsd_mean')} "
                  f"corr={d1.get('corr_deviation')} score={d1.get('utility_score')}")

            d2 = evaluate_privacy(train_df, test_df, synth_df, seed, dcr_holdout_cache)
            if dcr_holdout_cache is None and d2.get("_dcr_holdout_raw") is not None:
                # Holdout-to-real DCR depends only on the split, so it is
                # computed once per seed and reused across generators.
                dcr_holdout_cache = d2.pop("_dcr_holdout_raw")
            d2.pop("_dcr_holdout_raw", None)
            print(f"    D2 MIA={d2.get('mia_accuracy')} DCR={d2.get('dcr_ratio')} "
                  f"score={d2.get('privacy_score')}")

            d3 = fairness_dimension(train_df, synth_aligned, PROTECTED_ATTRIBUTES,
                                    TSTR_TARGET, REPRESENTATION_THRESHOLD_LOW,
                                    REPRESENTATION_THRESHOLD_HIGH)
            print(f"    D3 repviol={d3.get('representation_violations')}/"
                  f"{d3.get('representation_total_groups')} "
                  f"BAR_eth={d3.get('BAR_ethnicity_grouped')} "
                  f"({d3.get('BAR_ethnicity_grouped_status')}) "
                  f"empty={d3.get('empty_group_names_ethnicity_grouped') or 'none'} "
                  f"score={d3.get('fairness_score')}")

            d4 = evaluate_fidelity(train_df, synth_aligned, test_df, seed)
            print(f"    D4 TRTR={d4.get('auc_trtr')} TSTR={d4.get('auc_tstr')} "
                  f"gap={d4.get('tstr_gap')} FIrho={d4.get('fi_rank_correlation')} "
                  f"score={d4.get('fidelity_score')}")

            d5 = evaluate_integrity(synth_aligned, cparams)
            print(f"    D5 CSR={d5.get('csr')} (range {d5.get('csr_range_only')}, "
                  f"structural {d5.get('csr_structural_only')}) "
                  f"below95={d5.get('constraints_below_95')}/{d5.get('total_constraints')}")

            dims = {"utility": d1, "privacy": d2, "fairness": d3,
                    "fidelity": d4, "integrity": d5}
            detailed[str(seed)][name] = dims
            tidy_rows.extend(flatten_to_tidy(seed, name, dims, gen_times.get(name)))

    # ---------------- persist ----------------
    tidy = pd.DataFrame(tidy_rows)
    tidy.to_csv(RESULTS_DIR / "tidy_results.csv", index=False)

    summary = (tidy.groupby(["generator", "metric"])["value"]
               .agg(mean="mean", sd="std", n="count", min="min", max="max")
               .reset_index())
    summary.to_csv(RESULTS_DIR / "summary_mean_sd.csv", index=False)

    pd.DataFrame(gender_audit).to_csv(RESULTS_DIR / "gender_audit.csv", index=False)
    with open(RESULTS_DIR / "detailed_metrics.json", "w") as f:
        json.dump(detailed, f, indent=2, default=str)
    with open(RESULTS_DIR / "constraint_params.json", "w") as f:
        json.dump({"per_seed": constraint_params_by_seed,
                   "definitions": CONSTRAINT_DOCS}, f, indent=2, default=str)
    with open(RESULTS_DIR / "run_metadata.json", "w") as f:
        json.dump({"seeds": SEEDS, "ctgan_epochs": CTGAN_EPOCHS,
                   "tvae_epochs": TVAE_EPOCHS, "test_split": TEST_SPLIT,
                   "cohort_n": int(len(cohort)), "dcr_sample_n": DCR_SAMPLE_N,
                   "perm_importance_repeats": PERM_IMPORTANCE_REPEATS,
                   "allowed_genders": ALLOWED_GENDERS,
                   "software": versions,
                   "total_runtime_min": round((time.time() - t_start) / 60, 2)},
                  f, indent=2)

    # Mean dimension scores, in the shape 03_generate_outputs.py expects
    def mval(gen, metric):
        s = summary[(summary.generator == gen) & (summary.metric == metric)]
        return float(s["mean"].iloc[0]) if len(s) else np.nan

    def sdval(gen, metric):
        s = summary[(summary.generator == gen) & (summary.metric == metric)]
        return float(s["sd"].iloc[0]) if len(s) else np.nan

    gens = sorted(tidy["generator"].unique())
    col_map = [
        ("Utility", "utility.utility_score"), ("Privacy", "privacy.privacy_score"),
        ("Fairness", "fairness.fairness_score"), ("Fidelity", "fidelity.fidelity_score"),
        ("Integrity", "integrity.integrity_score"), ("KS_mean", "utility.ks_mean"),
        ("JSD_mean", "utility.jsd_mean"), ("Corr_dev", "utility.corr_deviation"),
        ("MIA_acc", "privacy.mia_accuracy"), ("DCR_ratio", "privacy.dcr_ratio"),
        ("AUC_TRTR", "fidelity.auc_trtr"), ("AUC_TSTR", "fidelity.auc_tstr"),
        ("TSTR_gap", "fidelity.tstr_gap"), ("FI_corr", "fidelity.fi_rank_correlation"),
        ("CSR", "integrity.csr"), ("CSR_structural", "integrity.csr_structural_only"),
        ("CSR_range", "integrity.csr_range_only"),
        ("Subgroup_coverage", "fairness.subgroup_coverage_mean"),
        ("Gen_time_s", "gen_time_s"),
    ]
    rows = []
    for g in gens:
        row = {"Method": g}
        for label, metric in col_map:
            row[label] = mval(g, metric)
            row[label + "_SD"] = sdval(g, metric)
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "dimension_scores.csv", index=False)

    # ---------------- Task 2: TVAE subgroup-erasure count ----------------
    print("\n" + "=" * 72)
    print("  SUBGROUP ERASURE ACROSS SEEDS (Task 2)")
    print("=" * 72)
    erasure_rows = []
    for g in gens:
        for grp in ["HISPANIC", "ASIAN", "OTHER", "BLACK", "WHITE"]:
            n_zero = 0
            for seed in SEEDS:
                d = detailed.get(str(seed), {}).get(g)
                if not d:
                    continue
                empty = d["fairness"].get("empty_group_names_ethnicity_grouped", "")
                if grp in str(empty).split(";"):
                    n_zero += 1
            erasure_rows.append({"generator": g, "subgroup": grp,
                                 "seeds_with_zero_records": n_zero,
                                 "n_seeds": len(SEEDS)})
    erasure = pd.DataFrame(erasure_rows)
    erasure.to_csv(RESULTS_DIR / "subgroup_erasure.csv", index=False)
    piv = erasure.pivot(index="generator", columns="subgroup",
                        values="seeds_with_zero_records")
    print(piv.to_string())
    print(f"\n(cells are the number of the {len(SEEDS)} seeds in which the generator "
          f"produced zero records for that ethnicity group)")

    print("\n" + "=" * 72)
    print("  MEAN +/- SD DIMENSION SCORES")
    print("=" * 72)
    for g in gens:
        parts = [f"{lab}={mval(g, m):.3f}+/-{sdval(g, m):.3f}"
                 for lab, m in col_map[:5]]
        print(f"  {g:16s} " + "  ".join(parts))

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} min")
    print(f"Results written to {RESULTS_DIR}/")
    print("Run 03_generate_outputs.py to produce paper figures and LaTeX tables.")


if __name__ == "__main__":
    main()
