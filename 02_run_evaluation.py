"""
02_run_evaluation.py
Main TrustSynth evaluation pipeline.
Trains generators, generates synthetic data, and evaluates across all 5 dimensions.

Input:  data/cohort.csv (from 01_extract_cohort.py)
Output: results/dimension_scores.csv, results/detailed_metrics.json
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, brier_score_loss
)

from config import *

# ============================================================================
# SDV METADATA HELPER (handles SDV 1.x API differences)
# ============================================================================

def make_sdv_metadata(df):
    """Create SDV single-table metadata, compatible with SDV >= 1.0."""
    try:
        # SDV >= 1.12: Metadata class with detect_from_dataframe classmethod
        from sdv.metadata import Metadata
        return Metadata.detect_from_dataframe(data=df)
    except (AttributeError, TypeError):
        pass
    # SDV 1.0–1.11: SingleTableMetadata
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
                # Handle unseen categories
                known = set(le.classes_)
                df_enc[col] = df_enc[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df_enc[col] = le.transform(df_enc[col])
    return df_enc, label_encoders


def compute_gower_distance_sample(df1, df2, n_sample=1000, seed=42):
    """Compute Gower distances between sampled rows of df1 and all of df2."""
    try:
        import gower
    except ImportError:
        print("  gower package not found; using Euclidean distance on encoded features.")
        return _fallback_distance(df1, df2, n_sample, seed)

    np.random.seed(seed)
    idx = np.random.choice(len(df1), size=min(n_sample, len(df1)), replace=False)
    sample = df1.iloc[idx].reset_index(drop=True)

    # gower expects a boolean array (one entry per column): True = categorical
    cat_features = [col in CATEGORICAL_FEATURES for col in sample.columns]

    try:
        dist_matrix = gower.gower_matrix(sample, df2, cat_features=cat_features)
    except Exception as e:
        print(f"  gower.gower_matrix failed ({e}); falling back to Euclidean distance.")
        return _fallback_distance(df1, df2, n_sample, seed)

    min_distances = dist_matrix.min(axis=1)
    return min_distances


def _fallback_distance(df1, df2, n_sample=1000, seed=42):
    """Euclidean distance fallback if gower is unavailable."""
    enc1, les = encode_dataframe(df1)
    enc2, _ = encode_dataframe(df2, les)

    # Normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cols = [c for c in enc1.columns if c in enc2.columns]
    X1 = scaler.fit_transform(enc1[cols].values.astype(float))
    X2 = scaler.transform(enc2[cols].values.astype(float))

    np.random.seed(seed)
    idx = np.random.choice(len(X1), size=min(n_sample, len(X1)), replace=False)

    from scipy.spatial.distance import cdist
    dists = cdist(X1[idx], X2, metric="euclidean")
    return dists.min(axis=1)


# ============================================================================
# STEP 1: DATA LOADING AND SPLITTING
# ============================================================================

def load_and_split():
    """Load cohort and split into train/test."""
    print("\n[Step 1] Loading cohort and splitting data...")
    df = pd.read_csv(DATA_DIR / "cohort.csv")
    print(f"  Loaded {len(df)} records with {len(df.columns)} features")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
        stratify=df[TSTR_TARGET] if TSTR_TARGET in df.columns else None
    )
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================================
# STEP 2: SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic(train_df):
    """Train generators and produce synthetic datasets. Loads from cache if available."""
    print("\n[Step 2] Training generators and generating synthetic data...")
    synthetic_datasets = {}
    generation_times = {}
    n_synth = int(len(train_df) * SYNTHETIC_SIZE_MULTIPLIER)

    # Load any already-generated synthetic datasets from disk (skip re-training)
    cached_names = []
    for name in ["GaussianCopula", "CTGAN", "TVAE", "DPGAN", "PATEGAN"]:
        path = DATA_DIR / f"synthetic_{name}.csv"
        if path.exists():
            print(f"  Found cached {name} at {path} — loading instead of re-training")
            synthetic_datasets[name] = pd.read_csv(path)
            generation_times[name] = None  # Already measured on first run
            cached_names.append(name)

    if cached_names:
        print(f"  Loaded {len(cached_names)} cached datasets: {cached_names}")
        return synthetic_datasets, generation_times

    # --- 2a: GaussianCopula ---
    print("\n  [2a] GaussianCopula (statistical baseline)...")
    try:
        from sdv.single_table import GaussianCopulaSynthesizer

        metadata = make_sdv_metadata(train_df)
        gc = GaussianCopulaSynthesizer(metadata)

        t0 = time.time()
        gc.fit(train_df)
        synth_gc = gc.sample(num_rows=n_synth)
        generation_times["GaussianCopula"] = time.time() - t0

        synthetic_datasets["GaussianCopula"] = synth_gc
        print(f"    Done in {generation_times['GaussianCopula']:.1f}s, {len(synth_gc)} rows")
    except Exception as e:
        print(f"    FAILED: {e}")

    # --- 2b: CTGAN ---
    print("\n  [2b] CTGAN...")
    try:
        from sdv.single_table import CTGANSynthesizer

        metadata = make_sdv_metadata(train_df)
        try:
            ctgan = CTGANSynthesizer(metadata, epochs=CTGAN_EPOCHS, verbose=False)
        except TypeError:
            ctgan = CTGANSynthesizer(metadata, epochs=CTGAN_EPOCHS)

        t0 = time.time()
        ctgan.fit(train_df)
        synth_ctgan = ctgan.sample(num_rows=n_synth)
        generation_times["CTGAN"] = time.time() - t0

        synthetic_datasets["CTGAN"] = synth_ctgan
        print(f"    Done in {generation_times['CTGAN']:.1f}s, {len(synth_ctgan)} rows")
    except Exception as e:
        print(f"    FAILED: {e}")

    # --- 2c: TVAE ---
    print("\n  [2c] TVAE...")
    try:
        from sdv.single_table import TVAESynthesizer

        metadata = make_sdv_metadata(train_df)
        tvae = TVAESynthesizer(metadata, epochs=TVAE_EPOCHS)

        t0 = time.time()
        tvae.fit(train_df)
        synth_tvae = tvae.sample(num_rows=n_synth)
        generation_times["TVAE"] = time.time() - t0

        synthetic_datasets["TVAE"] = synth_tvae
        print(f"    Done in {generation_times['TVAE']:.1f}s, {len(synth_tvae)} rows")
    except Exception as e:
        print(f"    FAILED: {e}")

    # --- 2d: DPGAN (via synthcity) ---
    print("\n  [2d] DPGAN (differentially private)...")
    try:
        from synthcity.plugins import Plugins

        t0 = time.time()
        dpgan = Plugins().get("dpgan", epsilon=DPGAN_EPSILON, n_iter=200)
        dpgan.fit(train_df)
        synth_dpgan = dpgan.generate(count=n_synth).dataframe()
        generation_times["DPGAN"] = time.time() - t0

        synthetic_datasets["DPGAN"] = synth_dpgan
        print(f"    Done in {generation_times['DPGAN']:.1f}s, {len(synth_dpgan)} rows")
    except Exception as e:
        print(f"    DPGAN failed: {e}")
        print("    Trying PATEGAN as fallback...")
        try:
            from synthcity.plugins import Plugins
            t0 = time.time()
            pategan = Plugins().get("pategan", n_iter=200)
            pategan.fit(train_df)
            synth_pategan = pategan.generate(count=n_synth).dataframe()
            generation_times["PATEGAN"] = time.time() - t0
            synthetic_datasets["PATEGAN"] = synth_pategan
            print(f"    PATEGAN done in {generation_times['PATEGAN']:.1f}s")
        except Exception as e2:
            print(f"    PATEGAN also failed: {e2}")
            print("    Continuing without a DP generator.")

    # Save all synthetic datasets
    for name, sdf in synthetic_datasets.items():
        sdf.to_csv(DATA_DIR / f"synthetic_{name}.csv", index=False)

    print(f"\n  Generated {len(synthetic_datasets)} synthetic datasets: {list(synthetic_datasets.keys())}")
    return synthetic_datasets, generation_times


# ============================================================================
# DIMENSION 1: STATISTICAL UTILITY
# ============================================================================

def evaluate_utility(real_df, synth_df, method_name):
    """Evaluate Dimension 1: Statistical Utility."""
    results = {}

    # 1a: KS statistic for continuous features
    ks_scores = {}
    for col in CONTINUOUS_FEATURES:
        if col in real_df.columns and col in synth_df.columns:
            real_vals = real_df[col].dropna().values
            synth_vals = synth_df[col].dropna().values
            if len(real_vals) > 0 and len(synth_vals) > 0:
                ks_stat, _ = stats.ks_2samp(real_vals, synth_vals)
                ks_scores[col] = round(ks_stat, 4)
    results["ks_per_feature"] = ks_scores
    results["ks_mean"] = round(np.mean(list(ks_scores.values())), 4) if ks_scores else None

    # 1b: JSD for categorical features
    jsd_scores = {}
    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        if col in real_df.columns and col in synth_df.columns:
            real_counts = real_df[col].value_counts(normalize=True).sort_index()
            synth_counts = synth_df[col].value_counts(normalize=True).sort_index()
            # Align categories
            all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
            p = np.array([real_counts.get(c, 0) for c in all_cats])
            q = np.array([synth_counts.get(c, 0) for c in all_cats])
            # Add small epsilon to avoid log(0)
            p = p + 1e-10
            q = q + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            jsd = jensenshannon(p, q) ** 2  # scipy returns sqrt(JSD)
            jsd_scores[col] = round(jsd, 4)
    results["jsd_per_feature"] = jsd_scores
    results["jsd_mean"] = round(np.mean(list(jsd_scores.values())), 4) if jsd_scores else None

    # 1c: Correlation deviation
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in real_df.columns and c in synth_df.columns]
    if len(cont_cols) >= 2:
        corr_real = real_df[cont_cols].corr().values
        corr_synth = synth_df[cont_cols].corr().values
        mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)
        diffs = np.abs(corr_real[mask] - corr_synth[mask])
        results["corr_deviation"] = round(np.mean(diffs), 4)
    else:
        results["corr_deviation"] = None

    # Composite utility score (0-1, higher is better)
    scores = []
    if results["ks_mean"] is not None:
        scores.append(max(0, 1 - results["ks_mean"] / 0.5))  # Normalize: 0.5 KS → score 0
    if results["jsd_mean"] is not None:
        scores.append(max(0, 1 - results["jsd_mean"] / 0.25))
    if results["corr_deviation"] is not None:
        scores.append(max(0, 1 - results["corr_deviation"] / 0.5))
    results["utility_score"] = round(np.mean(scores), 4) if scores else None

    return results


# ============================================================================
# DIMENSION 2: PRIVACY PRESERVATION
# ============================================================================

def evaluate_privacy(train_df, test_df, synth_df, method_name):
    """Evaluate Dimension 2: Privacy Preservation."""
    results = {}

    # Encode all dataframes
    enc_train, les = encode_dataframe(train_df)
    enc_test, _ = encode_dataframe(test_df, les)
    enc_synth, _ = encode_dataframe(synth_df, les)

    all_cols = [c for c in enc_train.columns if c in enc_synth.columns and c in enc_test.columns]

    # 2a: Membership Inference Attack (simplified)
    # Idea: train a classifier to distinguish "member" (train) vs "non-member" (test)
    # using distances to synthetic data as features
    print(f"    Running MIA for {method_name}...")
    try:
        n_sample = min(2000, len(enc_train), len(enc_test))
        members = enc_train[all_cols].sample(n=n_sample, random_state=RANDOM_SEED)
        non_members = enc_test[all_cols].sample(n=n_sample, random_state=RANDOM_SEED)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_synth = scaler.fit_transform(enc_synth[all_cols].values.astype(float))
        X_members = scaler.transform(members.values.astype(float))
        X_non_members = scaler.transform(non_members.values.astype(float))

        from scipy.spatial.distance import cdist

        # Features: distance statistics to synthetic data (sample for speed)
        synth_sample = X_synth[:min(5000, len(X_synth))]

        def distance_features(X):
            dists = cdist(X, synth_sample, metric="euclidean")
            return np.column_stack([
                dists.min(axis=1),
                dists.mean(axis=1),
                np.percentile(dists, 10, axis=1),
                np.std(dists, axis=1),
            ])

        feat_mem = distance_features(X_members)
        feat_non = distance_features(X_non_members)

        X_attack = np.vstack([feat_mem, feat_non])
        y_attack = np.array([1]*len(feat_mem) + [0]*len(feat_non))

        # Train attack model with cross-validation
        from sklearn.model_selection import cross_val_score
        attack_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        mia_scores = cross_val_score(attack_model, X_attack, y_attack, cv=5, scoring="accuracy")
        results["mia_accuracy"] = round(np.mean(mia_scores), 4)
        results["mia_std"] = round(np.std(mia_scores), 4)
    except Exception as e:
        print(f"    MIA failed: {e}")
        results["mia_accuracy"] = None

    # 2b: Distance to Closest Record (DCR)
    print(f"    Computing DCR for {method_name}...")
    try:
        dcr_synth_to_real = compute_gower_distance_sample(synth_df, train_df, n_sample=2000)
        dcr_holdout_to_real = compute_gower_distance_sample(test_df, train_df, n_sample=2000)

        results["dcr_synth_median"] = round(float(np.median(dcr_synth_to_real)), 4)
        results["dcr_synth_5pct"] = round(float(np.percentile(dcr_synth_to_real, 5)), 4)
        results["dcr_holdout_median"] = round(float(np.median(dcr_holdout_to_real)), 4)
        results["dcr_holdout_5pct"] = round(float(np.percentile(dcr_holdout_to_real, 5)), 4)
        results["dcr_ratio"] = round(results["dcr_synth_median"] / max(results["dcr_holdout_median"], 1e-10), 4)
    except Exception as e:
        print(f"    DCR failed: {e}")
        results["dcr_synth_median"] = None

    # Composite privacy score (0-1, higher is better = more private)
    scores = []
    if results.get("mia_accuracy") is not None:
        # MIA at 0.5 = perfect privacy, at 1.0 = no privacy
        scores.append(max(0, 1 - 2 * (results["mia_accuracy"] - 0.5)))
    if results.get("dcr_ratio") is not None:
        # DCR ratio near 1.0 = good (synthetic distances similar to holdout distances)
        scores.append(min(1.0, results["dcr_ratio"]))
    results["privacy_score"] = round(min(1.0, np.mean(scores)), 4) if scores else None

    return results


# ============================================================================
# DIMENSION 3: FAIRNESS
# ============================================================================

def evaluate_fairness(train_df, synth_df, test_df, method_name):
    """Evaluate Dimension 3: Fairness."""
    results = {}

    # 3a: Subgroup representation fidelity
    rep_ratios = {}
    for attr in PROTECTED_ATTRIBUTES:
        if attr in train_df.columns and attr in synth_df.columns:
            real_props = train_df[attr].value_counts(normalize=True)
            synth_props = synth_df[attr].value_counts(normalize=True)
            for group in real_props.index:
                real_p = real_props.get(group, 0)
                synth_p = synth_props.get(group, 0)
                ratio = synth_p / max(real_p, 1e-10)
                rep_ratios[f"{attr}_{group}"] = round(ratio, 4)
    results["representation_ratios"] = rep_ratios

    # Count violations
    violations = sum(
        1 for r in rep_ratios.values()
        if r < REPRESENTATION_THRESHOLD_LOW or r > REPRESENTATION_THRESHOLD_HIGH
    )
    results["representation_violations"] = violations
    results["representation_total_groups"] = len(rep_ratios)

    # 3b: Bias Amplification Ratio (BAR) on downstream task
    target = TSTR_TARGET
    if target in train_df.columns:
        for attr in PROTECTED_ATTRIBUTES:
            if attr in train_df.columns and train_df[attr].nunique() >= 2:
                # Compute DPD on real data
                groups = train_df[attr].unique()
                if len(groups) >= 2:
                    g0, g1 = groups[0], groups[1]
                    real_rate_g0 = train_df[train_df[attr] == g0][target].mean()
                    real_rate_g1 = train_df[train_df[attr] == g1][target].mean()
                    dpd_real = abs(real_rate_g0 - real_rate_g1)

                    synth_rate_g0 = synth_df[synth_df[attr] == g0][target].mean() if g0 in synth_df[attr].values else 0
                    synth_rate_g1 = synth_df[synth_df[attr] == g1][target].mean() if g1 in synth_df[attr].values else 0
                    dpd_synth = abs(synth_rate_g0 - synth_rate_g1)

                    bar = dpd_synth / max(dpd_real, 1e-10)
                    results[f"BAR_{attr}"] = round(bar, 4)

    # Composite fairness score
    scores = []
    if rep_ratios:
        # Average distance from perfect representation (1.0)
        rep_score = 1 - np.mean([abs(r - 1.0) for r in rep_ratios.values()])
        scores.append(max(0, rep_score))
    bar_values = [v for k, v in results.items() if k.startswith("BAR_")]
    if bar_values:
        # BAR near 1.0 is ideal; penalize deviation
        bar_score = 1 - np.mean([abs(b - 1.0) for b in bar_values])
        scores.append(max(0, bar_score))
    results["fairness_score"] = round(np.mean(scores), 4) if scores else None

    return results


# ============================================================================
# DIMENSION 4: DOWNSTREAM TASK FIDELITY
# ============================================================================

def evaluate_fidelity(train_df, synth_df, test_df, method_name):
    """Evaluate Dimension 4: Downstream Task Fidelity."""
    results = {}
    target = TSTR_TARGET

    if target not in train_df.columns:
        print(f"    Target {target} not found. Skipping fidelity evaluation.")
        return {"fidelity_score": None}

    feature_cols = [c for c in ALL_FEATURES if c != target and c in train_df.columns]

    # Encode
    enc_train, les = encode_dataframe(train_df)
    enc_synth, _ = encode_dataframe(synth_df, les)
    enc_test, _ = encode_dataframe(test_df, les)

    X_train = enc_train[feature_cols].values.astype(float)
    y_train = enc_train[target].values.astype(int)
    X_synth = enc_synth[feature_cols].values.astype(float)
    y_synth = enc_synth[target].values.astype(int)
    X_test = enc_test[feature_cols].values.astype(float)
    y_test = enc_test[target].values.astype(int)

    # Replace NaN
    X_train = np.nan_to_num(X_train)
    X_synth = np.nan_to_num(X_synth)
    X_test = np.nan_to_num(X_test)

    # 4a: TSTR AUC gap
    try:
        from xgboost import XGBClassifier

        # TRTR (train on real, test on real)
        model_real = XGBClassifier(
            n_estimators=100, max_depth=4, random_state=RANDOM_SEED,
            eval_metric="logloss", verbosity=0
        )
        model_real.fit(X_train, y_train)
        auc_trtr = roc_auc_score(y_test, model_real.predict_proba(X_test)[:, 1])

        # TSTR (train on synthetic, test on real)
        model_synth = XGBClassifier(
            n_estimators=100, max_depth=4, random_state=RANDOM_SEED,
            eval_metric="logloss", verbosity=0
        )
        model_synth.fit(X_synth, y_synth)
        auc_tstr = roc_auc_score(y_test, model_synth.predict_proba(X_test)[:, 1])

        results["auc_trtr"] = round(auc_trtr, 4)
        results["auc_tstr"] = round(auc_tstr, 4)
        results["tstr_gap"] = round(abs(auc_trtr - auc_tstr), 4)

        # 4b: Feature importance rank correlation
        fi_real = model_real.feature_importances_
        fi_synth = model_synth.feature_importances_

        from scipy.stats import spearmanr
        fi_corr, _ = spearmanr(fi_real, fi_synth)
        results["fi_rank_correlation"] = round(fi_corr, 4)

        # 4c: Calibration (Brier score as proxy)
        prob_synth = model_synth.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, prob_synth)
        results["brier_score_tstr"] = round(brier, 4)

        prob_real = model_real.predict_proba(X_test)[:, 1]
        brier_real = brier_score_loss(y_test, prob_real)
        results["brier_score_trtr"] = round(brier_real, 4)

    except Exception as e:
        print(f"    Fidelity evaluation failed: {e}")
        results["tstr_gap"] = None

    # Composite fidelity score
    scores = []
    if results.get("tstr_gap") is not None:
        scores.append(max(0, 1 - results["tstr_gap"] / 0.2))  # 0.2 gap → score 0
    if results.get("fi_rank_correlation") is not None:
        scores.append(max(0, results["fi_rank_correlation"]))
    results["fidelity_score"] = round(np.mean(scores), 4) if scores else None

    return results


# ============================================================================
# DIMENSION 5: STRUCTURAL INTEGRITY
# ============================================================================

def evaluate_integrity(synth_df, method_name):
    """Evaluate Dimension 5: Structural Integrity."""
    results = {}
    n = len(synth_df)
    constraint_results = {}

    # Single-column constraints
    for name, (col, op, val) in DOMAIN_CONSTRAINTS.items():
        if col not in synth_df.columns:
            continue
        series = synth_df[col]
        if op == ">=":
            satisfied = (series >= val).sum()
        elif op == "<=":
            satisfied = (series <= val).sum()
        elif op == ">":
            satisfied = (series > val).sum()
        elif op == "<":
            satisfied = (series < val).sum()
        elif op == "in":
            satisfied = series.isin(val).sum()
        else:
            satisfied = n
        rate = satisfied / n if n > 0 else 0
        constraint_results[name] = round(rate, 4)

    # Cross-column constraints
    for col1, op, col2 in CROSS_CONSTRAINTS:
        if col1 in synth_df.columns and col2 in synth_df.columns:
            if op == ">":
                satisfied = (synth_df[col1] > synth_df[col2]).sum()
            elif op == "<":
                satisfied = (synth_df[col1] < synth_df[col2]).sum()
            else:
                satisfied = n
            name = f"{col1}_{op}_{col2}"
            rate = satisfied / n if n > 0 else 0
            constraint_results[name] = round(rate, 4)

    results["constraint_results"] = constraint_results
    results["csr"] = round(np.mean(list(constraint_results.values())), 4) if constraint_results else None

    # Count failures (constraints with < 95% satisfaction)
    results["constraints_below_95"] = sum(1 for v in constraint_results.values() if v < 0.95)
    results["total_constraints"] = len(constraint_results)

    # Composite integrity score (CSR directly maps to [0,1])
    results["integrity_score"] = results["csr"]

    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("  TrustSynth Experimental Evaluation Pipeline")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load and split
    train_df, test_df = load_and_split()

    # Generate synthetic data
    synthetic_datasets, gen_times = generate_synthetic(train_df)

    if not synthetic_datasets:
        print("\nERROR: No generators succeeded. Check dependencies and try again.")
        return

    # Evaluate all dimensions
    all_results = {}
    summary_rows = []

    for method_name, synth_df in synthetic_datasets.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {method_name}")
        print(f"{'='*60}")

        method_results = {}

        # Align columns
        common_cols = [c for c in train_df.columns if c in synth_df.columns]
        synth_aligned = synth_df[common_cols].copy()

        # Dimension 1: Statistical Utility
        print(f"\n  [D1] Statistical Utility...")
        d1 = evaluate_utility(train_df[common_cols], synth_aligned, method_name)
        method_results["utility"] = d1
        print(f"    KS mean: {d1.get('ks_mean')}, JSD mean: {d1.get('jsd_mean')}, "
              f"Corr dev: {d1.get('corr_deviation')}, Score: {d1.get('utility_score')}")

        # Dimension 2: Privacy
        print(f"\n  [D2] Privacy Preservation...")
        d2 = evaluate_privacy(train_df, test_df, synth_df, method_name)
        method_results["privacy"] = d2
        print(f"    MIA acc: {d2.get('mia_accuracy')}, DCR ratio: {d2.get('dcr_ratio')}, "
              f"Score: {d2.get('privacy_score')}")

        # Dimension 3: Fairness
        print(f"\n  [D3] Fairness...")
        d3 = evaluate_fairness(train_df, synth_aligned, test_df, method_name)
        method_results["fairness"] = d3
        print(f"    Rep violations: {d3.get('representation_violations')}/{d3.get('representation_total_groups')}, "
              f"Score: {d3.get('fairness_score')}")

        # Dimension 4: Downstream Fidelity
        print(f"\n  [D4] Downstream Task Fidelity...")
        d4 = evaluate_fidelity(train_df, synth_aligned, test_df, method_name)
        method_results["fidelity"] = d4
        print(f"    AUC TRTR: {d4.get('auc_trtr')}, AUC TSTR: {d4.get('auc_tstr')}, "
              f"Gap: {d4.get('tstr_gap')}, FI corr: {d4.get('fi_rank_correlation')}, "
              f"Score: {d4.get('fidelity_score')}")

        # Dimension 5: Structural Integrity
        print(f"\n  [D5] Structural Integrity...")
        d5 = evaluate_integrity(synth_aligned, method_name)
        method_results["integrity"] = d5
        print(f"    CSR: {d5.get('csr')}, Constraints below 95%: "
              f"{d5.get('constraints_below_95')}/{d5.get('total_constraints')}, "
              f"Score: {d5.get('integrity_score')}")

        all_results[method_name] = method_results

        # Summary row
        summary_rows.append({
            "Method": method_name,
            "Utility": d1.get("utility_score"),
            "Privacy": d2.get("privacy_score"),
            "Fairness": d3.get("fairness_score"),
            "Fidelity": d4.get("fidelity_score"),
            "Integrity": d5.get("integrity_score"),
            "KS_mean": d1.get("ks_mean"),
            "JSD_mean": d1.get("jsd_mean"),
            "Corr_dev": d1.get("corr_deviation"),
            "MIA_acc": d2.get("mia_accuracy"),
            "DCR_ratio": d2.get("dcr_ratio"),
            "TSTR_gap": d4.get("tstr_gap"),
            "FI_corr": d4.get("fi_rank_correlation"),
            "CSR": d5.get("csr"),
            "Gen_time_s": gen_times.get(method_name),
        })

    # Save results
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "dimension_scores.csv", index=False)

    with open(RESULTS_DIR / "detailed_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 70)
    print("  TRUSTSYNTH EVALUATION SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Run 03_generate_outputs.py to produce paper figures and LaTeX tables.")


if __name__ == "__main__":
    main()
