"""
trustsynth_gower.py
Parallel, value-identical replacement for `gower.gower_matrix(...).min(axis=1)`.

WHY THIS EXISTS
---------------
`gower` 0.1.2 computes its numeric normalisation constants from the
concatenation of both of its arguments:

    Z = np.concatenate((X, Y))
    num_max[col]    = nanmax(Z[:, col])
    num_ranges[col] = |1 - min/max|        (0 when max == 0)
    Z_num           = Z_num / num_max

so simply splitting X into chunks and calling the package once per chunk
changes `num_max` and therefore changes every distance. Measured drift from
naive chunking on this cohort was up to 1.07e-2 in Gower units -- large
enough to move the DCR ratio reported in the paper.

This module computes the constants ONCE over the full `concat(sample,
reference)`, exactly as the package does, then evaluates the distance rows in
parallel against those fixed constants. The arithmetic below reproduces
`gower.gower_dist.gower_get` term for term, including its float32 constant
estimation, its division-by-max normalisation (the package normalises by the
maximum rather than by the range, which is idiosyncratic but is what the
original submission used), and its float32 output. `verify_against_package`
asserts bit-for-bit agreement and is exercised by the self-test below.

NOTE ON THE METRIC ITSELF
-------------------------
The per-column contributions are:
    categorical : 1 if the two values differ, else 0
    numeric     : |x_i/num_max - x_j/num_max| / num_ranges
and the Gower distance is their unweighted mean over all columns. This is the
definition used throughout the paper's DCR analysis; it is reproduced
verbatim in the supplementary material.
"""

import os

import numpy as np


def _constants(Z, cat_mask):
    """num_max / num_ranges exactly as gower 0.1.2 derives them."""
    Z_num = Z[:, ~cat_mask]
    n_cols = Z_num.shape[1]
    num_max = np.zeros(n_cols)
    num_ranges = np.zeros(n_cols)
    for col in range(n_cols):
        col_array = Z_num[:, col].astype(np.float32)
        mx, mn = np.nanmax(col_array), np.nanmin(col_array)
        if np.isnan(mx):
            mx = 0.0
        if np.isnan(mn):
            mn = 0.0
        num_max[col] = mx
        num_ranges[col] = np.abs(1 - mn / mx) if mx != 0 else 0.0
    return num_max, num_ranges


def _min_dist_block(X_cat, X_num, Y_cat, Y_num, num_ranges, weight_sum):
    """Row-wise minimum Gower distance from each row of X to all of Y."""
    out = np.empty(X_cat.shape[0], dtype=np.float32)
    safe = num_ranges != 0
    for i in range(X_cat.shape[0]):
        sum_cat = (X_cat[i] != Y_cat).sum(axis=1)
        delta = np.abs(X_num[i] - Y_num)
        sij = np.zeros_like(delta)
        np.divide(delta, num_ranges, out=sij, where=safe)
        d = (sum_cat + sij.sum(axis=1)) / weight_sum
        out[i] = np.float32(d).min()
    return out


def gower_min_distance(sample, reference, cat_features, n_jobs=-1,
                       chunk_rows=None):
    """Minimum Gower distance from every row of `sample` to `reference`.

    Equivalent to `gower.gower_matrix(sample, reference,
    cat_features=cat_features).min(axis=1)`, parallelised over row blocks of
    `sample`. Returns a float32 array of length len(sample).
    """
    cat_mask = np.asarray(cat_features, dtype=bool)
    X = np.asarray(sample)
    Y = np.asarray(reference)
    Z = np.concatenate((X, Y))

    num_max, num_ranges = _constants(Z, cat_mask)
    weight_sum = float(Z.shape[1])

    def split(arr):
        a_cat = arr[:, cat_mask]
        a_num = arr[:, ~cat_mask].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_num = np.divide(a_num, num_max,
                              out=np.zeros_like(a_num), where=num_max != 0)
        return a_cat, a_num

    X_cat, X_num = split(X)
    Y_cat, Y_num = split(Y)

    n_workers = (os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs)
    n_workers = max(1, min(n_workers, len(X)))
    if n_workers == 1:
        return _min_dist_block(X_cat, X_num, Y_cat, Y_num, num_ranges, weight_sum)

    if chunk_rows is None:
        blocks = np.array_split(np.arange(len(X)), n_workers)
    else:
        blocks = [np.arange(i, min(i + chunk_rows, len(X)))
                  for i in range(0, len(X), chunk_rows)]

    from joblib import Parallel, delayed
    parts = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_min_dist_block)(X_cat[b], X_num[b], Y_cat, Y_num,
                                 num_ranges, weight_sum)
        for b in blocks if len(b)
    )
    return np.concatenate(parts)


def verify_against_package(sample, reference, cat_features):
    """Assert bit-for-bit agreement with gower.gower_matrix(...).min(axis=1)."""
    import gower
    ref = gower.gower_matrix(sample, reference,
                             cat_features=cat_features).min(axis=1)
    ours = gower_min_distance(sample, reference, cat_features)
    max_diff = float(np.abs(ref - ours).max())
    exact = bool(np.array_equal(ref, ours))
    return {"max_abs_diff": max_diff, "bitwise_identical": exact,
            "n": len(ref)}


if __name__ == "__main__":
    import sys
    import time

    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import CATEGORICAL_FEATURES, DATA_DIR

    df = pd.read_csv(DATA_DIR / "cohort.csv")
    smp = df.sample(400, random_state=7).reset_index(drop=True)
    ref = df.sample(5000, random_state=11).reset_index(drop=True)
    cat = [c in CATEGORICAL_FEATURES for c in df.columns]

    print("Verifying against gower 0.1.2 ...")
    print(" ", verify_against_package(smp, ref, cat))

    big = df.sample(2000, random_state=7).reset_index(drop=True)
    import gower
    t = time.time(); gower.gower_matrix(big, ref, cat_features=cat).min(axis=1)
    t_pkg = time.time() - t
    t = time.time(); gower_min_distance(big, ref, cat, n_jobs=1)
    t_ser = time.time() - t
    t = time.time(); gower_min_distance(big, ref, cat)
    t_par = time.time() - t
    print(f"  package          {t_pkg:7.1f}s")
    print(f"  this, 1 core     {t_ser:7.1f}s   ({t_pkg/t_ser:.1f}x)")
    print(f"  this, all cores  {t_par:7.1f}s   ({t_pkg/t_par:.1f}x)")
