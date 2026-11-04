"""
03_generate_outputs.py
Produces publication-ready figures and LaTeX tables from TrustSynth evaluation results.

Input:  results/dimension_scores.csv, results/detailed_metrics.json
Output: figures/*.pdf, results/summary_table.tex
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

from config import *

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_results():
    scores = pd.read_csv(RESULTS_DIR / "dimension_scores.csv")
    with open(RESULTS_DIR / "detailed_metrics.json") as f:
        detailed = json.load(f)
    return scores, detailed


# ============================================================================
# Figure 1: Radar Chart (Spider Plot) comparing methods across 5 dimensions
# ============================================================================

def plot_radar_chart(scores):
    """Create a radar/spider chart comparing all methods across 5 TrustSynth dimensions."""
    dimensions = ["Utility", "Privacy", "Fairness", "Fidelity", "Integrity"]
    methods = scores["Method"].tolist()

    # Prepare data
    values = scores[dimensions].values
    values = np.nan_to_num(values, nan=0.0)

    # Radar chart setup
    N = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#2CA58D"]
    for i, method in enumerate(methods):
        vals = values[i].tolist()
        vals += vals[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, vals, "o-", linewidth=2, label=method, color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "radar_chart.pdf")
    plt.savefig(FIGURES_DIR / "radar_chart.png")
    plt.close()
    print("  Saved radar_chart.pdf")


# ============================================================================
# Figure 2: Distribution comparison (KS per feature)
# ============================================================================

def plot_ks_comparison(detailed):
    """Bar chart of per-feature KS statistics for each method."""
    methods = list(detailed.keys())
    all_features = set()
    for m in methods:
        ks = detailed[m].get("utility", {}).get("ks_per_feature", {})
        all_features.update(ks.keys())
    features = sorted(all_features)

    if not features:
        print("  No KS data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(features))
    width = 0.8 / len(methods)
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#2CA58D"]

    for i, method in enumerate(methods):
        ks = detailed[method].get("utility", {}).get("ks_per_feature", {})
        vals = [ks.get(f, 0) for f in features]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=method, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Feature")
    ax.set_ylabel("KS Statistic")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_mean", "").replace("_", " ") for f in features],
                       rotation=45, ha="right")
    ax.axhline(y=KS_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({KS_THRESHOLD})")
    ax.legend()
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.1))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ks_comparison.pdf")
    plt.savefig(FIGURES_DIR / "ks_comparison.png")
    plt.close()
    print("  Saved ks_comparison.pdf")


# ============================================================================
# Figure 3: DCR distributions
# ============================================================================

def plot_dcr_distributions(scores):
    """Compare DCR ratios across methods."""
    methods = scores["Method"].tolist()
    dcr_ratios = scores["DCR_ratio"].tolist()

    if all(pd.isna(dcr_ratios)):
        print("  No DCR data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#2CA58D"]

    bars = ax.bar(methods, dcr_ratios, color=[colors[i % len(colors)] for i in range(len(methods))],
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.7, label="Ideal (ratio = 1.0)")
    ax.set_ylabel("DCR Ratio\n(synthetic-to-real / holdout-to-real)")
    ax.set_xlabel("Generator")
    ax.legend()

    for bar, val in zip(bars, dcr_ratios):
        if not pd.isna(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dcr_comparison.pdf")
    plt.savefig(FIGURES_DIR / "dcr_comparison.png")
    plt.close()
    print("  Saved dcr_comparison.pdf")


# ============================================================================
# Figure 4: TSTR comparison
# ============================================================================

def plot_tstr_comparison(scores):
    """Compare TSTR AUC across methods."""
    methods = scores["Method"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#2CA58D"]

    x = np.arange(len(methods))
    width = 0.35

    auc_trtr = scores["TSTR_gap"].values  # We'll show TSTR AUC directly
    # Get TSTR AUC from detailed if available
    trtr_vals = []
    tstr_vals = []
    for _, row in scores.iterrows():
        trtr_vals.append(row.get("AUC_TRTR", None))
        tstr_vals.append(row.get("AUC_TSTR", None))

    # If we don't have separate TRTR/TSTR in summary, reconstruct from gap
    if all(pd.isna(trtr_vals)):
        # Just plot the gap
        gaps = scores["TSTR_gap"].values
        bars = ax.bar(methods, gaps, color=[colors[i % len(colors)] for i in range(len(methods))],
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(y=TSTR_AUC_THRESHOLD, color="red", linestyle="--",
                    alpha=0.7, label=f"Threshold ({TSTR_AUC_THRESHOLD})")
        ax.set_ylabel("|AUC_TRTR - AUC_TSTR|")
        ax.set_xlabel("Generator")
        ax.legend()
    else:
        ax.bar(x - width / 2, trtr_vals, width, label="TRTR (baseline)", color="#555555", alpha=0.7)
        ax.bar(x + width / 2, tstr_vals, width, label="TSTR",
               color=[colors[i % len(colors)] for i in range(len(methods))], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("AUC-ROC")
        ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tstr_comparison.pdf")
    plt.savefig(FIGURES_DIR / "tstr_comparison.png")
    plt.close()
    print("  Saved tstr_comparison.pdf")


# ============================================================================
# Figure 5: Structural integrity heatmap
# ============================================================================

def plot_integrity_heatmap(detailed):
    """Heatmap of constraint satisfaction rates across methods."""
    methods = list(detailed.keys())

    all_constraints = set()
    for m in methods:
        cr = detailed[m].get("integrity", {}).get("constraint_results", {})
        all_constraints.update(cr.keys())
    constraints = sorted(all_constraints)

    if not constraints:
        print("  No integrity data to plot.")
        return

    matrix = []
    for m in methods:
        cr = detailed[m].get("integrity", {}).get("constraint_results", {})
        row = [cr.get(c, np.nan) for c in constraints]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, max(4, len(methods) * 1.2)))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=[c.replace("_", " ") for c in constraints],
        yticklabels=methods,
        vmin=0.8, vmax=1.0, ax=ax,
        cbar_kws={"label": "Satisfaction Rate"}
    )
    ax.set_xlabel("Constraint")
    ax.set_ylabel("Generator")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "integrity_heatmap.pdf")
    plt.savefig(FIGURES_DIR / "integrity_heatmap.png")
    plt.close()
    print("  Saved integrity_heatmap.pdf")


# ============================================================================
# LaTeX Summary Table
# ============================================================================

def generate_latex_table(scores):
    """Generate a LaTeX table suitable for the paper."""
    methods = scores["Method"].tolist()

    # Build the table
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{TrustSynth evaluation results on eICU-CRD v2.0 ICU cohort. "
                 r"Dimension scores are normalized to $[0,1]$ (higher is better). "
                 r"Key metrics are reported in their original units.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l ccccc cccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{5}{c}{\textbf{Dimension Scores}} "
                 r"& \multicolumn{4}{c}{\textbf{Key Metrics}} \\")
    lines.append(r"\cmidrule(lr){2-6} \cmidrule(lr){7-10}")
    lines.append(r"\textbf{Method} & Util. & Priv. & Fair. & Fidel. & Integ. "
                 r"& KS$_\mu$ & MIA & $\Delta$TSTR & CSR \\")
    lines.append(r"\midrule")

    for _, row in scores.iterrows():
        def fmt(val, digits=3):
            if pd.isna(val) or val is None:
                return "--"
            return f"{val:.{digits}f}"

        line = (f"{row['Method']} & "
                f"{fmt(row['Utility'])} & {fmt(row['Privacy'])} & "
                f"{fmt(row['Fairness'])} & {fmt(row['Fidelity'])} & "
                f"{fmt(row['Integrity'])} & "
                f"{fmt(row['KS_mean'])} & {fmt(row['MIA_acc'])} & "
                f"{fmt(row['TSTR_gap'])} & {fmt(row['CSR'])} \\\\")
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)

    with open(RESULTS_DIR / "summary_table.tex", "w") as f:
        f.write(table_str)

    print(f"  Saved summary_table.tex")
    print("\n  LaTeX table preview:")
    print(table_str)

    return table_str


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  TrustSynth: Generating Paper Outputs")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    scores, detailed = load_results()
    print(f"\n  Loaded results for {len(scores)} methods: {scores['Method'].tolist()}")

    print("\n  Generating figures...")
    plot_radar_chart(scores)
    plot_ks_comparison(detailed)
    plot_dcr_distributions(scores)
    plot_tstr_comparison(scores)
    plot_integrity_heatmap(detailed)

    print("\n  Generating LaTeX table...")
    generate_latex_table(scores)

    print(f"\n  All outputs saved to {FIGURES_DIR}/ and {RESULTS_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
