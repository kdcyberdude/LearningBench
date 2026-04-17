"""
B2: Cross-Category Correlation Analysis (Hypothesis H1)
- Spearman rank correlation between model rankings across all 5 categories
- Tests H1: "Learning is not monolithic"
- Outputs: cross_category_correlations.csv, cross_category_rank_correlations.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
CATEGORIES = ["associative", "concept", "language", "observational", "rl"]
CAT_LABELS = {
    "associative": "Assoc.",
    "concept": "Concept",
    "language": "Language",
    "observational": "Obs.",
    "rl": "RL",
}


def build_category_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Build model × category pivot of mean scores."""
    pivot = (
        df.groupby(["model", "category"])["score"]
        .mean()
        .unstack("category")
    )
    # Ensure column order
    cols = [c for c in CATEGORIES if c in pivot.columns]
    return pivot[cols]


def compute_spearman_matrix(pivot: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank correlation matrix (score-based)."""
    n = len(pivot.columns)
    corr = np.zeros((n, n))
    p_vals = np.zeros((n, n))
    cols = list(pivot.columns)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                corr[i, j] = 1.0
                p_vals[i, j] = 0.0
                continue
            shared = pivot[[c1, c2]].dropna()
            if len(shared) < 3:
                corr[i, j] = np.nan
                p_vals[i, j] = np.nan
            else:
                result = stats.spearmanr(shared[c1], shared[c2])
                corr[i, j] = float(result.correlation)
                p_vals[i, j] = float(result.pvalue)

    corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    p_df = pd.DataFrame(p_vals, index=cols, columns=cols)
    return corr_df, p_df


def compute_rank_correlation_matrix(pivot: pd.DataFrame) -> pd.DataFrame:
    """Compute rank-based Spearman using model ranks within each category."""
    rank_pivot = pivot.rank(ascending=False)
    return compute_spearman_matrix(rank_pivot)


def print_summary(pivot: pd.DataFrame, corr_df: pd.DataFrame, p_df: pd.DataFrame,
                  rank_corr_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("B2: CROSS-CATEGORY CORRELATION ANALYSIS (H1)")
    print("=" * 60)

    print("\n--- Model × Category Score Pivot ---")
    print(pivot.round(3).to_string())

    print("\n--- Spearman Correlation Matrix (scores) ---")
    print(corr_df.round(3).to_string())

    print("\n--- P-values (Spearman) ---")
    print(p_df.round(3).to_string())

    print("\n--- Key pairwise correlations ---")
    cols = list(corr_df.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_df.iloc[i, j]
            p = p_df.iloc[i, j]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            pairs.append((cols[i], cols[j], r, p, sig))

    pairs.sort(key=lambda x: x[2])  # sort by correlation
    for c1, c2, r, p, sig in pairs:
        print(f"  {CAT_LABELS[c1]:<10} vs {CAT_LABELS[c2]:<10}  r={r:+.3f}  p={p:.3f} {sig}")

    # H1 verdict
    off_diag = [corr_df.iloc[i, j] for i in range(len(cols)) for j in range(i + 1, len(cols))
                if not np.isnan(corr_df.iloc[i, j])]
    mean_r = np.mean(off_diag) if off_diag else np.nan
    print(f"\n  Mean pairwise correlation: {mean_r:.3f}")
    if mean_r < 0.7:
        print("  → H1 SUPPORTED: Categories are not strongly correlated. Learning is multi-dimensional.")
    else:
        print("  → H1 CHALLENGED: High correlations suggest learning sub-abilities may co-vary.")

    print("\n--- Rank Correlation Matrix (model ranks) ---")
    print(rank_corr_df.round(3).to_string())


def main():
    df = load_score_matrix()
    pivot = build_category_pivot(df)
    corr_df, p_df = compute_spearman_matrix(pivot)
    rank_corr_df, rank_p_df = compute_rank_correlation_matrix(pivot)

    print_summary(pivot, corr_df, p_df, rank_corr_df)

    corr_df.to_csv(OUTPUT_DIR / "cross_category_correlations.csv")
    p_df.to_csv(OUTPUT_DIR / "cross_category_pvalues.csv")
    rank_corr_df.to_csv(OUTPUT_DIR / "cross_category_rank_correlations.csv")
    pivot.to_csv(OUTPUT_DIR / "category_pivot.csv")
    print(f"\nSaved → {OUTPUT_DIR}/cross_category_*.csv")
    return corr_df, p_df, rank_corr_df


if __name__ == "__main__":
    main()
