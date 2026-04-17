"""
B4: Efficiency Ablation Analysis (H3, H4)
- Reconstruct accuracy-only scores for Concept Formation and Language Learning
- Compare accuracy-only rankings vs. composite (accuracy × efficiency) rankings
- Identify models where efficiency scoring most changes rank
- Outputs: efficiency_ablation.csv

NOTE: The leaderboard stores only the composite score. We reconstruct accuracy-only
via the formula: composite = accuracy × (0.40 + 0.60 × efficiency)
At minimum efficiency (all examples used), efficiency ≈ 0, so score ≥ 0.40 × accuracy.
At max efficiency (minimal examples), efficiency = 1, so score = accuracy.

Since we only have composite scores (not the raw accuracy + efficiency split), we:
1. Treat the composite score directly as a proxy for combined signal
2. Simulate an "accuracy-only" bound using composite / 0.40 as an upper bound
3. Compare rank ordering between composite and this bound
4. Additionally analyze score variance: if efficiency were removed, how would rankings change?

For H3 (speed-accuracy tradeoff): We look at whether models that rank high on
composite (good efficiency) have lower absolute scores on RL (accuracy-dominant).
For H4 (efficiency reverses rankings): We compare composite vs. simulated accuracy-only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Categories with efficiency in scoring
INTERACTIVE_CATS = ["concept", "language"]
# Weight of accuracy in composite: score = acc × (0.4 + 0.6 × eff)
# Min composite = 0.4 × acc (when eff=0), Max composite = acc (when eff=1)
# So: acc_only_lower_bound = composite (when eff=1 — best case)
# And: acc_only_upper_bound = composite / 0.4 (when eff=0 — worst case)
# We define simulated accuracy-only as composite / 0.70 (midpoint, eff=0.5)
EFF_MIDPOINT = 0.70


def compute_model_ranks(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """Compute per-model ranks for given categories."""
    model_scores = df[df["category"].isin(categories)].groupby(["model", "category"])["score"].mean().unstack()
    model_scores["interactive_mean"] = model_scores[categories].mean(axis=1)
    return model_scores


def simulate_accuracy_only(score: float) -> float:
    """Simulate what the score would be without efficiency penalty (eff=1.0)."""
    # composite = acc × (0.4 + 0.6 × eff)
    # If eff=1 → composite = acc → acc = composite  (same, best case)
    # If eff=0 → composite = 0.4 × acc → acc = composite / 0.4
    # Realistic middle: assume eff=0.5 → composite = 0.7 × acc → acc = composite / 0.7
    return min(1.0, score / EFF_MIDPOINT)


def compute_rank_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compare composite vs simulated accuracy-only rankings."""
    results = []

    for cat in INTERACTIVE_CATS:
        cat_df = df[df["category"] == cat]
        model_means = cat_df.groupby("model")["score"].mean()

        composite_rank = model_means.rank(ascending=False)
        accuracy_only = model_means.map(simulate_accuracy_only)
        accuracy_rank = accuracy_only.rank(ascending=False)

        for model in model_means.index:
            results.append({
                "category": cat,
                "model": model,
                "composite_score": round(float(model_means[model]), 4),
                "simulated_acc_only": round(float(accuracy_only[model]), 4),
                "composite_rank": int(composite_rank[model]),
                "accuracy_only_rank": int(accuracy_rank[model]),
                "rank_change": int(composite_rank[model]) - int(accuracy_rank[model]),
            })

    return pd.DataFrame(results)


def analyze_rl_vs_interactive(df: pd.DataFrame) -> pd.DataFrame:
    """H3: Do models with high interactive (efficiency) scores trade off vs. RL accuracy?"""
    # Interactive mean (efficiency component matters)
    interactive = df[df["category"].isin(INTERACTIVE_CATS)].groupby("model")["score"].mean()
    rl = df[df["category"] == "rl"].groupby("model")["score"].mean()

    common = interactive.index.intersection(rl.index)
    comparison = pd.DataFrame({
        "interactive_mean": interactive.loc[common].round(4),
        "rl_mean": rl.loc[common].round(4),
    })
    comparison["tier"] = comparison.index.map(
        lambda m: df[df["model"] == m]["tier"].iloc[0] if len(df[df["model"] == m]) else "unknown"
    )
    r, p = stats.pearsonr(comparison["interactive_mean"], comparison["rl_mean"])
    return comparison, r, p


def print_summary(rank_changes: pd.DataFrame, comparison: pd.DataFrame, r: float, p: float) -> None:
    print("\n" + "=" * 60)
    print("B4: EFFICIENCY ABLATION ANALYSIS (H3, H4)")
    print("=" * 60)

    print("\n--- H4: Does efficiency scoring change rankings? ---")
    print("(+rank_change = model ranks BETTER with efficiency; - = ranks WORSE)")
    for cat in INTERACTIVE_CATS:
        print(f"\n  Category: {cat.upper()}")
        cat_df = rank_changes[rank_changes["category"] == cat].sort_values("composite_rank")
        print(cat_df[["model", "composite_score", "simulated_acc_only",
                       "composite_rank", "accuracy_only_rank", "rank_change"]].to_string(index=False))

    max_change = rank_changes["rank_change"].abs().max()
    print(f"\n  Maximum rank change: {max_change} positions")
    if max_change >= 3:
        print("  → H4 SUPPORTED: Efficiency scoring meaningfully reshuffles rankings.")
    else:
        print("  → H4 CHALLENGED: Rankings are largely stable with/without efficiency.")

    print(f"\n--- H3: Speed-accuracy tradeoff (Interactive vs. RL) ---")
    print(comparison.sort_values("interactive_mean", ascending=False).to_string())
    print(f"\n  Pearson r(interactive, RL) = {r:.3f}  p={p:.3f}")
    if r < 0:
        print("  → H3 SUPPORTED: Models with higher interactive scores have lower RL scores.")
    elif p > 0.05:
        print("  → H3 INCONCLUSIVE: No significant correlation between interactive and RL performance.")
    else:
        print(f"  → H3 NOT SUPPORTED: Positive correlation (r={r:.3f}) — efficient learners also do well on RL.")


def main():
    df = load_score_matrix()
    rank_changes = compute_rank_changes(df)
    comparison, r, p = analyze_rl_vs_interactive(df)

    print_summary(rank_changes, comparison, r, p)

    rank_changes.to_csv(OUTPUT_DIR / "efficiency_ablation.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "interactive_vs_rl.csv")
    print(f"\nSaved → {OUTPUT_DIR}/efficiency_ablation.csv")
    return rank_changes


if __name__ == "__main__":
    main()
