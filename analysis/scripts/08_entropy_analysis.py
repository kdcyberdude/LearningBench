"""
B7: Entropy Analysis Per Category (H12)
- Shannon entropy per task: high entropy = models spread evenly (good signal)
- Per-category aggregate entropy comparison
- Identify which categories are most/least informative
- Identify low-entropy tasks (candidates for removal)
- Outputs: entropy_report.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix
from utils.stats import shannon_entropy

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
CATEGORIES = ["associative", "concept", "language", "observational", "rl"]
MAX_ENTROPY = np.log2(10)  # max entropy with 10 bins ≈ 3.32 bits


def compute_task_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Shannon entropy for each task."""
    rows = []
    for (cat, task), task_df in df.groupby(["category", "task_name"]):
        scores = task_df["score"].dropna().values
        h = shannon_entropy(scores, n_bins=10)
        normalized_h = h / MAX_ENTROPY  # 0–1 scale
        rows.append({
            "category": cat,
            "task_name": task,
            "entropy": round(h, 4),
            "normalized_entropy": round(normalized_h, 4),
            "n_models": len(scores),
            "mean_score": round(float(np.mean(scores)), 4),
            "std_score": round(float(np.std(scores)), 4),
            "entropy_class": classify_entropy(normalized_h),
        })
    return pd.DataFrame(rows)


def classify_entropy(norm_h: float) -> str:
    if norm_h >= 0.65:
        return "high_entropy"
    if norm_h >= 0.40:
        return "medium_entropy"
    if norm_h >= 0.20:
        return "low_entropy"
    return "very_low_entropy"


def compute_category_entropy(task_entropy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate entropy stats per category."""
    rows = []
    for cat, cat_df in task_entropy.groupby("category"):
        rows.append({
            "category": cat,
            "n_tasks": len(cat_df),
            "mean_entropy": round(float(cat_df["entropy"].mean()), 4),
            "mean_normalized_entropy": round(float(cat_df["normalized_entropy"].mean()), 4),
            "std_entropy": round(float(cat_df["entropy"].std()), 4),
            "min_entropy": round(float(cat_df["entropy"].min()), 4),
            "max_entropy": round(float(cat_df["entropy"].max()), 4),
            "n_high_entropy": int((cat_df["entropy_class"] == "high_entropy").sum()),
            "n_low_entropy": int((cat_df["entropy_class"].isin(["low_entropy", "very_low_entropy"])).sum()),
            "pct_high_entropy": round(float((cat_df["entropy_class"] == "high_entropy").mean() * 100), 1),
        })
    return pd.DataFrame(rows).sort_values("mean_entropy", ascending=False)


def compute_entropy_discrimination_correlation(task_entropy: pd.DataFrame,
                                                task_stats: pd.DataFrame) -> tuple:
    """Correlate entropy with item discrimination index."""
    try:
        merged = task_entropy.merge(
            task_stats[["category", "task_name", "item_discrimination"]],
            on=["category", "task_name"]
        ).dropna(subset=["item_discrimination"])
        if len(merged) < 5:
            return np.nan, np.nan
        r, p = stats.pearsonr(merged["entropy"], merged["item_discrimination"])
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def print_summary(task_entropy: pd.DataFrame, cat_entropy: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("B7: ENTROPY ANALYSIS PER CATEGORY (H12)")
    print("=" * 60)

    print("\n--- Per-category entropy summary (sorted by mean entropy) ---")
    print(cat_entropy[["category", "n_tasks", "mean_entropy", "mean_normalized_entropy",
                        "n_high_entropy", "n_low_entropy", "pct_high_entropy"]].to_string(index=False))

    print("\n--- Entropy class breakdown across all tasks ---")
    counts = task_entropy["entropy_class"].value_counts()
    for cls, cnt in counts.items():
        pct = cnt / len(task_entropy) * 100
        print(f"  {cls:<25} {cnt:>3} tasks  ({pct:.1f}%)")

    print("\n--- Top 10 highest entropy tasks (most informative) ---")
    top = task_entropy.nlargest(10, "entropy")[["category", "task_name", "entropy",
                                                  "normalized_entropy", "std_score"]]
    print(top.to_string(index=False))

    print("\n--- Bottom 10 lowest entropy tasks (least informative) ---")
    bot = task_entropy.nsmallest(10, "entropy")[["category", "task_name", "entropy",
                                                   "normalized_entropy", "mean_score", "std_score"]]
    print(bot.to_string(index=False))

    print("\n--- H12 verdict ---")
    ordered = cat_entropy.sort_values("mean_entropy", ascending=False)
    print("  Category entropy ranking (most → least informative):")
    for _, row in ordered.iterrows():
        bar = "█" * int(row["mean_normalized_entropy"] * 20)
        print(f"  {row['category']:<15} H={row['mean_entropy']:.3f}  {bar}")

    top_cat = ordered.iloc[0]["category"]
    bot_cat = ordered.iloc[-1]["category"]
    print(f"\n  Most informative: {top_cat}")
    print(f"  Least informative: {bot_cat}")

    low_entropy_tasks = task_entropy[task_entropy["entropy_class"].isin(["very_low_entropy"])]
    print(f"\n  Very-low-entropy tasks (removal candidates): {len(low_entropy_tasks)}")
    if len(low_entropy_tasks):
        print(low_entropy_tasks[["category", "task_name", "entropy", "mean_score"]].to_string(index=False))


def main():
    df = load_score_matrix()
    task_entropy = compute_task_entropy(df)
    cat_entropy = compute_category_entropy(task_entropy)

    # Try to correlate with discrimination if task_stats available
    task_stats_path = OUTPUT_DIR / "task_stats.csv"
    if task_stats_path.exists():
        task_stats = pd.read_csv(task_stats_path)
        r, p = compute_entropy_discrimination_correlation(task_entropy, task_stats)
        if not np.isnan(r):
            print(f"\n  Entropy ↔ Discrimination correlation: r={r:.3f}  p={p:.3f}")

    print_summary(task_entropy, cat_entropy)

    task_entropy.to_csv(OUTPUT_DIR / "entropy_report.csv", index=False)
    cat_entropy.to_csv(OUTPUT_DIR / "category_entropy.csv", index=False)
    print(f"\nSaved → {OUTPUT_DIR}/entropy_report.csv")
    return task_entropy, cat_entropy


if __name__ == "__main__":
    main()
