"""
B3 + B10: Scale-Performance Analysis & Tier Deep-Dive (H2)
- Score distributions by tier per category
- Identifies specific scale inversions (small > mid > frontier unexpected cases)
- Explains WHY tier separations happen (which tasks drive them)
- Outputs: scale_analysis.csv, tier_inversion_report.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

TIER_ORDER = ["small", "mid", "frontier"]
CATEGORIES = ["associative", "concept", "language", "observational", "rl"]


def compute_tier_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category, per-tier mean scores and variance."""
    rows = []
    for cat, cat_df in df.groupby("category"):
        for tier, tier_df in cat_df.groupby("tier"):
            model_means = tier_df.groupby("model")["score"].mean()
            rows.append({
                "category": cat,
                "tier": tier,
                "n_models": len(model_means),
                "mean_score": float(model_means.mean()),
                "std_score": float(model_means.std()),
                "min_score": float(model_means.min()),
                "max_score": float(model_means.max()),
                "models": ", ".join(sorted(model_means.index)),
            })
    return pd.DataFrame(rows)


def find_inversions(df: pd.DataFrame) -> pd.DataFrame:
    """Find model-level inversions: smaller model beats larger model on a category."""
    # Get per-model, per-category means
    model_cat = df.groupby(["model", "category", "tier"])["score"].mean().reset_index()
    model_cat = model_cat.rename(columns={"score": "mean_score"})

    inversions = []
    for cat, cat_df in model_cat.groupby("category"):
        for _, row_a in cat_df.iterrows():
            for _, row_b in cat_df.iterrows():
                if row_a["model"] == row_b["model"]:
                    continue
                tier_a = TIER_ORDER.index(row_a["tier"]) if row_a["tier"] in TIER_ORDER else -1
                tier_b = TIER_ORDER.index(row_b["tier"]) if row_b["tier"] in TIER_ORDER else -1
                # Inversion: smaller tier model (lower index) beats larger tier model
                if tier_a < tier_b and row_a["mean_score"] > row_b["mean_score"]:
                    diff = row_a["mean_score"] - row_b["mean_score"]
                    if diff > 0.05:  # meaningful gap only
                        inversions.append({
                            "category": cat,
                            "smaller_model": row_a["model"],
                            "smaller_tier": row_a["tier"],
                            "smaller_score": round(row_a["mean_score"], 4),
                            "larger_model": row_b["model"],
                            "larger_tier": row_b["tier"],
                            "larger_score": round(row_b["mean_score"], 4),
                            "gap": round(diff, 4),
                        })
    df_inv = pd.DataFrame(inversions) if inversions else pd.DataFrame()
    if len(df_inv):
        df_inv = df_inv.sort_values("gap", ascending=False)
    return df_inv


def find_tier_driving_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """For each category, which tasks show the largest frontier-small gap?"""
    rows = []
    for cat, cat_df in df.groupby("category"):
        for task, task_df in cat_df.groupby("task_name"):
            tier_means = task_df.groupby("tier")["score"].mean()
            f_mean = tier_means.get("frontier", np.nan)
            s_mean = tier_means.get("small", np.nan)
            m_mean = tier_means.get("mid", np.nan)
            if not np.isnan(f_mean) and not np.isnan(s_mean):
                gap_fs = f_mean - s_mean
            else:
                gap_fs = np.nan
            rows.append({
                "category": cat,
                "task_name": task,
                "frontier_mean": round(f_mean, 4) if not np.isnan(f_mean) else None,
                "mid_mean": round(m_mean, 4) if not np.isnan(m_mean) else None,
                "small_mean": round(s_mean, 4) if not np.isnan(s_mean) else None,
                "frontier_small_gap": round(gap_fs, 4) if not np.isnan(gap_fs) else None,
            })
    df_tasks = pd.DataFrame(rows)
    return df_tasks


def print_summary(tier_stats: pd.DataFrame, inversions: pd.DataFrame,
                  tier_tasks: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("B3 + B10: SCALE-PERFORMANCE & TIER DEEP-DIVE (H2)")
    print("=" * 60)

    print("\n--- Per-tier, per-category mean scores ---")
    pivot = tier_stats.pivot_table(
        values="mean_score", index="tier", columns="category"
    ).reindex(TIER_ORDER)
    print(pivot.round(3).to_string())

    print("\n--- H2 verdict per category (does scale predict performance?) ---")
    for cat in CATEGORIES:
        cat_df = tier_stats[tier_stats["category"] == cat].set_index("tier")
        if "frontier" in cat_df.index and "small" in cat_df.index:
            f_score = cat_df.loc["frontier", "mean_score"]
            s_score = cat_df.loc["small", "mean_score"]
            m_score = cat_df.loc["mid", "mean_score"] if "mid" in cat_df.index else np.nan
            # Check monotone: frontier > mid > small
            if not np.isnan(m_score):
                monotone = f_score > m_score > s_score
            else:
                monotone = f_score > s_score
            verdict = "MONOTONE (scale works)" if monotone else "NON-MONOTONE (scale fails)"
            print(f"  {cat:<15} frontier={f_score:.3f}  mid={m_score:.3f}  small={s_score:.3f}  → {verdict}")

    print(f"\n--- Scale inversions (smaller model beats larger, gap > 0.05) ---")
    if len(inversions):
        print(f"  Total inversions: {len(inversions)}")
        print(inversions.head(20).to_string(index=False))
    else:
        print("  No inversions found with gap > 0.05")

    print("\n--- Top 5 tier-discriminating tasks per category ---")
    for cat in CATEGORIES:
        cat_tasks = tier_tasks[tier_tasks["category"] == cat].dropna(subset=["frontier_small_gap"])
        top5 = cat_tasks.nlargest(5, "frontier_small_gap")
        print(f"\n  {cat.upper()}")
        print(top5[["task_name", "frontier_mean", "mid_mean", "small_mean", "frontier_small_gap"]].to_string(index=False))

    print("\n--- Bottom 5 (weakest tier discrimination) per category ---")
    for cat in CATEGORIES:
        cat_tasks = tier_tasks[tier_tasks["category"] == cat].dropna(subset=["frontier_small_gap"])
        bot5 = cat_tasks.nsmallest(5, "frontier_small_gap")
        print(f"\n  {cat.upper()}")
        print(bot5[["task_name", "frontier_mean", "mid_mean", "small_mean", "frontier_small_gap"]].to_string(index=False))


def main():
    df = load_score_matrix()
    tier_stats = compute_tier_stats(df)
    inversions = find_inversions(df)
    tier_tasks = find_tier_driving_tasks(df)

    print_summary(tier_stats, inversions, tier_tasks)

    tier_stats.to_csv(OUTPUT_DIR / "tier_stats.csv", index=False)
    tier_tasks.to_csv(OUTPUT_DIR / "tier_task_gaps.csv", index=False)
    if len(inversions):
        inversions.to_csv(OUTPUT_DIR / "tier_inversions.csv", index=False)
    print(f"\nSaved → {OUTPUT_DIR}/tier_*.csv")
    return tier_stats, inversions, tier_tasks


if __name__ == "__main__":
    main()
