"""
B1: Discriminatory Power Analysis
- Item Response Theory (IRT)-style analysis for each task
- Item discrimination index (point-biserial correlation with total category score)
- Identifies tasks that are highly discriminating vs. those that add no signal
- Outputs: discrimination_report.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def compute_discrimination_report(df: pd.DataFrame) -> pd.DataFrame:
    """For each task, compute item discrimination and related stats."""
    rows = []

    for category, cat_df in df.groupby("category"):
        # Model total score within this category (used as ability estimate)
        model_totals = cat_df.groupby("model")["score"].mean().rename("total_score")

        for task_name, task_df in cat_df.groupby("task_name"):
            task_scores = task_df.set_index("model")["score"]
            common_models = task_scores.index.intersection(model_totals.index)
            if len(common_models) < 3:
                continue

            t_scores = task_scores.loc[common_models].values
            tot_scores = model_totals.loc[common_models].values

            # Point-biserial / Pearson correlation
            if t_scores.std() < 1e-9 or tot_scores.std() < 1e-9:
                discrimination = 0.0
                p_value = 1.0
            else:
                discrimination, p_value = stats.pearsonr(t_scores, tot_scores)

            # Upper 27% / Lower 27% split (classical item analysis)
            n = len(tot_scores)
            n_split = max(1, int(n * 0.27))
            sorted_idx = np.argsort(tot_scores)
            low_group = t_scores[sorted_idx[:n_split]]
            high_group = t_scores[sorted_idx[-n_split:]]
            d_index = float(np.mean(high_group) - np.mean(low_group))

            rows.append({
                "category": category,
                "task_name": task_name,
                "mean_score": float(np.mean(t_scores)),
                "std_score": float(np.std(t_scores)),
                "discrimination_r": float(discrimination),
                "discrimination_p": float(p_value),
                "d_index_27pct": d_index,
                "n_models": len(common_models),
                "classification": classify_discrimination(float(discrimination), float(np.mean(t_scores))),
            })

    return pd.DataFrame(rows)


def classify_discrimination(r: float, mean: float) -> str:
    """Classify task quality based on discrimination and difficulty."""
    if mean < 0.02:
        return "all_zero"
    if mean > 0.98:
        return "all_perfect"
    if r < 0.0:
        return "negative_discrimination"
    if r < 0.1:
        return "poor_discrimination"
    if r < 0.3:
        return "fair_discrimination"
    if r < 0.5:
        return "good_discrimination"
    return "excellent_discrimination"


def print_summary(report: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("B1: DISCRIMINATORY POWER ANALYSIS")
    print("=" * 60)

    print("\n--- Classification breakdown ---")
    counts = report["classification"].value_counts()
    for cls, cnt in counts.items():
        print(f"  {cls:<30} {cnt:>3} tasks")

    print("\n--- Top 10 most discriminating tasks ---")
    top = report.nlargest(10, "discrimination_r")[
        ["category", "task_name", "discrimination_r", "d_index_27pct", "mean_score"]
    ]
    print(top.to_string(index=False))

    print("\n--- Bottom 10 (least/negatively discriminating) ---")
    bottom = report.nsmallest(10, "discrimination_r")[
        ["category", "task_name", "discrimination_r", "mean_score"]
    ]
    print(bottom.to_string(index=False))

    print("\n--- Per-category mean discrimination ---")
    cat_disc = report.groupby("category")["discrimination_r"].agg(["mean", "std", "min", "max"])
    print(cat_disc.round(3).to_string())

    print("\n--- Tasks with negative discrimination (problems) ---")
    neg = report[report["discrimination_r"] < 0][
        ["category", "task_name", "discrimination_r", "mean_score"]
    ]
    if len(neg):
        print(neg.to_string(index=False))
    else:
        print("  None")


def main():
    df = load_score_matrix()
    report = compute_discrimination_report(df)
    out_path = OUTPUT_DIR / "discrimination_report.csv"
    report.to_csv(out_path, index=False)
    print_summary(report)
    print(f"\nSaved → {out_path}")
    return report


if __name__ == "__main__":
    main()
