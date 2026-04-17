"""
C1: Task Removal Sensitivity Analysis (R4 — robustness)

Leave-one-out (LOO) analysis: for each task, compute how aggregate model
rankings change when that task is excluded.

Also: targeted removal of the flagged tasks from Phase B — show stability
of the cleaned benchmark vs the original.

Outputs:
  - loo_ranking_stability.csv  (per-task: max rank change, mean rank change)
  - flagged_removal_impact.csv (effect of removing all flagged tasks at once)
  - benchmark_stability_summary.csv (overall stability metrics)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


# --------------------------------------------------------------------------
# Tasks flagged for removal from Phase B
# --------------------------------------------------------------------------
HIGH_PRIORITY_REMOVE = {
    "blocking_effect_assoc_learning",
    "custom_gravity_simulation_obs_learning",
    "dual_recurrence_concept_learning",
    "hapax_prime_concept_learning",
    "manhattan_point_rf_learning",
    "minesweeper_1d_rf_learning",
    "grid_nav_rf_learning",
    "euler_totient_rf_learning",
    "semantic_override_concept_learning",
}

MEDIUM_PRIORITY_REMOVE = {
    "nim_heap_rf_learning",
    "hidden_modal_logic_kripke2_obs_learning",
    "voronoi_custom_metric_obs_learning",
    "digit_square_error_rf_learning",
    "vigenere_variant_cipher_obs_learning",
    "hanoi_two_rf_learning",
    "hanoi_three_rf_learning",
    "levenshtein_words_rf_learning",
    "hangman_lite_rf_learning",
    "lights_out_2x2_rf_learning",
    "linear_equation_rf_learning",
    "linear_polynomial_rf_learning",
    "parity_groups_rf_learning",
    "grid_seven_rf_learning",
}

TIER_ORDER = ["frontier", "mid", "small"]


def _get_rankings(df: pd.DataFrame) -> pd.Series:
    """Compute per-model mean score across all tasks, return rank (1=best)."""
    model_means = df.groupby("model")["score"].mean()
    return model_means.rank(ascending=False, method="min")


def _spearman_rank_correlation(ranks_a: pd.Series, ranks_b: pd.Series) -> float:
    """Spearman correlation between two ranking Series."""
    models = ranks_a.index.intersection(ranks_b.index)
    if len(models) < 3:
        return float("nan")
    return float(stats.spearmanr(ranks_a[models], ranks_b[models]).statistic)


def leave_one_out_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """For each task, exclude it and compute ranking changes."""
    baseline_ranks = _get_rankings(df)
    tasks = df["task_name"].unique()

    rows = []
    for task in tasks:
        df_minus = df[df["task_name"] != task]
        if df_minus.empty:
            continue
        new_ranks = _get_rankings(df_minus)
        models = baseline_ranks.index.intersection(new_ranks.index)

        rank_changes = (new_ranks[models] - baseline_ranks[models]).abs()
        max_change = rank_changes.max()
        mean_change = rank_changes.mean()
        models_affected = (rank_changes > 0).sum()
        models_moved_2plus = (rank_changes >= 2).sum()
        spearman_corr = _spearman_rank_correlation(baseline_ranks, new_ranks)

        # Which model changed most?
        if rank_changes.max() > 0:
            top_mover = rank_changes.idxmax()
            top_change = rank_changes.max()
        else:
            top_mover = ""
            top_change = 0.0

        category = df[df["task_name"] == task]["category"].iloc[0]
        rows.append(
            {
                "task_name": task,
                "category": category,
                "max_rank_change": max_change,
                "mean_rank_change": mean_change,
                "models_affected": int(models_affected),
                "models_moved_2plus": int(models_moved_2plus),
                "spearman_with_baseline": spearman_corr,
                "top_mover": top_mover,
                "top_change": top_change,
            }
        )

    result = pd.DataFrame(rows).sort_values("max_rank_change", ascending=False)
    return result


def per_category_loo(df: pd.DataFrame) -> pd.DataFrame:
    """LOO within each category: removes task, recomputes category ranking."""
    rows = []
    for category, cat_df in df.groupby("category"):
        baseline_ranks = _get_rankings(cat_df)
        tasks = cat_df["task_name"].unique()

        for task in tasks:
            df_minus = cat_df[cat_df["task_name"] != task]
            if df_minus.empty:
                continue
            new_ranks = _get_rankings(df_minus)
            models = baseline_ranks.index.intersection(new_ranks.index)
            rank_changes = (new_ranks[models] - baseline_ranks[models]).abs()

            rows.append(
                {
                    "task_name": task,
                    "category": category,
                    "max_rank_change": rank_changes.max(),
                    "mean_rank_change": rank_changes.mean(),
                    "models_moved_2plus": int((rank_changes >= 2).sum()),
                    "spearman_with_baseline": _spearman_rank_correlation(
                        baseline_ranks, new_ranks
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values("max_rank_change", ascending=False)


def flagged_removal_impact(df: pd.DataFrame) -> dict:
    """Compute rankings before/after removing all flagged tasks."""
    # Normalize task names for matching (the matrix may include category suffix)
    df_tasks = set(df["task_name"].unique())

    # Match flagged tasks to actual task names (fuzzy prefix match)
    def match_tasks(flag_set):
        matched = set()
        for flag in flag_set:
            # Try exact match first
            if flag in df_tasks:
                matched.add(flag)
                continue
            # Try prefix match (flag is prefix of task_name)
            for t in df_tasks:
                if t.startswith(flag.replace("_rf_learning", "").replace("_assoc_learning", "").replace("_concept_learning", "").replace("_obs_learning", "").replace("_concept_learning", "")):
                    matched.add(t)
        return matched

    high_matched = match_tasks(HIGH_PRIORITY_REMOVE)
    medium_matched = match_tasks(MEDIUM_PRIORITY_REMOVE)
    all_matched = high_matched | medium_matched

    baseline_ranks = _get_rankings(df)

    # After removing high-priority only
    df_no_high = df[~df["task_name"].isin(high_matched)]
    ranks_no_high = _get_rankings(df_no_high)

    # After removing all flagged
    df_clean = df[~df["task_name"].isin(all_matched)]
    ranks_clean = _get_rankings(df_clean)

    models = baseline_ranks.index

    def rank_delta_table(r_new):
        deltas = {}
        for m in models:
            if m in r_new:
                deltas[m] = int(r_new[m] - baseline_ranks[m])
        return deltas

    return {
        "baseline_ranks": baseline_ranks.to_dict(),
        "ranks_after_high_removal": ranks_no_high.to_dict(),
        "ranks_after_all_removal": ranks_clean.to_dict(),
        "delta_high": rank_delta_table(ranks_no_high),
        "delta_all": rank_delta_table(ranks_clean),
        "n_high_matched": len(high_matched),
        "n_medium_matched": len(medium_matched),
        "n_all_matched": len(all_matched),
        "tasks_remaining_after_all": len(df_clean["task_name"].unique()),
        "high_matched": sorted(high_matched),
        "medium_matched": sorted(medium_matched),
    }


def overall_stability_metrics(loo_global: pd.DataFrame, loo_category: pd.DataFrame) -> dict:
    """Summarise overall benchmark stability."""
    return {
        "global_loo": {
            "max_rank_change_any_task": float(loo_global["max_rank_change"].max()),
            "median_max_rank_change": float(loo_global["max_rank_change"].median()),
            "mean_spearman_with_baseline": float(loo_global["spearman_with_baseline"].mean()),
            "min_spearman_with_baseline": float(loo_global["spearman_with_baseline"].min()),
            "pct_tasks_zero_change": float(
                (loo_global["max_rank_change"] == 0).mean() * 100
            ),
            "pct_tasks_change_le_1": float(
                (loo_global["max_rank_change"] <= 1).mean() * 100
            ),
        },
        "category_loo": {
            "max_rank_change_any_task": float(loo_category["max_rank_change"].max()),
            "median_max_rank_change": float(loo_category["max_rank_change"].median()),
            "mean_spearman_with_baseline": float(loo_category["spearman_with_baseline"].mean()),
        },
    }


def main():
    print("=== C1: Task Removal Sensitivity Analysis ===\n")

    df = load_score_matrix()
    print(f"Loaded {len(df)} rows, {df['task_name'].nunique()} tasks, {df['model'].nunique()} models\n")

    # --- Global LOO (across all categories) ---
    print("Running global leave-one-out analysis...")
    loo_global = leave_one_out_analysis(df)
    loo_global.to_csv(OUTPUT_DIR / "loo_global.csv", index=False)
    print(f"  Saved loo_global.csv ({len(loo_global)} rows)")

    # --- Per-category LOO ---
    print("Running per-category leave-one-out analysis...")
    loo_cat = per_category_loo(df)
    loo_cat.to_csv(OUTPUT_DIR / "loo_category.csv", index=False)
    print(f"  Saved loo_category.csv ({len(loo_cat)} rows)")

    # --- Flagged removal impact ---
    print("\nComputing flagged-task removal impact...")
    removal = flagged_removal_impact(df)

    # Build a nice comparison table
    baseline = removal["baseline_ranks"]
    after_high = removal["ranks_after_high_removal"]
    after_all = removal["ranks_after_all_removal"]

    comparison_rows = []
    for m in sorted(baseline.keys()):
        comparison_rows.append(
            {
                "model": m,
                "rank_original": baseline.get(m, float("nan")),
                "rank_after_high_removal": after_high.get(m, float("nan")),
                "rank_after_all_removal": after_all.get(m, float("nan")),
                "delta_high": removal["delta_high"].get(m, 0),
                "delta_all": removal["delta_all"].get(m, 0),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("rank_original")
    comparison_df.to_csv(OUTPUT_DIR / "flagged_removal_impact.csv", index=False)
    print(f"  Saved flagged_removal_impact.csv")
    print(f"  Matched {removal['n_high_matched']} high-priority + {removal['n_medium_matched']} medium-priority flagged tasks")
    print(f"  Tasks remaining after all removals: {removal['tasks_remaining_after_all']}")

    # --- Stability summary ---
    stability = overall_stability_metrics(loo_global, loo_cat)

    summary_rows = []
    for scope, metrics in stability.items():
        for k, v in metrics.items():
            summary_rows.append({"scope": scope, "metric": k, "value": round(v, 4)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "benchmark_stability_summary.csv", index=False)
    print(f"\n  Saved benchmark_stability_summary.csv")

    # --- Print key findings ---
    print("\n--- KEY FINDINGS ---")

    glo = stability["global_loo"]
    print(f"\nGlobal LOO stability:")
    print(f"  Max rank change from removing any single task: {glo['max_rank_change_any_task']:.0f} positions")
    print(f"  Median max rank change: {glo['median_max_rank_change']:.1f}")
    print(f"  Mean Spearman with baseline: {glo['mean_spearman_with_baseline']:.4f}")
    print(f"  Min Spearman with baseline: {glo['min_spearman_with_baseline']:.4f}")
    print(f"  % tasks causing zero rank change: {glo['pct_tasks_zero_change']:.1f}%")
    print(f"  % tasks causing ≤1 position change: {glo['pct_tasks_change_le_1']:.1f}%")

    print(f"\nTop 10 most impactful tasks to remove:")
    print(loo_global[["task_name", "category", "max_rank_change", "models_moved_2plus", "spearman_with_baseline"]].head(10).to_string(index=False))

    print(f"\nFlagged task removal impact on global rankings:")
    print(comparison_df[["model", "rank_original", "rank_after_high_removal", "rank_after_all_removal", "delta_all"]].to_string(index=False))


if __name__ == "__main__":
    main()
