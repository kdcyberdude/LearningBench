"""
C5: Comprehensive Phase C Summary Analysis

Covers:
  - Overall benchmark health summary (all stats in one place)
  - H5 revised interpretation (UNKNOWN tasks: H5 INVERTED — models score BETTER)
  - Final ranking stability confirmation
  - Key numbers table for the Kaggle writeup
  - Phase D input: definitive flagged task list with reasons

Outputs:
  - phase_c_summary.csv  (key numbers for writeup)
  - final_flagged_tasks.csv  (definitive removal list for Phase D)
  - writeup_numbers.csv  (all numbers the writeup cites)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def load_phase_b_outputs():
    """Load all Phase B CSVs for synthesis."""
    files = {
        "discrimination": "discrimination_report.csv",
        "cross_cat": "cross_category_correlations.csv",
        "cross_cat_pval": "cross_category_pvalues.csv",
        "tier_stats": "tier_stats.csv",
        "tier_task_gaps": "tier_task_gaps.csv",
        "tier_inversions": "tier_inversions.csv",
        "entropy": "entropy_report.csv",
        "cat_entropy": "category_entropy.csv",
        "provider": "provider_pivot.csv",
        "bimodal": "bimodal_report.csv",
        "rank1": "rank1_counts.csv",
        "thinking": "thinking_comparison.csv",
        "gemini": "gemini_ceiling.csv",
        "loo_global": "loo_global.csv",
        "loo_cat": "loo_category.csv",
        "flagged_removal": "flagged_removal_impact.csv",
        "random_baseline": "random_baseline.csv",
        "model_stats": "model_stats.csv",
        "task_stats": "task_stats.csv",
    }
    result = {}
    for key, fname in files.items():
        path = OUTPUT_DIR / fname
        if path.exists():
            result[key] = pd.read_csv(path)
    return result


def build_writeup_numbers(data: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Extract all key numbers needed for the Kaggle writeup."""
    rows = []

    def add(metric, value, source, notes=""):
        rows.append({"metric": metric, "value": str(value), "source": source, "notes": notes})

    # Basic counts
    add("total_tasks", df["task_name"].nunique(), "score_matrix", "Before curation")
    add("total_models", df["model"].nunique(), "score_matrix", "")
    add("total_categories", df["category"].nunique(), "score_matrix", "")

    for cat, n in df.groupby("category")["task_name"].nunique().items():
        add(f"tasks_{cat}", n, "score_matrix", "")

    # After cleaning (134 tasks)
    if "flagged_removal" in data:
        fr = data["flagged_removal"]
        # Find from stability summary if available
    add("tasks_after_curation_estimate", 134, "C1_analysis", "After removing 23 flagged tasks (9 high + 14 medium)")

    # Discrimination
    if "discrimination" in data:
        disc = data["discrimination"]
        # Find the classification and r column
        r_col = next((c for c in ["discrimination_r", "r"] if c in disc.columns), None)
        class_col = next((c for c in ["classification", "discrimination_class"] if c in disc.columns), None)
        if class_col:
            excellent = disc[class_col].str.contains("excellent", na=False).sum()
            neg = disc[class_col].str.contains("negative", na=False).sum()
        elif r_col:
            excellent = (disc[r_col] >= 0.50).sum()
            neg = (disc[r_col] < 0).sum()
        else:
            excellent, neg = None, None
        if excellent is not None:
            add("pct_tasks_excellent_discrimination",
                f"{excellent/len(disc)*100:.1f}%",
                "B1_discrimination",
                f"{excellent}/157 tasks with r ≥ 0.50")
        if neg is not None:
            add("tasks_negative_discrimination", int(neg), "B1_discrimination", "Candidates for removal")

    # Cross-category correlations
    if "cross_cat" in data:
        cc = data["cross_cat"]
        # Get upper triangle values
        cc_vals = []
        for col in cc.columns[1:]:
            for idx, row in cc.iterrows():
                val = row[col]
                if isinstance(val, (int, float)) and not np.isnan(val) and val < 0.999:
                    cc_vals.append(val)
        if cc_vals:
            add("mean_cross_category_spearman", f"{np.mean(cc_vals):.3f}", "B2_cross_category",
                "Mean pairwise Spearman r across all category pairs")
        # Weakest pair
        add("weakest_cross_category_pair", "Concept × Observational (r=0.552)",
            "B2_cross_category", "Most distinct learning sub-abilities")
        add("strongest_cross_category_pair", "Concept × RL (r=0.886)",
            "B2_cross_category", "Hypothesis testing link")

    # Scale inversions
    if "tier_inversions" in data:
        n_inv = len(data["tier_inversions"])
        add("n_scale_inversions", n_inv, "B3_scaling", "Model-level inversions with gap > 0.05")

    # Most striking inversions
    add("biggest_inversion", "Qwen Thinking (mid) > GPT-5.4 (frontier) on Observational by 0.289",
        "B3_scaling", "H2 evidence")

    # Efficiency
    add("efficiency_max_rank_change", 0, "B4_efficiency",
        "Rankings identical with/without efficiency — monotonic transformation")

    # Entropy
    if "cat_entropy" in data:
        cat_ent = data["cat_entropy"]
        # Find most/least informative
        if "mean_entropy" in cat_ent.columns:
            best = cat_ent.loc[cat_ent["mean_entropy"].idxmax(), "category"]
            worst = cat_ent.loc[cat_ent["mean_entropy"].idxmin(), "category"]
            add("most_informative_category_by_entropy", best, "B7_entropy", "Highest mean task entropy")
            add("least_informative_category_by_entropy", worst, "B7_entropy", "Lowest mean task entropy")

    # Thinking vs non-thinking
    if "thinking" in data:
        t = data["thinking"]
        if "thinking_advantage" in t.columns:
            max_adv = t["thinking_advantage"].max()
            max_cat = t.loc[t["thinking_advantage"].idxmax(), "category"]
            add("max_thinking_advantage", f"+{max_adv:.3f}", "B8_thinking",
                f"On {max_cat} — thinking models vs non-thinking")
        add("mean_thinking_advantage_pct", "+241%", "B8_thinking",
            "Mean improvement across all categories (Qwen Thinking vs Instruct)")

    # Gemini dominance
    if "rank1" in data:
        r1 = data["rank1"]
        if "model" in r1.columns and "rank1_count" in r1.columns:
            gemini_row = r1[r1["model"].str.contains("Gemini 3.1 Pro", na=False)]
            if not gemini_row.empty:
                add("gemini_rank1_tasks", int(gemini_row["rank1_count"].iloc[0]),
                    "B6_dominance", "Out of 157 tasks (expected 11 by random)")
        add("gemini_rank1_pct", "23.6%", "B6_dominance", "vs 7.1% random expectation (3.3× baseline)")

    # Bimodality
    if "bimodal" in data:
        n_bimodal = len(data["bimodal"])
        add("n_bimodal_tasks", n_bimodal, "B5_bimodality",
            f"{n_bimodal}/157 tasks ({n_bimodal/157*100:.0f}%) show bimodal score distribution")

    # Provider analysis
    add("provider_openai_concept_score", 0.233, "B9_provider", "Weakest provider on rule induction")
    add("provider_anthropic_concept_score", 0.263, "B9_provider", "Concept formation deficit")
    add("provider_google_best_overall", 0.490, "B9_provider", "Highest mean across all categories")

    # LOO stability
    if "loo_global" in data:
        loo = data["loo_global"]
        max_change = loo["max_rank_change"].max()
        mean_spearman = loo["spearman_with_baseline"].mean()
        add("loo_max_rank_change", int(max_change), "C1_loo",
            "Maximum rank change from removing any single task")
        add("loo_mean_spearman", f"{mean_spearman:.4f}", "C1_loo",
            "Mean Spearman correlation after LOO removal")
        pct_stable = float((loo["max_rank_change"] == 0).mean() * 100)
        add("loo_pct_zero_change", f"{pct_stable:.1f}%", "C1_loo",
            "% tasks whose removal causes zero rank changes")

    # Random baseline
    if "random_baseline" in data:
        rb = data["random_baseline"]
        for _, row in rb.iterrows():
            add(f"signal_ratio_{row['category']}",
                f"{row['signal_ratio_mean']:.1f}×",
                "C2_random_baseline",
                f"Mean model score is {row['signal_ratio_mean']:.1f}× above random")

    # H5 revised finding
    add("h5_epistemic_verdict", "INVERTED — models score BETTER on UNKNOWN tasks",
        "C4_epistemic",
        "Models score 0.695 on UNKNOWN tasks vs 0.582 on normal tasks. "
        "Frontier models (Claude Opus/Sonnet 1.0, GPT-5.4 1.0) show epistemic awareness. "
        "Gemini Pro scores LOWER (0.5) on blocking_effect — possibly overthinking. "
        "H5 is incorrect: models show epistemic humility on explicit UNKNOWN traps.")

    add("n_unknown_tasks", 3, "C4_epistemic",
        "Tasks with UNKNOWN as correct answer: blocking_effect, xor_attribute_binding, deep_second_order_extinction")

    return pd.DataFrame(rows)


def build_final_flagged_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate all flagged task signals from Phase B + C into final list."""
    # Load Phase B outputs
    disc_path = OUTPUT_DIR / "discrimination_report.csv"
    entropy_path = OUTPUT_DIR / "entropy_report.csv"
    tier_gaps_path = OUTPUT_DIR / "tier_task_gaps.csv"
    bimodal_path = OUTPUT_DIR / "bimodal_report.csv"

    disc = pd.read_csv(disc_path) if disc_path.exists() else pd.DataFrame()
    entropy = pd.read_csv(entropy_path) if entropy_path.exists() else pd.DataFrame()
    tier_gaps = pd.read_csv(tier_gaps_path) if tier_gaps_path.exists() else pd.DataFrame()
    bimodal = pd.read_csv(bimodal_path) if bimodal_path.exists() else pd.DataFrame()

    task_list = []
    all_tasks = df["task_name"].unique()

    for task in all_tasks:
        task_df = df[df["task_name"] == task]
        category = task_df["category"].iloc[0]
        mean_score = task_df["score"].mean()
        std_score = task_df["score"].std()

        # Get discrimination
        r_val = float("nan")
        if not disc.empty:
            r_col = next((c for c in ["discrimination_r", "r"] if c in disc.columns), None)
            row = disc[disc["task_name"] == task]
            if not row.empty and r_col:
                r_val = float(row[r_col].iloc[0])

        # Get entropy
        ent_val = float("nan")
        if not entropy.empty:
            row = entropy[entropy["task_name"] == task]
            if not row.empty:
                ent_col = "entropy" if "entropy" in entropy.columns else entropy.columns[1]
                ent_val = float(row[ent_col].iloc[0])

        # Get tier gap
        tier_gap = float("nan")
        if not tier_gaps.empty:
            row = tier_gaps[tier_gaps["task_name"] == task]
            if not row.empty:
                gap_col = next(
                    (c for c in tier_gaps.columns if "gap" in c.lower()),
                    None
                )
                if gap_col:
                    tier_gap = float(row[gap_col].iloc[0])

        # Determine flags
        flags = []
        removal_priority = "keep"

        if mean_score < 0.02:
            flags.append("all_zero")
            removal_priority = "high"
        if mean_score > 0.95:
            flags.append("near_ceiling")
            removal_priority = "high"
        if not np.isnan(r_val) and r_val < 0:
            flags.append(f"negative_discrimination(r={r_val:.3f})")
            removal_priority = max(removal_priority, "high") if r_val < -0.1 else max(removal_priority, "medium")
        if not np.isnan(ent_val) and ent_val < 0.4:
            flags.append(f"low_entropy({ent_val:.3f})")
            if removal_priority == "keep":
                removal_priority = "medium"
        if not np.isnan(tier_gap) and tier_gap < -0.15:
            flags.append(f"inverted_tier_gap({tier_gap:.3f})")
            removal_priority = max(removal_priority, "medium")

        # Check extreme bimodal
        if not bimodal.empty:
            brow = bimodal[bimodal["task_name"] == task]
            if not brow.empty:
                qual_col = next(
                    (c for c in ["quality", "bimodal_class", "type"] if c in bimodal.columns),
                    None
                )
                if qual_col:
                    bc = str(brow[qual_col].iloc[0])
                    if "extreme" in bc.lower() or "inverted" in bc.lower():
                        flags.append("extreme_or_inverted_bimodal")
                        if removal_priority == "keep":
                            removal_priority = "medium"

        def priority_key(p):
            return {"high": 0, "medium": 1, "keep": 2}.get(p, 2)

        task_list.append(
            {
                "task_name": task,
                "category": category,
                "mean_score": round(mean_score, 4),
                "std_score": round(std_score, 4),
                "discrimination_r": round(r_val, 4) if not np.isnan(r_val) else None,
                "entropy": round(ent_val, 4) if not np.isnan(ent_val) else None,
                "tier_gap": round(tier_gap, 4) if not np.isnan(tier_gap) else None,
                "flags": " | ".join(flags) if flags else "",
                "n_flags": len(flags),
                "removal_priority": removal_priority,
            }
        )

    result = pd.DataFrame(task_list)
    result = result.sort_values(
        ["removal_priority", "n_flags"],
        key=lambda x: x.map({"high": 0, "medium": 1, "keep": 2}) if x.name == "removal_priority" else x,
        ascending=[True, False],
    )
    return result


def main():
    print("=== C5: Phase C Summary Analysis ===\n")

    df = load_score_matrix()
    data = load_phase_b_outputs()
    print(f"Loaded {len(data)} Phase B/C output files\n")

    # Writeup numbers
    print("Building writeup numbers table...")
    writeup_numbers = build_writeup_numbers(data, df)
    writeup_numbers.to_csv(OUTPUT_DIR / "writeup_numbers.csv", index=False)
    print(f"  Saved writeup_numbers.csv ({len(writeup_numbers)} metrics)\n")

    # Final flagged tasks
    print("Building final flagged task list...")
    flagged = build_final_flagged_tasks(df)
    flagged.to_csv(OUTPUT_DIR / "final_flagged_tasks.csv", index=False)

    high_count = (flagged["removal_priority"] == "high").sum()
    medium_count = (flagged["removal_priority"] == "medium").sum()
    keep_count = (flagged["removal_priority"] == "keep").sum()
    print(f"  High-priority removal: {high_count} tasks")
    print(f"  Medium-priority removal: {medium_count} tasks")
    print(f"  Keep: {keep_count} tasks")
    print(f"  Estimated tasks after curation: {157 - high_count - medium_count} (removing all flagged)\n")
    print("  Saved final_flagged_tasks.csv\n")

    # Print high-priority flagged tasks
    print("--- HIGH PRIORITY REMOVAL TASKS ---")
    high_flagged = flagged[flagged["removal_priority"] == "high"]
    print(high_flagged[["task_name", "category", "mean_score", "discrimination_r", "entropy", "flags"]].to_string(index=False))

    print("\n--- MEDIUM PRIORITY TASKS (investigate) ---")
    medium_flagged = flagged[flagged["removal_priority"] == "medium"]
    print(medium_flagged[["task_name", "category", "mean_score", "discrimination_r", "entropy", "flags"]].to_string(index=False))

    print("\n--- KEY WRITEUP NUMBERS ---")
    key_metrics = [
        "total_tasks", "total_models", "total_categories",
        "pct_tasks_excellent_discrimination", "tasks_negative_discrimination",
        "mean_cross_category_spearman", "weakest_cross_category_pair",
        "n_scale_inversions", "biggest_inversion",
        "gemini_rank1_pct", "n_bimodal_tasks",
        "loo_max_rank_change", "loo_mean_spearman",
        "h5_epistemic_verdict",
        "max_thinking_advantage", "mean_thinking_advantage_pct",
    ]
    display = writeup_numbers[writeup_numbers["metric"].isin(key_metrics)]
    print(display[["metric", "value", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
