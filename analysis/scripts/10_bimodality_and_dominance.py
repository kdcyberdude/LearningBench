"""
B5 + B6 + B8 + B10: Bimodality, Gemini Dominance, Thinking vs. Non-thinking (H6, H7, H10)

B5: Bimodal task classification — which models are in the "high" group?
B6: Gemini dominance analysis — how many tasks does it rank #1?
B8: Thinking vs. Non-thinking (Qwen pair comparison, H10)
B9b: Gemini 3.1 Pro ceiling analysis (Q7 from backward track)

Outputs: bimodal_report.csv, rank1_counts.csv, thinking_comparison.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix
from utils.stats import bimodality_coefficient

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

THINKING_MODEL = "Qwen 3 Next 80B Thinking"
INSTRUCT_MODEL = "Qwen 3 Next 80B Instruct"
GEMINI_PRO = "Gemini 3.1 Pro Preview"
BIMODALITY_THRESHOLD = 0.555  # Sarle's threshold


def classify_bimodal_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """For each bimodal task, classify which tier the high-scoring group belongs to."""
    rows = []
    for (cat, task), task_df in df.groupby(["category", "task_name"]):
        scores = task_df["score"].dropna().values
        if len(scores) < 4:
            continue

        b_coeff = bimodality_coefficient(scores)
        is_bimodal = b_coeff > BIMODALITY_THRESHOLD if not np.isnan(b_coeff) else False

        if not is_bimodal:
            continue

        # Identify high-scoring models (above mean + 0.5 std)
        threshold = np.mean(scores) + 0.3 * np.std(scores)
        high_models = task_df[task_df["score"] >= threshold].copy()
        low_models = task_df[task_df["score"] < threshold].copy()

        high_tiers = set(high_models["tier"].tolist())
        high_model_names = sorted(high_models["model"].tolist())
        mean_high = float(np.mean(high_models["score"])) if len(high_models) else np.nan
        mean_low = float(np.mean(low_models["score"])) if len(low_models) else np.nan

        # Classification
        if "frontier" in high_tiers and "small" not in high_tiers:
            quality = "good_bimodal"  # frontier on top, small on bottom
        elif "small" in high_tiers and "frontier" not in high_tiers:
            quality = "inverted_bimodal"  # small on top, suspicious
        elif high_tiers == {"frontier"}:
            quality = "frontier_only"
        elif len(high_models) <= 2:
            quality = "extreme_bimodal"  # only 1-2 models score well
        else:
            quality = "mixed_bimodal"

        rows.append({
            "category": cat,
            "task_name": task,
            "bimodality_coeff": round(b_coeff, 4),
            "quality": quality,
            "n_high": len(high_models),
            "n_low": len(low_models),
            "mean_high": round(mean_high, 4) if not np.isnan(mean_high) else None,
            "mean_low": round(mean_low, 4) if not np.isnan(mean_low) else None,
            "high_tiers": str(sorted(high_tiers)),
            "high_models": ", ".join(high_model_names[:5]),
        })

    return pd.DataFrame(rows).sort_values("bimodality_coeff", ascending=False) if rows else pd.DataFrame()


def compute_rank1_counts(df: pd.DataFrame) -> pd.DataFrame:
    """For each task, rank all models. Count how many tasks each model ranks #1."""
    rows = []
    for (cat, task), task_df in df.groupby(["category", "task_name"]):
        if len(task_df) == 0:
            continue
        max_score = task_df["score"].max()
        # All models tied at max score get #1
        top_models = task_df[task_df["score"] == max_score]["model"].tolist()
        rank1_model = top_models[0] if len(top_models) == 1 else "TIE:" + ",".join(top_models[:3])
        rows.append({
            "category": cat,
            "task_name": task,
            "rank1_model": rank1_model,
            "rank1_score": round(max_score, 4),
        })

    task_ranks = pd.DataFrame(rows)

    # Count rank-1 appearances per model
    rank1_counts = {}
    for _, row in task_ranks.iterrows():
        models = [row["rank1_model"]] if not row["rank1_model"].startswith("TIE:") else []
        for m in models:
            rank1_counts[m] = rank1_counts.get(m, 0) + 1

    rank1_df = pd.DataFrame.from_dict(rank1_counts, orient="index", columns=["rank1_count"])
    rank1_df.index.name = "model"
    rank1_df = rank1_df.reset_index().sort_values("rank1_count", ascending=False)

    # Add tier and provider
    tier_map = df.groupby("model")["tier"].first()
    prov_map = df.groupby("model")["provider"].first()
    rank1_df["tier"] = rank1_df["model"].map(tier_map)
    rank1_df["provider"] = rank1_df["model"].map(prov_map)

    return rank1_df, task_ranks


def compute_thinking_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """H10: Thinking vs. Non-thinking (Qwen pair)."""
    thinking = df[df["model"] == THINKING_MODEL].groupby("category")["score"].mean()
    instruct = df[df["model"] == INSTRUCT_MODEL].groupby("category")["score"].mean()

    rows = []
    for cat in df["category"].unique():
        t = float(thinking.get(cat, np.nan))
        i = float(instruct.get(cat, np.nan))
        if not np.isnan(t) and not np.isnan(i):
            rows.append({
                "category": cat,
                "thinking_score": round(t, 4),
                "instruct_score": round(i, 4),
                "thinking_advantage": round(t - i, 4),
                "pct_improvement": round((t - i) / i * 100, 1) if i > 0 else np.nan,
            })

    return pd.DataFrame(rows).sort_values("thinking_advantage", ascending=False)


def compute_gemini_ceiling(df: pd.DataFrame) -> pd.DataFrame:
    """Q7: Does Gemini 3.1 Pro dominate or have hidden weaknesses?"""
    gemini_df = df[df["model"] == GEMINI_PRO].copy()

    rows = []
    for (cat, task), task_df in df.groupby(["category", "task_name"]):
        gemini_score = gemini_df[(gemini_df["category"] == cat) &
                                  (gemini_df["task_name"] == task)]["score"]
        if len(gemini_score) == 0:
            continue
        g_score = float(gemini_score.iloc[0])
        all_scores = task_df["score"].values
        max_score = float(np.max(all_scores))
        rank = int((task_df["score"] > g_score).sum()) + 1

        rows.append({
            "category": cat,
            "task_name": task,
            "gemini_score": round(g_score, 4),
            "max_score": round(max_score, 4),
            "gemini_rank": rank,
            "beaten_by": ", ".join(task_df[task_df["score"] > g_score]["model"].tolist()),
        })

    result = pd.DataFrame(rows)
    return result


def print_summary(bimodal: pd.DataFrame, rank1: pd.DataFrame, task_ranks: pd.DataFrame,
                  thinking: pd.DataFrame, gemini_ceiling: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("B5/B6/B8: BIMODALITY, DOMINANCE, THINKING ANALYSIS (H6, H7, H10)")
    print("=" * 60)

    print("\n--- B5: Bimodal task classification ---")
    if len(bimodal):
        print(f"  Total bimodal tasks (B > 0.555): {len(bimodal)}")
        quality_counts = bimodal["quality"].value_counts()
        for q, cnt in quality_counts.items():
            print(f"  {q:<30} {cnt:>3}")
        print(f"\n  Extreme bimodal tasks (≤2 models score well):")
        extreme = bimodal[bimodal["quality"] == "extreme_bimodal"]
        if len(extreme):
            print(extreme[["category", "task_name", "bimodality_coeff", "n_high", "high_models"]].to_string(index=False))
        else:
            print("  None")
        print(f"\n  Inverted bimodal (small model on top — suspicious):")
        inverted = bimodal[bimodal["quality"] == "inverted_bimodal"]
        if len(inverted):
            print(inverted[["category", "task_name", "bimodality_coeff", "high_models", "high_tiers"]].to_string(index=False))
        else:
            print("  None")
    else:
        print("  No bimodal tasks detected.")

    print(f"\n--- B6: Gemini dominance — Rank #1 counts (total tasks: {len(task_ranks)}) ---")
    print(rank1.to_string(index=False))

    if GEMINI_PRO in rank1["model"].values:
        gemini_rank1 = int(rank1[rank1["model"] == GEMINI_PRO]["rank1_count"].iloc[0])
        pct = gemini_rank1 / len(task_ranks) * 100
        print(f"\n  {GEMINI_PRO} ranks #1 on {gemini_rank1}/{len(task_ranks)} tasks ({pct:.1f}%)")
    else:
        print(f"\n  {GEMINI_PRO} has no exclusive #1 ranks (often tied)")

    print(f"\n  Tasks where {GEMINI_PRO} is NOT ranked #1:")
    gemini_non_top = gemini_ceiling[gemini_ceiling["gemini_rank"] > 1].sort_values("gemini_rank", ascending=False)
    print(f"  Count: {len(gemini_non_top)}")
    if len(gemini_non_top):
        print(gemini_non_top.head(10)[["category", "task_name", "gemini_score", "gemini_rank", "beaten_by"]].to_string(index=False))

    print(f"\n--- B8 / H10: Thinking vs. Non-thinking (Qwen pair) ---")
    if len(thinking):
        print(thinking.to_string(index=False))
        overall_adv = float(thinking["thinking_advantage"].mean())
        print(f"\n  Mean thinking advantage: +{overall_adv:.3f}")
        if overall_adv > 0.03:
            print("  → H10 SUPPORTED: Thinking capability provides consistent advantage.")
        else:
            print("  → H10 CHALLENGED: Thinking advantage is marginal or inconsistent.")
    else:
        print(f"  {THINKING_MODEL} or {INSTRUCT_MODEL} not found in data.")


def main():
    df = load_score_matrix()
    bimodal = classify_bimodal_tasks(df)
    rank1, task_ranks = compute_rank1_counts(df)
    thinking = compute_thinking_comparison(df)
    gemini_ceiling = compute_gemini_ceiling(df)

    print_summary(bimodal, rank1, task_ranks, thinking, gemini_ceiling)

    if len(bimodal):
        bimodal.to_csv(OUTPUT_DIR / "bimodal_report.csv", index=False)
    rank1.to_csv(OUTPUT_DIR / "rank1_counts.csv", index=False)
    task_ranks.to_csv(OUTPUT_DIR / "task_rank1.csv", index=False)
    if len(thinking):
        thinking.to_csv(OUTPUT_DIR / "thinking_comparison.csv", index=False)
    gemini_ceiling.to_csv(OUTPUT_DIR / "gemini_ceiling.csv", index=False)
    print(f"\nSaved → {OUTPUT_DIR}/bimodal_report.csv, rank1_counts.csv, thinking_comparison.csv, gemini_ceiling.csv")
    return bimodal, rank1, thinking


if __name__ == "__main__":
    main()
