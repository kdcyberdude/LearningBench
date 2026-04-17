"""
Phase A: Data Extraction & Analysis Foundation
================================================
A1  – Parse all 5 leaderboard JSONs into unified score_matrix.csv
A2  – Compute Associative Learning aggregate scores (confirm vs. JSON aggregate)
A3  – Per-task statistics: mean, std, min, max, entropy, % zero, % perfect
A4  – Per-model statistics: mean per category, overall rank, tier
A5  – Flag problematic tasks (all-zero, all-perfect, near-uniform, low-entropy, etc.)

Outputs (written to analysis/outputs/):
  score_matrix.csv        – long format: category, task_name, model, score
  aggregates.csv          – per-model per-category aggregates (JSON vs computed)
  task_stats.csv          – per-task statistics
  model_stats.csv         – per-model statistics
  flagged_tasks.csv       – tasks failing at least one quality criterion
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the analysis package is importable when run from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_aggregates, load_score_matrix
from utils.stats import (
    bimodality_coefficient,
    item_discrimination_index,
    shannon_entropy,
    tier_discrimination,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIER_ORDER = ["small", "mid", "frontier"]


# ---------------------------------------------------------------------------
# A1 – Score matrix
# ---------------------------------------------------------------------------

def build_score_matrix() -> pd.DataFrame:
    print("A1 – Building score matrix …")
    df = load_score_matrix()
    out = OUTPUT_DIR / "score_matrix.csv"
    df.to_csv(out, index=False)
    print(f"   Saved {len(df):,} rows → {out}")
    # Quick sanity check
    n_models = df["model"].nunique()
    n_tasks = df["task_name"].nunique()
    n_cats = df["category"].nunique()
    print(f"   {n_models} models × {n_tasks} tasks across {n_cats} categories")
    for cat, grp in df.groupby("category"):
        print(f"   [{cat}] {grp['task_name'].nunique()} tasks, {grp['model'].nunique()} models")
    return df


# ---------------------------------------------------------------------------
# A2 – Aggregates
# ---------------------------------------------------------------------------

def build_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    print("\nA2 – Computing aggregate scores …")
    agg = load_aggregates()
    out = OUTPUT_DIR / "aggregates.csv"
    agg.to_csv(out, index=False)

    print(f"   Saved → {out}")
    print("\n   Category aggregate comparison (JSON vs computed from task means):")
    print(f"   {'Category':<14} {'Model':<35} {'JSON':>8} {'Computed':>10} {'Diff':>8}")
    print("   " + "-" * 80)
    for _, row in agg.sort_values(["category", "model"]).iterrows():
        diff = row["json_aggregate"] - row["computed_aggregate"]
        flag = " !" if abs(diff) > 0.01 else ""
        print(f"   {row['category']:<14} {row['model']:<35} {row['json_aggregate']:>8.4f} "
              f"{row['computed_aggregate']:>10.4f} {diff:>+8.4f}{flag}")
    return agg


# ---------------------------------------------------------------------------
# A3 – Per-task statistics
# ---------------------------------------------------------------------------

def build_task_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("\nA3 – Computing per-task statistics …")

    # Per-category totals for discrimination index
    category_totals = (
        df.groupby(["category", "model"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "cat_total"})
    )
    df = df.merge(category_totals, on=["category", "model"])

    records = []
    for (cat, task), grp in df.groupby(["category", "task_name"]):
        scores = grp["score"].values
        cat_totals = grp["cat_total"].values

        tier_scores = {
            t: grp.loc[grp["tier"] == t, "score"].values for t in TIER_ORDER
        }

        tier_info = tier_discrimination(tier_scores)
        bc = bimodality_coefficient(scores)
        ent = shannon_entropy(scores)
        disc = item_discrimination_index(scores, cat_totals)

        records.append(
            {
                "category": cat,
                "task_name": task,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "n_models": len(scores),
                "n_zero": int(np.sum(scores == 0.0)),
                "n_perfect": int(np.sum(scores == 1.0)),
                "pct_zero": float(np.mean(scores == 0.0)),
                "pct_perfect": float(np.mean(scores == 1.0)),
                "entropy": ent,
                "bimodality_coeff": bc,
                "item_discrimination": disc,
                "kw_stat": tier_info["kw_stat"],
                "kw_p": tier_info["kw_p"],
                "tier_discriminates": tier_info["discriminates"],
                "mean_small": tier_info["tier_means"].get("small", float("nan")),
                "mean_mid": tier_info["tier_means"].get("mid", float("nan")),
                "mean_frontier": tier_info["tier_means"].get("frontier", float("nan")),
            }
        )

    stats_df = pd.DataFrame(records).sort_values(["category", "mean"]).reset_index(drop=True)
    out = OUTPUT_DIR / "task_stats.csv"
    stats_df.to_csv(out, index=False)
    print(f"   Saved {len(stats_df)} task rows → {out}")
    return stats_df


# ---------------------------------------------------------------------------
# A4 – Per-model statistics
# ---------------------------------------------------------------------------

def build_model_stats(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    print("\nA4 – Computing per-model statistics …")

    # Per-category mean from task-level data (more accurate than aggregate)
    cat_means = (
        df.groupby(["model", "category"])["score"]
        .mean()
        .unstack("category")
        .reset_index()
    )

    # Overall mean across categories
    cat_means["overall_mean"] = cat_means[
        [c for c in cat_means.columns if c != "model"]
    ].mean(axis=1)

    # Ranks per category (1 = best)
    for cat in ["associative", "concept", "language", "observational", "rl"]:
        if cat in cat_means.columns:
            cat_means[f"rank_{cat}"] = cat_means[cat].rank(ascending=False, method="min")

    cat_means["rank_overall"] = cat_means["overall_mean"].rank(ascending=False, method="min")

    # Attach tier and provider
    from utils.data_loader import MODEL_PROVIDERS, MODEL_TIERS
    cat_means["tier"] = cat_means["model"].map(MODEL_TIERS).fillna("unknown")
    cat_means["provider"] = cat_means["model"].map(MODEL_PROVIDERS).fillna("unknown")

    cat_means = cat_means.sort_values("overall_mean", ascending=False).reset_index(drop=True)

    out = OUTPUT_DIR / "model_stats.csv"
    cat_means.to_csv(out, index=False)
    print(f"   Saved {len(cat_means)} model rows → {out}")

    # Quick print of final rankings
    print("\n   Overall model rankings:")
    cols = ["model", "tier", "provider", "overall_mean", "rank_overall"]
    print(cat_means[cols].to_string(index=False))
    return cat_means


# ---------------------------------------------------------------------------
# A5 – Flag problematic tasks
# ---------------------------------------------------------------------------

FLAGGING_CRITERIA = {
    "all_zero":        lambda r: r["mean"] == 0.0,
    "all_perfect":     lambda r: r["mean"] == 1.0,
    "near_uniform":    lambda r: r["std"] < 0.05,
    "low_entropy":     lambda r: r["entropy"] < 0.5 and r["mean"] not in (0.0, 1.0),
    "no_discrimination": lambda r: not np.isnan(r["item_discrimination"]) and r["item_discrimination"] < 0.1,
    "no_tier_diff":    lambda r: not r["tier_discriminates"] and r["std"] > 0.05,
    "bimodal":         lambda r: not np.isnan(r["bimodality_coeff"]) and r["bimodality_coeff"] > 0.555,
    "too_easy":        lambda r: r["mean"] > 0.90,
    "too_hard":        lambda r: r["mean"] < 0.05 and r["mean"] != 0.0,
}


def build_flagged_tasks(task_stats: pd.DataFrame) -> pd.DataFrame:
    print("\nA5 – Flagging problematic tasks …")

    flag_records = []
    for _, row in task_stats.iterrows():
        flags = [name for name, fn in FLAGGING_CRITERIA.items() if fn(row)]
        if flags:
            rec = row.to_dict()
            rec["flags"] = "|".join(flags)
            rec["n_flags"] = len(flags)
            flag_records.append(rec)

    flagged_df = pd.DataFrame(flag_records).sort_values(
        ["n_flags", "category"], ascending=[False, True]
    ).reset_index(drop=True)

    out = OUTPUT_DIR / "flagged_tasks.csv"
    flagged_df.to_csv(out, index=False)
    print(f"   Flagged {len(flagged_df)} / {len(task_stats)} tasks → {out}")

    # Summary by flag type
    from collections import Counter
    all_flags = []
    for _, row in flagged_df.iterrows():
        all_flags.extend(row["flags"].split("|"))
    counts = Counter(all_flags)
    print("\n   Flag summary:")
    for flag, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"   {flag:<22} {count:>3} tasks")

    # Per-category counts
    print("\n   Flagged tasks per category:")
    for cat, grp in flagged_df.groupby("category"):
        print(f"   {cat}: {len(grp)} flagged")

    return flagged_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE A: Data Extraction & Analysis Foundation")
    print("=" * 70)

    df = build_score_matrix()
    agg = build_aggregates(df)
    task_stats = build_task_stats(df)
    model_stats = build_model_stats(df, agg)
    flagged = build_flagged_tasks(task_stats)

    print("\n" + "=" * 70)
    print("Phase A complete. Outputs in analysis/outputs/")
    print("  score_matrix.csv    – raw scores")
    print("  aggregates.csv      – per-category aggregates")
    print("  task_stats.csv      – per-task stats")
    print("  model_stats.csv     – per-model stats & rankings")
    print("  flagged_tasks.csv   – tasks needing review")
    print("=" * 70)
