"""
Find the 4 models to drop from 14 so that the remaining 10 show
maximum spread / significant step-changes on a bar chart.

Strategy:
  1. Compute mean score per model.
  2. Sort models by mean score.
  3. Find the 4 models to *remove* whose removal maximises the average
     gap between consecutive models (i.e. we keep the 10 models whose
     scores are most spread out / evenly spaced).
     We do this with exhaustive search over C(14,4) = 1001 combinations.
  4. Also flag "redundant clusters" – pairs of models whose scores are
     very close together (< some threshold) – as natural removal candidates.
"""

import pandas as pd
import numpy as np
from itertools import combinations

CSV = "/Users/kd/Desktop/proj/learning_eval/analysis/outputs/notebook_logs/all_notebook_logs.csv"

# ── 1. Load & aggregate ────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(CSV, usecols=["model_display_name", "score_fraction", "score_value", "status"])

# Keep only rows with a valid status
df = df[df["status"] == "ok"].copy()

# score_fraction is only filled for assoc-learning tasks;
# all other categories store the score in score_value.
# Use score_fraction when available, fall back to score_value.
df["score"] = pd.to_numeric(df["score_fraction"], errors="coerce").combine_first(
              pd.to_numeric(df["score_value"],    errors="coerce"))
df = df.dropna(subset=["score"])

model_stats = (
    df.groupby("model_display_name")["score"]
    .agg(mean_score="mean", std_score="std", n_tasks="count")
    .reset_index()
    .sort_values("mean_score", ascending=False)
    .reset_index(drop=True)
)

print(f"\nAll {len(model_stats)} models (sorted by mean score, all 6 categories, unified score):")
print(model_stats[["model_display_name", "mean_score", "std_score", "n_tasks"]].to_string(index=False))

# Drop models with too few tasks (likely incomplete runs) — keep only those
# with at least 50 tasks so scores are meaningful.
model_stats = model_stats[model_stats["n_tasks"] >= 50].reset_index(drop=True)
n_models = len(model_stats)
print(f"\nAfter filtering to models with ≥5 tasks: {n_models} models")
assert n_models == 14, f"Expected 14 models, found {n_models}"

scores = model_stats["mean_score"].values
names  = model_stats["model_display_name"].values

# ── 2. Exhaustive search for the best 10-model subset ─────────────────────────
# Objective: maximise the minimum gap between consecutive scores
#            (secondary: maximise the average gap = total spread / 9)

best_min_gap   = -1
best_avg_gap   = -1
best_keep_idx  = None
best_drop_idx  = None

all_idx = list(range(n_models))

for drop_combo in combinations(all_idx, 4):
    keep_idx = sorted(set(all_idx) - set(drop_combo))
    kept_scores = scores[keep_idx]  # already sorted desc – reverse for gaps
    kept_sorted = np.sort(kept_scores)  # ascending for gap calc
    gaps = np.diff(kept_sorted)
    min_gap = gaps.min()
    avg_gap = gaps.mean()

    # Primary: maximise min gap  (makes every consecutive pair distinct)
    # Tie-break: maximise avg gap (maximises total spread)
    if (min_gap > best_min_gap) or (min_gap == best_min_gap and avg_gap > best_avg_gap):
        best_min_gap  = min_gap
        best_avg_gap  = avg_gap
        best_keep_idx = keep_idx
        best_drop_idx = list(drop_combo)

# ── 3. Results ─────────────────────────────────────────────────────────────────
kept_scores = np.sort(scores[best_keep_idx])
gaps = np.diff(kept_scores)

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

print("\n✅  10 MODELS TO KEEP (sorted by score, low → high):")
keep_rows = model_stats.iloc[sorted(best_keep_idx)].sort_values("mean_score")
for _, row in keep_rows.iterrows():
    print(f"  {row['mean_score']:.4f}  {row['model_display_name']}")

print(f"\n❌  4 MODELS TO REMOVE:")
for i in sorted(best_drop_idx):
    row = model_stats.iloc[i]
    print(f"  {row['mean_score']:.4f}  {row['model_display_name']}")

print(f"\n📊  Gap statistics for the 10-model set:")
print(f"  Min gap  : {best_min_gap:.4f}")
print(f"  Max gap  : {gaps.max():.4f}")
print(f"  Avg gap  : {best_avg_gap:.4f}")
print(f"  Std gap  : {gaps.std():.4f}")
print(f"  Total spread (max−min): {kept_scores[-1] - kept_scores[0]:.4f}")

# ── 4. Show why those 4 are the ones to drop ──────────────────────────────────
print("\n💡  Why these 4?")
print("  The removed models are the ones whose scores were closest to")
print("  a neighbour — i.e. they were inside a dense cluster. Removing")
print("  them maximises the *minimum* bar-to-bar gap on the chart.\n")

# Show the full original ranking with a marker
print("Full ranking with removal markers:")
for i, (name, score) in enumerate(zip(names, scores)):
    marker = "  ❌ REMOVE" if i in best_drop_idx else ""
    print(f"  #{i+1:2d}  {score:.4f}  {name}{marker}")
