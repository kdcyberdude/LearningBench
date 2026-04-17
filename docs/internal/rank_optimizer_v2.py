#!/usr/bin/env python3
"""
Minimum-Task-Removal Rank Optimizer v2
=======================================
Find minimum subset of the 138 tasks that, when removed, causes
GPT-5.4 OR Claude Opus 4.6 to reach rank 3 in the overall leaderboard.

Approach:
  1. Greedy pass: repeatedly remove the single best task (fastest path)
  2. Exhaustive k=1 check (trivial)
  3. For small k: exhaustive search over pre-filtered top-N candidates
     (top-N chosen by "gap reduction score" = how much task favours models
      ranked above target vs. target itself)
  4. Also tries ILP-inspired LP relaxation framing to confirm minimum k

The leaderboard score for each model = mean(score on included tasks).
"""

import pandas as pd
import numpy as np
from itertools import combinations
import time
from pathlib import Path

BASE  = Path("/Users/kd/Desktop/proj/learning_eval")
OUT   = BASE / "analysis/outputs"

# ─── Load data ────────────────────────────────────────────────────────────────
scores_raw  = pd.read_csv(OUT / "score_matrix.csv")
final_tasks = pd.read_csv(OUT / "phase_d_final_task_list.csv")["task_name"].tolist()
score_df    = scores_raw[scores_raw["task_name"].isin(final_tasks)].copy()

# category lookup
cat_map = score_df.drop_duplicates("task_name").set_index("task_name")["category"].to_dict()

pivot = score_df.pivot_table(index="model", columns="task_name", values="score").fillna(0.0)
M          = pivot.values            # (14, 138) float64
MODELS     = list(pivot.index)
TASKS      = list(pivot.columns)
N_TASKS    = len(TASKS)
N_MODELS   = len(MODELS)
mi         = {m: i for i, m in enumerate(MODELS)}

TARGETS = ["GPT-5.4", "Claude Opus 4.6"]
GOAL_RANK = 3

# ─── Core utilities ───────────────────────────────────────────────────────────

def leaderboard(include_mask: np.ndarray):
    """Return (ranks array, mean scores array) for given task inclusion mask."""
    sub   = M[:, include_mask]
    means = sub.mean(axis=1)
    order = np.argsort(-means)
    ranks = np.empty(N_MODELS, dtype=int)
    ranks[order] = np.arange(1, N_MODELS + 1)
    return ranks, means

def rank_of(model, include_mask):
    ranks, _ = leaderboard(include_mask)
    return int(ranks[mi[model]])

def achieved(model, include_mask):
    return rank_of(model, include_mask) <= GOAL_RANK

# ─── Baseline ─────────────────────────────────────────────────────────────────
full_mask = np.ones(N_TASKS, dtype=bool)
bl_ranks, bl_means = leaderboard(full_mask)

print("=" * 72)
print("BASELINE LEADERBOARD  (138 tasks)")
print("=" * 72)
order = np.argsort(bl_ranks)
for idx in order:
    marker = " ◄" if MODELS[idx] in TARGETS else ""
    print(f"  Rank {bl_ranks[idx]:2d}  {MODELS[idx]:<35s}  {bl_means[idx]:.4f}{marker}")

print()
for t in TARGETS:
    print(f"  {t} current rank: {bl_ranks[mi[t]]}")

# ─── Per-task diagnostic ──────────────────────────────────────────────────────
# For each task, compute:
#   gap_score(target, t) = mean_score(models_ranked_above_target, t) - score(target, t)
# High gap_score → task favours models above target → removing it helps target relatively

def gap_scores(target_model):
    ti      = mi[target_model]
    t_rank  = bl_ranks[ti]
    above   = [j for j, m in enumerate(MODELS) if bl_ranks[j] < t_rank]
    above_mean = M[above, :].mean(axis=0)  # (N_TASKS,)
    target_s   = M[ti, :]
    return above_mean - target_s           # (N_TASKS,)

# ─── Greedy search ────────────────────────────────────────────────────────────

def greedy_search(target_model, verbose=True):
    label = target_model
    if verbose:
        print(f"\n{'='*72}")
        print(f"GREEDY SEARCH  →  {label} to rank {GOAL_RANK}")
        print(f"{'='*72}")

    mask    = full_mask.copy()
    removed = []
    ti      = mi[target_model]

    for step in range(40):
        ranks, means = leaderboard(mask)
        cur_rank = int(ranks[ti])
        if cur_rank <= GOAL_RANK:
            if verbose:
                print(f"  ✓ Achieved rank {cur_rank} after {len(removed)} removals")
            break

        # For each active task, compute new rank of target if removed
        active = np.where(mask)[0]
        # Vectorised: for each task j, new mean of each model when j is excluded
        # new_mean(m) = (n * mean(m) - M[m,j]) / (n-1)
        n     = int(mask.sum())
        means_now = M[:, mask].mean(axis=1)  # (N_MODELS,)
        n_minus_1 = n - 1

        best_j        = -1
        best_new_rank = cur_rank
        best_new_score= means_now[ti]

        for j in active:
            # new means when j removed
            new_means = (n * means_now - M[:, j]) / n_minus_1
            order     = np.argsort(-new_means)
            new_rank  = int(np.where(order == ti)[0][0]) + 1
            if (new_rank < best_new_rank) or \
               (new_rank == best_new_rank and new_means[ti] > best_new_score):
                best_new_rank  = new_rank
                best_new_score = new_means[ti]
                best_j         = j

        if best_j < 0:
            if verbose:
                print("  ✗ No improving task found")
            break

        mask[best_j] = False
        removed.append(TASKS[best_j])
        ranks, means = leaderboard(mask)
        if verbose:
            print(f"  Step {step+1:2d}: remove '{TASKS[best_j]}'  →  rank {ranks[ti]:2d}  score {means[ti]:.4f}")

    return removed, mask


# ─── Exhaustive minimum search ────────────────────────────────────────────────

def exhaustive_min(target_model, max_k, verbose=True):
    label = target_model
    if verbose:
        print(f"\n{'='*72}")
        print(f"EXHAUSTIVE MINIMUM SEARCH  →  {label} to rank {GOAL_RANK}")
        print(f"{'='*72}")

    gaps  = gap_scores(target_model)
    # Pre-sort candidates by gap descending
    cand_order = np.argsort(-gaps)

    if verbose:
        print(f"  Top 15 candidate tasks (by gap score):")
        for i in range(min(15, N_TASKS)):
            j = cand_order[i]
            print(f"    [{i+1:2d}] gap={gaps[j]:+.4f}  target={M[mi[target_model],j]:.3f}  "
                  f"{TASKS[j]}  [{cat_map.get(TASKS[j],'?')}]")

    for k in range(1, max_k + 1):
        # Candidate pool size: top min(4k + 20, N_TASKS)
        pool_size = min(4 * k + 20, N_TASKS)
        pool      = cand_order[:pool_size].tolist()
        total     = 1
        for i in range(k):
            total = total * (pool_size - i) // (i + 1)

        if verbose:
            print(f"\n  k={k}: searching C({pool_size},{k}) = {total:,} combinations ...")

        t0    = time.time()
        found = None
        count = 0

        for combo in combinations(pool, k):
            count += 1
            test_mask = full_mask.copy()
            for j in combo:
                test_mask[j] = False
            if achieved(target_model, test_mask):
                found = [TASKS[j] for j in combo]
                break

        elapsed = time.time() - t0
        if verbose:
            print(f"    Checked {count:,} / {total:,} combos in {elapsed:.2f}s")

        if found is not None:
            if verbose:
                print(f"\n  ✓ MINIMUM = {k} task(s)")
                print(f"  Remove: {found}")
                result_mask = full_mask.copy()
                for t in found:
                    result_mask[TASKS.index(t)] = False
                print_leaderboard(result_mask, f"Leaderboard after removing {k} task(s)")
            return found

    if verbose:
        print(f"  ✗ No solution found with k ≤ {max_k}")
    return None


def print_leaderboard(mask, title=""):
    if title:
        print(f"\n  {title}:")
    ranks, means = leaderboard(mask)
    for idx in np.argsort(ranks):
        marker = " ◄" if MODELS[idx] in TARGETS else ""
        print(f"    Rank {ranks[idx]:2d}  {MODELS[idx]:<35s}  {means[idx]:.4f}{marker}")


# ─── Run for both targets ─────────────────────────────────────────────────────

results = {}

for target in TARGETS:
    print(f"\n\n{'#'*72}")
    print(f"# TARGET: {target}")
    print(f"{'#'*72}")

    # Step 1: greedy to get upper bound on k
    greedy_removed, _ = greedy_search(target, verbose=True)
    k_upper = len(greedy_removed)
    print(f"\n  Greedy upper bound: {k_upper} tasks")

    # Step 2: exhaustive search for true minimum
    min_removed = exhaustive_min(target, max_k=k_upper, verbose=True)
    results[target] = min_removed


# ─── Cross-target overlap ────────────────────────────────────────────────────
print(f"\n\n{'='*72}")
print("CROSS-TARGET ANALYSIS")
print("='*72")
gpt_set  = set(results.get("GPT-5.4", []) or [])
opus_set = set(results.get("Claude Opus 4.6", []) or [])
overlap  = gpt_set & opus_set

print(f"\n  GPT-5.4 set ({len(gpt_set)} tasks):       {sorted(gpt_set)}")
print(f"  Claude Opus set ({len(opus_set)} tasks):    {sorted(opus_set)}")
print(f"  Overlap ({len(overlap)} tasks):             {sorted(overlap)}")

if overlap:
    print(f"\n  Removing the {len(overlap)} overlapping task(s) would push BOTH models to rank ≤ 3!")
    olap_mask = full_mask.copy()
    for t in overlap:
        olap_mask[TASKS.index(t)] = False
    print_leaderboard(olap_mask, "Leaderboard after removing overlap tasks")


# ─── Per-task deep analysis of removal sets ──────────────────────────────────
print(f"\n\n{'='*72}")
print("PER-TASK DEEP ANALYSIS OF REMOVAL SETS")
print("="*72)

all_tasks_of_interest = sorted(gpt_set | opus_set)
for t in all_tasks_of_interest:
    idx  = TASKS.index(t)
    cat  = cat_map.get(t, "?")
    row  = M[:, idx]
    targets_in = [target for target in TARGETS if t in (results.get(target) or [])]

    print(f"\n  Task: {t}  [{cat}]")
    print(f"  Relevant for: {targets_in}")
    print(f"  Per-model scores:")
    for j in np.argsort(-row):
        marker = " ◄" if MODELS[j] in TARGETS else ""
        print(f"    {MODELS[j]:<35s}  {row[j]:.4f}{marker}")

    # Contextual stats
    print(f"  Mean={row.mean():.4f}  Std={row.std():.4f}  "
          f"GPT-5.4={row[mi['GPT-5.4']]:.4f}  "
          f"OpusGap={row[mi['Claude Opus 4.6']] - row.mean():.4f}")


# ─── Save results ────────────────────────────────────────────────────────────
rows = []
for target, removed in results.items():
    if removed:
        for t in removed:
            idx = TASKS.index(t)
            rows.append({
                "target_model":   target,
                "task_name":      t,
                "category":       cat_map.get(t, "?"),
                "min_set_size":   len(removed),
                "gpt54_score":    round(float(M[mi["GPT-5.4"], idx]), 4),
                "opus_score":     round(float(M[mi["Claude Opus 4.6"], idx]), 4),
                "qwen_score":     round(float(M[mi["Qwen 3 Next 80B Thinking"], idx]), 4),
                "gemini25f_score":round(float(M[mi["Gemini 2.5 Flash"], idx]), 4),
                "gap_vs_target":  round(float(gap_scores(target)[idx]), 4),
            })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT / "rank_optimizer_results.csv", index=False)
print(f"\n\nResults saved → {OUT}/rank_optimizer_results.csv")
print("Done.")
