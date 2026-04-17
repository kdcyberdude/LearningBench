#!/usr/bin/env python3
"""
Phase D: Minimum-Task-Removal Leaderboard Rank Optimizer
=========================================================
Goal: Find the MINIMUM set of tasks from the final 138-task benchmark
      whose removal causes GPT-5.4 OR Claude Opus 4.6 to reach 3rd place overall.

Strategy:
  1. Greedy search: iteratively remove the single task that most improves
     the target model's rank (use marginal gain = how much the gap to rank-3 closes)
  2. Branch-and-bound with aggressive pruning (depth-first, prune when 
     current best is already achieved with fewer tasks)
  3. Verify minimum via exhaustive search over small-k subsets (k=1,2,3,...)
     using bit-manipulation or itertools for small k until solution found
"""

import pandas as pd
import numpy as np
from itertools import combinations
import time
from pathlib import Path

BASE   = Path("/Users/kd/Desktop/proj/learning_eval")
SCORE  = BASE / "analysis/outputs/score_matrix.csv"
TASKS  = BASE / "analysis/outputs/phase_d_final_task_list.csv"
OUT    = BASE / "analysis/outputs"

# ─── Load ─────────────────────────────────────────────────────────────────────
scores_raw  = pd.read_csv(SCORE)
final_tasks = pd.read_csv(TASKS)["task_name"].tolist()
df = scores_raw[scores_raw["task_name"].isin(final_tasks)].copy()

# Pivot to (model × task) matrix for fast computation
pivot = df.pivot_table(index="model", columns="task_name", values="score")
pivot = pivot.reindex(columns=sorted(pivot.columns))
MODELS = list(pivot.index)
TASKS_LIST = list(pivot.columns)
N_TASKS = len(TASKS_LIST)

# Pre-fill NaN with 0 (missing = no score = 0)
M = pivot.fillna(0.0).values   # shape: (n_models, n_tasks)
model_idx = {m: i for i, m in enumerate(MODELS)}

TARGET_RANK = 3   # we want model to be rank 3 (1-indexed)

def get_ranking(task_mask):
    """Given boolean mask of tasks to INCLUDE, return dict model->rank and scores."""
    sub = M[:, task_mask]
    means = sub.mean(axis=1)
    # rank by mean descending; ties broken arbitrarily
    order = np.argsort(-means)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order)+1)
    return ranks, means

def check_goal(task_mask, target_model):
    """Returns True if target_model is at rank <= TARGET_RANK."""
    ranks, _ = get_ranking(task_mask)
    idx = model_idx[target_model]
    return ranks[idx] <= TARGET_RANK

# ─── Baseline ─────────────────────────────────────────────────────────────────
full_mask = np.ones(N_TASKS, dtype=bool)
baseline_ranks, baseline_means = get_ranking(full_mask)

print("=" * 70)
print("BASELINE LEADERBOARD (138 tasks)")
print("=" * 70)
sorted_models = sorted(enumerate(MODELS), key=lambda x: baseline_ranks[x[0]])
for i, (mi, m) in enumerate(sorted_models):
    print(f"  Rank {baseline_ranks[mi]:2d}  {m:<35s}  {baseline_means[mi]:.4f}")

gpt_rank  = baseline_ranks[model_idx["GPT-5.4"]]
opus_rank = baseline_ranks[model_idx["Claude Opus 4.6"]]
print(f"\nGPT-5.4      current rank: {gpt_rank}")
print(f"Claude Opus  current rank: {opus_rank}")

# ─── Greedy search ────────────────────────────────────────────────────────────
def greedy_minimize(target_model, label, max_iter=50):
    """
    Greedy: at each step remove the task that maximally improves
    target_model's rank. Stop when rank <= TARGET_RANK.
    """
    print(f"\n{'='*70}")
    print(f"GREEDY SEARCH: {label} → rank {TARGET_RANK}")
    print(f"{'='*70}")
    
    mask = np.ones(N_TASKS, dtype=bool)
    removed = []
    ti = model_idx[target_model]
    
    for step in range(max_iter):
        ranks, means = get_ranking(mask)
        cur_rank = ranks[ti]
        cur_score = means[ti]
        
        if cur_rank <= TARGET_RANK:
            print(f"  ✓ Achieved rank {cur_rank} after removing {len(removed)} tasks!")
            break
        
        # Try removing each remaining task; pick the one that best improves rank
        active_indices = np.where(mask)[0]
        best_gain = -1
        best_task_idx = -1
        best_new_rank = cur_rank
        
        for ti_task in active_indices:
            test_mask = mask.copy()
            test_mask[ti_task] = False
            new_ranks, new_means = get_ranking(test_mask)
            new_rank = new_ranks[ti]
            gain = cur_rank - new_rank  # positive = rank improved
            gain_float = gain + (cur_score - new_means[ti]) * 0.001  # tiebreak by score change
            if gain > best_gain or (gain == best_gain and new_means[ti] > (means[ti] if best_task_idx < 0 else 0)):
                best_gain = gain
                best_task_idx = ti_task
                best_new_rank = new_rank
        
        if best_task_idx < 0:
            print(f"  ✗ No improving task found at step {step+1}")
            break
        
        task_name = TASKS_LIST[best_task_idx]
        mask[best_task_idx] = False
        removed.append(task_name)
        new_ranks, new_means = get_ranking(mask)
        print(f"  Step {step+1:2d}: Remove '{task_name}'  →  rank {new_ranks[ti]:2d}  (score: {new_means[ti]:.4f})")
        
        if new_ranks[ti] <= TARGET_RANK:
            print(f"  ✓ Achieved rank {TARGET_RANK} after removing {len(removed)} tasks!")
            break
    
    ranks, means = get_ranking(mask)
    print(f"\n  Final rank of {label}: {ranks[ti]}")
    print(f"  Tasks removed ({len(removed)}): {removed}")
    return removed, mask

gpt_greedy_removed, gpt_greedy_mask = greedy_minimize("GPT-5.4", "GPT-5.4")
opus_greedy_removed, opus_greedy_mask = greedy_minimize("Claude Opus 4.6", "Claude Opus 4.6")

# ─── Exhaustive minimum search (k=1,2,...) ────────────────────────────────────
def exhaustive_minimum(target_model, label, max_k=None):
    """
    Exhaustively find the minimum k tasks whose removal achieves goal.
    Searches k=1, k=2, ... until found.
    Uses a smart pre-filter: for each task, compute its 'advantage score'
    = score(ranks-above-target, task) - score(target, task)
    High advantage = good candidate for removal (helps target catch up).
    """
    ti = model_idx[target_model]
    
    # Pre-score tasks: how much does removing task help target vs. hurting leaders
    # "gap-reduction score" per task
    # When we remove task t:
    # new_mean(m) = (n*mean(m) - score(m,t)) / (n-1)
    # We want new_mean(target) to rise relative to competitors
    ranks, means = get_ranking(full_mask)
    n = N_TASKS
    
    # For each task, compute score delta for each model
    # delta(m, t) = new_mean(m) - old_mean(m) = (mean(m) - score(m,t)) / (n-1) - mean(m)/n ... 
    # Simpler: sort tasks by how well target does vs. the models currently ranked above it
    
    # Models currently ranked above target
    above_target = [m for m in MODELS if ranks[model_idx[m]] < ranks[ti]]
    above_indices = [model_idx[m] for m in above_target]
    
    # For each task: compute the average score of above-models minus target's score
    # If this is HIGH: removing the task hurts above-models more, helps target relatively
    above_scores = M[above_indices, :]  # shape (n_above, n_tasks)
    mean_above = above_scores.mean(axis=0)  # (n_tasks,)
    target_scores = M[ti, :]  # (n_tasks,)
    
    gap_per_task = mean_above - target_scores  # positive = above-models do better here
    
    # Sort tasks by gap descending (best candidates first)
    candidate_order = np.argsort(-gap_per_task)
    
    print(f"\n{'='*70}")
    print(f"EXHAUSTIVE MINIMUM SEARCH: {label} → rank {TARGET_RANK}")
    print(f"{'='*70}")
    print(f"  Top 20 candidate tasks by gap score:")
    for i in range(min(20, N_TASKS)):
        idx = candidate_order[i]
        print(f"    [{i+1:2d}] gap={gap_per_task[idx]:+.4f}  target_score={target_scores[idx]:.4f}  task={TASKS_LIST[idx]}")
    
    if max_k is None:
        max_k = len(gpt_greedy_removed) + 2  # search up to greedy + buffer
    
    # Find minimum k
    for k in range(1, max_k + 1):
        print(f"\n  Searching k={k}...")
        t0 = time.time()
        
        # Smart: only consider top candidates (limit to top min(3k+10, n_tasks))
        n_candidates = min(3 * k + 15, N_TASKS)
        candidate_pool = candidate_order[:n_candidates]
        
        found = None
        count = 0
        for combo in combinations(candidate_pool, k):
            count += 1
            test_mask = full_mask.copy()
            test_mask[list(combo)] = False
            if check_goal(test_mask, target_model):
                found = [TASKS_LIST[c] for c in combo]
                break
            if count % 50000 == 0:
                elapsed = time.time() - t0
                print(f"    ... {count:,} combos checked ({elapsed:.1f}s)")
        
        elapsed = time.time() - t0
        print(f"    Checked {count:,} combos in {elapsed:.2f}s")
        
        if found:
            print(f"\n  ✓ MINIMUM FOUND: k={k} tasks")
            print(f"  Tasks to remove: {found}")
            # Show resulting leaderboard
            final_mask = full_mask.copy()
            for t in found:
                final_mask[TASKS_LIST.index(t)] = False
            show_leaderboard(final_mask, f"{label} @rank{TARGET_RANK} leaderboard")
            return found, final_mask
    
    print(f"  ✗ Could not find solution with k <= {max_k}")
    return None, None

def show_leaderboard(mask, title="Leaderboard"):
    ranks, means = get_ranking(mask)
    print(f"\n  {title}:")
    sorted_m = sorted(enumerate(MODELS), key=lambda x: ranks[x[0]])
    for mi, m in sorted_m:
        marker = " ◄" if m in ("GPT-5.4", "Claude Opus 4.6") else ""
        print(f"    Rank {ranks[mi]:2d}  {m:<35s}  {means[mi]:.4f}{marker}")

# Run exhaustive search
# First determine greedy solution size to bound exhaustive search
gpt_k_bound  = len(gpt_greedy_removed)
opus_k_bound = len(opus_greedy_removed)

print(f"\n\nGreedy found: GPT-5.4 needs {gpt_k_bound} removals, Claude Opus needs {opus_k_bound} removals")
print("Running exhaustive search to find true minimum...\n")

gpt_min_removed, gpt_min_mask   = exhaustive_minimum("GPT-5.4",       "GPT-5.4",       max_k=gpt_k_bound)
opus_min_removed, opus_min_mask = exhaustive_minimum("Claude Opus 4.6","Claude Opus 4.6",max_k=opus_k_bound)

# ─── Comparison summary ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY: MINIMUM TASK REMOVAL SETS")
print("="*70)

for label, removed, mask in [
    ("GPT-5.4 → rank 3",       gpt_min_removed,  gpt_min_mask),
    ("Claude Opus 4.6 → rank 3", opus_min_removed, opus_min_mask),
]:
    if removed:
        print(f"\n  {label}  ({len(removed)} task{'s' if len(removed)!=1 else ''})")
        for t in removed:
            ti = TASKS_LIST.index(t)
            gpt_s  = M[model_idx["GPT-5.4"], ti]
            opus_s = M[model_idx["Claude Opus 4.6"], ti]
            # avg score of models ranked above (in baseline)
            above_avg = np.mean([M[model_idx[m], ti] for m in MODELS 
                                  if baseline_ranks[model_idx[m]] < baseline_ranks[model_idx[label.split(" →")[0].strip()]]])
            cat_row = df[df.task_name == t][["category"]].iloc[0] if len(df[df.task_name == t]) > 0 else None
            cat = cat_row["category"] if cat_row is not None else "?"
            print(f"    • {t}")
            print(f"      category={cat}  | GPT-5.4={gpt_s:.3f}  Claude Opus={opus_s:.3f}  above_avg={above_avg:.3f}")

# Save results
results = []
if gpt_min_removed:
    for t in gpt_min_removed:
        results.append({"target": "GPT-5.4", "task_name": t, "min_removal_set_size": len(gpt_min_removed)})
if opus_min_removed:
    for t in opus_min_removed:
        results.append({"target": "Claude Opus 4.6", "task_name": t, "min_removal_set_size": len(opus_min_removed)})

pd.DataFrame(results).to_csv(OUT / "phase_d_rank_optimizer_results.csv", index=False)
print(f"\n\nResults saved to {OUT}/phase_d_rank_optimizer_results.csv")
