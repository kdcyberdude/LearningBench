#!/usr/bin/env python3
"""
Rank Optimizer – vectorised, output to file
"""
import sys, os
# Force all output to be immediately flushed
sys.stdout = open('/tmp/rank_opt_v3.txt', 'w', buffering=1)
sys.stderr = sys.stdout

import pandas as pd
import numpy as np
from itertools import combinations
import time
from pathlib import Path

BASE  = Path("/Users/kd/Desktop/proj/learning_eval")
OUT   = BASE / "analysis/outputs"

# ─── Load ─────────────────────────────────────────────────────────────────────
scores_raw  = pd.read_csv(OUT / "score_matrix.csv")
final_tasks = pd.read_csv(OUT / "phase_d_final_task_list.csv")["task_name"].tolist()
df          = scores_raw[scores_raw["task_name"].isin(final_tasks)].copy()
cat_map     = df.drop_duplicates("task_name").set_index("task_name")["category"].to_dict()

pivot  = df.pivot_table(index="model", columns="task_name", values="score").fillna(0.0)
M      = pivot.values            # (14, 138)
MODELS = list(pivot.index)
TASKS  = list(pivot.columns)
N      = len(TASKS)
mi     = {m: i for i, m in enumerate(MODELS)}

TARGETS   = ["GPT-5.4", "Claude Opus 4.6"]
GOAL_RANK = 3

print("Data loaded OK")
print(f"Models: {len(MODELS)}, Tasks: {N}")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def leaderboard(mask):
    sub   = M[:, mask]
    means = sub.mean(axis=1)
    order = np.argsort(-means)
    ranks = np.empty(len(MODELS), dtype=int)
    ranks[order] = np.arange(1, len(MODELS)+1)
    return ranks, means

def show_lb(mask, title=""):
    if title: print(f"\n  --- {title} ---")
    ranks, means = leaderboard(mask)
    for i in np.argsort(ranks):
        m = MODELS[i]
        marker = " ◄" if m in TARGETS else ""
        print(f"    Rank {ranks[i]:2d}  {m:<35s}  {means[i]:.4f}{marker}")

# ─── Baseline ─────────────────────────────────────────────────────────────────
full = np.ones(N, dtype=bool)
bl_ranks, bl_means = leaderboard(full)

print("\n" + "="*70)
print("BASELINE LEADERBOARD (138 tasks)")
print("="*70)
show_lb(full)
for t in TARGETS:
    print(f"  {t} current rank: {bl_ranks[mi[t]]}")

# ─── Vectorised greedy ────────────────────────────────────────────────────────

def greedy(target):
    ti   = mi[target]
    mask = full.copy()
    removed = []
    print(f"\n{'='*70}")
    print(f"GREEDY SEARCH: {target} → rank {GOAL_RANK}")
    print(f"{'='*70}")

    for step in range(50):
        n    = int(mask.sum())
        sub  = M[:, mask]
        means_now = sub.mean(axis=1)
        cur_rank  = int(bl_ranks[ti])
        if step > 0:
            r2, _ = leaderboard(mask)
            cur_rank = int(r2[ti])

        if cur_rank <= GOAL_RANK:
            print(f"  ✓ Rank {cur_rank} achieved after {len(removed)} removals")
            break

        # Vectorised: for each active task j, compute new mean of every model
        active_idx = np.where(mask)[0]
        n1 = n - 1

        # new_means_all[:, k] = new means when active_idx[k] is removed
        active_M    = M[:, active_idx]          # (n_models, n_active)
        new_means_T = (n * means_now[:, None] - active_M) / n1  # (n_models, n_active)

        # Rank of ti under each scenario
        rank_matrix = np.argsort(-new_means_T, axis=0)   # (n_models, n_active) sorted indices
        ti_new_rank = (rank_matrix == ti).argmax(axis=0) + 1   # (n_active,)

        # Best: lowest new rank (primary), highest new score (secondary)
        best_local = int(np.argmin(ti_new_rank))
        best_j     = active_idx[best_local]
        best_rank  = ti_new_rank[best_local]

        if best_rank >= cur_rank:
            print(f"  ✗ No improvement possible at step {step+1}")
            break

        mask[best_j] = False
        removed.append(TASKS[best_j])
        r2, m2 = leaderboard(mask)
        print(f"  Step {step+1:2d}: remove '{TASKS[best_j]}'  → rank {r2[ti]}  score {m2[ti]:.4f}")

        if r2[ti] <= GOAL_RANK:
            print(f"  ✓ Rank {r2[ti]} achieved after {len(removed)} removals")
            break

    return removed

# ─── Exhaustive search ────────────────────────────────────────────────────────

def gap_scores(target):
    ti     = mi[target]
    above  = [j for j in range(len(MODELS)) if bl_ranks[j] < bl_ranks[ti]]
    above_mean = M[above, :].mean(axis=0)
    return above_mean - M[ti, :]

def exhaustive(target, max_k):
    ti   = mi[target]
    gaps = gap_scores(target)
    cand = np.argsort(-gaps)   # best candidates first

    print(f"\n{'='*70}")
    print(f"EXHAUSTIVE MINIMUM: {target} → rank {GOAL_RANK}")
    print(f"{'='*70}")
    print(f"  Top 15 candidates:")
    for i in range(min(15, N)):
        j = cand[i]
        print(f"    [{i+1:2d}] gap={gaps[j]:+.4f}  target={M[ti,j]:.3f}  "
              f"task={TASKS[j]}  [{cat_map.get(TASKS[j],'?')}]")

    for k in range(1, max_k + 1):
        pool_size = min(5 * k + 20, N)
        pool      = cand[:pool_size].tolist()
        total     = 1
        for i in range(k):
            total = total * (pool_size - i) // (i + 1)
        print(f"\n  k={k}: C({pool_size},{k})={total:,}")
        t0    = time.time()
        found = None
        for count, combo in enumerate(combinations(pool, k), 1):
            test = full.copy()
            for j in combo: test[j] = False
            if leaderboard(test)[0][ti] <= GOAL_RANK:
                found = [TASKS[j] for j in combo]
                break
            if count % 200000 == 0:
                print(f"    ... {count:,} checked ({time.time()-t0:.0f}s)")

        elapsed = time.time() - t0
        print(f"    Checked {count:,} in {elapsed:.2f}s")
        if found:
            print(f"  ✓ MINIMUM = {k}  →  {found}")
            result = full.copy()
            for t in found: result[TASKS.index(t)] = False
            show_lb(result, f"Leaderboard after removing {k} task(s) for {target}")
            return found

    print(f"  ✗ Not found in k≤{max_k}")
    return None

# ─── Run ──────────────────────────────────────────────────────────────────────

results = {}
for target in TARGETS:
    print(f"\n\n{'#'*70}")
    print(f"# {target}")
    print(f"{'#'*70}")
    gr = greedy(target)
    print(f"\nGreedy bound: {len(gr)} tasks → {gr}")
    mn = exhaustive(target, max_k=len(gr))
    results[target] = mn

# ─── Deep analysis ────────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("DEEP ANALYSIS OF REMOVAL SETS")
print("="*70)

all_tasks = sorted(set().union(*[set(v) for v in results.values() if v]))
for t in all_tasks:
    idx  = TASKS.index(t)
    cat  = cat_map.get(t, "?")
    row  = M[:, idx]
    for_targets = [tgt for tgt, lst in results.items() if lst and t in lst]
    print(f"\n  {t}  [{cat}]")
    print(f"  Used for: {for_targets}")
    for j in np.argsort(-row):
        marker = " ◄" if MODELS[j] in TARGETS else ""
        print(f"    {MODELS[j]:<35s}  {row[j]:.4f}{marker}")
    print(f"  mean={row.mean():.4f}  std={row.std():.4f}")

# ─── Save CSV ─────────────────────────────────────────────────────────────────
rows = []
for target, removed in results.items():
    if not removed: continue
    for t in removed:
        idx = TASKS.index(t)
        rows.append({
            "target_model": target,
            "task_name": t,
            "category": cat_map.get(t, "?"),
            "min_set_size": len(removed),
            "gpt54_score": round(float(M[mi["GPT-5.4"], idx]), 4),
            "opus_score":  round(float(M[mi["Claude Opus 4.6"], idx]), 4),
            "qwen_score":  round(float(M[mi["Qwen 3 Next 80B Thinking"], idx]), 4),
            "gemini25f_score": round(float(M[mi["Gemini 2.5 Flash"], idx]), 4),
        })

pd.DataFrame(rows).to_csv(OUT / "rank_optimizer_results.csv", index=False)
print(f"\nSaved → {OUT}/rank_optimizer_results.csv")
print("DONE")
sys.stdout.flush()
