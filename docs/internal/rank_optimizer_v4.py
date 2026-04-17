#!/usr/bin/env python3
"""
Rank Optimizer v4 – proper multi-step greedy + exhaustive for k up to ~15
"""
import sys
sys.stdout = open('/tmp/rank_opt_v4.txt', 'w', buffering=1)
sys.stderr = sys.stdout

import pandas as pd
import numpy as np
from itertools import combinations
import time
from pathlib import Path

BASE  = Path("/Users/kd/Desktop/proj/learning_eval")
OUT   = BASE / "analysis/outputs"

scores_raw  = pd.read_csv(OUT / "score_matrix.csv")
final_tasks = pd.read_csv(OUT / "phase_d_final_task_list.csv")["task_name"].tolist()
df          = scores_raw[scores_raw["task_name"].isin(final_tasks)].copy()
cat_map     = df.drop_duplicates("task_name").set_index("task_name")["category"].to_dict()

pivot  = df.pivot_table(index="model", columns="task_name", values="score").fillna(0.0)
M      = pivot.values
MODELS = list(pivot.index)
TASKS  = list(pivot.columns)
N      = len(TASKS)
mi     = {m: i for i, m in enumerate(MODELS)}

TARGETS   = ["GPT-5.4", "Claude Opus 4.6"]
GOAL_RANK = 3

print(f"Loaded: {len(MODELS)} models × {N} tasks")

# ─── Core ─────────────────────────────────────────────────────────────────────

def leaderboard(mask):
    means = M[:, mask].mean(axis=1)
    order = np.argsort(-means)
    ranks = np.empty(len(MODELS), dtype=int)
    ranks[order] = np.arange(1, len(MODELS)+1)
    return ranks, means

def show_lb(mask, title=""):
    if title: print(f"\n  {title}")
    ranks, means = leaderboard(mask)
    for i in np.argsort(ranks):
        m = MODELS[i]
        marker = " ◄" if m in TARGETS else ""
        print(f"    Rank {ranks[i]:2d}  {m:<35s}  {means[i]:.4f}{marker}")

full = np.ones(N, dtype=bool)
bl_ranks, bl_means = leaderboard(full)

print("\n" + "="*70)
print("BASELINE (138 tasks)")
print("="*70)
show_lb(full)
for t in TARGETS:
    print(f"  {t}: rank {bl_ranks[mi[t]]}")

# ─── Greedy (fully vectorised, tracks actual rank via leaderboard) ─────────────

def greedy(target, max_steps=40, verbose=True):
    """
    At each step: remove the task that lowers the rank of `target` the most.
    If multiple tasks give the same best rank, break tie by target's new mean score.
    """
    ti   = mi[target]
    mask = full.copy()
    removed = []
    if verbose:
        print(f"\n{'='*70}")
        print(f"GREEDY: {target} → rank {GOAL_RANK}")
        print(f"{'='*70}")

    for step in range(max_steps):
        n         = int(mask.sum())
        means_now = M[:, mask].mean(axis=1)
        ranks_now, _ = leaderboard(mask)
        cur_rank  = int(ranks_now[ti])

        if cur_rank <= GOAL_RANK:
            if verbose: print(f"  ✓ Achieved rank {cur_rank} after {len(removed)} removals")
            break

        active = np.where(mask)[0]
        n1     = n - 1

        # Vectorised: new mean of each model when active[k] is removed
        active_M = M[:, active]                             # (14, n_active)
        nm       = (n * means_now[:, None] - active_M) / n1  # (14, n_active)

        # New rank of ti for each removal candidate
        # rank(ti) = position when nm is sorted descending per column
        # Faster: count how many models have nm > nm[ti]
        nm_ti    = nm[ti, :]                    # (n_active,)
        beats_ti = (nm > nm_ti[None, :]).sum(axis=0)  # (n_active,) = #models scoring > ti
        new_rank = beats_ti + 1                 # 1-indexed rank

        best_rank = int(new_rank.min())
        if best_rank >= cur_rank:
            # Try random restarts / look-ahead?
            if verbose: print(f"  ✗ No single removal improves rank at step {step+1}")
            break

        # Among tasks giving best_rank, pick the one that maximises target's new score
        best_cands = np.where(new_rank == best_rank)[0]
        best_local = best_cands[int(np.argmax(nm_ti[best_cands]))]
        best_j     = active[best_local]

        mask[best_j] = False
        removed.append(TASKS[best_j])
        r2, m2 = leaderboard(mask)
        if verbose:
            print(f"  Step {step+1:2d}: remove '{TASKS[best_j]}'  → rank {r2[ti]}  "
                  f"score {m2[ti]:.4f}  [gap closed: rank {cur_rank}→{r2[ti]}]")

        if r2[ti] <= GOAL_RANK:
            if verbose: print(f"  ✓ Achieved rank {r2[ti]} after {len(removed)} removals!")
            break
    else:
        if verbose: print("  ✗ max_steps reached")

    final_r, _ = leaderboard(mask)
    if verbose: print(f"\n  Final rank of {target}: {final_r[ti]}  (removed {len(removed)} tasks)")
    return removed

# ─── Beam-search greedy (width=5) ─────────────────────────────────────────────

def beam_greedy(target, beam_width=5, max_steps=40, verbose=True):
    """
    Like greedy but maintains top-B candidate masks at each step.
    Returns the best (fewest removals) successful beam.
    """
    ti   = mi[target]
    if verbose:
        print(f"\n{'='*70}")
        print(f"BEAM SEARCH (width={beam_width}): {target} → rank {GOAL_RANK}")
        print(f"{'='*70}")

    beams = [(full.copy(), [])]   # (mask, removed_list)

    for step in range(max_steps):
        next_beams = []
        any_success = False

        for mask, removed in beams:
            ranks_now, _ = leaderboard(mask)
            if int(ranks_now[ti]) <= GOAL_RANK:
                if verbose:
                    print(f"  ✓ Beam succeeded with {len(removed)} tasks at step {step}")
                return removed

            n         = int(mask.sum())
            means_now = M[:, mask].mean(axis=1)
            active    = np.where(mask)[0]
            n1        = n - 1
            active_M  = M[:, active]
            nm        = (n * means_now[:, None] - active_M) / n1
            nm_ti     = nm[ti, :]
            beats_ti  = (nm > nm_ti[None, :]).sum(axis=0)
            new_rank  = beats_ti + 1

            # Take top-B candidates from this beam
            order = np.argsort(new_rank)   # best rank first
            for local_k in order[:beam_width]:
                j = active[local_k]
                new_mask    = mask.copy(); new_mask[j] = False
                new_removed = removed + [TASKS[j]]
                r2, m2 = leaderboard(new_mask)
                if int(r2[ti]) <= GOAL_RANK:
                    if verbose:
                        print(f"  ✓ Found solution: {len(new_removed)} tasks → {new_removed}")
                    return new_removed
                next_beams.append((new_mask, new_removed, int(r2[ti]), float(m2[ti])))

        if not next_beams:
            break

        # Prune: keep beam_width best beams by (rank, -score)
        next_beams.sort(key=lambda x: (x[2], -x[3]))
        beams = [(m, r) for m, r, _, _ in next_beams[:beam_width]]
        best_rank = next_beams[0][2]
        if verbose:
            print(f"  Step {step+1:2d}: best rank in beam = {best_rank}")

    if verbose: print("  ✗ Beam search exhausted")
    return None

# ─── Exhaustive for small k ────────────────────────────────────────────────────

def exhaustive(target, max_k, verbose=True):
    ti   = mi[target]
    # Rank candidates by gap score
    above    = [j for j in range(len(MODELS)) if bl_ranks[j] < bl_ranks[ti]]
    gaps     = M[above, :].mean(axis=0) - M[ti, :]
    cand_ord = np.argsort(-gaps)

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXHAUSTIVE MIN: {target} → rank {GOAL_RANK}")
        print(f"{'='*70}")
        print("  Top 20 candidates (by gap score):")
        for i in range(min(20, N)):
            j = cand_ord[i]
            print(f"    [{i+1:2d}] gap={gaps[j]:+.4f}  t={M[ti,j]:.3f}  "
                  f"{TASKS[j]}  [{cat_map.get(TASKS[j],'?')}]")

    for k in range(1, max_k + 1):
        # Use a growing pool: top min(5k+20, N) candidates
        pool_size = min(5 * k + 20, N)
        pool      = cand_ord[:pool_size].tolist()
        total     = 1
        for i in range(k):
            total = total * (pool_size - i) // (i + 1)

        if total > 5_000_000:
            # Too many combinations; restrict pool
            pool_size = min(4 * k + 15, N)
            pool = cand_ord[:pool_size].tolist()
            total = 1
            for i in range(k):
                total = total * (pool_size - i) // (i + 1)

        if verbose:
            print(f"\n  k={k}: C({pool_size},{k}) = {total:,}")

        t0    = time.time()
        found = None
        for count, combo in enumerate(combinations(pool, k), 1):
            test = full.copy()
            for j in combo: test[j] = False
            if leaderboard(test)[0][ti] <= GOAL_RANK:
                found = [TASKS[j] for j in combo]
                break
            if count % 500_000 == 0:
                print(f"    ... {count:,} ({time.time()-t0:.0f}s)")

        elapsed = time.time() - t0
        if verbose:
            print(f"    Checked {count:,} in {elapsed:.2f}s")

        if found:
            if verbose:
                print(f"  ✓ MINIMUM = {k}: {found}")
                rm = full.copy()
                for t in found: rm[TASKS.index(t)] = False
                show_lb(rm, f"Leaderboard after removing {k} task(s)")
            return found

    if verbose: print(f"  ✗ Not found k≤{max_k}")
    return None

# ─── Run ──────────────────────────────────────────────────────────────────────

results = {}
for target in TARGETS:
    print(f"\n\n{'#'*70}")
    print(f"# {target}")
    print(f"{'#'*70}")

    # Greedy upper bound
    gr = greedy(target, verbose=True)
    ub = len(gr)
    print(f"\n  Greedy upper bound: {ub}")

    # Beam greedy (wider search)
    bg = beam_greedy(target, beam_width=10, verbose=True)
    if bg and len(bg) < ub:
        ub = len(bg)
        print(f"  Beam improved upper bound to: {ub}")

    print(f"\n  Running exhaustive search up to k={ub}...")
    mn = exhaustive(target, max_k=ub, verbose=True)
    results[target] = mn if mn else gr

# ─── Analysis ─────────────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("FULL ANALYSIS OF REMOVAL SETS")
print("="*70)

all_t = sorted(set().union(*[set(v) for v in results.values() if v]))
for t in all_t:
    idx = TASKS.index(t)
    cat = cat_map.get(t, "?")
    row = M[:, idx]
    for_tgts = [tgt for tgt, lst in results.items() if lst and t in lst]
    print(f"\n  TASK: {t}  [{cat}]")
    print(f"  In removal set for: {for_tgts}")
    print(f"  Mean={row.mean():.4f}  Std={row.std():.4f}")
    for j in np.argsort(-row):
        mk = " ◄" if MODELS[j] in TARGETS else ("  [target]" if MODELS[j] in TARGETS else "")
        print(f"    {MODELS[j]:<35s}  {row[j]:.4f}{mk}")

# GPT-5.4 and Opus comparison
print(f"\n\n{'='*70}")
print("SUMMARY")
print("="*70)
for target, removed in results.items():
    if removed:
        print(f"\n  {target} → rank {GOAL_RANK}: remove {len(removed)} task(s)")
        for t in removed:
            print(f"    • {t}  [{cat_map.get(t,'?')}]")
        rm = full.copy()
        for t in removed: rm[TASKS.index(t)] = False
        show_lb(rm, f"Final leaderboard ({target} scenario)")
    else:
        print(f"\n  {target}: no solution found")

# Save CSV
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
            "gemini_pro_score": round(float(M[mi["Gemini 3.1 Pro Preview"], idx]), 4),
        })

pd.DataFrame(rows).to_csv(OUT / "rank_optimizer_results.csv", index=False)
print(f"\nSaved → {OUT}/rank_optimizer_results.csv")
print("DONE")
sys.stdout.flush()
