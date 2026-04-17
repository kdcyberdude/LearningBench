#!/usr/bin/env python3
"""
Phase D: Final Benchmark Curation Analysis
Performs per-task inspection and generates verdict table with score impact simulation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/Users/kd/Desktop/proj/learning_eval")
FLAGGED_CSV  = BASE / "analysis/outputs/final_flagged_tasks.csv"
SCORE_CSV    = BASE / "analysis/outputs/score_matrix.csv"
OUT_DIR      = BASE / "analysis/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
flagged = pd.read_csv(FLAGGED_CSV)
scores  = pd.read_csv(SCORE_CSV)

# ─── Per-Task Verdicts (Based on Code Inspection) ─────────────────────────────
# verdict: REMOVE | KEEP | FIX (minor update possible)
# reason : concise human reasoning
# phase_c_flag: what the automated analysis flagged
# code_inspection: what manual code review found

VERDICTS = [
    # ══════════════════════ HIGH PRIORITY (4 tasks) ══════════════════════════
    {
        "task_name": "euler_totient_rf_learning",
        "category": "rl",
        "removal_priority": "high",
        "verdict": "REMOVE",
        "removal_reason": "All-zero scores across all models. Task requires discovering hidden integer n in [12,200] via MOD k probes with only 40 turns. Feedback uses ambiguous WITHIN/OUTSIDE/EXACT notation with no clear instruction. Even discovering the query interface takes most of the budget. Zero discriminatory value.",
        "code_issue": "No implementation bug. Task is simply too hard for the current RL budget (40 turns) and provides no gradient signal to models that don't already know the exact query format.",
        "verdict_short": "REMOVE – all-zero, zero discrimination, opaque feedback interface",
    },
    {
        "task_name": "hangman_lite_rf_learning",
        "category": "rl",
        "removal_priority": "high",
        "verdict": "REMOVE",
        "removal_reason": "Near-zero scores for all models except Gemini 2.5 Flash. Task uses 5-symbol string over reduced alphabet with custom feedback bands (BULL/COW/MISS) and hidden wrong-guess penalty. Gemini's success appears to be due to format familiarity, not genuine inference-time learning. The hidden penalty creates unexplainable loss events that prevent systematic learning.",
        "code_issue": "Hidden wrong-guess penalty (-0.5 per duplicate guess) is undisclosed to the model, creating a hostile reward signal that undermines learning.",
        "verdict_short": "REMOVE – hidden undisclosed penalty sabotages learning; near-zero except one model",
    },
    {
        "task_name": "levenshtein_words_rf_learning",
        "category": "rl",
        "removal_priority": "high",
        "verdict": "REMOVE",
        "removal_reason": "Near-zero scores. Task reveals secret word via non-standard Levenshtein costs (insert=1, delete=2, substitute=3). Models must infer the cost table from feedback alone. With 30 turns, convergence is extremely unlikely. Only Gemini 2.5 Flash passes, suggesting rote pattern matching rather than distance-cost inference.",
        "code_issue": "No bug. Task is legitimately hard but fails to discriminate between models systematically – it's a random walk with one outlier.",
        "verdict_short": "REMOVE – non-standard costs not learnable in 30 turns; near-zero across all models",
    },
    {
        "task_name": "lights_out_2x2_rf_learning",
        "category": "rl",
        "removal_priority": "high",
        "verdict": "REMOVE",
        "removal_reason": "Near-zero scores. Task is actually a 4×4 binary lattice (mislabeled 2x2) with hidden XOR chord patterns and graduated telemetry. Finding the chord requires brute-force XOR exploration which is infeasible in 40 turns. Gemini 2.5 Flash is the only model to score, and its advantage appears to be grid-state parsing, not chord inference.",
        "code_issue": "Task name '2x2' is misleading – it's a 4×4 grid. This may confuse models reading the task header. Consider renaming if kept, but recommend removal due to zero discrimination.",
        "verdict_short": "REMOVE – mislabeled task name, all-zero, no discrimination across tiers",
    },

    # ══════════════════════ MEDIUM PRIORITY – CONCEPT ════════════════════════
    {
        "task_name": "hapax_prime_concept_learning",
        "category": "concept",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Negative discrimination (stronger models score lower). The task requires knowing the term 'hapax legomenon' (letters appearing exactly once) and checking if that count is prime. Larger models likely have prior knowledge that creates memorization shortcuts on easy cases but fails on edge cases, inverting the expected tier ordering.",
        "code_issue": "No implementation bug. But the task measures prior knowledge rather than inference-time learning. Prime-of-hapax-count is a narrow factual lookup, not a structural rule inference.",
        "verdict_short": "REMOVE – negative discrimination; tests prior knowledge not inference-time learning",
    },
    {
        "task_name": "semantic_override_concept_learning",
        "category": "concept",
        "removal_priority": "medium",
        "verdict": "KEEP",
        "removal_reason": "N/A – kept intentionally.",
        "code_issue": "No issue. Task reveals 'semantic rigidity': GPT-5.4 and Gemini fail while smaller models succeed because they override structural pattern (double letters) with semantic meaning. This is a novel capability insight of significant research value.",
        "verdict_short": "KEEP – reveals semantic rigidity in frontier models; high research value",
    },

    # ══════════════════════ MEDIUM PRIORITY – OBSERVATIONAL ══════════════════
    {
        "task_name": "custom_gravity_simulation_obs_learning",
        "category": "observational",
        "removal_priority": "medium",
        "verdict": "KEEP",
        "removal_reason": "N/A – borderline but kept.",
        "code_issue": "No bug. Near-ceiling scores for most models but GLM-5 fails completely, revealing a specific capability gap in physics simulation reasoning. Retained per Phase C revised philosophy.",
        "verdict_short": "KEEP – GLM-5 failure reveals physics sim gap; near-ceiling but informative",
    },
    {
        "task_name": "vigenere_variant_cipher_obs_learning",
        "category": "observational",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Low entropy (scores cluster tightly). Task requires inferring hidden key length and positional modulus from ciphertext samples. Models either fully solve it or give up – no partial learning gradient. Low variance means the task cannot rank models within tiers.",
        "code_issue": "No implementation bug. The task is well-designed but the cipher's structure allows brute-force key enumeration in a few probes, making it binary (solve/fail).",
        "verdict_short": "REMOVE – low entropy; bimodal solve/fail with no within-tier discrimination",
    },
    {
        "task_name": "voronoi_custom_metric_obs_learning",
        "category": "observational",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Negative discrimination. Task requires inferring directionally-asymmetric hub routing rule from probe demonstrations. Frontier models over-engineer the solution, hypothesizing complex rules when a simpler one suffices. Mid-tier models apply Occam's razor and score better.",
        "code_issue": "No implementation bug, but task description may inadvertently prime models toward geometric reasoning instead of graph-routing reasoning.",
        "verdict_short": "REMOVE – negative discrimination; frontier models over-engineer the simple rule",
    },

    # ══════════════════════ MEDIUM PRIORITY – RL ══════════════════════════════
    {
        "task_name": "cyclic_distance_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Low entropy. Task infers hidden residue on cyclic ring Z_M with noisy RING_GAP feedback. Models converge to similar mid-range scores – the task doesn't separate tiers. The noise in feedback makes systematic narrowing unreliable, collapsing score variance.",
        "code_issue": "No bug, but the noise level is too high relative to the search space, making all models equivalently lost.",
        "verdict_short": "REMOVE – low entropy; noise level collapses score variance across tiers",
    },
    {
        "task_name": "digit_square_error_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Low entropy. Task requires finding a 3-digit secret with ENERGY feedback (per-position errors raised to hidden exponents 2 or 3). Models cannot reliably infer the exponents from energy readings alone without algebraic reconstruction. Scores cluster with no tier separation.",
        "code_issue": "No bug, but the exponent ambiguity combined with multi-digit coupling creates a confounded search space that is not learnable in 40 turns.",
        "verdict_short": "REMOVE – low entropy; exponent ambiguity makes learning infeasible in budget",
    },
    {
        "task_name": "parity_groups_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Low entropy. Task infers a 12-bit secret using noisy block-parity XOR feedback. XOR parity requires structured query design which language models struggle with systematically. Noisy feedback further reduces learning gradient. No tier separation.",
        "code_issue": "No bug. The XOR parity feedback is theoretically recoverable but requires information-theoretically optimal queries – beyond current LLM sequential reasoning.",
        "verdict_short": "REMOVE – low entropy; XOR parity + noise beyond sequential LLM reasoning",
    },
    {
        "task_name": "grid_seven_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Low entropy. Task navigates a 7×7 fog-of-war grid with walls and hazard tiles. The 7×7 size combined with fog makes systematic exploration infeasible within budget. All models end up at similar low scores, providing no discrimination.",
        "code_issue": "Grid size (7×7 = 49 cells) with fog and hazards exceeds what can be mapped in 30-40 turns. Compare to grid_octile (6×6) which performs better.",
        "verdict_short": "REMOVE – low entropy; 7×7 fog grid too large for budget, no tier separation",
    },
    {
        "task_name": "linear_polynomial_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Low entropy. Task requires inferring coefficients of a quadratic f(x)=Ax²+Bx+C via black-box queries. Models can recover linear functions well but the quadratic term creates a 3-parameter search space. With budget constraints, models converge to same intermediate accuracy, eliminating tier separation.",
        "code_issue": "No bug. A cubic or higher polynomial would worsen this further. Consider simplifying to linear-only or increasing the probe budget if keeping.",
        "verdict_short": "REMOVE – low entropy; 3-param quadratic collapses within-tier variance",
    },
    {
        "task_name": "linear_equation_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Low entropy. Task infers affine map f(x)=(Ax+B) mod 1009 which may shift mid-episode. Modular arithmetic with mid-episode drift is extremely hard to recover from. Scores cluster at low values with no differentiation.",
        "code_issue": "The mid-episode shift is a design flaw for inference-time learning – it invalidates previously gathered evidence, making systematic learning impossible.",
        "verdict_short": "REMOVE – mid-episode concept drift invalidates prior evidence; no discrimination",
    },
    {
        "task_name": "hanoi_two_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "KEEP",
        "removal_reason": "N/A – kept intentionally.",
        "code_issue": "No bug. 3-disk Hanoi with hidden goal peg and hidden forbidden disk-peg rule. Inverted bimodal pattern (small models better) reveals RL-specific strengths in structured problem solving. Valuable capability insight per Phase C.",
        "verdict_short": "KEEP – inverted bimodal reveals small-model RL strengths; high research value",
    },
    {
        "task_name": "letter_overlap_word_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "KEEP",
        "removal_reason": "N/A – borderline keep.",
        "code_issue": "No bug. Task finds 5-symbol word over synthetic alphabet via multiset overlap score. Moderate entropy and positive discrimination. Score variance is acceptable for a well-functioning RL task.",
        "verdict_short": "KEEP – moderate entropy, positive discrimination, well-functioning task",
    },
    {
        "task_name": "minesweeper_1d_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Negative discrimination + inverted tier gap. Task infers 3 hidden hazards on a 1D field with noisy adjacent-hazard count telemetry. Smaller models accidentally stumble upon correct answers while larger models over-think the noisy feedback, creating a perverse tier ordering.",
        "code_issue": "The noise in adjacency counts combined with a sparse hazard density makes the problem under-constrained. Models cannot distinguish signal from noise reliably.",
        "verdict_short": "REMOVE – negative discrimination; noisy feedback creates perverse tier inversion",
    },
    {
        "task_name": "verbal_bandit_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "KEEP",
        "removal_reason": "N/A – kept despite medium flag.",
        "code_issue": "No bug. Multi-armed bandit with nonstationary verbal payoffs and commit phase tests genuine exploration-exploitation trade-off. This is a clean RL task with real-world relevance. Retain pending deeper tier analysis.",
        "verdict_short": "KEEP – clean exploration-exploitation task; real-world relevance",
    },
    {
        "task_name": "digitwise_l1_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "KEEP",
        "removal_reason": "N/A – kept.",
        "code_issue": "No bug. Find 4-digit code with weighted L1 distance feedback (hidden per-position weights). Good information-theoretic structure allows binary-search-like convergence. Positive discrimination expected.",
        "verdict_short": "KEEP – good information-theoretic structure; positive discrimination",
    },
    {
        "task_name": "grid_octile_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "KEEP",
        "removal_reason": "N/A – kept despite medium flag.",
        "code_issue": "No bug. 6×6 grid with 8-way movement and slip noise. Despite inverted tier gap within one model family, overall pattern is informative. Gemini-5-Flash outperforms Gemini-5 Pro – revealing a model-family-specific RL anomaly worth studying.",
        "verdict_short": "KEEP – Gemini family inversion reveals fine-tuning trade-offs; research value",
    },
    {
        "task_name": "grid_nav_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Inverted tier gap. Larger models navigate the 7×7 fog grid worse than smaller models. This appears to be a systematic issue where larger models generate more verbose reasoning that exceeds context for action parsing. Not a genuine capability gap – an artifact.",
        "code_issue": "Action parsing may fail on verbose model outputs. The run() loop should be more lenient in accepting action strings.",
        "verdict_short": "REMOVE – inverted tier gap is parsing artifact, not genuine capability signal",
    },
    {
        "task_name": "hanoi_three_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Extreme bimodal. Models either solve it completely or score near zero – no middle ground. Without a partial-credit gradient, the task cannot rank models within tiers. The 4-disk (or 3-disk more complex variant) Hanoi with hidden constraints collapses to a binary outcome.",
        "code_issue": "No bug, but the lack of intermediate reward shaping means models either discover the pattern or give up. Consider adding step-level partial credit.",
        "verdict_short": "REMOVE – extreme bimodal; no partial credit gradient, binary solve/fail",
    },
    {
        "task_name": "interval_contains_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Extreme bimodal. Task requires inferring interval [L, R] via INSIDE/OUTSIDE feedback. Models either quickly bracket the interval (solve) or chase random endpoints (fail). No intermediate performance level exists, collapsing tier discrimination.",
        "code_issue": "No bug. Interval search with binary feedback naturally produces bimodal outcomes. Adding a 'distance to nearest boundary' hint could smooth performance but changes task semantics.",
        "verdict_short": "REMOVE – binary feedback produces extreme bimodal; no tier discrimination",
    },
    {
        "task_name": "hot_cold_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Extreme bimodal. Classic hot-cold search with hidden target. Models either understand the gradient structure (warm/cold feedback) and converge quickly, or they don't and score near zero. Similar to interval_contains, feedback structure produces binary outcomes.",
        "code_issue": "No bug. The hot-cold metaphor is very well-known, potentially causing LLMs to rely on prior knowledge rather than learn from the specific feedback in context.",
        "verdict_short": "REMOVE – extreme bimodal; well-known metaphor may bypass inference-time learning",
    },
    {
        "task_name": "mastermind_aggregate_rf_learning",
        "category": "rl",
        "removal_priority": "medium",
        "verdict": "REMOVE",
        "removal_reason": "Extreme bimodal. Mastermind variant with aggregate (not per-position) feedback. Models familiar with Mastermind apply standard strategies but the aggregate feedback changes the information structure significantly. This causes unpredictable performance: experts over-apply prior strategy, novices under-apply. Bimodal outcome.",
        "code_issue": "No bug. The aggregate feedback is a deliberate twist, but it interacts poorly with LLM Mastermind priors.",
        "verdict_short": "REMOVE – extreme bimodal; aggregate feedback clashes with LLM Mastermind priors",
    },
]

verdict_df = pd.DataFrame(VERDICTS)

# ─── Score Impact Simulation ───────────────────────────────────────────────────
remove_tasks = set(v["task_name"] for v in VERDICTS if v["verdict"] == "REMOVE")
keep_tasks   = set(v["task_name"] for v in VERDICTS if v["verdict"] in ("KEEP", "FIX"))

print(f"Tasks to REMOVE: {len(remove_tasks)}")
print(f"Tasks to KEEP:   {len(keep_tasks)}")
print()

# Overall score impact per model
def compute_leaderboard(scores_df, exclude_tasks=None):
    df = scores_df.copy()
    if exclude_tasks:
        df = df[~df["task_name"].isin(exclude_tasks)]
    return df.groupby(["model", "tier"])["score"].mean().reset_index().rename(columns={"score": "mean_score"})

lb_before = compute_leaderboard(scores)
lb_after  = compute_leaderboard(scores, exclude_tasks=remove_tasks)

lb_merged = lb_before.merge(lb_after, on=["model", "tier"], suffixes=("_before", "_after"))
lb_merged["delta"] = lb_merged["mean_score_after"] - lb_merged["mean_score_before"]
lb_merged = lb_merged.sort_values(["tier", "mean_score_after"], ascending=[True, False])

print("=" * 80)
print("LEADERBOARD IMPACT (mean score per model)")
print("=" * 80)
print(lb_merged[["model", "tier", "mean_score_before", "mean_score_after", "delta"]].to_string(index=False))
print()

# Category-level impact
cat_before = scores.groupby("category")["score"].mean().rename("before")
cat_after  = scores[~scores["task_name"].isin(remove_tasks)].groupby("category")["score"].mean().rename("after")
cat_impact = pd.concat([cat_before, cat_after], axis=1)
cat_impact["delta"] = cat_impact["after"] - cat_impact["before"]
cat_impact["tasks_removed"] = scores[scores["task_name"].isin(remove_tasks)].groupby("category")["task_name"].nunique()
cat_impact["tasks_removed"] = cat_impact["tasks_removed"].fillna(0).astype(int)

print("=" * 80)
print("CATEGORY-LEVEL IMPACT")
print("=" * 80)
print(cat_impact.round(4).to_string())
print()

# Task count summary
total_before = scores["task_name"].nunique()
total_after  = scores[~scores["task_name"].isin(remove_tasks)]["task_name"].nunique()
print(f"Total tasks before: {total_before}")
print(f"Total tasks after:  {total_after}")
print(f"Tasks removed:      {total_before - total_after}")
print()

# ─── Final Decision Table ──────────────────────────────────────────────────────
final_table = verdict_df[["task_name", "category", "removal_priority", "verdict", "verdict_short"]].copy()
final_table.columns = ["Task", "Category", "Phase C Priority", "Phase D Verdict", "Reasoning"]

print("=" * 80)
print("PHASE D FINAL DECISION TABLE")
print("=" * 80)
removes = final_table[final_table["Phase D Verdict"] == "REMOVE"]
keeps   = final_table[final_table["Phase D Verdict"] == "KEEP"]

print(f"\n--- REMOVE ({len(removes)} tasks) ---")
for _, row in removes.iterrows():
    print(f"  [{row['Phase C Priority'].upper():6}] {row['Task']}")
    print(f"         {row['Reasoning']}")
    print()

print(f"\n--- KEEP ({len(keeps)} tasks) ---")
for _, row in keeps.iterrows():
    print(f"  [{row['Phase C Priority'].upper():6}] {row['Task']}")
    print(f"         {row['Reasoning']}")
    print()

# ─── Save Outputs ─────────────────────────────────────────────────────────────
verdict_df.to_csv(OUT_DIR / "phase_d_verdicts.csv", index=False)
lb_merged.to_csv(OUT_DIR / "phase_d_leaderboard_impact.csv", index=False)
cat_impact.to_csv(OUT_DIR / "phase_d_category_impact.csv")

# Save the final clean task list (removes removed tasks from flagged)
all_tasks = scores["task_name"].unique()
final_task_list = [t for t in all_tasks if t not in remove_tasks]
pd.Series(sorted(final_task_list), name="task_name").to_csv(OUT_DIR / "phase_d_final_task_list.csv", index=False)

print(f"\nOutputs saved to {OUT_DIR}")
print("  - phase_d_verdicts.csv")
print("  - phase_d_leaderboard_impact.csv")
print("  - phase_d_category_impact.csv")
print("  - phase_d_final_task_list.csv")
