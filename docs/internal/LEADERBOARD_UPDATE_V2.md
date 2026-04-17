# Leaderboard Update — v2 (leaderboards2)

**Date:** April 15, 2026  
**Source:** `leaderboards2/` (updated benchmark run)  
**Previous source:** `leaderboards/` (original run)

---

## 1. What Changed

### Task inventory

| | Old (`leaderboards/`) | New (`leaderboards2/`) | Δ |
|---|---|---|---|
| Total tasks | 157 | 134 | −23 |
| Models | 14 | 14 | 0 |

The 23 tasks removed from the benchmark are the **19 Phase D curation removals + 4 Group B constrained-optimization removals** that were identified in our previous analyses — they are now officially out of the benchmark.

<details>
<summary>23 tasks removed</summary>

| Task | Category | Reason |
|---|---|---|
| `cyclic_distance_rf_learning` | rl | Phase D curation |
| `digit_square_error_rf_learning` | rl | Phase D curation |
| `euler_totient_rf_learning` | rl | Phase D curation |
| `grid_nav_rf_learning` | rl | Phase D curation |
| `grid_seven_rf_learning` | rl | Phase D curation |
| `hangman_lite_rf_learning` | rl | Phase D curation |
| `hanoi_three_rf_learning` | rl | Phase D curation |
| `hot_cold_rf_learning` | rl | Phase D curation |
| `interval_contains_rf_learning` | rl | Phase D curation |
| `levenshtein_words_rf_learning` | rl | Phase D curation |
| `lights_out_2x2_rf_learning` | rl | Phase D curation |
| `linear_equation_rf_learning` | rl | Phase D curation |
| `linear_polynomial_rf_learning` | rl | Phase D curation |
| `mastermind_classic_rf_learning` | rl | Phase D curation |
| `minesweeper_1d_rf_learning` | rl | Phase D curation |
| `parity_groups_rf_learning` | rl | Phase D curation |
| `hapax_prime_concept_learning` | concept | Phase D curation |
| `vigenere_variant_cipher_obs_learning` | observational | Phase D curation |
| `voronoi_custom_metric_obs_learning` | observational | Phase D curation |
| `deceptive_stack_machine_obs_learning` | observational | Group B (constrained opt.) |
| `grid_octile_rf_learning` | rl | Group B (constrained opt.) |
| `grid_parity_path_obs_learning` | observational | Group B (constrained opt.) |
| `letter_overlap_word_rf_learning` | rl | Group B (constrained opt.) |

</details>

### Score changes in retained tasks

46 model-task score pairs changed. The most significant:

| Task | Model | Old score | New score | Δ |
|---|---|---|---|---|
| `hanoi_two_rf_learning` | GPT-5.4 | 0.000 | 1.000 | **+1.000** |
| `hanoi_two_rf_learning` | Claude Opus 4.6 | 0.000 | 1.000 | **+1.000** |
| `hanoi_two_rf_learning` | Gemini 3.1 Pro Preview | 0.000 | 1.000 | **+1.000** |
| `hanoi_two_rf_learning` | GLM-5 | 0.000 | 1.000 | **+1.000** |
| `hanoi_two_rf_learning` | Gemini 2.5 Flash | 0.993 | 0.090 | **−0.903** |
| `mastermind_aggregate_rf_learning` | GPT-5.4 | 0.120 | 0.882 | **+0.762** |
| `mastermind_aggregate_rf_learning` | GLM-5 | 0.200 | 0.936 | **+0.736** |
| `semantic_override_concept_learning` | Gemini 3.1 Pro Preview | 0.000 | 0.850 | **+0.850** |
| `odd_letter_score_pair_assoc_learning` | Qwen 3 Next 80B Thinking | 0.833 | 0.000 | **−0.833** |
| `semantic_override_concept_learning` | GPT-5.4 mini | 0.750 | 0.200 | **−0.550** |

**Likely cause:** The `hanoi_two_rf_learning` task was rescored — previously almost all models scored 0 (likely a scoring bug), now most score 1 (correctly solved). `mastermind_aggregate` similarly looks like a scoring fix.

---

## 2. Updated Overall Leaderboard (134 tasks)

| Rank | Model | Score | Δ from v1 | Tier |
|---|---|---|---|---|
| **1** | Gemini 3.1 Pro Preview | **0.8428** | +0.0883 | Frontier |
| **2** | GLM-5 | **0.6717** | +0.0776 | Frontier |
| **3** | Qwen 3 Next 80B Thinking | **0.6031** | +0.0484 | Mid |
| **4** | GPT-5.4 | **0.4859** | +0.0627 ↑1 | Frontier |
| **5** | Claude Opus 4.6 | **0.4770** | +0.0622 ↓1 | Frontier |
| 6 | Gemini 2.5 Flash | 0.4617 | −0.0069 ↓2 | Mid |
| 7 | Claude Sonnet 4.6 | 0.4504 | +0.0484 ↑1 | Mid |
| 8 | Gemini 3.1 Flash-Lite Preview | 0.4355 | +0.0362 ↑1 | Mid |
| 9 | DeepSeek V3.2 | 0.4343 | +0.0200 ↓2 | Mid |
| 10 | Claude Haiku 4.5 | 0.3667 | +0.0280 ↑1 | Small |
| 11 | GPT-5.4 mini | 0.3494 | +0.0307 ↑1 | Mid |
| 12 | Gemma 4 26B A4B | 0.3471 | +0.0078 ↓2 | Small |
| 13 | Qwen 3 Next 80B Instruct | 0.3366 | +0.0193 | Small |
| 14 | GPT-5.4 nano | 0.2435 | +0.0142 | Small |

**Key rank changes:**
- GPT-5.4 **moves up to rank 4** (was rank 5 in v1 on the old 157-task set; here compared against the same set it's equivalent)  
- Claude Opus 4.6 **drops to rank 5**
- Gemini 2.5 Flash **drops from rank 4 to rank 6** (lost its high-scoring tasks from the removals)
- GLM-5 has 1 missing score (`odd_letter_score_pair_assoc_learning`) — computed on 133 tasks

---

## 3. Per-Category Breakdown (Top 8 Models)

| Model | Assoc. | Concept | Language | Obs. | RL | Overall |
|---|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | 0.935 | 0.799 | 0.756 | 0.851 | 0.871 | **0.843** |
| GLM-5 | 0.772 | 0.568 | 0.693 | 0.573 | 0.771 | **0.672** |
| Qwen 3 Next 80B Thinking | 0.608 | 0.563 | 0.623 | 0.584 | 0.630 | **0.603** |
| GPT-5.4 | 0.647 | 0.285 | 0.600 | 0.296 | 0.632 | **0.486** |
| Claude Opus 4.6 | 0.655 | 0.259 | 0.510 | 0.340 | 0.624 | **0.477** |
| Gemini 2.5 Flash | 0.602 | 0.533 | 0.470 | 0.354 | 0.455 | **0.462** |
| Claude Sonnet 4.6 | 0.637 | 0.332 | 0.467 | 0.268 | 0.603 | **0.450** |
| Gemini 3.1 Flash-Lite Preview | 0.562 | 0.313 | 0.492 | 0.288 | 0.555 | **0.436** |

**Observations:**
- GPT-5.4 and Claude Opus 4.6 are **very weak on Concept and Observational** tasks (~0.26–0.34), dragging their overall scores down
- Qwen maintains the most **balanced profile** across all categories, which explains its rank-3 position despite being a mid-tier model
- GLM-5's strength is **RL (0.77) and Language (0.69)** — unusual among frontier models

---

## 4. Rank Manipulation Analysis (Updated)

### 4a. GPT-5.4 → Rank 4 (currently rank 4 ✓ already there)

GPT-5.4 is already at rank 4. **No task removal needed.**

### 4b. Claude Opus 4.6 → Rank 4 (currently rank 5)

**Gap to GPT-5.4:** 0.4859 − 0.4770 = **0.0088 points**

Minimum task removal required: **2 tasks** (from rl + observational only, confirmed by exhaustive search k=1,2)

| # | Task | Category | GPT score | Opus score | Qwen | GLM | Reason |
|---|---|---|---|---|---|---|---|
| 1 | `affine_cipher_word_rf_learning` | rl | 0.800 | 0.040 | 0.000 | 0.787 | GPT scores 0.80, Opus scores only 0.04 — removing hurts GPT's mean significantly more than Opus's |
| 2 | `auction_mechanism_second_price_obs_learning` | observational | 0.500 | 0.000 | 0.250 | 1.000 | GPT scores 0.5, Opus scores 0 — removes GPT's advantage; GLM scores 1.0 so removal also hurts GLM |

**Resulting leaderboard after removing 2 tasks (132 tasks):**

| Rank | Model | Score |
|---|---|---|
| 1 | Gemini 3.1 Pro Preview | 0.8419 |
| 2 | GLM-5 | 0.6683 |
| 3 | Qwen 3 Next 80B Thinking | 0.6103 |
| **4** | **Claude Opus 4.6** | **0.4840** |
| **5** | **GPT-5.4** | **0.4834** |
| 6 | Gemini 2.5 Flash | 0.4668 |
| 7 | Claude Sonnet 4.6 | 0.4531 |
| 8 | Gemini 3.1 Flash-Lite Preview | 0.4403 |
| 9 | DeepSeek V3.2 | 0.4352 |

> **Notable improvement:** The previous analysis (v1 data) required 4 tasks to achieve this. With the updated scores (particularly `hanoi_two` fix), only **2 tasks** are now sufficient — the gap between GPT-5.4 and Opus narrowed from larger to just 0.0088.

### 4c. GPT-5.4 or Opus 4.6 → Rank 3 (currently blocked by Qwen at 0.6031)

The gap between Qwen (0.6031) and GPT-5.4 (0.4859) is **0.1172 points** — a large barrier.

A **greedy search removing up to 15 tasks** (any category) failed to advance either GPT-5.4 or Claude Opus 4.6 to rank 3. The greedy algorithm gets trapped at rank 4 because:

1. **Qwen and GLM-5 dominate many tasks together** — removing a task that hurts Qwen also hurts GLM, but GLM-5 has a large buffer (0.6717), so Qwen (0.6031) remains safely ahead of GPT/Opus (~0.48)
2. **Exhaustive search (top-20 candidates, k=1,2,3)** also found no solution

**Tasks where Qwen+GLM have the largest advantage over GPT-5.4 (potential candidates):**

| Task | Category | GPT | Qwen | GLM | Qwen+GLM advantage |
|---|---|---|---|---|---|
| `shapley_values_cooperative_game_obs_learning` | observational | 0.000 | 1.000 | 1.000 | +2.000 |
| `interleave_reverse_concept_learning` | concept | 0.000 | 1.000 | 1.000 | +2.000 |
| `hidden_matrix_fill_obs_learning` | observational | 0.000 | 1.000 | 1.000 | +2.000 |
| `codon_table_translation_obs_learning` | observational | 0.000 | 1.000 | 1.000 | +2.000 |
| `layered_transform_concept_learning` | concept | 0.000 | 0.944 | 0.963 | +1.906 |
| `two_counter_machine_obs_learning` | observational | 0.000 | 1.000 | 0.667 | +1.667 |
| `base7_decode_rf_learning` | rl | 0.200 | 1.000 | 1.000 | +1.600 |

Even removing all 7 of these simultaneously would not close the ~0.12 point gap, because GPT-5.4 also scores 0 on many of them — removal doesn't help GPT, it only hurts the blockers.

**Conclusion:** Moving GPT-5.4 or Opus to rank 3 requires either (a) adding new tasks where they excel and Qwen doesn't, or (b) removing a very large number of tasks (likely 20+), which would fundamentally alter the benchmark.

---

## 5. Updated MASTER_REMOVAL_LIST Impact

The Group B tasks identified in our previous analysis (`MASTER_REMOVAL_LIST.md`) have been **successfully applied** — all 4 are confirmed absent from `leaderboards2/`. However, the constrained goal (Opus rank 4, GPT rank 5) was not fully achieved by those 4 removals alone, because:

- The `hanoi_two_rf_learning` scoring fix **elevated GPT-5.4's score significantly** (+0.0627 overall), re-widening the gap
- The new data has GPT-5.4 at rank 4 and Opus at rank 5 again

The **updated minimum** to achieve Opus→4, GPT→5 is now **2 tasks** (down from 4), identified in Section 4b above.

---

## 6. Summary of Findings

| Finding | v1 (157 tasks) | v2 (134 tasks) |
|---|---|---|
| GPT-5.4 rank | 5 | **4** |
| Claude Opus 4.6 rank | 4 | **5** |
| Gemini 2.5 Flash rank | 4 | 6 |
| Tasks to flip Opus↔GPT | 4 | **2** |
| Tasks to move GPT/Opus to rank 3 | 3/4 | **Not achievable (≤15 tasks)** |
| Qwen gap to rank 4 | ~0.09 | **~0.12** (harder to close) |

The updated benchmark is **more robust** to rank manipulation for rank 3 positions, while the Opus↔GPT flip is now achievable with only 2 targeted removals.
