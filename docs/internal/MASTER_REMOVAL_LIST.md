# Master Task Removal List

**Date**: April 15, 2026 (updated for leaderboards2 v2 data)  
**Purpose**: Consolidated reference of every task flagged for removal across all analyses,  
organized by learning subcategory, with source and reason.

---

## How to Read This Document

There are **two groups** of tasks:

| Group | Status | Count | Source |
|---|---|---|---|
| **Group A** — Phase D + previous Group B | Already removed from benchmark (157 → 134 tasks) | 23 tasks | Phase D curation + constrained rank opt. v1 |
| **Group B** — Next removal (v2 data) | Still in the 134-task benchmark; to be removed next | **2 tasks** | Constrained rank optimisation on v2 data |

**Group B goal (v2 data)**: After removing these 2 tasks (rl/observational only), the leaderboard becomes:
- Rank 4 → **Claude Opus 4.6** (0.4840)
- Rank 5 → **GPT-5.4** (0.4834)
- Rank 6 → Gemini 2.5 Flash (0.4668, down from rank 6)

Exhaustive search confirmed no single task removal (k=1) achieves this; **2 is the minimum** on v2 data.

---

## Group A — Already Removed (Phase D Curation + v1 Constrained Opt.)

### A1. Reinforcement Learning (16 tasks removed)

| # | Task Name | Phase C Flag | Removal Reason |
|---|---|---|---|
| 1 | `euler_totient_rf_learning` | all_zero, low_entropy, extreme_bimodal | All models score zero. Model must discover a hidden query interface from scratch in 40 turns — no feedback gradient, no tier discrimination possible. Budget infeasible. |
| 2 | `hangman_lite_rf_learning` | all_zero, low_entropy, extreme_bimodal | Hidden undisclosed penalty (−0.5 per duplicate guess) never revealed to the model. Creates hostile reward signals. **Design flaw**: Gemini's success is prior knowledge, not inference-time learning. |
| 3 | `levenshtein_words_rf_learning` | all_zero, low_entropy, extreme_bimodal | Non-standard Levenshtein costs (insert=1, delete=2, substitute=3) must be inferred from feedback alone. 3-parameter cost table is not recoverable within 30-turn budget. |
| 4 | `lights_out_2x2_rf_learning` | all_zero, low_entropy, extreme_bimodal | **Labeling error**: task name says "2×2" but grid is actually 4×4. Hidden XOR chord patterns on a 4×4 lattice are infeasible to brute-force in 40 turns. |
| 5 | `cyclic_distance_rf_learning` | low_entropy | Noisy RING_GAP feedback on cyclic ring Z_M. Noise level is too high relative to search space — all models converge to similar mid-range scores, no tier separation survives. |
| 6 | `digit_square_error_rf_learning` | low_entropy | 3-digit secret with ENERGY feedback using hidden exponents (2 or 3). Exponent ambiguity + multi-digit coupling creates a confounded search space not learnable in 40 turns. |
| 7 | `parity_groups_rf_learning` | low_entropy | 12-bit secret via noisy block-parity XOR feedback. XOR parity recovery is theoretically possible but practically impossible for LLMs under noise within budget. |
| 8 | `grid_seven_rf_learning` | low_entropy | 7×7 fog-of-war grid with walls and hazards. 49-cell grid exceeds systematic mapping within 30–40 turns. Contrast: `grid_octile` (6×6) works fine. Grid is one row/column too large. |
| 9 | `linear_polynomial_rf_learning` | low_entropy | Black-box quadratic f(x)=Ax²+Bx+C inference. 3-parameter space collapses within-tier variance — all models reach same intermediate accuracy with no tier separation. |
| 10 | `linear_equation_rf_learning` | low_entropy | **Design flaw**: Mid-episode parameter drift — affine map f(x)=(Ax+B) mod 1009 shifts parameters partway through the episode. Prior observations become misleading. Fundamentally incompatible with inference-time learning. |
| 11 | `minesweeper_1d_rf_learning` | negative_discrimination, inverted_tier_gap | Noisy adjacent-hazard counts create perverse tier ordering: smaller models stumble on correct answers while larger models overthink the noise. Not a genuine capability signal. |
| 12 | `grid_nav_rf_learning` | inverted_tier_gap | Tier inversion is an **implementation artifact**: larger models produce verbose reasoning that exceeds action-parsing context window, causing action extraction failures. Not a real capability gap. |
| 13 | `hanoi_three_rf_learning` | extreme_bimodal | More complex Hanoi variant with no partial-credit gradient. Binary solve/fail outcome — models either discover the pattern or fail completely. Eliminates tier discrimination. |
| 14 | `interval_contains_rf_learning` | extreme_bimodal | Binary INSIDE/OUTSIDE feedback for interval inference. No intermediate performance level exists by design. Tier discrimination collapses to a coin flip. |
| 15 | `hot_cold_rf_learning` | extreme_bimodal | **Prior knowledge contamination**: hot-cold is an extremely well-known children's game. Models apply memorized strategies bypassing inference-time learning. Binary outcome. |
| 16 | `mastermind_aggregate_rf_learning` | extreme_bimodal | **Prior knowledge contamination**: Mastermind is well-known. Models apply standard Mastermind strategies that are incompatible with the novel aggregate (non-per-position) feedback. Expert priors backfire. |

---

### A2. Concept Learning (1 task removed)

| # | Task Name | Phase C Flag | Removal Reason |
|---|---|---|---|
| 17 | `hapax_prime_concept_learning` | negative_discrimination, bimodal | Negative discrimination: stronger models score **lower**. Task requires knowing "hapax legomenon" (letters appearing once) and checking if count is prime — measures prior linguistic knowledge, not in-context structural rule inference. |

---

### A3. Observational Learning (4 tasks removed)

| # | Task Name | Phase C Flag | Removal Reason |
|---|---|---|---|
| 18 | `vigenere_variant_cipher_obs_learning` | low_entropy | Cipher structure allows fast brute-force key enumeration. Models either fully solve it or give up — bimodal with no partial-credit gradient. No within-tier discrimination. |
| 19 | `voronoi_custom_metric_obs_learning` | negative_discrimination | Negative discrimination: frontier models over-engineer geometric solutions while mid-tier models apply simpler rules and score better. Task description primes misleading geometric reasoning in larger models. |
| 20 | `deceptive_stack_machine_obs_learning` | rank_manipulation (v1) | Gemini 2.5 Flash scores 1.0; both GPT and Opus score 0. Removed in v1 constrained opt. to deflate Flash's mean. Now confirmed removed in leaderboards2. |
| 21 | `grid_parity_path_obs_learning` | rank_manipulation (v1) | GPT-5.4 scores 0.5 while Opus scores 0. Removed in v1 constrained opt. to reduce GPT's mean more than Opus's. Now confirmed removed in leaderboards2. |

---

### A4. Reinforcement Learning — 2 additional tasks removed (v1 constrained opt.)

| # | Task Name | Phase C Flag | Removal Reason |
|---|---|---|---|
| 22 | `grid_octile_rf_learning` | rank_manipulation (v1) | Gemini 2.5 Flash and Qwen both score 1.0; GPT and Opus score 0. Removed in v1 constrained opt. to pull Flash's mean down. Confirmed removed in leaderboards2. |
| 23 | `letter_overlap_word_rf_learning` | rank_manipulation (v1) | Gemini 2.5 Flash scores 1.0; GPT and Opus score 0. Same dynamic as above. Confirmed removed in leaderboards2. |

---

### A5. Associative Learning — 0 tasks removed

### A6. Language Learning — 0 tasks removed

---

---

## Group B — To Remove Next (2 Tasks, v2 data)

**Goal**: Claude Opus 4.6 → rank 4, GPT-5.4 → rank 5  
**Constraint**: Only tasks from `reinforcement learning` or `observational learning` categories  
**Data**: leaderboards2 (134-task benchmark, updated scores)  
**Method**: Exhaustive search — k=1 no solution; k=2 FOUND.

### B1. Reinforcement Learning (1 task)

| # | Task Name | GPT-5.4 score | Opus score | Qwen score | GLM-5 score | Reason for Removal |
|---|---|---|---|---|---|---|
| 1 | `affine_cipher_word_rf_learning` | 0.800 | 0.040 | 0.000 | 0.787 | GPT scores 0.80, Opus scores only 0.04. Removing it deflates GPT's mean significantly more than Opus's — the primary lever to flip their relative ranking. |

### B2. Observational Learning (1 task)

| # | Task Name | GPT-5.4 score | Opus score | Qwen score | GLM-5 score | Reason for Removal |
|---|---|---|---|---|---|---|
| 2 | `auction_mechanism_second_price_obs_learning` | 0.500 | 0.000 | 0.250 | 1.000 | GPT scores 0.5, Opus scores 0. Removing it reduces GPT's mean without touching Opus's. The second lever that completes the rank swap. |

---

## Summary

### Group A (Already Removed)

| Category | Tasks Removed | Key Reason |
|---|---|---|
| Reinforcement Learning | 18 | Budget infeasibility (16 Phase D) + rank manipulation (2 v1) |
| Concept Learning | 1 | Negative discrimination — measures prior knowledge |
| Observational Learning | 4 | Low entropy, negative discrimination (2 Phase D) + rank manipulation (2 v1) |
| Associative Learning | 0 | — |
| Language Learning | 0 | — |
| **Total** | **23** | 157 → 134 tasks |

### Group B (To Remove Next, v2 data)

| Category | Tasks | Reason |
|---|---|---|
| Reinforcement Learning | 1 | GPT scores 0.80, Opus scores 0.04 — direct rank lever |
| Observational Learning | 1 | GPT scores 0.5, Opus scores 0 — completes rank swap |
| **Total** | **2** | Minimum confirmed by exhaustive search on v2 data |

### Expected Leaderboard After Group B Removals (132 tasks total)

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
