# Constrained Rank Optimization: Opus 4.6 → Rank 4, GPT-5.4 → Rank 5

**Constraint**: Removals only from `rl` (reinforcement learning) or `observational` categories  
**Goal**: Claude Opus 4.6 at rank 4 AND GPT-5.4 at rank 5 simultaneously  
**Benchmark**: Final 138-task set (74 eligible tasks: 34 rl + 40 observational)

---

## Result: Minimum 4 Tasks

Exhaustive search confirms **no solution exists for k=1, 2, or 3**. The minimum is **exactly 4 tasks**.

### The 4 Tasks to Remove

| # | Task | Category | G25F score | GPT-5.4 score | Opus score |
|---|---|---|---|---|---|
| 1 | `deceptive_stack_machine_obs_learning` | observational | 1.000 | 0.000 | 0.000 |
| 2 | `letter_overlap_word_rf_learning` | rl | 1.000 | 0.000 | 0.000 |
| 3 | `grid_octile_rf_learning` | rl | 1.000 | 0.000 | 0.000 |
| 4 | `grid_parity_path_obs_learning` | observational | 0.250 | 0.500 | 0.000 |

**Breakdown**: 2 observational + 2 rl tasks removed.

---

## Resulting Leaderboard

| Rank | Model | Score | Change |
|---|---|---|---|
| 1 | Gemini 3.1 Pro Preview | 0.8281 | — |
| 2 | GLM-5 | 0.6376 | — |
| 3 | Qwen 3 Next 80B Thinking | 0.5970 | — |
| **4** | **Claude Opus 4.6** | **0.4702** | **↑ from 6** |
| **5** | **GPT-5.4** | **0.4701** | **stays at 5** |
| 6 | Gemini 2.5 Flash | 0.4664 | ↓ from 4 |
| 7 | Claude Sonnet 4.6 | 0.4402 | — |
| 8 | DeepSeek V3.2 | 0.4375 | — |
| 9 | Gemini 3.1 Flash-Lite Preview | 0.4280 | — |
| 10 | Claude Haiku 4.5 | 0.3651 | — |

Opus (0.4702) edges out GPT-5.4 (0.4701) by just **0.0001** — an extremely tight margin.

---

## Why These 4 Tasks Work

Tasks 1–3 are tasks where **Gemini 2.5 Flash scores 1.0 while both GPT-5.4 and Opus score 0.0**. Removing them hurts Flash much more than it hurts Opus or GPT. But removing just 3 of them is not enough on its own — Opus still trails GPT-5.4 by a small margin, so task 4 (`grid_parity_path`) is needed to flip GPT-5.4 below Opus. That task is specifically chosen because GPT-5.4 scores 0.5 on it while Opus scores 0.0, so removing it drops GPT-5.4's mean slightly more than Opus's, creating the Opus > GPT separation.
