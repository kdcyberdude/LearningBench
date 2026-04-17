# Phase A Insights — Data Extraction & Analysis Foundation

**Generated:** 2026-04-14  
**Script:** `analysis/phase_a.py`  
**Outputs:** `analysis/outputs/` (score_matrix.csv, aggregates.csv, task_stats.csv, model_stats.csv, flagged_tasks.csv)

---

## What Phase A Did

Parsed all 5 leaderboard JSON files into a unified dataset and computed three layers of statistics:
1. **Per-task statistics** — how each of the 157 tasks performed across 14 models
2. **Per-model statistics** — how each model ranked within and across categories
3. **Flagged tasks** — tasks that may have quality issues based on statistical criteria

---

## Key Finding 1 — Overall Model Rankings

| Rank | Model | Tier | Overall Mean |
|---|---|---|---|
| 1 | Gemini 3.1 Pro Preview | frontier | 0.775 |
| 2 | GLM-5 | frontier | 0.624 |
| 3 | Qwen 3 Next 80B Thinking | mid | 0.569 |
| 4 | Gemini 2.5 Flash | mid | 0.486 |
| 5 | GPT-5.4 | frontier | 0.444 |
| 6 | Claude Opus 4.6 | frontier | 0.431 |
| 7 | Claude Sonnet 4.6 | mid | 0.424 |
| 8 | Gemini 3.1 Flash-Lite Preview | mid | 0.415 |
| 9 | DeepSeek V3.2 | mid | 0.406 |
| 10 | GPT-5.4 mini | mid | 0.344 |
| 11 | Gemma 4 26B A4B | small | 0.341 |
| 12 | Claude Haiku 4.5 | small | 0.339 |
| 13 | Qwen 3 Next 80B Instruct | small | 0.328 |
| 14 | GPT-5.4 nano | small | 0.251 |

**Notable:** Gemini 3.1 Pro Preview dominates by a large margin (0.775 vs 0.624 for #2). Among frontier models, GLM-5 (0.624) substantially outperforms GPT-5.4 (0.444) and Claude Opus 4.6 (0.431). See Phase B for deeper investigation.

---

## Key Finding 2 — Associative Learning Aggregates Were Broken in JSON

**The problem:** The leaderboard JSON for Associative Learning stored the overall summary score as `0.0` for two models (GLM-5, Gemma 4 26B) and as missing/NaN for the other 12 models. The platform had not yet computed these scores (marked as "pending recalculation" in PROJECT_MASTER.md).

**The fix:** We compute the aggregate ourselves as the mean of each model's 20 individual task scores. This is identical to how the other categories compute their aggregate.

**Verified correct computed aggregates for Associative Learning:**

| Model | Computed Aggregate |
|---|---|
| Gemini 3.1 Pro Preview | 0.935 |
| GLM-5 | 0.792 |
| Claude Opus 4.6 | 0.662 |
| GPT-5.4 | 0.650 |
| Claude Sonnet 4.6 | 0.651 |
| Qwen 3 Next 80B Thinking | 0.628 |
| Gemini 2.5 Flash | 0.595 |
| Gemini 3.1 Flash-Lite Preview | 0.562 |
| DeepSeek V3.2 | 0.523 |
| Claude Haiku 4.5 | 0.502 |
| Gemma 4 26B A4B | 0.501 |
| GPT-5.4 mini | 0.487 |
| Qwen 3 Next 80B Instruct | 0.464 |
| GPT-5.4 nano | 0.437 |

**Action required:** Update the Associative Learning leaderboard on the platform with these computed values, or confirm the platform will eventually recalculate them automatically.

---

## Key Finding 3 — Flagged Tasks (145 / 157 flagged on at least one criterion)

The flagging system identifies tasks that might have quality issues. A task can be flagged on multiple criteria simultaneously. **Being flagged does not mean the task is bad** — it means it needs human review.

### Flag types explained (plain language):

| Flag | What it means | How many tasks | Action |
|---|---|---|---|
| `no_tier_diff` | Statistically, we can't confirm that frontier models do better than small models on this task | 123 | Low priority — expected artifact of only 14 models. Check manually if also flagged for `no_discrimination`. |
| `bimodal` | Scores cluster into two groups: some models score ~1.0, others score ~0.0. Nothing in the middle. | 84 | Review in Phase B. Bimodal tasks can be *good* (hard threshold) or *bad* (broken task). |
| `no_discrimination` | The task score doesn't correlate with a model's overall performance. A weak model gets the same score as a strong one. | 14 | High priority — these tasks may not measure learning ability at all. |
| `low_entropy` | Scores are very concentrated (not spread out). E.g., most models score around 0.5 with little variation. | 11 | Medium priority — low spread reduces the task's ability to distinguish models. |
| `near_uniform` | Scores have very low standard deviation (< 0.05). All models score about the same. | 7 | High priority — task adds no signal. Consider removing. |
| `too_hard` | Mean score < 5% but not zero. Only 1-2 models score anything at all. | 5 | Investigate: is the task broken, or is it a genuine frontier-failure task (which is valuable)? |
| `too_easy` | Mean score > 90%. Almost all models score near perfect. | 4 | Remove or increase difficulty. No discriminatory power. |
| `all_zero` | Every single model scored 0.0. | 1 | Likely broken. Investigate immediately. |

### Flagged tasks by category:

| Category | Total Tasks | Flagged | % Flagged |
|---|---|---|---|
| Associative Learning | 20 | 15 | 75% |
| Concept Formation | 19 | 19 | 100% |
| Language Learning | 26 | 20 | 77% |
| Observational Learning | 42 | 42 | 100% |
| Reinforcement Learning | 50 | 49 | 98% |

**Important context:** The high flagging rates for Concept Formation, Observational Learning, and RL are driven almost entirely by `no_tier_diff` — the statistical test that fails because we have too few models per tier (4 frontier, 6 mid, 4 small). This is expected and does NOT mean these categories have poor tasks.

---

## Key Finding 4 — Bimodal Score Distributions (84 tasks)

**What bimodal means (concrete example):**  
Imagine scoring all 14 models on a hard grammar task. Bimodal looks like this:
- Gemini 3.1 Pro Preview: 0.95, GLM-5: 0.90, Qwen Thinking: 0.85
- Everyone else: 0.05–0.10

There's a clear break. Either you get the rule or you don't. No model scores 0.5.

**Is bimodality expected?** Yes, for this type of benchmark. Our tasks are designed to test whether a model can induce a hidden rule — it's a cognitive threshold, not a gradual skill. The correct interpretation depends on *which* models are in the high group:
- **Good bimodality:** The high-scoring group is the frontier/strong models — the task genuinely discriminates capability tiers.
- **Concerning bimodality:** The high-scoring group is random (e.g., one small model scores 1.0 while frontier models score 0) — may indicate task ambiguity or a lucky shortcut.

Phase B will classify bimodal tasks by which tier the high-scorers belong to.

---

## Questions for Phase B to Answer

1. **The 14 tasks with no discrimination** — which specific tasks are these, and why do they fail? (Task design issues vs. genuine ceiling/floor effects?)
2. **The 1 all-zero task** — which task? Is it broken or genuinely impossible?
3. **The 5 "too hard" tasks** — are these good frontier-failure tasks or broken?
4. **The 4 "too easy" tasks** — should these be removed or difficulty-increased?
5. **Bimodal task classification** — for each of the 84 bimodal tasks, is the "high scoring" group the expected strong models?
6. **GLM-5 anomaly** — why does a frontier model (GLM-5) rank #2 overall while GPT-5.4 and Claude Opus, also frontier, rank #5 and #6? Category-level breakdown needed.
7. **Associative Learning dominance** — Gemini 3.1 Pro Preview scores 0.935 on Associative Learning. Is this a ceiling effect? How many tasks does it get perfect?

---

## Files Produced

| File | Rows | Description |
|---|---|---|
| `score_matrix.csv` | 2,193 | One row per (model, task, category). Columns: category, task_name, model, score, tier, provider |
| `aggregates.csv` | 70 | Per-model per-category aggregates. Columns: category, model, json_aggregate, computed_aggregate, tier, provider |
| `task_stats.csv` | 157 | Per-task statistics. Columns: mean, std, min, max, pct_zero, pct_perfect, entropy, item_discrimination, bimodality_coeff, tier_discriminates, kw_p, flag_* |
| `model_stats.csv` | 14 | Per-model statistics. Columns: tier, provider, overall_mean, mean per category, rank per category, rank_overall |
| `flagged_tasks.csv` | 145 | Tasks with at least one flag. Includes all flag columns and task stats |
