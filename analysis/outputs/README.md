# Analysis Outputs

All CSV files produced by the LearningBench analysis pipeline. Pre-computed outputs are committed here so you can run any analysis script directly without re-running the full pipeline.

---

## Primary Data Files

These are the starting point for any analysis.

### `score_matrix.csv`
**The raw score matrix — all 157 tasks before curation.**

| Column | Type | Description |
|---|---|---|
| `category` | str | Sub-ability key (`associative`, `concept`, `language`, `observational`, `procedural`, `rl`) |
| `category_full` | str | Human-readable category name |
| `task_name` | str | Task display name |
| `model` | str | Model display name |
| `score` | float | Final composite score in [0, 1] |
| `tier` | str | Model tier (`frontier`, `mid`, `small`) |
| `provider` | str | Provider (`Google`, `OpenAI`, `Anthropic`, `Open-source`) |

~2,193 rows · 7 columns · 218 KB

---

### `score_matrix_phase_d.csv`
**The curated score matrix — 138 tasks after Phase D curation.**

Same schema as `score_matrix.csv`. Use this for all primary analyses. 19 tasks were removed (see `CURATION_DECISIONS.md` for rationale).

~1,928 rows · 7 columns · 193 KB

---

### `full_task_model_stats.csv`
**Per-(model, task) statistics: scores plus rank within task.**

Extends the score matrix with additional computed columns like within-task rank and category-normalized score.

~1,835 rows · 219 KB

---

## Per-Model Statistics

### `model_stats.csv`
**One row per model. Overall statistics and ranks.**

| Column | Description |
|---|---|
| `model` | Model display name |
| `tier` | `frontier`, `mid`, `small` |
| `provider` | Provider |
| `overall_score` | Mean score across all retained tasks |
| `overall_rank` | 1 = best |
| `<category>_score` | Mean score for each sub-ability |
| `<category>_rank` | Rank for each sub-ability |

14 rows · 2.5 KB

---

### `aggregates.csv`
**Per-(model, category) aggregate scores.**

71 rows · 5.1 KB

### `aggregate_stats.csv`
**Aggregate statistics with bootstrap CIs.**

85 rows · 6.7 KB

### `category_pivot.csv`
**Pivot: model × category mean scores.**

15 rows × 7 columns · 1.5 KB

### `tier_stats.csv`
**Per-tier, per-category mean scores.**

16 rows · 2.7 KB

### `rank1_counts.csv`
**Per-model count of tasks where that model ranks #1.**

12 rows · 410 B

---

## Per-Task Statistics

### `task_stats.csv`
**Master per-task quality table. The primary output of Phase C analysis.**

| Column | Description |
|---|---|
| `task_name` | Task display name |
| `category` | Sub-ability |
| `mean` | Mean score across 14 models |
| `std` | Standard deviation |
| `entropy` | Shannon entropy of score distribution |
| `discrimination_r` | Point-biserial correlation with category total |
| `discrimination_class` | `excellent`, `good`, `marginal`, `poor` |
| `loo_max_rank_change` | Maximum rank change in LOO analysis |
| `loo_spearman` | Spearman correlation with full leaderboard under LOO |
| `signal_ratio` | Score / random baseline |
| `flags` | Comma-separated quality flags |
| `flag_count` | Number of active flags |

158 rows · 42 KB

---

### `flagged_tasks.csv`
**All tasks with at least one quality flag, with flag details.**

146 rows · 41 KB

### `final_flagged_tasks.csv`
**Per-task priority assessment for Phase D curation decisions.**

158 rows · 14 KB

---

## Item Analysis

### `discrimination_report.csv`
**Per-task item discrimination: r, D-index, classification.**

158 rows · 25 KB. Used by Phase D to identify poorly discriminating tasks.

### `entropy_report.csv`
**Per-task Shannon entropy.**

158 rows · 13 KB. Near-zero entropy = all models score the same (ceiling or floor collapse).

### `category_entropy.csv`
**Per-category entropy summary.**

6 rows · 412 B

### `bimodal_report.csv`
**Bimodal tasks: tasks where models either fully solve or fully fail.**

85 rows · 13 KB. Classified by whether the bimodality reflects a real ability split vs. task noise.

### `gemini_ceiling.csv`
**Gemini Pro score and rank on every task.**

158 rows · 12 KB. Used to measure Gemini ceiling dominance.

---

## Cross-Category Analysis

### `cross_category_correlations.csv`
**5×5 Spearman correlation matrix across sub-abilities.**

6 rows · 493 B (5 categories + header)

### `cross_category_pvalues.csv`
**P-values for cross-category correlations.**

6 rows · 533 B

### `cross_category_rank_correlations.csv`
**Rank-based cross-category Spearman correlations.**

6 rows · 493 B

---

## Scaling & Tier Analysis

### `tier_inversions.csv`
**22 model-level scale inversions (smaller model outranks larger, same provider).**

23 rows · 1.7 KB

### `tier_task_gaps.csv`
**Per-task frontier-minus-small tier score gap.**

158 rows · 10 KB

---

## Robustness Analysis

### `loo_global.csv`
**Leave-one-out global stability: max rank change and Spearman per task.**

158 rows · 14 KB

### `loo_category.csv`
**Leave-one-out within-category stability.**

158 rows · 12 KB

### `benchmark_stability_summary.csv`
**Overall LOO stability summary.**

10 rows · 398 B

---

## Efficiency & Ablation

### `efficiency_ablation.csv`
**Composite vs. accuracy-only rank comparison.**

29 rows · 1.4 KB. Tests whether the efficiency component of the scoring formula changes results.

### `efficiency_ablation_stability.csv` (in `novelty_claims/`)
**Stability of efficiency ablation findings.**

---

## Provider Analysis

### `provider_analysis.csv`
**Per-(provider, category) mean scores, sample sizes.**

21 rows · 2.3 KB

### `provider_overall.csv`
**Provider overall mean scores.**

5 rows · 246 B

### `provider_pivot.csv`
**Provider × Category pivot.**

5 rows × 7 cols · 229 B

---

## Thinking vs. Instruct

### `thinking_comparison.csv`
**Qwen 3 Thinking vs. Instruct per-category: mean score, win rate, p-value.**

6 rows · 248 B

### `interactive_vs_rl.csv`
**Per-model mean on interactive tasks vs. RL tasks.**

15 rows · 555 B

---

## Epistemic Analysis

### `epistemic_analysis.csv`
**Per-(model, task-type) mean scores: UNKNOWN vs. normal tasks.**

21 rows · 1.2 KB

### `unknown_task_scores.csv`
**Per-model comparison of UNKNOWN vs. normal task scores.**

15 rows · 811 B

---

## Random Baseline

### `random_baseline.csv`
**Per-category expected random score and actual mean. Signal ratios.**

6 rows · 1.3 KB

### `model_signal_above_random.csv`
**Per-(model, category) signal above random baseline.**

71 rows · 4.2 KB

---

## Ground Truth Verification

### `ground_truth_spotcheck.csv`
**Results of 15-task ground truth integrity check.**

16 rows · 3.6 KB

---

## Curation (Phase D)

### `phase_d_final_task_list.csv`
**The final list of 138 retained tasks after Phase D curation.**

139 rows · 4.5 KB

### `phase_d_verdicts.csv`
**Per-task remove/keep decision table with rationale.**

27 rows · 13 KB

### `phase_d_leaderboard_impact.csv`
**Model scores before vs. after Phase D curation.**

15 rows · 1.2 KB

### `phase_d_category_impact.csv`
**Category-level impact of Phase D curation.**

### `flagged_removal_impact.csv`
**Model rank changes before vs. after removing all flagged tasks.**

15 rows · 570 B

---

## Kernel Logs (Token/Cost/Timing)

### `kernel_logs_parsed.csv`
**Per-model timing, token counts, and costs from Kaggle kernel runs.**

14 rows · 2.8 KB. The source data for H14–H18 in `30_hypothesis_tests.py`.

| Column | Description |
|---|---|
| `model` | Model display name |
| `total_tokens` | Total tokens consumed across all tasks |
| `prompt_tokens` | Input tokens |
| `completion_tokens` | Output tokens |
| `thinking_tokens` | Thinking chain tokens (where applicable) |
| `total_cost_usd` | Estimated API cost |
| `latency_p50_ms` | Median latency per task |

---

## Hypothesis Tests

### `hypothesis_tests.json`
**All formal hypothesis test results in machine-readable JSON.**

13 KB. One entry per hypothesis with test statistic, p-value, effect size, confidence interval, and BH-adjusted q-value.

### `hypothesis_tests.md`
**Hypothesis test results in human-readable markdown.**

11 KB. The same data as the JSON, formatted as a table.

---

## Writeup Numbers

### `writeup_numbers.csv`
**All key metrics cited in WRITEUP.md, with their values.**

38 rows · 3.1 KB. Useful for verifying any number quoted in the writeup against the underlying computation.

---

## Novelty Claims (`novelty_claims/`)

Supplementary CSVs supporting the novelty section of the writeup.

| File | Description |
|---|---|
| `leaderboard.csv` | Full leaderboard used in novelty comparison |
| `active_querying_per_model.csv` | Per-model evidence probe counts |
| `active_querying_summary.csv` | Summary statistics for active querying |
| `model_rule_induction_table.csv` | Per-model rule induction scores |
| `provider_rule_induction.csv` | Provider-level rule induction breakdown |
| `provider_rule_induction_summary.csv` | Provider rule induction summary |
| `thinking_vs_instruct.csv` | Thinking vs. instruct comparison |
| `token_failure_by_quartile.csv` | Token spend by score quartile |
| `token_failure_summary.csv` | Token failure signal summary |
| `benchmark_summary.csv` | Benchmark comparison summary |
| `efficiency_ablation_stability.csv` | Efficiency ablation stability |
| `efficiency_ablation_summary.csv` | Efficiency ablation summary |

---

## Charts (`figures/`, `charts/`, `efficiency_charts/`)

### `figures/` — Publication figures for WRITEUP.md

| File | Description |
|---|---|
| `fig_leaderboard_ci.png` | Leaderboard bar chart with 95% bootstrap CIs |
| `fig_radar_profiles.png` | Per-model radar charts across 6 sub-abilities |
| `fig_task_difficulty.png` | Task difficulty distribution (mean score histogram) |
| `fig_thinking_vs_instruct.png` | Qwen Thinking vs. Instruct per-category comparison |
| `fig_tokens_vs_score.png` | Token spend vs. score (RL category) |
| `fig_trajectory_orthogonal.png` | Procedural trajectory slope vs. asymptote scatter |
| `fig_meta_calibration.png` | Evidence probe ratio vs. overall score scatter |

### `charts/` — Phase C comparison charts

Before/after curation comparison: radar, grouped bar, and heatmap.

### `efficiency_charts/` — Token/cost efficiency

10 charts from `18_timing_hypotheses.py` and `20_efficiency_analysis.py`.

---

## Procedural Trajectory (`../procedural_trajectory_ablation/`)

### `trajectories.csv`
**Raw per-(model, task, episode) accuracy data for procedural learning tasks.**

52 KB. One row per episode within a run.

### `trajectory_summary.csv`
**Aggregated trajectory statistics: OLS slope, asymptote, R², per (model, task).**

12 KB. This is the source of the ρ(slope, asymptote) = −0.02 finding.
