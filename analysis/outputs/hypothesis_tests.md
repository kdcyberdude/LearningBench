# LearningBench — Hypothesis Test Report

- **Runs analysed:** 1901
- **Tasks:** 135
- **Models:** 14
- **Bootstrap iterations:** 10000
- **Multiple-comparisons correction:** Benjamini-Hochberg

All correlations are Spearman rank (robust to outliers and non-linearity). Paired comparisons use Wilcoxon signed-rank. Group comparisons report Mann-Whitney U and Cliff's δ.

---

## H1. Evidence-seeking efficiency negatively predicts accuracy on interactive learning tasks (concept formation + language learning).
**Test:** Spearman rank correlation on (examples_requested, score), pooled and by category; bootstrap 95% CI.

**Result:** ρ = -0.516, 95% CI [-0.591, -0.436] (p = 8.5e-29, n = 403). Negative sign ⇒ models that request more examples score lower. Across models, mean examples-requested ranges from 1.8 to 11.9 (×6.5 spread).
**BH-adjusted p = 2.1e-28**

<details><summary>Raw statistics</summary>

```json
{
  "pooled": {
    "rho": -0.5159722454687398,
    "p": 8.504196318474056e-29,
    "n": 403,
    "ci_low": -0.5906217974530321,
    "ci_high": -0.43632686597114984
  },
  "per_category": {
    "concept": {
      "rho": -0.4555560844879723,
      "p": 1.8063808609891856e-09,
      "n": 158,
      "ci_low": -0.5778089124271047,
      "ci_high": -0.3182624733120842
    },
    "language": {
      "rho": -0.5675554953701369,
      "p": 2.7299485429734437e-22,
      "n": 245,
      "ci_low": -0.6549675069483963,
      "ci_high": -0.4711172691092502
    }
  },
  "evidence_seeking_spread_across_models": {
    "min": 1.8181818181818181,
    "max": 11.863636363636363,
    "ratio": 6.525,
    "per_model_mean": {
      "Qwen 3 Next 80B Thinking": 1.8181818181818181,
      "Gemini 3.1 Flash-Lite Preview": 2.022727272727273,
      "GLM-5": 2.6363636363636362,
      "GPT-5.4 mini": 3.0454545454545454,
      "Gemini 3.1 Pro Preview": 3.6136363636363638,
      "Gemini 3 Flash Preview": 5.0,
      "GPT-5.4": 5.7272727272727275,
      "Gemini 2.5 Flash": 6.136363636363637,
      "Qwen 3 Next 80B Instruct": 6.5227272727272725,
      "Claude Sonnet 4.6": 6.75,
      "DeepSeek V3.2": 7.636363636363637,
      "GPT-5.4 nano": 8.068181818181818,
      "Gemma 4 26B A4B": 9.772727272727273,
      "Claude Opus 4.6": 10.931818181818182,
      "Claude Haiku 4.5": 11.863636363636363
    }
  },
  "bh_adjusted_p": 2.1260490796185137e-28
}
```
</details>

## H2. Token consumption inversely predicts success in reinforcement-learning runs; failed runs spend dramatically more tokens than solved runs.
**Test:** Spearman rank correlation on (total_tokens, score) across all RL runs; Mann-Whitney U comparing solved (score≥0.5) vs failed (score≤0.1); Cliff's delta effect size.

**Result:** ρ = -0.533, 95% CI [-0.599, -0.462] (p = 1.6e-30, n = 397). Solved runs: 41328 avg tokens (n=237); failed runs: 177495 (n=44); ratio = ×4.3. Mann-Whitney U = 2109, p = 3.6e-10, Cliff's δ = -0.596.
**BH-adjusted p = 8.1e-30**

<details><summary>Raw statistics</summary>

```json
{
  "correlation": {
    "rho": -0.5330084024209704,
    "p": 1.6125177005508023e-30,
    "n": 397,
    "ci_low": -0.5992239469267049,
    "ci_high": -0.46175750457938686
  },
  "solved_mean_tokens": 41327.624472573836,
  "partial_mean_tokens": 122653.5775862069,
  "failed_mean_tokens": 177495.20454545456,
  "solved_vs_failed_mw": {
    "U": 2109.0,
    "p": 3.580499772152567e-10,
    "delta": -0.595512082853855,
    "n_a": 237,
    "n_b": 44,
    "mean_a": 41327.624472573836,
    "mean_b": 177495.20454545456
  },
  "token_ratio_failed_to_solved": 4.294832011533722,
  "bh_adjusted_p": 8.062588502754011e-30
}
```
</details>

## H3. On rule induction from evidence (concept formation + observational), Google+Open-source models outperform Anthropic+OpenAI models.
**Test:** Mann-Whitney U on model-level mean scores (rule-induction categories only); Cliff's delta effect size.

**Result:** Google+OSS mean = 0.591 (n=8); Anthropic+OpenAI mean = 0.420 (n=6). Relative gap = +40.8%. Mann-Whitney U = 41, p = 0.029, Cliff's δ = 0.708.
**BH-adjusted p = 0.029**

<details><summary>Raw statistics</summary>

```json
{
  "google_or_oss_mean": 0.5907644824195164,
  "anthropic_or_openai_mean": 0.4196052627030778,
  "relative_gap_pct": 40.790532181088196,
  "mw": {
    "U": 41.0,
    "p": 0.029304029304029304,
    "delta": 0.7083333333333334,
    "n_a": 8,
    "n_b": 6,
    "mean_a": 0.5907644824195164,
    "mean_b": 0.4196052627030778
  },
  "models_google_or_oss": [
    "DeepSeek V3.2",
    "GLM-5",
    "Gemini 2.5 Flash",
    "Gemini 3.1 Flash-Lite Preview",
    "Gemini 3.1 Pro Preview",
    "Gemma 4 26B A4B",
    "Qwen 3 Next 80B Instruct",
    "Qwen 3 Next 80B Thinking"
  ],
  "models_anthropic_or_openai": [
    "Claude Haiku 4.5",
    "Claude Opus 4.6",
    "Claude Sonnet 4.6",
    "GPT-5.4",
    "GPT-5.4 mini",
    "GPT-5.4 nano"
  ],
  "bh_adjusted_p": 0.029304029304029304
}
```
</details>

## H4. On concept formation, Qwen 3 Next 80B Thinking outperforms Qwen 3 Next 80B Instruct (same base model, reasoning toggled).
**Test:** Wilcoxon signed-rank test on matched per-task score pairs.

**Result:** Thinking mean = 0.563, Instruct mean = 0.186, uplift = +202.6%. Thinking wins 14/18 tasks. Wilcoxon W = 6.0, p = 0.0022.
**BH-adjusted p = 0.0027**

<details><summary>Raw statistics</summary>

```json
{
  "n_pairs": 18,
  "thinking_mean": 0.5628923333333333,
  "instruct_mean": 0.18601638888888888,
  "pct_uplift": 202.6036236353129,
  "median_diff": 0.365,
  "wins_thinking": 14,
  "wins_instruct": 1,
  "ties": 3,
  "W_stat": 6.0,
  "p": 0.002157762679075225,
  "bh_adjusted_p": 0.0026972033488440314
}
```
</details>

## H5. Practice trajectory (OLS slope) and final asymptote capture different signals and are not reducible to each other.
**Test:** Spearman ρ(slope, asymptote) across all parsed procedural (model, task) pairs; R² of slope~asymptote linear fit. A low R² means the trajectory score carries information the asymptote does not.

**Result:** ρ(slope, asymptote) = -0.017 (p = 0.86; note: an equivalence-style test — we *want* this p to be large, confirming no correlation) across n = 112 (model, task) pairs. R² = 0.011 — only 1.1% of trajectory-slope variance is explained by the asymptote, so trajectory carries ~99% orthogonal signal that a traditional final-score benchmark misses.

<details><summary>Raw statistics</summary>

```json
{
  "n": 112,
  "rho": -0.016724781185531015,
  "p": 0.8610599648908398,
  "r_squared_slope_on_asymptote": 0.011209219100524187,
  "pct_variance_orthogonal": 98.87907808994758,
  "example_same_asymptote_different_slope": {
    "slope_gap": 0.48000000000000004,
    "a": {
      "model": "Qwen 3 Next 80B Thinking",
      "task": "sql-reverse-engineering-proc-learning",
      "slope": -0.30000000000000004,
      "asymptote": 0.5
    },
    "b": {
      "model": "Qwen 3 Next 80B Instruct",
      "task": "state-machine-password-proc-learning",
      "slope": 0.18,
      "asymptote": 0.5
    }
  }
}
```
</details>

## H6. The overall model ranking is stable under leave-one-out task removal (no single task dominates).
**Test:** For each of N tasks, drop it, re-rank models by overall mean score, compute Spearman ρ with full-data ranking.

**Result:** Mean ρ = 0.9985, min ρ = 0.9912 across 135 leave-one-out iterations. Maximum rank change for any model after dropping any single task = 1 position(s). The ranking is robust to single-task removal.

<details><summary>Raw statistics</summary>

```json
{
  "n_tasks": 135,
  "n_models": 14,
  "spearman_mean": 0.9985347985347984,
  "spearman_min": 0.9912087912087912,
  "spearman_max": 1.0,
  "max_rank_change_any_model": 1
}
```
</details>

## H7. The six cognitive sub-abilities are not reducible to a single latent factor; they measure distinct dimensions.
**Test:** (a) PCA on the category-level model-score matrix — variance explained by PC1; (b) off-diagonal cross-category Spearman ρ range.

**Result:** PC1 explains 81.0% of category-level variance; if learning were a single latent factor this would approach 100%. Pairwise cross-category Spearman ρ among model rankings ranges from 0.29 to 0.93 (mean 0.65) — high but not unity, so the sub-abilities share variance yet each measures something distinct.

<details><summary>Raw statistics</summary>

```json
{
  "pc1_variance_explained": 0.8099835371365177,
  "pc2_variance_explained": 0.07216954850760493,
  "pc1_plus_pc2": 0.8821530856441226,
  "cross_category_rho_min": 0.2907496039850294,
  "cross_category_rho_max": 0.9296703296703297,
  "cross_category_rho_mean": 0.6540707576358769,
  "categories": [
    "assoc-learning",
    "concept-learning",
    "lang-learning",
    "obs-learning",
    "proc-learning",
    "rf-learning",
    "unknown"
  ]
}
```
</details>

## H8. The 'repeated-action failure mode' — the model issues the same action many turns in a row — is more common in low-performing RL runs and in weaker models.
**Test:** (a) Spearman ρ on (max_repeat_run, score) across all RL runs; (b) Mann-Whitney U comparing max_repeat_run in top-4 vs bottom-4 models.

**Result:** ρ(max_repeat_run, score) = -0.291, 95% CI [-0.385, -0.193] (p = 1.3e-08, n = 368). Top-4 models avg max streak = 3.4; bottom-4 models avg = 5.7 (Mann-Whitney p = 0.00084, Cliff's δ = -0.251). 43 runs show streaks of 10+ identical actions.
**BH-adjusted p = 2.1e-08**

<details><summary>Raw statistics</summary>

```json
{
  "correlation": {
    "rho": -0.29128387563235475,
    "p": 1.250497988189846e-08,
    "n": 368,
    "ci_low": -0.385116654897115,
    "ci_high": -0.1932534315156788
  },
  "top4_mean_repeat": 3.4339622641509435,
  "bot4_mean_repeat": 5.6875,
  "top_vs_bot_mw": {
    "U": 4448.5,
    "p": 0.00084491230820842,
    "delta": -0.2505896226415094,
    "n_a": 106,
    "n_b": 112,
    "mean_a": 3.4339622641509435,
    "mean_b": 5.6875
  },
  "n_runs_with_repeat_gte_10": 43,
  "worst_cases": [
    {
      "model": "Qwen 3 Next 80B Thinking",
      "task": "battleship-1d-rf-learning",
      "max_repeat_run": 42,
      "n_assistant_turns": 44,
      "score": null
    },
    {
      "model": "Qwen 3 Next 80B Instruct",
      "task": "arithmetic-next-rf-learning",
      "max_repeat_run": 33,
      "n_assistant_turns": 34,
      "score": 0.1
    },
    {
      "model": "GPT-5.4 nano",
      "task": "arithmetic-next-rf-learning",
      "max_repeat_run": 33,
      "n_assistant_turns": 34,
      "score": 0.1
    },
    {
      "model": "Gemma 4 26B A4B",
      "task": "chebyshev-point-rf-learning",
      "max_repeat_run": 32,
      "n_assistant_turns": 34,
      "score": 0.1643
    },
    {
      "model": "GPT-5.4 nano",
      "task": "battleship-1d-rf-learning",
      "max_repeat_run": 32,
      "n_assistant_turns": 44,
      "score": null
    },
    {
      "model": "GPT-5.4 mini",
      "task": "quadratic-root-rf-learning",
      "max_repeat_run": 31,
      "n_assistant_turns": 32,
      "score": 0.1845
    },
    {
      "model": "GPT-5.4 nano",
      "task": "divisor-count-rf-learning",
      "max_repeat_run": 29,
      "n_assistant_turns": 30,
      "score": null
    },
    {
      "model": "Qwen 3 Next 80B Instruct",
      "task": "divisor-count-rf-learning",
      "max_repeat_run": 29,
      "n_assistant_turns": 30,
      "score": null
    },
    {
      "model": "GPT-5.4 nano",
      "task": "gray-hamming-rf-learning",
      "max_repeat_run": 29,
      "n_assistant_turns": 30,
      "score": 0.1
    },
    {
      "model": "Qwen 3 Next 80B Thinking",
      "task": "digitwise-l1-rf-learning",
      "max_repeat_run": 27,
      "n_assistant_turns": 28,
      "score": 0.1481
    }
  ],
  "bh_adjusted_p": 2.084163313649743e-08
}
```
</details>
