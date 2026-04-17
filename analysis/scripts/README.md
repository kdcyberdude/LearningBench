# Analysis Scripts

This directory contains the numbered analysis pipeline for LearningBench. Scripts are designed to run in order, but each can be run independently since all intermediate outputs are pre-computed in `../outputs/`.

## Prerequisites

From the repo root:

```bash
pip install -r requirements.txt
```

All scripts expect to be run from either the repo root or the `analysis/scripts/` directory — both work because paths are resolved relative to the script's own location via `Path(__file__)`.

---

## Pipeline Overview

```
Kaggle leaderboard JSONs
        │
        ▼
[utils/data_loader.py]  ← loads & parses all leaderboard data
        │
        ├── 04_discriminatory_power.py   ─► discrimination_report.csv
        ├── 05_cross_category.py         ─► cross_category_*.csv
        ├── 06_scaling_analysis.py       ─► tier_stats.csv, tier_inversions.csv
        ├── 07_efficiency_ablation.py    ─► efficiency_ablation.csv
        ├── 08_entropy_analysis.py       ─► entropy_report.csv
        ├── 09_provider_analysis.py      ─► provider_*.csv
        ├── 10_bimodality_and_dominance  ─► bimodal_report.csv, thinking_comparison.csv
        ├── 11_task_removal_sensitivity  ─► loo_global.csv, loo_category.csv
        ├── 12_random_baseline.py        ─► random_baseline.csv, model_signal_above_random.csv
        ├── 13_ground_truth_spotcheck    ─► ground_truth_spotcheck.csv
        ├── 14_epistemic_analysis.py     ─► epistemic_analysis.csv, unknown_task_scores.csv
        ├── 15_phase_c_summary.py        ─► task_stats.csv, flagged_tasks.csv, aggregates.csv
        ├── 16_model_capability_charts   ─► outputs/charts/*.png
        ├── 18_timing_hypotheses.py      ─► kernel_logs_parsed.csv, efficiency_charts/*.png
        ├── 19_full_stats_pipeline.py    ─► full_task_model_stats.csv, aggregate_stats.csv
        ├── 20_efficiency_analysis.py    ─► efficiency_charts/*.png
        ├── 21_novelty_claims_analysis   ─► novelty_claims/*.csv
        ├── 30_hypothesis_tests.py       ─► hypothesis_tests.json, hypothesis_tests.md
        └── make_writeup_figures.py      ─► outputs/figures/*.png
```

---

## Script Reference

### Core Analysis (Phase B — Item-Level)

#### `04_discriminatory_power.py`
**Item discrimination analysis** (IRT-style).

Computes point-biserial correlation between each task score and the model's category total. Classifies tasks as excellent (r ≥ 0.50), good, marginal, or poor discriminators. Also computes a D-index (top/bottom quartile gap).

**Outputs:** `outputs/discrimination_report.csv`

---

#### `05_cross_category.py`
**Cross-category Spearman correlation matrix.**

Computes rank-based correlations between the six sub-ability scores across models. Tests whether being good at one ability predicts performance on another.

**Outputs:** `outputs/cross_category_correlations.csv`, `outputs/cross_category_pvalues.csv`, `outputs/cross_category_rank_correlations.csv`

---

#### `06_scaling_analysis.py`
**Scaling and tier analysis.**

Checks whether frontier > mid > small consistently, per category. Identifies scale inversions (cases where a smaller model outranks a larger one within the same provider).

**Outputs:** `outputs/tier_stats.csv`, `outputs/tier_inversions.csv`, `outputs/tier_task_gaps.csv`

---

#### `07_efficiency_ablation.py`
**Efficiency scoring ablation.**

Compares composite scores (accuracy × efficiency) against accuracy-only scoring. Quantifies how much the efficiency component changes model rankings.

**Outputs:** `outputs/efficiency_ablation.csv`

---

#### `08_entropy_analysis.py`
**Shannon entropy analysis.**

Computes entropy of score distributions across models per task. High entropy = genuinely discriminating; near-zero entropy = all models score the same (ceiling or floor).

**Outputs:** `outputs/entropy_report.csv`, `outputs/category_entropy.csv`

---

#### `09_provider_analysis.py`
**Provider-level performance breakdown.**

Aggregates scores by provider (Google, OpenAI, Anthropic, Open-source) per category. Tests the Google + OSS vs. Anthropic + OpenAI rule-induction gap.

**Outputs:** `outputs/provider_analysis.csv`, `outputs/provider_overall.csv`, `outputs/provider_pivot.csv`

---

#### `10_bimodality_and_dominance.py`
**Bimodal task classification + Gemini dominance + Qwen thinking comparison.**

Identifies tasks where the score distribution is bimodal (models either solve it or don't). Measures Gemini ceiling dominance. Compares Qwen Thinking vs. Instruct on matched tasks.

**Outputs:** `outputs/bimodal_report.csv`, `outputs/gemini_ceiling.csv`, `outputs/thinking_comparison.csv`, `outputs/interactive_vs_rl.csv`, `outputs/rank1_counts.csv`

---

#### `11_task_removal_sensitivity.py`
**Leave-one-out robustness analysis.**

For every task, removes it and recomputes the leaderboard. Reports maximum rank change and Spearman correlation with the full-task leaderboard. A Spearman > 0.99 means no single task can swing results.

**Outputs:** `outputs/loo_global.csv`, `outputs/loo_category.csv`, `outputs/benchmark_stability_summary.csv`

---

#### `12_random_baseline.py`
**Random baseline and signal-to-noise ratios.**

Computes expected scores from random behavior per category (based on task structure). Computes signal ratio = actual score / random baseline. Verifies the benchmark is above chance.

**Outputs:** `outputs/random_baseline.csv`, `outputs/model_signal_above_random.csv`

---

#### `13_ground_truth_spotcheck.py`
**Ground truth integrity verification.**

Manually spot-checks 15 tasks by re-running the scoring function with held-out test cases and verifying answers match the expected output. Catches scoring bugs and mis-generated answers.

**Outputs:** `outputs/ground_truth_spotcheck.csv`

---

#### `14_epistemic_analysis.py`
**UNKNOWN-answer / epistemic uncertainty analysis.**

Tests whether models score better or worse on tasks with provably unknowable answers (the correct response is "UNKNOWN"). Measures epistemic calibration.

**Outputs:** `outputs/epistemic_analysis.csv`, `outputs/unknown_task_scores.csv`

---

### Phase C — Synthesis

#### `15_phase_c_summary.py`
**Phase C synthesis: combine all per-task signals into a unified quality assessment.**

Merges discrimination, entropy, LOO stability, random baseline, and ground truth signals. Produces the master task quality table and flags tasks by priority for curation review.

**Outputs:** `outputs/task_stats.csv`, `outputs/flagged_tasks.csv`, `outputs/final_flagged_tasks.csv`, `outputs/aggregates.csv`, `outputs/aggregate_stats.csv`, `outputs/category_pivot.csv`, `outputs/model_stats.csv`, `outputs/task_rank1.csv`

---

#### `16_model_capability_charts.py`
**Visualization: radar charts, grouped bar charts, heatmaps.**

Generates capability profile charts for Phase C. Produces before/after curation comparison charts.

**Outputs:** `outputs/charts/fig1_radar_before.png`, `fig2_grouped_bar_before.png`, `fig3_heatmap_before_after.png`, `fig4_radar_after.png`, `fig5_grouped_bar_after.png`, `fig6_rank_change_table.png`, `fig7_retention_recommendation.png`

---

### Kernel Log & Token Analysis

#### `17_download_kernel_logs.py`
**Download per-model run logs from Kaggle kernels.**

Uses the Kaggle API to download `run.json` files containing per-task timing, token counts, and costs. Requires a valid Kaggle API token. Saves to `outputs/kernel_logs/<model>/`.

> ⚠️ **Requires Kaggle API access.** Pre-downloaded logs are in `outputs/kernel_logs/` — most users can skip this script.

**Outputs:** `outputs/kernel_logs/<model>/<task>.run.json`

---

#### `17b_download_from_manifest.py`
**Download from a pre-built manifest file.**

Alternative to `17_download_kernel_logs.py`. Uses `outputs/kernel_logs/manifest.json` to download only specific kernel versions.

---

#### `18_timing_hypotheses.py`
**Token/cost/timing hypotheses (H14–H18).**

Tests five hypotheses about thinking token usage, cost vs. performance, verbosity as a failure signal, cost-per-point efficiency, and efficiency ranking by model. Requires kernel logs in `outputs/kernel_logs/`.

**Key findings tested:**
- H14: Thinking models use more tokens but score higher
- H15: Cost vs. performance correlation
- H16: Token verbosity predicts failure (RL tasks: ρ = −0.53)
- H17: Cost-per-point efficiency ranking
- H18: Evidence-seeking efficiency ≠ token efficiency

**Outputs:** `outputs/kernel_logs_parsed.csv`, `outputs/efficiency_charts/*.png`

---

#### `20_efficiency_analysis.py`
**Token spend vs. score efficiency analysis (extended).**

Full efficiency analysis: scatter plots, frontier models, hard vs. easy tasks, provider verbosity. Extended version of the findings in WRITEUP.md Finding 3.

**Outputs:** `outputs/efficiency_charts/*.png`

---

### Supporting Scripts

#### `19_fetch_task_runs.py`
Downloads individual task run JSON files from Kaggle.

#### `19_full_stats_pipeline.py`
Full statistics pipeline: computes `full_task_model_stats.csv` and `aggregate_stats.csv` from scratch.

**Outputs:** `outputs/full_task_model_stats.csv`, `outputs/aggregate_stats.csv`

---

#### `21_novelty_claims_analysis.py`
**Novelty and benchmark comparison analysis.**

Tests novelty-specific claims: active querying patterns, rule induction vs. other benchmarks, efficiency ablation stability. Produces tables for the novelty section of the writeup.

**Outputs:** `outputs/novelty_claims/*.csv`

---

#### `30_hypothesis_tests.py`
**All formal hypothesis tests.**

Runs the complete set of statistical tests referenced in WRITEUP.md: Spearman correlations, Mann-Whitney U tests, Wilcoxon signed-rank tests, with Benjamini-Hochberg FDR correction throughout. Produces a machine-readable JSON and a human-readable markdown table.

**Outputs:** `outputs/hypothesis_tests.json`, `outputs/hypothesis_tests.md`

---

#### `make_writeup_figures.py`
**Generate the seven publication figures for WRITEUP.md.**

Produces all figures referenced in the writeup. Reads from pre-computed CSVs — no Kaggle API required. The output figures are also committed directly to `figures/` at the repo root.

**Outputs:**
- `outputs/figures/fig_leaderboard_ci.png` — Leaderboard with 95% CIs
- `outputs/figures/fig_radar_profiles.png` — Per-model cognitive profiles
- `outputs/figures/fig_task_difficulty.png` — Task difficulty distribution
- `outputs/figures/fig_thinking_vs_instruct.png` — Thinking vs. instruct
- `outputs/figures/fig_tokens_vs_score.png` — Token spend vs. RL score
- `outputs/figures/fig_trajectory_orthogonal.png` — Procedural trajectory orthogonality
- `outputs/figures/fig_meta_calibration.png` — Evidence-seeking efficiency

---

### Utility Scripts

#### `find_diverse_models.py`
Finds the most diverse subset of models for display purposes (maximize tier/provider spread in limited-slot visualizations).

#### `12_kernel_version_urls.py`
Constructs Kaggle kernel version URLs from model slugs and version IDs. Used to build `outputs/kernel_logs/manifest.json`.

---

## Utility Modules (`../utils/`)

#### `utils/data_loader.py`
Parses Kaggle leaderboard JSONs into a unified long-format DataFrame. Also defines model tier and provider metadata. All analysis scripts import from here via:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix, MODEL_TIERS, MODEL_PROVIDERS
```

#### `utils/stats.py`
Shared statistical helpers: Spearman + bootstrap CI, Mann-Whitney wrapper with effect size, Wilcoxon wrapper, Benjamini-Hochberg correction.

---

## Output Files Index

See [`../outputs/README.md`](../outputs/README.md) for a complete description of every CSV produced by this pipeline.
