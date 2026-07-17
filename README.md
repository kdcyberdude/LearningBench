# LearningBench

**Measuring Inference-Time Learning in LLMs** — [Kaggle Benchmark](https://www.kaggle.com/benchmarks/kdcyberdude/learningbench) · [Project Page](https://learningbench-project-page-918170344855.us-west1.run.app)

Existing benchmarks measure what models already *know*. **LearningBench measures how they learn** — from scratch, inside a single conversation, on systems that have never existed before. No memorisation can help. Every correct answer must be derived from in-context evidence alone.

---

## Contents

1. [What We Measure](#1-what-we-measure)
2. [Task Design](#2-task-design)
3. [Benchmark Curation](#3-benchmark-curation)
4. [Scoring](#4-scoring)
5. [Results](#5-results)
6. [Validity & Robustness](#6-validity--robustness)
7. [Reproduce](#7-reproduce)
8. [Data Collection](#8-data-collection)
9. [Models Evaluated](#9-models-evaluated)
10. [Repository Structure](#10-repository-structure)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. What We Measure

LearningBench covers **six cognitive sub-abilities of inference-time learning**, each targeting a distinct cognitive act across **135 tasks**:


| Sub-ability                                                                                        | Tasks | Protocol      | What it isolates                                                             |
| -------------------------------------------------------------------------------------------------- | ----- | ------------- | ---------------------------------------------------------------------------- |
| **[Associative Learning](https://www.kaggle.com/benchmarks/kdcyberdude/associativelearning/)**     | 17    | Single-turn   | Causal inference vs. correlation — blocking, spurious cues, epistemic limits |
| **[Concept Formation](https://www.kaggle.com/benchmarks/kdcyberdude/conceptlearning/)**            | 18    | Interactive   | Meta-calibration: *does the model know when it has seen enough?*             |
| **[Language Learning](https://www.kaggle.com/benchmarks/kdcyberdude/languagelearning/)**           | 26    | Interactive   | Productive rule induction — wug-test on invented phonologies                 |
| **[Observational Learning](https://www.kaggle.com/benchmarks/kdcyberdude/observationallearning/)** | 30    | Single-shot   | Structural inference from demonstrated behavior alone                        |
| **[Procedural Learning](https://www.kaggle.com/benchmarks/kdcyberdude/procedurallearningbench)**   | 11    | Multi-episode | Learning *trajectory*: did performance improve with practice?                |
| **[Reinforcement Learning](https://www.kaggle.com/benchmarks/kdcyberdude/reinforcementlearning/)** | 30    | Multi-turn    | Hypothesis updating from feedback under a finite action budget               |


**Three interaction protocols** cover distinct failure modes that no single protocol can expose:


| Protocol                  | Sub-abilities              | What it stresses                                              |
| ------------------------- | -------------------------- | ------------------------------------------------------------- |
| Single-turn / single-shot | Associative, Observational | Inductive inference without the ability to seek more evidence |
| Interactive               | Concept, Language          | Calibration: knowing when to stop requesting examples         |
| Multi-turn / episodic     | RL, Procedural             | Hypothesis generation, update, and abandonment under feedback |


**Domain coverage** spans 6 broad domains — mathematical structures, linguistics, formal systems, cryptography and computation, causal and strategic reasoning, and dynamical systems — ensuring no subject-area familiarity can substitute for genuine in-context learning. Full task catalog: `[sub_benchmarks/](sub_benchmarks/)`.

---

## 2. Task Design

### 2.1 The Novelty Constraint

Every task presents a system that does not exist anywhere. Invented languages with phonologies built from scratch, hidden Boolean circuits with randomised wiring, physics with alien damping constants, counterintuitive causal structures. These models have consumed the internet — any rule they can recall is not one they had to learn. **Novelty is the entire benchmark.**

### 2.2 Construction Pipeline

**Facet enumeration →** Each sub-ability is decomposed into distinct cognitive facets. A task seed is designed per facet.

**Programmatic expansion →** Each seed becomes a Python environment that generates training examples, manages the interaction protocol, and computes ground truth at runtime. No lookup tables. The grader *is* the rule.

**Five-point validation gate →** Every candidate must pass all five criteria. Failing any one means reject, rewrite, or harden — the loop runs several rounds per task:


| Criterion                | The question it answers                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| **Human feasibility**    | Could an expert human solve it with the same inputs?                                         |
| **Solution uniqueness**  | Does exactly one rule fit all training examples *and* held-out tests?                        |
| **Logical consistency**  | Does the rule produce deterministic, reproducible outputs?                                   |
| **Anti-contamination**   | Is the system genuinely invented — no plausible training-data trace?                         |
| **PhD-level difficulty** | Hard enough to require learning — yet solvable by at least one frontier LLM or expert human? |


Task construction pipeline

### 2.3 Difficulty Calibration

Each candidate task is evaluated on a pilot set of models during development. Tasks where every model scores perfectly (ceiling) or no model scores above chance (floor) are rewritten or discarded. This loop ran through a large number of candidates — roughly 95% were rejected before reaching the final 135. The survivors produce a clean difficulty gradient: no task trivially solved by all models, and none unsolved by all.

Task difficulty distribution

---

## 3. Benchmark Curation

The 135 tasks in this repository are the final, curated benchmark. After the initial benchmark run across all 14 models, results underwent a systematic four-phase analysis to verify that each retained task is measuring what it claims to measure.


| Phase | Focus                                                                 | Outcome                                                         |
| ----- | --------------------------------------------------------------------- | --------------------------------------------------------------- |
| **A** | Data extraction and validation                                        | Verified scores across all 14 models                            |
| **B** | Item discrimination, score entropy, scaling analysis                  | Identified tasks with potential signal issues                   |
| **C** | Leave-one-out stability, ground-truth spot-checks, epistemic analysis | Confirmed ranking stability (Spearman ρ = 0.997 under LOO)      |
| **D** | Full Python source inspection of flagged tasks                        | Removed tasks not suitable for evaluation → **135 final tasks** |


Tasks were removed when code inspection revealed they were unsuitable for evaluating learning — not because of surprising results. A surprising result is not a removal reason; it may be the finding. Four tasks initially flagged as outliers were retained after inspection confirmed they capture genuine and reproducible behavioral signals (see §6.3).

---

## 4. Scoring

### 4.1 Shared Primitives

Two rules apply to every scoring formula across all six sub-abilities:

- **Free exploration zone** — efficiency penalties begin only *after* the minimum evidence structurally required to see the pattern. Only avoidable over-querying is penalised.
- **Zero-accuracy floor** — every efficiency-weighted task returns 0.0 if accuracy is zero. No reward for being fast at being wrong.

### 4.2 Per-Sub-Ability Formulas


| Sub-ability       | Formula                                                               | Efficiency component?                            |
| ----------------- | --------------------------------------------------------------------- | ------------------------------------------------ |
| Associative       | `correct / total_questions`                                           | No (single-turn, model cannot request more data) |
| Concept Formation | `accuracy × (0.40 + 0.60 × efficiency)`                               | Yes — active retrieval under probe budget        |
| Language Learning | `accuracy × (0.40 + 0.60 × efficiency)`                               | Yes — same formula; exact surface-form matching  |
| Observational     | `fraction of sequences fully correct`                                 | No (all demonstrations provided at once)         |
| Procedural        | `0.30×transfer + 0.25×asymptote + 0.25×trajectory + 0.20×consistency` | Yes — within each practice round                 |
| Reinforcement     | `0.55×solved + 0.25×efficiency + 0.20×progress`                       | Yes — step budget with free exploration zone     |


**Why the Procedural trajectory component is unique.** The `trajectory` term is the OLS slope of accuracy across practice rounds, normalised to [0, 1]. It is the only component in any major benchmark that directly measures *whether learning occurred* — independently of the final level reached. Two models ending at the same asymptote can have opposite slopes.

Across 112 (model, task) procedural runs: **Spearman ρ(slope, asymptote) = −0.02** — the trajectory signal is 99% orthogonal to the final score.

Trajectory orthogonality

*Two runs landing at the same final score (x ≈ 0.5) have slopes of +0.18 and −0.30 — one model is still climbing, the other is falling back. Same destination, opposite journeys.*

Full technical specification for all formulas: `[SCORING.md](SCORING.md)`.

---

## 5. Results

> 14 models · 135 tasks · every claim below is confirmed with formal hypothesis tests (Spearman, Mann-Whitney, Wilcoxon), 10,000-sample bootstrap 95% CIs, and Benjamini-Hochberg FDR correction. Full test output: `[analysis/outputs/hypothesis_tests.md](analysis/outputs/hypothesis_tests.md)`.

### 5.1 Leaderboard

- Only **Gemini 3.1 Pro Preview** clears 0.70 (scores 0.85)
- **11 of 14 models** score below 0.50 — scale alone does not close this gap; a small model with extended reasoning outperforms a larger model without it on every induction-heavy learning sub-ability
- **GLM-5 (open-source)** ranks #2, outranking every closed-source lab except Google
- **Google + open-source** score +40.8% higher on rule induction than Anthropic + OpenAI (Mann-Whitney p = 0.029, Cliff's δ = 0.71, large effect)

### 5.2 Finding 1 — Larger models are not better learners

**Qwen 3 Next 80B Thinking vs. Instruct** comparison makes this concrete: enabling extended reasoning lifts Concept Formation by 183% (0.191 → 0.541), Observational by 91%, and RL by 77%. A small model with extended reasoning outperforms a larger model without it on every induction-heavy sub-ability.

- Across 132 matched tasks: Thinking wins **87**, Instruct wins **19**, ties **26**
- Largest gains on induction-heavy abilities: **Observational +0.43, Concept +0.38, RL +0.30** (all p ≤ 0.001)
- Apparent dip on **Procedural −0.11 (p = 0.55, not significant)** — when feedback rounds arrive in rapid succession, extended deliberation per round may blunt the iteration loop itself

### 5.3 Finding 2 — The best learners need the least evidence

The best learners need the least evidence. Across 403 interactive runs (Concept Formation + Language Learning), models requesting fewer examples score higher (ρ = **−0.52**, p < 10⁻²⁸, n = 403). Evidence appetite spans 1.8 to 11.9 avg examples — a **6.5× spread** across the 14 models.

**Insight:** Models do not fail because the tasks are too hard — they fail because they do not know when to stop asking. The well-calibrated models commit earlier and more accurately; the underconfident ones exhaust their probe budget without reaching a better hypothesis. Claude Opus is the sharpest anomaly: it ranks #4 overall yet probes as aggressively as the weakest models, held up by strength on other sub-abilities.

**A deeper pattern:** This evidence-seeking style is not a response to domain difficulty — it is a fixed property of the model. Across Concept Formation and Language Learning (structurally different tasks), a model's mean probe count correlates strongly across both (Spearman ρ = 0.793, p = 0.0007). A model that over-probes on invented Boolean rules also over-probes on invented phonological rules. Evidence appetite is not calibrated to the task — it travels with the model.

Evidence-seeking efficiency scatter

*The underconfident model is the one in deployment that keeps asking "can you clarify?" instead of just answering.*

### 5.4 Finding 3 — Token spend is a failure signal, not a success signal

Across 397 RL runs: ρ(tokens, score) = **−0.53** (p < 10⁻³⁰). Tokens consumed track not just outcome but descent into failure:


| Score quartile      | Avg tokens consumed |
| ------------------- | ------------------- |
| Q4 — highest scores | 10,434              |
| Q3                  | 68,409              |
| Q2                  | 104,895             |
| Q1 — lowest scores  | 178,754             |


- **Solved runs:** 41K tokens average
- **Failed runs:** 177K tokens average — a **4.3× gap** (Cliff's δ = −0.60)
- **43 runs** show ≥10 consecutive identical actions — when the first hypothesis is wrong, many models cannot update at all

**Insight:** Token spend is not merely correlated with failure — it monotonically tracks how deeply a model is stuck. A model that cannot update its hypothesis does not stop; it repeats. This is observable in real time: a production monitor can flag likely failures **before** the wrong answer returns.

Token spend vs. score

### 5.5 Conclusion — Three axes separate genuine learners from pattern matchers

The three findings converge on the same three axes:


| Axis            | What fails                                       | Benchmark signal                                                            |
| --------------- | ------------------------------------------------ | --------------------------------------------------------------------------- |
| **Generation**  | Cannot form a new hypothesis from novel evidence | Concept/Observational scores                                                |
| **Sufficiency** | Cannot judge when evidence is enough to commit   | Evidence appetite ρ = −0.52, 6.5× spread (H1)                               |
| **Update**      | Cannot revise a wrong guess under feedback       | Token-score ρ = −0.53, 4.3× solved/failed gap, repeat-streak count (H2, H8) |


These three axes had not been simultaneously isolated in any existing LLM benchmark before LearningBench.

---

## 6. Validity & Robustness

### 6.1 Benchmark Integrity

*"How do I know the score differences between models are real and not noise?"* Two checks cover the three ways a benchmark can fail.

**1 — Are models actually learning, or just guessing?**

The *random baseline* is the score you'd get by answering uniformly at random — pure chance, zero learning. It comes from each task's answer format: ~3 choices on Associative tasks gives a floor of 0.333; exact string production on Language/Observational gives ≈0.010. The *signal ratio* (best model ÷ random) and *spread ratio* (best model ÷ worst model) together show the benchmark contains real signal and meaningfully separates strong from weak learners.


| Sub-ability   | Random baseline | Worst model | Best model | Signal ratio | Spread ratio |
| ------------- | --------------- | ----------- | ---------- | ------------ | ------------ |
| Associative   | 0.333           | 0.43        | 0.95       | 2.8×         | 2.2×         |
| Concept       | 0.020           | 0.15        | 0.80       | 40.0×        | 5.4×         |
| Language      | 0.010           | 0.25        | 0.78       | 78.2×        | 3.1×         |
| Observational | 0.010           | 0.14        | 0.85       | 85.3×        | 5.9×         |
| RL            | 0.020           | 0.25        | 0.93       | 46.5×        | 3.7×         |


*Signal ratio* = best model ÷ random — confirms the benchmark is not noise. *Spread ratio* = best ÷ worst — confirms the benchmark discriminates between models. Note that every worst-model score is well above the random floor, meaning even the weakest models are learning — they are just learning much less effectively. Associative's lower signal ratio (2.8×) is an artefact of its 3-choice format raising the mathematical floor, not a weakness in the task signal; its absolute top score of 0.95 is the highest of any sub-ability.

These baselines are mathematical (derived from answer-space size). They are equivalent to a model's true prior only if the novelty constraint holds — every task uses invented systems absent from training data, so a model with zero examples has nothing to draw on. This assumption is inherent to the design rather than verified by a separate zero-shot empirical run.

**2 — Is the grader correct?**

The same function that generates training examples also grades responses — no static answer keys that could be wrong. 15 tasks spot-checked manually: 15/15 confirmed. RL tasks use seeded generation — correct by construction.

### 6.2 Leaderboard Stability

**Leave-one-out across 135 tasks:** mean Spearman ρ with full ranking = **0.9985**, minimum = 0.9912. Maximum rank change from removing any single task: **1 position**. No reviewer can argue the leaderboard is driven by a handful of cherry-picked items.

**Efficiency component ablation:** removing efficiency scoring from the formula changes maximum rank by **0 positions**. Ranking is determined by accuracy; efficiency amplifies differentiation at the top without reordering it.

### 6.3 Anomalous Findings (Retained)

Four tasks initially flagged for removal were confirmed as genuine signals after code inspection:

`**semantic_override`** — Gemini Pro = 0.00, GPT-5.4 = 0.00. Gemini 2.5 Flash = 0.95, Gemma 4 26B = 0.90. Frontier models cannot override semantic priors with in-context evidence. Larger models are worse precisely because they have stronger priors. Retained as a flagship example of **semantic rigidity under RLHF**.

`**hidden_priority_order`** — Qwen Thinking = 1.00, Qwen Instruct = 1.00. All 12 other models: ≤ 0.75, 11 scoring 0.00. Reveals a systematic Qwen advantage on priority-ordering inference — a **provider-specific capability signal**.

`**manhattan_point`** — Gemini Flash-Lite = 1.00, Gemini 2.5 Flash = 0.15, Gemini Pro = 0.00 on the same task. Non-monotonic capability within a model family. Evidence that **larger is not always better on inductive reasoning**.

`**blocking_effect`** — Claude Opus and GPT-5.4 score 1.0; Gemini Pro scores 0.5. A pure epistemic trap where every correct answer is UNKNOWN. Retained as a measure of Gemini's specific calibration blind spot — **real and reproducible**.

### 6.4 Sub-ability Independence

The sub-abilities are distinct but correlated. Cross-category Spearman ρ ranges from 0.55 to 0.89. PCA PC1 explains **81%** of category variance — a strong general learning factor exists, but not unity. The remaining 19% means sub-ability profiles are not reducible to one number.

Model profile inversions confirm non-monolithic structure:

- Gemma 4 26B ranks **#5 on RL** but **#12 on Language**
- GPT-5.4 ranks **#4 on Language** but **#9 on Concept**
- DeepSeek V3.2 ranks **#4 on Observational** but **#12 on Concept**

If the benchmark were measuring one thing, these inversions would not exist.

### 6.5 What LearningBench Uniquely Captures


| Measurement                                                              | Status in existing benchmarks     | LearningBench                                                   |
| ------------------------------------------------------------------------ | --------------------------------- | --------------------------------------------------------------- |
| Novel rule inference from examples                                       | Rare (ARC-AGI, BIG-Bench subsets) | 135 tasks × 3 protocols                                         |
| Does score improve across practice rounds?                               | **Not in any major benchmark**    | Procedural trajectory slope (99% orthogonal to final score)     |
| How many examples does the model request before committing?              | **Not measured anywhere**         | Evidence appetite per model (6.5× spread, ρ = −0.52 with score) |
| Does token spend predict failure in real time?                           | **Not measured**                  | RL token-score ρ = −0.53, 4.3× gap solved vs. failed            |
| Can the model override a strong semantic prior with in-context evidence? | **Not measured**                  | `semantic_override` — frontier models score 0.00                |
| Stability under task removal                                             | Varies                            | LOO Spearman ρ = 0.9985, max rank shift = 1                     |


---

## 7. Reproduce

### Prerequisites

```bash
pip install -r requirements.txt
```

### Option A — Pre-computed outputs (recommended)

All analysis outputs are in `analysis/outputs/`. Load directly:

```python
import pandas as pd

# Curated score matrix (135 tasks, 14 models)
df = pd.read_csv("analysis/outputs/score_matrix_phase_d.csv")

# Per-model statistics
model_stats = pd.read_csv("analysis/outputs/model_stats.csv")

# Formal hypothesis test results
import json
tests = json.load(open("analysis/outputs/hypothesis_tests.json"))
```

### Option B — Full analysis pipeline

Scripts are numbered in dependency order. Run from repo root:

```bash
cd analysis/scripts

# Core analysis (item discrimination, cross-category, scaling, provider)
python 04_discriminatory_power.py
python 05_cross_category.py
python 06_scaling_analysis.py
python 09_provider_analysis.py

# Robustness (LOO stability, random baselines, ground truth, epistemic)
python 11_task_removal_sensitivity.py
python 12_random_baseline.py
python 13_ground_truth_spotcheck.py
python 14_epistemic_analysis.py

# Formal hypothesis tests (H1–H8)
python 30_hypothesis_tests.py

# Regenerate all figures
python make_writeup_figures.py
```

See `[analysis/scripts/README.md](analysis/scripts/README.md)` for the full script catalog.

### Option C — Download raw task files from Kaggle

```bash
python scripts/download_tasks.py
```

Requires a [Kaggle API token](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`.

---

## 8. Data Collection for Benchmark Robustness evaluation

### Where the data comes from

Every task in the benchmark runs as a **Kaggle notebook kernel**. When a model evaluation completes, Kaggle stores two artefacts per `(task, model)` pair:

1. **Aggregated metrics** — score, token counts (input / output / thinking), inference latency, cost. These are accessible via Kaggle's internal benchmark API.
2. **Full conversation log** — the complete multi-turn exchange between the task environment and the model, stored as a `.run.json` sidecar file attached to the notebook output. This includes every request and response turn, assertion results, and a human-readable stdout trace showing per-item correctness and the final score.

These two artefacts are the raw data for all analysis in this repository.

### Why browser session cookies are required

Kaggle exposes a **public REST API** (`/api/v1/`) authenticated with an API key. That API does not provide access to benchmark task run metrics or notebook output files. The data lives behind Kaggle's **internal API** (`/api/i/`), which is only accessible with a live browser session — the same cookies a logged-in user carries when navigating kaggle.com.

Two specific internal endpoints are used:


| Endpoint                                        | What it returns                                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `BenchmarkTaskRunService/ListBenchmarkTaskRuns` | Aggregated metrics for all 14 models on a given task (scores, tokens, cost, latency, session ID) |
| `KernelsService/GetKernelViewModel`             | Signed CDN download URLs for the stdout log and `.run.json` conversation file                    |


Both require `XSRF-TOKEN`, `ka_sessionid`, `__Host-KAGGLEID`, and several supporting cookies extracted from a live browser session. The scripts use `browser-cookie3` to extract these automatically from Brave or Chrome.

### Fetching the data

```bash
# Step 1 — extract browser session cookies and fetch all metrics
python3 analysis/scripts/19_fetch_task_runs.py --refresh-cookies --workers 6
# → analysis/outputs/task_runs/all_task_runs.csv  (~135 tasks × 14 models)

# Step 2 — fetch stdout traces and full conversation run.json files
python3 analysis/scripts/20_fetch_notebook_logs.py --workers 2
# → analysis/outputs/notebook_logs/<task-slug>/<model>.json
```

Script 20 must run after script 19 (it reads the `output_kernel_session_id` column produced by script 19). Keep `--workers 2` for script 20 — higher concurrency triggers Kaggle's reCAPTCHA.

Session cookies are cached in `.kaggle_session.json` (gitignored — never commit). Re-run with `--refresh-cookies` when the session expires (typically after a few days). Full details and troubleshooting: `[docs/KERNEL_LOG_DOWNLOAD.md](docs/KERNEL_LOG_DOWNLOAD.md)`.

### What a single data record looks like

Each `(task, model)` record in `analysis/outputs/notebook_logs/` contains:

```json
{
  "task_slug": "spurious-hue-true-edge-assoc-learning",
  "model_display_name": "Gemini 3.1 Pro Preview",
  "score_fraction": "1.0",
  "input_tokens": "517",
  "output_tokens": "330",
  "thinking_tokens": "0",
  "total_latency_ms": "30679",
  "cost_usd": "0.022454",
  "stdout_log": "=== spurious_hue_true_edge ... SCORE: 1.0000 ===",
  "run_json": {
    "conversations": [
      { "requests": [{ "contents": [
        { "role": "CONTENT_ROLE_USER", "parts": [{ "text": "Items are classified..." }] },
        { "role": "CONTENT_ROLE_ASSISTANT", "parts": [{ "text": "GROUP_X\nGROUP_Y..." }] }
      ]}]}
    ],
    "assertions": [ ... ]
  },
  "notebook_url": "https://www.kaggle.com/code/kdcyberdude/spurious-hue-true-edge-assoc-learning?scriptVersionId=310210849"
}
```

The `run_json.conversations` field is the complete evidence for every analysis claim about model behavior in this repository.

---

## 9. Models Evaluated


| Model                         | Provider    | Overall Score |
| ----------------------------- | ----------- | ------------- |
| Gemini 3.1 Pro Preview        | Google      | 0.851         |
| GLM-5                         | Open-source | 0.692         |
| Qwen 3 Next 80B Thinking      | Open-source | 0.606         |
| Claude Opus 4.6               | Anthropic   | 0.508         |
| Gemini 2.5 Flash              | Google      | 0.507         |
| GPT-5.4                       | OpenAI      | 0.493         |
| Claude Sonnet 4.6             | Anthropic   | 0.466         |
| Gemini 3.1 Flash-Lite Preview | Google      | 0.458         |
| DeepSeek V3.2                 | Open-source | 0.432         |
| Claude Haiku 4.5              | Anthropic   | 0.384         |
| GPT-5.4 mini                  | OpenAI      | 0.362         |
| Gemma 4 26B A4B               | Google      | 0.346         |
| Qwen 3 Next 80B Instruct      | Open-source | 0.340         |
| GPT-5.4 nano                  | OpenAI      | 0.246         |


---

## 10. Repository Structure

```
learning_eval/
│
├── README.md                        ← This file — full benchmark guide
├── WRITEUP.md                       ← Kaggle competition writeup (narrative)
├── SCORING.md                       ← Complete scoring formulas (technical spec)
├── LICENSE                          ← CC BY 4.0
├── requirements.txt
│
├── figures/                         ← All publication figures (PNG)
│   ├── fig_leaderboard_ci.png
│   ├── fig_radar_profiles.png
│   ├── fig_task_difficulty.png
│   ├── fig_thinking_vs_instruct.png
│   ├── fig_tokens_vs_score.png
│   ├── fig_trajectory_orthogonal.png
│   └── fig_meta_calibration.png
│
├── sub_benchmarks/                  ← Per-sub-ability task documentation
│   ├── learningbench_details.md     ← Overall benchmark design (Kaggle-facing)
│   ├── associative_learning.md
│   ├── concept_formation.md
│   ├── language_learning.md
│   ├── observational_learning.md
│   ├── procedural_learning.md
│   └── reinforcement_learning.md
│
├── leaderboard/                     ← Leaderboard CSVs
│   ├── leaderboard_flat.csv
│   ├── leaderboard_model_ranks.csv
│   └── leaderboard_score_matrix.csv
│
├── analysis/
│   ├── scripts/                     ← Analysis pipeline (numbered in run order)
│   │   └── README.md                ← Script catalog
│   ├── outputs/                     ← All generated CSVs and figures
│   │   ├── score_matrix_phase_d.csv ← Primary data: 135 tasks × 14 models
│   │   ├── hypothesis_tests.json    ← All formal test results
│   │   ├── hypothesis_tests.md      ← Human-readable test report
│   │   └── README.md                ← Full output file catalog
│   └── PHASE_D_INSIGHTS.md          ← Curation decisions (Phase D)
│
├── downloaded_tasks/                ← Task source files by category
└── docs/internal/                   ← Working documents (not needed to reproduce)
```

---

## 11. Citation

```bibtex
@misc{learningbench2026,
  title   = {{LearningBench}: Measuring Inference-Time Learning in {LLMs}},
  author  = {Singh, Karandeep},
  year    = {2026},
  url     = {https://github.com/kdcyberdude/learning_eval},
  note    = {Treow Intelligence}
}
```

**References**

1. Chollet, F. (2019). *On the Measure of Intelligence*. arXiv:1911.01547.
2. Srivastava, A. et al. (2022). *Beyond the Imitation Game (BIG-Bench)*. TMLR.
3. Chollet, F. et al. (2025). *ARC-AGI-2*. ARC Prize Foundation.
4. Morris, J. et al. (2026). *Measuring Progress Toward AGI: A Cognitive Framework*. Google DeepMind.

---

## 12. License

This work is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). See [`LICENSE`](LICENSE) for the full text.

