# LearningBench Methodology

## Overview

LearningBench is built around a single design principle: **novelty is the entire test**. Every task presents a system that has never existed anywhere — so any correct answer must come from learning within the conversation, not recalling from training data.

---

## Task Construction Pipeline

### 1. Facet Enumeration

Each sub-ability is decomposed into distinct cognitive facets. For example, Reinforcement Learning facets include:
- Blocking (learned irrelevance)
- Spurious correlation rejection
- Action-consequence mapping
- Multi-dimensional reward tracking

For each facet, a task seed is designed.

### 2. Programmatic Expansion

Each seed becomes a programmatic environment: a Python function that generates training examples, manages the interaction protocol, and computes ground truth. This means:
- Every run produces a fresh instance (randomized parameters)
- Ground truth is computed by the same function that generated the task — no lookup tables
- The environment is the judge

### 3. Five-Point Validation Gate

Every candidate task must pass all five criteria. Failing any one means reject, rewrite, or harden — the loop can take several rounds per task:

| Criterion | The question it answers |
|---|---|
| **Human feasibility** | Could an expert human solve it with the same inputs? |
| **Solution uniqueness** | Does exactly one rule fit all training examples *and* the held-out tests? |
| **Logical consistency** | Does the rule produce deterministic, reproducible outputs? |
| **Anti-contamination** | Is the system genuinely invented — no plausible web or training-data trace? |
| **PhD-level difficulty** | Hard enough to require learning — yet solvable by at least one frontier LLM or expert human? |

### 4. Multi-Model Calibration

Every validated candidate is run across all 14 models. Two automatic removal conditions:
- **Too easy**: All 14 models score perfectly → ceiling task, no discrimination
- **Too hard**: No model scores above chance → floor task, measures nothing

This step removed ~95% of all candidates. The 157 survivors form a clean difficulty gradient.

### 5. Four-Phase Analysis and Curation

After the initial benchmark run, results went through a rigorous four-phase analysis:

| Phase | Focus | Outcome |
|---|---|---|
| A | Data extraction and validation | Verified 157 tasks × 14 models |
| B | Item discrimination, entropy, scaling, provider analysis | Identified 145 tasks with at least one signal concern |
| C | LOO robustness, epistemic analysis, ground truth spot-checks | Confirmed benchmark stability (Spearman = 0.997 under LOO) |
| D | Full code inspection of 26 flagged tasks | Removed 19 tasks → **138 final tasks** |

---

## Curation Philosophy

The core principle: **a surprising result is not a reason to remove a task — it may be the finding.**

Four tasks illustrate this:

### `semantic_override`
Gemini Pro and GPT-5.4 score 0.00 while Gemma and Flash score 0.95. Frontier models cannot override semantic priors with evidence. This is not noise — it is a systematic failure of the most capable models on a specific cognitive demand. Retained as a flagship example of **semantic rigidity**.

### `hidden_priority_order`
Only both Qwen variants (thinking + instruct) score 1.00; all 12 other models score ≤ 0.75. This reveals a systematic Qwen advantage on priority-ordering inference. Retained as a **provider-specific capability signal**.

### `manhattan_point`
Gemini Flash-Lite = 1.00, Gemini 2.5 Flash = 0.15, Gemini 3.1 Pro = 0.00 on the same task. Non-monotonic capability within a model family. Retained as evidence that **larger is not always better on inductive reasoning**.

### `blocking_effect`
Anthropic and OpenAI models score 1.0; Gemini Pro scores 0.5. Kept despite negative discrimination because it measures Gemini's specific epistemic calibration blind spot — a real and reproducible finding.

---

## Removal Criteria

A task was removed only if it had one or more of:

| Category | Description | Count |
|---|---|---|
| Design flaw | The task logic does not correctly isolate the intended ability | 4 |
| Budget infeasibility | The interaction budget is structurally insufficient to solve the task | 6 |
| Prior-knowledge contamination | The rule can be recalled from training data | 2 |
| Noise-induced bimodality | Score distribution is bimodal due to random seed variance, not genuine ability difference | 3 |
| Bimodal collapse | The task has completely collapsed to near-uniform low performance | 4 |
| **Total removed** | | **19** |

---

## Scoring Design

### Shared Primitives

**Free exploration zone**: Each interactive task has a minimum evidence threshold — the number of examples structurally required to distinguish the correct rule from all plausible alternatives. Efficiency is only penalized *after* this threshold. A model is not punished for consuming the minimum necessary evidence.

**Zero-accuracy floor**: If a model achieves zero accuracy, the composite score is 0.0 regardless of efficiency. A model cannot score points by being fast at being wrong.

### Per-Sub-Ability Formulas

See [`SCORING.md`](../SCORING.md) for the complete technical specification.

| Sub-ability | Formula | Key design choice |
|---|---|---|
| Associative | `correct / total` | Pure accuracy — no efficiency component (single-turn) |
| Concept Formation | `accuracy × (0.40 + 0.60 × efficiency)` | Meta-calibration: knowing when you've seen enough |
| Language Learning | `accuracy × (0.40 + 0.60 × efficiency)` | Same formula; efficiency = probe ratio below threshold |
| Observational | `fraction of sequences fully correct` | No efficiency — all information is given at once |
| Procedural | `0.30×transfer + 0.25×asymptote + 0.25×trajectory + 0.20×consistency` | **Trajectory (OLS slope)** = the only direct learning-occurred measure |
| Reinforcement | `0.55×solved + 0.25×efficiency + 0.20×progress` | Partial progress credited; efficiency tracks action budget |

### Why Trajectory Is Special

The procedural scoring formula includes an OLS slope component that measures whether accuracy *improved* across practice rounds — independently of the final level. Two models with identical asymptotic accuracy but different trajectories receive different scores.

Across 112 (model, task) procedural runs: **Spearman ρ(slope, asymptote) = −0.02** — the trajectory signal is 99% orthogonal to the final score. This confirms the slope captures a genuinely distinct cognitive dimension: not what level you reached, but whether you climbed.

---

## Reproducibility

All scoring is deterministic and self-contained:
- Task environments are seeded with fixed random seeds for reproducibility
- Scoring functions are pure Python with no external dependencies
- Pre-computed score matrices are committed to the repository — no re-running required for analysis

See the [main README](../README.md) for how to reproduce the full analysis.
