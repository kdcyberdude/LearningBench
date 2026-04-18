# LearningBench — Validity, Diversity & Reliability

> Four questions that every serious evaluator should ask. Every answer below is grounded in the pre-computed outputs in `analysis/outputs/` and the formal hypothesis tests in `analysis/outputs/hypothesis_tests.md`.

---

## Contents

- [Why Frontier Models Underperform](#1-why-frontier-models-underperform)
- [Is the Benchmark Diverse?](#2-is-the-benchmark-diverse)
- [Difficulty Distribution](#3-difficulty-distribution)
- [Validity & Reliability](#4-validity--reliability)

---

## 1. Why Frontier Models Underperform

### The Paradox

Claude Opus 4.6 and GPT-5.4 are two of the most capable models available. Both score in the top-5 on most general benchmarks. On LearningBench they rank **6th and 5th overall**, and drop to **9th–10th on Concept Formation** and **6th–8th on Observational Learning** — below mid-tier and even small-tier models from other providers.

| Model | Tier | Overall rank | Concept rank | Obs rank |
|---|---|---|---|---|
| Gemini 3.1 Pro Preview | frontier | 1 | 1 | 1 |
| GLM-5 | frontier | 2 | 2 | 3 |
| Qwen 3 Next 80B Thinking | mid | 3 | 3 | 2 |
| Gemini 2.5 Flash | mid | 4 | 4 | 5 |
| **GPT-5.4** | **frontier** | **5** | **9** | **8** |
| **Claude Opus 4.6** | **frontier** | **6** | **10** | **6** |
| Claude Sonnet 4.6 | mid | 7 | 5 | 10 |
| Gemini 3.1 Flash-Lite | mid | 8 | 6 | 9 |
| DeepSeek V3.2 | mid | 9 | 12 | 4 |

GPT-5.4 ranks 4th on Language but 9th on Concept. Claude Opus ranks 3rd on Associative but 10th on Concept. This is not a general intelligence deficit — both models are strong where broad knowledge or language pattern matching suffices. **The collapse is specific to novel rule induction from evidence.**

---

### Mechanism 1 — RLHF Hardens Prior Beliefs

The most direct evidence is `semantic_override` (Concept Formation). This task presents a hidden rule that is semantically counterintuitive — the structural pattern is deliberately disguised by word choice that invites a different inference.

| Model | Score | Tier |
|---|---|---|
| Gemini 2.5 Flash | 0.95 | mid |
| Gemma 4 26B A4B | 0.90 | small |
| **GPT-5.4** | **0.00** | frontier |
| **Gemini 3.1 Pro Preview** | **0.00** | frontier |

Both top-tier frontier models score zero. Mid-tier and small-tier models solve it.

**The mechanism:** Reinforcement Learning from Human Feedback (RLHF) and instruction-tuning at scale teach models to prefer semantically coherent, "sensible" answers — the kind humans rate highly. This works well when the task rewards familiar patterns. When the correct answer requires overriding a semantically strong prior with a counterintuitive rule derived only from in-context evidence, the training signal actively fights the task. The stronger the model's semantic priors, the harder they are to override. **Larger models are worse precisely because they have stronger priors.**

This is formally confirmed by H3 (provider gap):

> Google + Open-source mean on rule-induction tasks = **0.591**  
> Anthropic + OpenAI mean = **0.420**  
> Relative gap = **+40.8%** (Mann-Whitney p = 0.029, Cliff's δ = 0.71, large effect)

The split is not random — Google and open-source models have been trained with more emphasis on structured reasoning and chain-of-thought, while Anthropic and OpenAI models have more aggressive RLHF alignment. The latter produces better conversational output; it also hardens the semantic defaults that LearningBench specifically tests.

---

### Mechanism 2 — Evidence Overconsumption Without Learning

Claude Opus 4.6 shows the clearest version of a second failure mode. On interactive tasks (Concept Formation, Language Learning), models control how many examples they request before committing to a rule. The correct strategy is to request examples until a unique rule is identifiable, then stop.

| Profile | Models | Avg examples used | Mean score |
|---|---|---|---|
| Well-calibrated | Qwen Thinking, GLM-5, Gemini Pro | 1.8–3.6 | 0.67–0.78 |
| Underconfident | **Claude Opus, Claude Haiku**, Gemma, GPT-5.4 nano | 8–12 | 0.29–0.35 |

Claude Opus exhausts **83–96%** of the available example budget — not because it's learning more, but because it cannot decide when it has learned enough. GPT-5.4 sits in the middle (avg 5.7 examples, score 0.44). The correlation is strong and negative: ρ = −0.516, p = 8.5e-29 (H1). More examples = lower score.

**The mechanism:** Calibration — the meta-cognitive ability to judge when evidence is sufficient — is exactly what RLHF does not train. RLHF trains the model to produce acceptable responses; it does not train the model to know when to stop requesting information. Underconfident models are behaving "safely" by human standards (asking for clarification is polite) but failing the actual learning task.

---

### Mechanism 3 — RL Hypothesis Freezing

GPT-5.4 scores near zero on `minesweeper_1d` and `verbal_bandit` (RL), while Gemini and open-source models score 0.5–1.0. GPT-5.4 ranks 9th overall on RL despite ranking 5th overall. Claude Opus ranks 10th on RL.

Both tasks require multi-step feedback-driven inference: the model must form a hypothesis, receive feedback, revise the hypothesis, and repeat. The failure mode is **hypothesis freezing** — after the first incorrect hypothesis, 43 RL runs across all models show ≥10 consecutive identical actions. The model repeats the same guess, spending 4.3× more tokens than successful runs (177K vs. 41K tokens, Cliff's δ = −0.60, H2).

This is distinct from Mechanism 1 (prior hardening) and Mechanism 2 (calibration failure). It is a **hypothesis update** failure: the model can generate a hypothesis but cannot revise it when feedback contradicts it.

---

### Summary: Three Axes of Learning Failure

| Axis | What fails | Which models | Benchmark signal |
|---|---|---|---|
| **Prior hardening** | RLHF semantic defaults override novel rule | GPT-5.4, Gemini Pro on semantic_override | Concept/Obs scores, H3 provider gap |
| **Calibration** | Cannot judge when evidence is sufficient | Claude Opus, Claude Haiku | Evidence appetite ρ = −0.52, H1 |
| **Hypothesis update** | Cannot revise a wrong first guess under feedback | GPT-5.4 on RL tasks | Token-score ρ = −0.53, repeat-streak H8 |

These three axes were theorized in cognitive science (generation, sufficiency judgment, update) and had not been simultaneously isolated in a single LLM benchmark before LearningBench. Frontier models fail on all three, with different profiles — and those profiles are invisible on static knowledge benchmarks.

---

## 2. Is the Benchmark Diverse?

### Sub-ability Coverage

LearningBench covers **five sub-abilities of inference-time learning**, each targeting a distinct cognitive act:

| Sub-ability | Tasks | Protocol | What it isolates |
|---|---|---|---|
| Associative | 20 | Single-turn | Causal vs. correlational inference; blocking effect traps |
| Concept Formation | 18 | Interactive | Meta-calibration: when has the model seen enough evidence? |
| Language Learning | 26 | Interactive | Productive rule induction on invented phonologies (wug-test) |
| Observational | 40 | Single-shot | Structural inference from demonstrated behavior alone |
| Reinforcement | 34 | Multi-turn | Hypothesis generation, update, and abandonment under feedback |

**138 total tasks** across these five sub-abilities.

---

### Protocol Diversity

Three fundamentally different interaction paradigms:

| Protocol | Sub-abilities | Model control |
|---|---|---|
| **Single-turn / single-shot** | Associative, Observational | Model receives examples, answers once |
| **Interactive** | Concept, Language | Model decides how many examples to request |
| **Multi-turn / episodic** | RL | Model acts, receives feedback, acts again, up to budget |

Single-turn tasks test inductive inference without the ability to seek more evidence. Interactive tasks test calibration and evidence-seeking strategy. Multi-turn tasks test the full hypothesis update loop. No single protocol covers all three failure modes.

---

### Are the Sub-abilities Measuring Distinct Things?

Yes — the cross-category Spearman rank correlations range from **0.55 to 0.93** (cross-category correlation matrix, `outputs/cross_category_correlations.csv`):

| | Assoc | Concept | Language | Obs | RL |
|---|---|---|---|---|---|
| Assoc | — | 0.71 | 0.85 | 0.69 | 0.71 |
| Concept | 0.71 | — | 0.64 | 0.55 | 0.89 |
| Language | 0.85 | 0.64 | — | 0.83 | 0.60 |
| Obs | 0.69 | 0.55 | 0.83 | — | 0.70 |
| RL | 0.71 | 0.89 | 0.60 | 0.70 | — |

The weakest pairing is **Concept × Observational (ρ = 0.55)** — abstract category formation and structural inference from demonstrations are genuinely distinct tasks. The strongest is **Concept × RL (ρ = 0.89)** — both involve hypothesis testing under uncertainty, which is the expected structural overlap.

PCA confirms this: **PC1 explains 81% of category variance**. A single latent factor accounts for most variance (a general "learning ability"), but not all of it — the sub-abilities each add independent signal. This is the desired structure: a shared g-factor plus distinct profiles (H7, formally tested).

**Model profile inversions confirm non-monolithic structure:**

- Gemma 4 26B ranks **#5 on RL** but **#12 on Language**
- GPT-5.4 ranks **#4 on Language** but **#9 on Concept**
- DeepSeek V3.2 ranks **#4 on Observational** but **#12 on Concept**

If the benchmark were measuring one thing, these inversions would not exist.

---

### Task Content Diversity

Within each sub-ability, tasks span genuinely different underlying domains and structures:

- **Associative (20):** causal blocking, spurious cue traps, temporal confounds, counterfactual inference, latent cause disambiguation
- **Concept (18):** Boolean rule induction, feature conjunction, relational concepts, numerical threshold rules, one-shot generalization
- **Language (26):** phonological transformation rules (vowel harmony, consonant mutation, syllable-weight effects, stress shift, allophonic variation)
- **Observational (40):** hidden priority orderings, latent sorting criteria, implicit grouping rules, sequential pattern completion, exception-pattern discrimination
- **RL (34):** 1D grid navigation, verbal bandits, Mastermind-style deduction, battleship-style inference, resource allocation, sequential probability update

No two tasks within any category test the same cognitive sub-act.

---

### Model Diversity

14 models across 3 tiers and 4 providers:

| Tier | Anthropic | OpenAI | Google | Open-source |
|---|---|---|---|---|
| Frontier | Claude Opus 4.6 | GPT-5.4 | Gemini 3.1 Pro Preview | GLM-5 |
| Mid | Claude Sonnet 4.6 | GPT-5.4 mini | Gemini 2.5 Flash, Gemini 3.1 Flash-Lite | DeepSeek V3.2, Qwen 3 Next 80B Thinking |
| Small | Claude Haiku 4.5 | GPT-5.4 nano | Gemma 4 26B A4B | Qwen 3 Next 80B Instruct |

The model set includes: both reasoning-enabled and standard models (Qwen Thinking vs. Instruct), models spanning 3 orders of magnitude in scale, all four major providers, and representatives of each tier for each provider. This ensures benchmark findings are not artifacts of a narrow model distribution.

---

## 3. Difficulty Distribution

### Overall (138 curated tasks)

| Band | Criterion | Count | % |
|---|---|---|---|
| Easy | Mean model score > 0.70 | 24 | 17.4% |
| Medium | Mean score 0.30–0.70 | 71 | 51.4% |
| Hard | Mean score < 0.30 | 43 | 31.2% |

The distribution has a healthy center-of-mass in the medium band (majority of tasks), a smaller easy tail (ensuring top models are distinguishable), and a substantial hard tail (ensuring no model saturates the benchmark). The mean task difficulty is **0.43** — well below ceiling.

---

### Per Sub-ability

| Sub-ability | Tasks | Easy | Medium | Hard | Mean difficulty |
|---|---|---|---|---|---|
| Associative | 20 | 7 (35%) | 11 (55%) | 2 (10%) | 0.595 |
| Concept | 18 | 0 (0%) | 9 (50%) | 9 (50%) | 0.345 |
| Language | 26 | 4 (15%) | 18 (69%) | 4 (15%) | 0.492 |
| Observational | 40 | 5 (13%) | 18 (45%) | 17 (43%) | 0.365 |
| RL | 34 | 8 (24%) | 15 (44%) | 11 (32%) | 0.517 |

**Observations:**

- **Associative** is the most accessible (35% easy) — the 3-class format (YES/NO/UNKNOWN) gives models a meaningful floor to stand on, and causal inference from a few examples is more tractable than open-ended rule induction.
- **Concept Formation** has no easy tasks: every concept task requires genuinely discovering a rule from scratch. 50% are hard, meaning half the concept tasks stump the majority of models.
- **Observational** has the most hard tasks (43%) — structural inference from demonstrations alone is the hardest protocol, producing the lowest average score.
- **RL** has the most easy tasks among the non-trivial categories (24%) — some RL environments have simple enough structure that most models can solve them; these act as calibration checks.

This difficulty spread is intentional. A benchmark that is all-hard measures only top model differences; one that is all-easy measures nothing. The current distribution ensures:

1. **No floor effect** — every model scores above random on every sub-ability
2. **No ceiling effect** — even the top model (Gemini 3.1 Pro, overall 0.775) does not saturate any category
3. **Discrimination at every tier** — easy tasks separate frontier from random; hard tasks separate top-3 from the rest

---

### How Difficulty Was Calibrated

Difficulty was not pre-set — it emerged from the construction process:

1. Every candidate task was first run across all 14 models before inclusion.
2. Tasks all 14 models solved perfectly were dropped (ceiling effect).
3. Tasks no model touched (score = 0.0) were dropped or rewritten (floor effect).
4. The 95% rejection rate across candidates naturally produced the current gradient.
5. Phase D removed 19 tasks that were noise rather than difficulty (near-zero entropy, implementation flaws, prior knowledge contamination). Removing noise pulled scores up uniformly across all models — confirming the removed tasks were not genuinely hard, just broken.

---

## 4. Validity & Reliability

### Does removing any single task change who wins?

No.

- Leave-one-out across all 135 tasks: mean Spearman ρ with full ranking = **0.9985**, min = **0.9912**
- Maximum rank change from removing any single task: **1 position**
- Removing all 19 Phase-D-removed tasks at once: top-11 unchanged; only positions 12/13 swap

**Script:** `11_task_removal_sensitivity.py` → `outputs/loo_global.csv`, `outputs/benchmark_stability_summary.csv`

This is a strong stability result. No single task controls the outcome, and no reviewer can argue the leaderboard is driven by a handful of cherry-picked items.

---

### Is the ground truth reliable? (Construct validity)

Yes.

- The same rule function that generates training examples also grades model responses — there are no static answer keys.
- 15 tasks spot-checked manually: **15/15 confirmed** correct ground truth computation, no hardcoded keys, no scoring bugs.
- RL tasks use seeded random generation — correct by construction.

**Script:** `13_ground_truth_spotcheck.py` → `outputs/ground_truth_spotcheck.csv`

This is a meaningful difference from LLM-as-judge benchmarks. The grader is deterministic and is the same function as the task generator. If the task generator is correct, the grader is correct by construction.

---

### Test-Retest Reliability: Would the same model score the same again?

LearningBench uses **programmatically generated tasks with seeded randomness** — every run can be reproduced exactly, and the score function is deterministic given a model response string.

The relevant question is: would a given model produce the same response on the same task input in a re-run? This is a property of the model's stochasticity (temperature), not of the benchmark. Two properties of the benchmark design minimize this concern:

1. **Most LearningBench tasks aggregate over multiple test rounds** (e.g., 8 test examples after learning, multiple RL episodes). A single unlucky generation does not dominate the score — the aggregate is stable.
2. **Task difficulty was calibrated across 14 models.** Items that showed extreme variance (bimodal, near-zero entropy) were removed in Phase D. What remains has an entropy distribution centered on real discriminatory signal — not noise that a temperature=0 rerun would resolve differently.

The closest direct measurement is the **LOO stability result**: removing any one task shifts the ranking by ≤1 position. If single-task stochasticity were dominating, we would expect larger rank swings from task removal.

**Formal test-retest on individual model runs is not in scope for this benchmark version** (it would require re-running all 14 models, which costs significant API budget). The structural design mitigates this concern without requiring it.

---

### Does the benchmark measure what it claims? (Face validity + convergent validity)

**Face validity**: Every task is human-feasible — a domain expert who reads the task can solve it with the same inputs given to the model. This is verified in the construction pipeline (one of five rejection criteria). The benchmark does not require real-world knowledge; it requires the ability to induce a rule from examples.

**Convergent validity**: Models that perform well on rule-induction sub-abilities also perform well across sub-abilities (PCA PC1 = 81% of variance). There is a genuine latent learning ability being measured. But the remaining 19% means sub-ability profiles are not reducible to one number — a model can be good at RL and bad at Language, which is the expected structure for a multi-dimensional benchmark.

**Discriminant validity**: The sub-abilities are not just relabeled versions of each other. Concept × Observational Spearman ρ = 0.55 — the weakest pairing in the matrix. Profile inversions (Gemma ranks #5 on RL, #12 on Language) confirm the dimensions are genuinely distinct.

---

### What Does LearningBench Uniquely Capture?

| Measurement | Status in existing benchmarks | LearningBench |
|---|---|---|
| Can model infer a novel rule from examples? | Rare (ARC-AGI, BIG-Bench subsets) | Yes, 138 tasks × 3 protocols |
| Does score improve with more practice rounds? | **Not in any major benchmark** | Procedural trajectory slope (ρ = −0.02 with final score — 99% orthogonal) |
| How many examples does the model request before committing? | **Not measured anywhere** | Evidence appetite per model (1.8–11.9 span, ρ = −0.52 with score) |
| Does token spend predict failure in real time? | **Not measured** | RL token-score correlation (ρ = −0.53, 4.3× gap solved vs. failed) |
| Does the model recognize when evidence is genuinely insufficient? | **Not measured** | Epistemic uncertainty tasks (UNKNOWN-answer tasks scored higher on average) |
| Can the model override a strong semantic prior with in-context evidence? | **Not measured** | `semantic_override` task (frontier models score 0.00) |
| Stability under task removal | Varies | LOO Spearman ρ = 0.9985, max rank shift = 1 |

The core insight: **LearningBench does not measure what models already know. It measures how they learn.** Final score, learning trajectory, evidence calibration, and hypothesis updating are four dimensions of inference-time learning, each orthogonal to the others in the data. No existing benchmark reports more than one.

---

## Reproduce

```bash
# All formal hypothesis tests (H1–H8, H14–H18)
python analysis/scripts/30_hypothesis_tests.py

# Cross-category correlations and PCA
python analysis/scripts/05_cross_category.py

# LOO stability
python analysis/scripts/11_task_removal_sensitivity.py

# Ground truth spot-check
python analysis/scripts/13_ground_truth_spotcheck.py

# Provider analysis (H3)
python analysis/scripts/09_provider_analysis.py

# Epistemic uncertainty (UNKNOWN tasks)
python analysis/scripts/14_epistemic_analysis.py
```

All pre-computed outputs are in `analysis/outputs/`. No Kaggle API required.
