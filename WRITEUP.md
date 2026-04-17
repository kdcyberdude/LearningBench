### Project Name
LearningBench: Measuring Inference-Time Learning in LLMs


### Your Team
Karandeep Singh


### Problem Statement

Existing benchmarks measure what models already know. **LearningBench measures how they learn** — from scratch, inside a single conversation, on systems that have never existed.

ARC-AGI showed that learning is best tested in novel environments the model must *interact with*, not static datasets it can be evaluated against. LearningBench brings that philosophy to text. Across 135 tasks in six cognitive learning sub-abilities, we score not just whether a model infers the hidden concept, but:

- *how many examples it needed*,
- *whether its performance improved with practice*, and
- *how efficiently it spent a finite interaction budget*.

No current benchmark measures any of this.


### Task & Benchmark Construction

**Novelty is the entire benchmark.** Every task presents a system that does not exist anywhere — invented languages with phonologies built from scratch, hidden Boolean circuits with randomized wiring, physics with alien damping constants, counterintuitive assumptions. These models have consumed the internet; any rule they can recall is not one they had to learn.

**Six sub-abilities, each targeting a distinct cognitive act:**

| Sub-ability | Tasks | Protocol | What it isolates |
|---|---|---|---|
| Associative | 17 | Single-turn | Causal inference vs. correlation (blocking, spurious cues) |
| Concept Formation | 18 | Interactive | Meta-calibration: *does the model know when it has seen enough?* |
| Language | 26 | Interactive | Productive rule induction (wug-test on invented phonologies) |
| Observational | 30 | Single-shot | Structural inference from demonstrated behavior alone |
| Procedural | 11 | Multi-episode | Learning *trajectory*: did performance improve with practice? |
| Reinforcement | 30 | Multi-turn | Hypothesis updating from feedback under a finite action budget |

**Scoring primitives shared across all 135 tasks:**

- **Free exploration zone** — efficiency penalties begin only *after* the minimum evidence structurally required to see the pattern. Only avoidable over-querying is punished.
- **Zero-accuracy floor** — every efficiency-weighted task returns 0.0 if accuracy is zero. No reward for being fast at being wrong.
- **Interactive tasks** (concept, language): `score = accuracy × (0.40 + 0.60 × efficiency)`.
- **RL tasks**: success (0.55) + step-efficiency (0.25) + partial progress (0.20).
- **Procedural tasks** score the OLS slope of practice-round accuracy *independently* of the asymptote — the only component in any major benchmark that directly measures whether learning occurred.

**No static dataset, one source of truth.** LearningBench is 135 programmatic scenarios/environments that generate examples and compute ground truth at runtime. Every correct answer comes from the same function that produced the task's training examples — no grading tables, no LLM-as-judge. **The grader *is* the rule.**

**Construction pipeline.** Enumerate every facet of each ability → seed a task per facet → expand programmatically → filter ruthlessly. A human reviewer is in the loop at every gate — rejecting, rewriting, or hardening until each survivor earns its place.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11780821%2Fa3031131e90be33079e1323c617c7263%2Fr9tidTx59EG4dkBnIp-d9_09o35ZIt.png?generation=1776378364686654&alt=media)

**Five-point validation, applied multiple times.** Every candidate is audited by a frontier LLM and a human reviewer. Failing any one criterion means reject, rewrite, or harden — the loop can take several rounds per task:

| Criterion | The question it answers |
|---|---|
| **Human feasibility**   | Could an expert human solve it with the same inputs? |
| **Solution uniqueness** | Does exactly one rule fit all training examples *and* the held-out tests? |
| **Logical consistency** | Does the rule produce deterministic, reproducible outputs? |
| **Anti-contamination**  | Is the system genuinely invented — no plausible web or training-data trace? |
| **PhD-level difficulty**| Hard enough to require learning rather than pattern-matching — yet still solvable by at least one frontier LLM or an expert human. |

**Multi-model calibration.** Every survivor is run across 14 models spanning small, mid-tier, and frontier. Tasks all 14 solve perfectly are dropped; tasks none can touch are dropped. Overall **~95% of candidates were rejected.** The 135 survivors produce a clean difficulty gradient — no task trivially solved by all, none unsolved by all.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11780821%2F6e2e332622695d7b60b7415cf5aa34b6%2Ffig_task_difficulty.png?generation=1776378437088515&alt=media)


### Technical Details — Trajectory is orthogonal to level(From Procedural Learning Tasks)

Imagine two students who both score 70% on the final exam. Student A started at 50% and climbed steadily — genuinely learning. Student B started at 90% and declined — forgetting. A transcript that records only the final grade treats them identically; **LearningBench does not.**

Across **112 (model, task) procedural-learning runs**:

- Spearman ρ(slope, asymptote) = **−0.02**, R² = 0.01
- **The trajectory signal is 99% orthogonal to the final score.**

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11780821%2F65ca8f838ae304973ebdf0e7092f0821%2Ffig_trajectory_orthogonal.png?generation=1776378495412524&alt=media)

*Two runs landing at the same final score (x ≈ 0.5) have slopes of +0.18 and −0.30 — one model is still climbing, the other is falling back. Same destination, opposite journeys.*


### Results & Insights

> 14 models · 135 tasks · every claim below is confirmed with formal hypothesis tests (Spearman, Mann-Whitney, Wilcoxon), 10,000-sample bootstrap 95% CIs, and Benjamini-Hochberg FDR correction.

---

#### Headline — today's frontier models largely cannot learn

- Only **Gemini 3.1 Pro Preview** clears 0.70 (scores 0.85).
- **11 of 14 models** score below 0.50.
- The #2 model — **GLM-5, open-source** — outranks every closed-source lab except Google.
- **Google + open-source** score **+40.8% higher** on rule induction (concept + observational) than **Anthropic + OpenAI** (Mann-Whitney p = 0.029, Cliff's δ = 0.71, large effect).

When tasks genuinely require counterintuitive rule discovery from novel evidence, **today's models demonstrate recall, not learning.**

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11780821%2F6dbe993fc67c374a9bbca4afb0066057%2Ffig_leaderboard_ci.png?generation=1776378597891082&alt=media)

---

#### Finding 1 — Reasoning is the cleanest controlled win

**Qwen 3 Next 80B Thinking vs. Instruct** is the cleanest A/B in the benchmark: same weights, same training, only the reasoning trace toggled.

- Across **132 matched tasks**: Thinking wins **87**, Instruct wins **19**, ties **26**.
- Largest gains land on induction-heavy abilities: **Observational +0.43, Concept +0.38, RL +0.30** (all p ≤ 0.001).
- Apparent dip on **Procedural −0.11 (p = 0.55, not significant)** — suggestive only.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11780821%2F72c914c8d12538919cc9618ff965e4c4%2Ffig_thinking_vs_instruct.png?generation=1776378735879864&alt=media)

*Reasoning is the **only** controlled intervention in the benchmark that lifts every induction-heavy sub-ability simultaneously. The Procedural negative is the hypothesis worth testing at scale: when feedback rounds arrive in rapid succession, extended deliberation per round may blunt the iteration loop itself. Reasoning is not free — it appears to trade speed of adaptation for depth of inference.*

---

#### Finding 2 — Evidence-seeking efficiency = epistemic calibration

**The best learners need the least evidence.** Across **201 interactive runs**, models requesting fewer examples score higher (ρ = **−0.52**, p < 10⁻¹⁴). Probe ratios span **37%–96%** across models — a 6.6× spread in evidence appetite.

Four calibration profiles emerge from the scatter:

- **Well-calibrated** (Gemini Pro, GLM-5, Qwen Thinking): use **37–47%** of available examples, score **0.67–0.78**.
- **Underconfident** (Claude Opus, Claude Haiku, DeepSeek, Gemma): exhaust **83–96%** of the budget, score **0.29–0.35** — burning through examples without learning from them.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11780821%2F89939551114ae401b6a4a67bccf84117%2Ffig_meta_calibration.png?generation=1776378803320159&alt=media)

*This is a directly productizable measurement. In deployment, the underconfident model is the one that keeps asking "can you clarify?" instead of just answering.*

---

#### Finding 3 — Token consumption is a real-time failure signal

Across **397 RL runs**, ρ(tokens, score) = **−0.53** (p < 10⁻³⁰).

- Solved runs average **41K tokens**.
- Failed runs average **177K tokens** — a **4.3× gap** (Cliff's δ = −0.60).
- **43 runs** show ≥10 consecutive identical actions: when the first hypothesis is wrong, many models cannot update at all.

*Token spend is a live diagnostic. Production monitors can flag likely failures **before** the wrong answer returns.*


### Conclusion — Learning is hypothesis management

The three findings rhyme. Reasoning helps where hypotheses must be **generated** (induction) and hurts where they must be **updated fast** (procedural). Calibration separates learners by **sufficiency judgment** — knowing when evidence is enough. Stuck-token runs expose the **update** failure — when the first guess is wrong, many models cannot revise at all.

**Generation · sufficiency · update.** Three axes of hypothesis management, each isolated by a different LearningBench protocol, none measured by any existing benchmark. The models that win here are not the ones that memorize more — they are the ones that manage hypotheses well.


### Organizational Affiliations

Treow Intelligence


### References & Citations

1. [Chollet, F. (2019). *On the Measure of Intelligence*. arXiv:1911.01547.](https://arxiv.org/abs/1911.01547)
2. [Srivastava, A. et al. (2022). *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models (BIG-Bench)*. TMLR.](https://arxiv.org/abs/2206.04615)
3. [Chollet, F. et al. (2025). *ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems*. ARC Prize Foundation.](https://arxiv.org/abs/2505.11831)
4. [Morris, J. et al. (2026). *Measuring Progress Toward AGI: A Cognitive Framework*. Google DeepMind.](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf)
