LearningBench measures **how efficiently language models learn** — not what they already know. Every task presents an invented system with no web trace: fabricated languages, randomized Boolean circuits, alien physics, counterintuitive causal structures. Models cannot recall their way to a correct answer.

**135 tasks across 6 sub-abilities**, each targeting a distinct cognitive act:

**Scoring by sub-ability** — each sub-ability is published as a standalone Kaggle benchmark as well containing only the tasks for that learning type. Click any sub-ability to explore its **dedicated leaderboard**, per-task scores, key findings and insights specific to that learning type.


| Sub-ability                                                                               | Tasks | Scoring                                                                                                                        | What it tells us                                                                                                            |
| ----------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **[Associative](https://www.kaggle.com/benchmarks/kdcyberdude/associativelearning/)**     | 17    | Fraction of causal-inference questions correct                                                                                 | Does the model distinguish genuine causes from spurious correlations — including knowing when evidence is insufficient?     |
| **[Concept Formation](https://www.kaggle.com/benchmarks/kdcyberdude/conceptlearning/)**   | 18    | `accuracy × (0.40 + 0.60 × efficiency)` — where efficiency measures how few examples beyond the structural minimum were needed | Can the model induce a hidden rule, and does it know when it has seen enough to commit?                                     |
| **[Language](https://www.kaggle.com/benchmarks/kdcyberdude/languagelearning/)**           | 26    | Same as Concept Formation; exact surface-form matching required                                                                | Can the model generalize a phonological rule to words it has never seen, not just memorize training pairs?                  |
| **[Observational](https://www.kaggle.com/benchmarks/kdcyberdude/observationallearning/)** | 30    | Fraction of test sequences fully correct                                                                                       | Can the model infer a hidden process (machine, operation, cipher) from demonstrations alone, with no interaction?           |
| **[Reinforcement](https://www.kaggle.com/benchmarks/kdcyberdude/reinforcementlearning/)** | 30    | `0.55 × success + 0.25 × efficiency + 0.20 × partial progress`                                                                 | Can the model explore an unknown environment, update its hypothesis from feedback, and solve the task within a step budget? |
| **[Procedural](https://www.kaggle.com/benchmarks/kdcyberdude/procedurallearningbench)**   | 11    | `0.30 × transfer + 0.25 × peak skill + 0.25 × learning slope + 0.20 × consistency`                                             | Did the model actually get better with practice — and can it apply what it learned to new instances with no hints?          |


All tasks use a deterministic programmatic grader. Expected answers are computed by the same function that generates the task — no LLM-as-judge, no hardcoded answer tables. Every scoring formula applies a zero-accuracy guard: no reward for being fast at being wrong.

---

## What this benchmark reveals

1. **Larger models are not better learners.** When tasks require genuine in-context learning, 11 of 14 models score below 0.50 — and model scale alone does not close this gap. The Qwen Thinking vs. Instruct comparison makes this concrete: enabling extended reasoning lifts Concept Formation by 183% (0.191 → 0.541), Observational by 91%, and RL by 77%. A small model with extended reasoning outperforms a larger model without it on every induction-heavy sub-ability.

2. **Reasoning helps generation but may hurt fast adaptation.** Enabling reasoning mode lifts induction-heavy sub-abilities significantly, but shows a suggestive dip on procedural tasks — where rapid hypothesis iteration matters more than deep deliberation.

3. **The best learners need the least evidence.** Evidence efficiency and accuracy are tightly linked (ρ = −0.52). Models that commit early score higher; models that exhaust their example budget without improving are not learning — they are stalling.

4. **How much evidence a model seeks predicts how well it learns.** Across Concept Formation and Language Learning — two structurally different interactive learning sub-abilities — mean probe(asking for examples) count per model correlates strongly across the two (Spearman ρ = 0.793, p = 0.0007). The spread is striking: Qwen 3 Thinking requests 1.8 examples on average; Claude Haiku requests 11.9 — a 6.5× gap. The pattern holds regardless of model size or tier: Claude Opus requests nearly as many examples as Claude Haiku, while Gemini Flash-Lite commits after just 2 probes on average. A model that over-probes on invented Boolean rules also over-probes on invented phonological rules. Evidence appetite is not calibrated to task difficulty or domain — it is a fixed property of the model. And it predicts performance: ρ = −0.52 between probe count and score, meaning the most aggressive evidence-seekers are also the lowest scorers.

5. **Token spend is a failure signal, not a success signal.** In reinforcement tasks, failed runs consume 4.3× more tokens than solved ones. Many models, once their first hypothesis is wrong, cannot update at all — 43 runs show streaks of 10 or more consecutive identical actions, the behavioral signature of a model that has no remaining hypothesis to test.

>These findings converge on three axes that separate genuine learners from pattern matchers: **generating** a candidate rule(s) from limited evidence, judging **sufficiency** — knowing when to stop — and **updating** when the hypothesis is wrong. No existing benchmark measures all three.

---

## Results and Analysis

### The Interactivity Paradox

Concept Formation (interactive: model requests examples) and Observational Learning (passive: all demonstrations given upfront) both require the same core act — inducing a hidden rule(s). The only structural difference is whether the model controls its evidence budget. 9 of 14 models score higher on the passive setting than the interactive one. The largest instance is DeepSeek V3.2: Concept = 0.194, Observational = 0.428. Giving models control over how much evidence they see degrades performance because it requires a meta-cognitive signal while learning — "I have seen enough to generalize" — that most models do not produce. 

### Learning Jaggedness Profiles

The jaggedness index ("Characterizing Model Jaggedness Supports Safety and Usability", Google DeepMind) measures how unevenly a model's strengths are distributed across capability domains. It is the standard deviation of per-domain z-scores: each domain score is first normalized by the cross-model mean and SD for that domain, then J = std of those z-scores across domains for a given model. A J of 0 means perfectly uniform; a higher J means strength on some sub-abilities coexists with weakness on others. Because this benchmark has no published human baseline, z-scores are normalized against the 14-model population rather than human performance (see `analysis/compute_jaggedness.py`).


| Model                         | Overall | J     | Assoc | Concept | Lang | Observ | RL   | Proc |
| ----------------------------- | ------- | ----- | ----- | ------- | ---- | ------ | ---- | ---- |
| Gemini 3.1 Pro Preview        | 0.840   | 0.246 | 0.95  | 0.80    | 0.78 | 0.85   | 0.93 | 0.73 |
| GLM-5                         | 0.678   | 0.164 | 0.77  | 0.57    | 0.75 | 0.65   | 0.79 | 0.54 |
| Qwen 3 Next 80B Thinking      | 0.583   | 0.589 | 0.65  | 0.56    | 0.64 | 0.67   | 0.63 | 0.34 |
| Gemini 2.5 Flash              | 0.512   | 0.490 | 0.61  | 0.53    | 0.50 | 0.44   | 0.50 | 0.49 |
| Claude Opus 4.6               | 0.498   | 0.386 | 0.68  | 0.26    | 0.52 | 0.39   | 0.69 | 0.45 |
| GPT-5.4                       | 0.470   | 0.560 | 0.66  | 0.28    | 0.62 | 0.31   | 0.66 | 0.28 |
| Gemini 3.1 Flash-Lite Preview | 0.455   | 0.149 | 0.59  | 0.31    | 0.50 | 0.33   | 0.58 | 0.41 |
| DeepSeek V3.2                 | 0.415   | 0.321 | 0.52  | 0.19    | 0.47 | 0.43   | 0.53 | 0.35 |
| Claude Haiku 4.5              | 0.373   | 0.332 | 0.50  | 0.19    | 0.36 | 0.27   | 0.57 | 0.35 |
| Qwen 3 Next 80B Instruct      | 0.355   | 0.592 | 0.48  | 0.19    | 0.44 | 0.24   | 0.33 | 0.45 |
| GPT-5.4 mini                  | 0.348   | 0.357 | 0.48  | 0.22    | 0.50 | 0.21   | 0.45 | 0.22 |
| Gemma 4 26B A4B               | 0.337   | 0.500 | 0.51  | 0.25    | 0.25 | 0.22   | 0.55 | 0.24 |
| Claude Sonnet 4.6             | 0.462   | 0.320 | 0.66  | 0.33    | 0.48 | 0.29   | 0.64 | 0.37 |
| GPT-5.4 nano                  | 0.247   | 0.283 | 0.43  | 0.15    | 0.30 | 0.14   | 0.25 | 0.20 |


GLM-5 is the most uniform learner (J = 0.164): its strength is distributed across all six sub learning abilities without a pronounced weak spot. Qwen 3 Instruct is the most jagged (J = 0.592): it scores comparably on Procedural (0.45) and Associative (0.48) but collapses on Concept Formation (0.19) and RL (0.33) — the two sub-abilities that most directly require active hypothesis updating. GPT-5.4 (J = 0.560) shows the same pattern in a different shape: strong on RL and Associative (0.66), near-floor on Concept and Procedural (0.28). The top two models — Gemini 3.1 Pro (J = 0.246) and GLM-5 (J = 0.164) — are also the most uniform, consistent with the general finding that higher overall learning ability comes with more even capability distribution across the six learning sub-abilities.

---

## Domain coverage

Tasks span 6 broad domains, ensuring no subject-area familiarity can substitute for genuine in-context learning.


| Domain                                | Tasks | Examples                                                                    |
| ------------------------------------- | ----- | --------------------------------------------------------------------------- |
| Mathematical & Algebraic Structures   | 18    | Modular recurrences, CRT reconstruction, ring operations, lattice meet/join |
| Linguistics & Morphological Systems   | 18    | Vowel harmony, tone sandhi, evidentiality, layered transforms               |
| Formal & Logical Systems              | 13    | DFA/PDA inference, Mealy machines, Boolean circuits, XOR/XNOR binding       |
| Cryptographic & Computational Systems | 13    | Feistel cipher, LFSR, pipeline hazards, LFU cache policy                    |
| Causal & Strategic Reasoning          | 13    | Overexpectation, spurious correlation, Nim variants, Shapley values         |
| Dynamical, Spatial & Natural Systems  | 11    | CA rule inference, shortest-path oracle, Lights-Out toggle, codon table     |


---

**PS:** If you want to try any task yourself or review how any model performed, open **View Notebook Output** on the relevant Kaggle Task model run. You can see the exact prompt submitted and the full model response, including multi-turn conversations. All tasks use a `log_trace` function that captures this for easy review and debugging.

**Why does Qwen 3 72B Thinking outperform GPT-5.4 and Claude Opus 4.6 on many tasks?** GPT-5.4 and Claude Opus 4.6 are reasoning models whose thinking effort can be set to `auto / low / medium / high / max`. When left on `auto`, the model may suppress chain-of-thought on tasks it judges as easy — degrading quality on tasks that actually require step-by-step simulation. This matters especially here: extended reasoning is not a luxury for learning tasks, it is foundational to them, as the performance gap between Qwen Thinking and Qwen Instruct makes clear. Additionally, enforcing a structured output schema further reduces effective thinking depth. Both factors likely explain the unexpected ranking.

[https://www.kaggle.com/benchmarks/kdcyberdude/learningbench](https://www.kaggle.com/benchmarks/kdcyberdude/learningbench)