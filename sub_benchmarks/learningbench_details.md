LearningBench measures **how efficiently language models learn** — not what they already know. Every task presents an invented system with no web trace: fabricated languages, randomized Boolean circuits, alien physics, counterintuitive causal structures. Models cannot recall their way to a correct answer.

**135 tasks across 6 sub-abilities**, each targeting a distinct cognitive act:

**Scoring by sub-ability** — each sub-ability is published as a standalone Kaggle benchmark as well containing only the tasks for that learning type. Click any sub-ability to explore its dedicated leaderboard, per-task scores, key findings and insights specific to that learning type.


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

**Frontier models largely demonstrate recall, not learning.** When tasks require genuinely novel rule discovery, 11 of 14 models evaluated score below 0.50.

**Reasoning helps generation but may hurt fast adaptation.** Enabling reasoning mode lifts induction-heavy sub-abilities significantly, but shows a suggestive dip on procedural tasks — where rapid hypothesis iteration matters more than deep deliberation.

**The best learners need the least evidence.** Evidence efficiency and accuracy are tightly linked (ρ = −0.52). Models that commit early score higher; models that exhaust their example budget without improving are not learning — they are stalling.

**Token spend is a failure signal, not a success signal.** In reinforcement tasks, failed runs consume 4.3× more tokens than solved ones. Many models, once their first hypothesis is wrong, cannot update at all.

The three findings point to the same three axes of hypothesis management: **generating** a candidate rule, judging **sufficiency** of evidence, and **updating** when wrong. None are measured by existing benchmarks.

---

## Domain coverage

Tasks span 17 distinct subject areas with no single domain exceeding 11% of the benchmark, ensuring no reasoning modality is over-represented.


| Domain                                | Tasks | Examples                                                                                        |
| ------------------------------------- | ----- | ----------------------------------------------------------------------------------------------- |
| Mathematics & Number Theory           | 14    | Modular recurrences, CRT reconstruction, polynomial interpolation, factorization, digit ciphers |
| Linguistics & Morphology              | 12    | Vowel harmony, tone sandhi, evidentiality, ergativity, reduplication, switch-reference          |
| Formal Language & Automata            | 7     | DFA/PDA inference, Mealy machines, FSTs, CFG identification                                     |
| Logic & Boolean Reasoning             | 6     | XOR/XNOR binding, Boolean circuits, nested logic, disjunctive rules                             |
| String & Sequence Manipulation        | 6     | Interleave-reverse, layered transforms, vowel rotation, counterfactual rewrite                  |
| Game Theory & Decision Theory         | 5     | Nim variants, iterated games, voting protocols, Shapley values, multi-armed bandit              |
| Cryptography                          | 5     | Feistel cipher, LFSR, affine ciphers, shift cipher, substitution                                |
| Computer Science & Systems            | 5     | Pipeline hazards, LFU cache policy, SQL query inference, firewall rules, instruction set RE     |
| Abstract Algebra                      | 4     | Ring operations, lattice meet/join, symbol grounding, modular addition                          |
| Causal & Statistical Reasoning        | 4     | Blocking effect, overexpectation, learned irrelevance, spurious correlation resistance          |
| Search & Localization                 | 4     | 1D/2D Battleship, Chebyshev search, Manhattan localization                                      |
| Information Theory                    | 3     | Hamming distance oracle, Gray codes, XOR subset identification                                  |
| Cellular Automata & Dynamical Systems | 3     | CA rule inference, Collatz maps, rule-90/30/110/150                                             |
| Graph Theory & Geometry               | 3     | Shortest path under noisy oracle, 2D point localization                                         |
| Planning & Constraint Satisfaction    | 2     | Tower of Hanoi with hidden constraints, 4×4 Latin square                                        |
| Spatial & Grid Reasoning              | 2     | Grid transposition, Lights-Out toggle mask discovery                                            |
| Molecular Biology                     | 1     | Codon table / genetic code translation                                                          |


---

**PS:** If you want to try any task yourself or review how any model performed, open **View Notebook Output** on the relevant Kaggle Task model run. You can see the exact prompt submitted and the full model response, including multi-turn conversations. All tasks use a `log_trace` function that captures this for easy review and debugging.

**Why does Qwen 3 72B Thinking outperform GPT-5.4 and Claude Opus 4.6 on many tasks?** GPT-5.4 and Claude Opus 4.6 are reasoning models whose thinking effort can be set to `auto / low / medium / high / max`. When left on `auto`, the model may suppress chain-of-thought on tasks it judges as easy — degrading quality on tasks that actually require step-by-step simulation. This matters especially here: extended reasoning is not a luxury for learning tasks, it is foundational to them, as the performance gap between Qwen Thinking and Qwen Instruct makes clear. Additionally, enforcing a structured output schema further reduces effective thinking depth. Both factors likely explain the unexpected ranking.