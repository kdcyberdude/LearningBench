# Learning Cognitive Ability Benchmark — Master Project Document (v2)

## Table of Contents

1. [Competition Context](#1-competition-context)
2. [Project Scope and Philosophy](#2-project-scope-and-philosophy)
3. [The Dataset Question](#3-the-dataset-question)
4. [Learning Sub-Abilities and Task Inventory](#4-learning-sub-abilities-and-task-inventory)
5. [Benchmark Construction Methodology](#5-benchmark-construction-methodology)
6. [Why Active-Retrieval Is a Learning Protocol](#6-why-active-retrieval-is-a-learning-protocol)
7. [Scoring Design](#7-scoring-design)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Task Quality Criteria](#9-task-quality-criteria)
10. [Ablation Studies and Robustness Checks](#10-ablation-studies-and-robustness-checks)
11. [Results Summary](#11-results-summary)
12. [Insights and Novel Hypotheses](#12-insights-and-novel-hypotheses)
13. [Deliverables](#13-deliverables)
14. [Open Questions and Future Work](#14-open-questions-and-future-work)
15. [Models Evaluated](#15-models-evaluated)
16. [References](#16-references)

---

## 1. Competition Context

**Competition:** Measuring Progress Toward AGI — Cognitive Abilities (Google DeepMind x Kaggle)

**Track:** Learning

**Deadline:** April 16, 2026

**Central question the judges will ask:** "What can this benchmark tell us about model behavior that we could not see before?"

**Evaluation criteria (from competition page):**

| Criteria | Weight | What judges look for |
|---|---|---|
| Dataset quality and task construction | 50% | Verifiably correct answers (no ambiguity). Sufficient sample size for statistical significance. Clean, readable code. Robust input prompts and output verification. |
| Writeup quality | 20% | Problem statement and motivation. Task and benchmark construction details. Dataset provenance, columns, data types. Technical implementation details. Results, insights, conclusions. Organizational affiliations. References and citations. |
| Novelty, insights, discriminatory power | 30% | New behavioral signal invisible to existing benchmarks. Meaningful performance gradient across models. Neither all-zero nor all-perfect scores. |

**What we submit:**
1. Kaggle Writeup (mandatory, 1500 words max, scientific paper style, cover image required)
2. Kaggle Benchmark with underlying tasks (mandatory, attached as project link)
3. Public Notebook (optional but we include it — full analysis, ablation studies, robustness)
4. Media Gallery (optional)

**Audience:** DeepMind researchers. Winning project shared publicly worldwide. Must be research-grade.

---

## 2. Project Scope and Philosophy

### The Core Problem

We lack an empirical framework to measure how AI models learn. Current evaluations cannot distinguish a model that truly solves a novel problem from one that simply recalls a scenario from its training data. Without benchmarks that isolate specific learning abilities, resist shortcut solutions, and expose systematic failure modes, progress toward human-level generality is difficult to interpret, model comparisons are noisy, and important weaknesses remain hidden until deployment.

### What We Measure

**Inference-time learning only.** We measure the ability to acquire new knowledge, skills, or behaviors within a single conversation — NOT learning during training. These frontier models have PhD-level intelligence and have consumed the entirety of the internet. Any standard task or question can be answered from pre-training data.

### The Novelty Imperative

> Every task must present something the model has never seen. Novel stimuli, novel rules, novel languages, novel systems. This is the only clean way to isolate learning ability from crystallized knowledge.

A model that scores well on our benchmark cannot have done so by recall. It must have genuinely learned from the examples provided in context.

### What Learning Means Here

Learning = encountering a genuinely novel environment (new concepts, new rules, new assumptions, fictional systems, contradictory conventions) and being able to understand, generalize, and apply that new information to unseen test cases. We quantify the ability to learn, not the ability to recall.

### Why Novel Synthetic Tasks Instead of Human-Like Learning Tasks

- Any textbook learning scenario already exists in pre-training data
- The model would "know" the answer before seeing any examples, defeating the purpose
- Models lack embodiment for physical/procedural tasks
- Synthetic tasks guarantee zero contamination — the model cannot pattern-match to memorized solutions

### Modalities Excluded

- **Video/image modality** — skipped
- **Very long context learning** — would require a fully novel, book-length fictional narrative not available anywhere on the internet. Deferred due to construction difficulty. (Though our RL tasks do involve long multi-turn conversations of 20-60 messages.)

---

## 3. The Dataset Question

### The Problem

The competition page states: "Your goal is to create a Kaggle benchmark using **datasets** that isolate specific cognitive abilities." The writeup template includes a "Dataset" section asking about provenance, columns, and data types.

Our benchmark does NOT use a traditional static CSV/JSON dataset in the way a classification or regression benchmark would. This section explains why, what we have instead, and the tradeoffs.

### Why a Static Dataset Is the Wrong Abstraction for Learning Benchmarks

A traditional static dataset (rows of inputs and expected outputs) is fundamentally unsuitable for measuring learning because:

1. **Contamination risk.** Any static dataset published on Kaggle becomes part of future training data. The benchmark's value decays to zero over time. A model that memorizes the dataset will score perfectly without learning anything.

2. **Learning requires interaction.** Several of our sub-abilities (concept formation, language learning, reinforcement learning) involve multi-turn protocols where the model actively requests evidence, receives feedback, and updates its hypothesis. This temporal, interactive structure cannot be captured in a flat table.

3. **Ground truth must be computed, not stored.** If answers are hardcoded in a static file, typos and hallucinations in the answer key silently corrupt the benchmark. Our approach computes ground truth from the same mathematical rule function used for grading — the dataset and the grader are one system.

4. **Novel tasks require generation, not curation.** We cannot curate "real-world" learning data because any real-world data exists in pre-training corpora. Every task must be synthetic and novel by construction.

### What We Have Instead

Each task IS its own self-contained dataset. Concretely, every task contains:

- **Training examples** (the "dataset" the model learns from): input-output pairs generated by a hidden rule function. These are the model's learning material.
- **Test items** (held-out evaluation set): novel inputs the model has never seen, with ground truth computed from the same rule function.
- **The rule function itself** (the generative process): the mathematical or logical system that produced the data. This is the source of truth.

For categories with static observation (associative learning, observational learning): the full set of examples is provided at once — this IS the dataset.

For categories with interactive protocols (concept formation, language learning, RL): the dataset is generated dynamically during the conversation as the model requests examples.

### What We Present in the Writeup "Dataset" Section

- **Provenance:** All data is synthetically generated. Zero overlap with any pre-training corpus.
- **Structure:** Per-task breakdown of example count, input/output format, rule complexity.
- **Data types:** Varies by category — string transformations, numerical sequences, symbolic tokens, phonological forms, state-action-reward tuples.
- **Size:** 157 tasks total across 5 categories, each containing 4-50+ examples plus held-out test items. The benchmark covers a breadth of domains: linguistics, mathematics, physics, computer science, abstract algebra, automata theory, chemistry, game theory, etc.
- **Integrity guarantee:** All ground truth is computed programmatically, not manually specified.

### Tradeoffs We Accept

| Traditional Static Dataset | Our Programmatic Approach |
|---|---|
| Easy to inspect, download, share | Tasks are code, not CSV files — requires running the notebook |
| Familiar to ML community | Less conventional — judges may need explanation |
| Subject to contamination over time | Immune to contamination by design |
| Cannot capture interactive protocols | Naturally supports multi-turn learning |
| Ground truth can have silent errors | Ground truth is computed and verified |
| Fixed difficulty | Difficulty is parameterizable |

---

## 4. Learning Sub-Abilities and Task Inventory

Based on Google DeepMind's cognitive framework, Learning decomposes into six sub-abilities. We have built benchmarks for all six (135 tasks across 6 leaderboarded categories).

### 4.1 Associative Learning — 20 tasks

**Definition:** The ability to learn relationships between events, objects, or stimuli that appear together. Inferring causal structure from contingent observations — which stimuli predict outcomes, which are confounded, which are indeterminate.

**Tasks:** `blocking_effect`, `contextual_flip`, `cumulative_state_rewrite`, `deep_second_order_extinction`, `gated_dual_signal_binding`, `glyph_bind`, `inference_dyad_operators`, `inhibitory_summation`, `latent_cross_binding`, `latent_inhibition`, `latent_set_binding`, `latent_set_variant`, `learned_irrelevance`, `occasion_setting`, `odd_letter_score_pair`, `overexpectation`, `retrospective_revaluation`, `spurious_hue_true_edge`, `temporal_pairing_tnr`, `xor_attribute_binding`

**Structure:** Fixed trial log provided upfront. 6-11 questions per task. Single-shot reasoning from the given data.

### 4.2 Concept Formation — 19 tasks

**Definition:** The ability to abstract key features from examples to form categories and schemas. Active induction of a hidden rule from I/O examples while controlling one's own evidence-gathering.

**Tasks:** `consonant_clusters`, `digit_cipher`, `disjunctive_noise`, `dual_recurrence`, `encoded_triple`, `grid_transform`, `hapax_prime`, `interleave_reverse`, `layered_transform`, `modular_subsequence`, `nested_logic`, `positional_encode`, `positional_mapping`, `relational_pairs`, `semantic_override`, `state_machine`, `triple_parity`, `violation_counter`, `vowel_rotation`

**Structure:** Interactive learning protocol (see Section 6). Model receives initial examples, may request more, then takes an exam on novel inputs.

### 4.3 Language Learning — 26 tasks

**Definition:** The ability to learn new language-related information (syntax, morphology, phonology, vocabulary). Inducting morphophonological rules from labeled form-meaning pairs and producing novel surface forms.

**Tasks:** `dimval_metathesis`, `drafnelt_switch_reference`, `dralven_tone_sandhi`, `drelkovak_harmony`, `drelvak_reduplication`, `grelkan_suppletion`, `gwelthar_mirative_evidential_tone`, `kelstran_tone`, `kophar_quantity`, `mixed_radix_number`, `norkvash_scalar`, `pelvan_agreement`, `prentova_allomorphy_wugtest`, `skelth_allomorph`, `sklonveth_root_pattern`, `skolvren_polysynthetic`, `skovar_deletion`, `strelkov_ergative`, `strevoklan_neg`, `telvari_evidentiality`, `threlkav_scope_ergativity`, `trenval_bleeding`, `trevkovan_gradation`, `vrelthan_rule_interaction`, `vrendel_templatic`, `wukal_tones`

**Anti-contamination:** All language names, morphemes, and phonological environments are invented. No pre-training signal exists for "DRELKOVAK" or "GWELTHAR."

**Structure:** Same interactive learning protocol as Concept Formation. Wug-test items (novel roots never in training) verify productive rule knowledge vs. memorization.

### 4.4 Observational Learning — 42 tasks

**Definition:** The ability to learn by watching others perform a skill or task. Inferring a hidden computational process from complete I/O demonstrations, then predicting outputs for novel inputs.

**Tasks:** `affine_transform_chain`, `agglutinative_morphology`, `arithmetic_entropy_coding`, `auction_mechanism_second_price`, `chemical_reaction_order`, `codon_table_translation`, `custom_gravity_simulation`, `deceptive_stack_machine`, `feistel_cipher_round`, `finite_state_transducer`, `flow_network_capacity`, `grid_parity_path`, `hidden_attribute_rule`, `hidden_damping_physics`, `hidden_graph_centrality`, `hidden_group_operation`, `hidden_matrix_fill`, `hidden_modal_logic_kripke`, `hidden_modal_logic_kripke2`, `hidden_priority_order`, `hidden_region_classifier`, `hidden_state_machine`, `hidden_token_filter`, `lattice_meet_join`, `lfu_cache_eviction`, `linear_feedback_shift_register`, `mealy_machine_output`, `phonological_alternation_harmony`, `pipeline_hazard_stall_counting`, `pushdown_automaton_inference`, `regex_intersection_membership`, `register_machine_trace`, `ring_operations_hidden_carry`, `shapley_values_cooperative_game`, `sigil_naming`, `singleton_gate_local_max`, `symbolic_glyphpair_scoring`, `syntax_tree_rewrite`, `titration_curve_diprotic`, `two_counter_machine`, `vigenere_variant_cipher`, `voronoi_custom_metric`

**Structure:** All demonstrations given at once (no interaction). Difficulty from process complexity: aliased states, deceptive patterns, interacting transformations.

### 4.5 Reinforcement Learning (Runtime RL) — 50 tasks

**Definition:** The ability to learn based on consequences of actions. Solving an unknown problem through sequential interaction: probing the environment, receiving feedback, updating hypotheses, converging within a step budget.

**Tasks:** `arithmetic_next`, `affine_cipher_word`, `chebyshev_point`, `battleship_two_ships`, `coin_balance`, `battleship_1d`, `bitstring_hamming`, `collatz_length`, `crt_unique`, `cyclic_distance`, `digit_square_error`, `base7_decode`, `cubic_eval`, `digitwise_l1`, `divisor_count`, `euler_totient`, `fib_like_next`, `graph_shortest_path`, `gray_hamming`, `grid_octile`, `grid_nav`, `grid_seven`, `hangman_lite`, `hanoi_three`, `hanoi_two`, `interval_contains`, `letter_overlap_word`, `hot_cold`, `levenshtein_words`, `lights_out_2x2`, `linear_polynomial`, `linear_equation`, `manhattan_point`, `mastermind_aggregate`, `mastermind_classic`, `minesweeper_1d`, `nim_heap`, `parity_groups`, `perm_fixed_points`, `perm_footrule`, `recurrence_second_order`, `quadratic_root`, `product_hotcold`, `rule90_step`, `shift_cipher`, `sudoku_2x2`, `sum_product_xy`, `verbal_bandit`, `wordle_micro`, `xor_subset_hamming`

(All suffixed with `_rf_learning` on Kaggle.)

**Structure:** Multi-turn interaction (20-60 messages). Explore, infer hidden state, exploit. Hidden variables make random action ineffective.

---

## 5. Benchmark Construction Methodology

### Design Philosophy

The goal is not to find tasks that make certain models look good or bad. The goal is to build a **measurement instrument** — like a cognitive test battery for humans — that produces a reliable, interpretable signal about each model's learning capabilities. Every design decision serves measurement validity, not model ranking.

### Phase 1: Cognitive Science Grounding

Before writing any code, we studied how each learning sub-ability is defined and measured in human cognition:

- **Associative learning** draws on classical and operant conditioning paradigms (Rescorla-Wagner model, blocking, latent inhibition, occasion setting).
- **Concept formation** draws on category learning research (prototype theory, exemplar models, hypothesis-testing strategies).
- **Language learning** draws on second language acquisition research (Carroll 1981 LLAMA tests, wug-test methodology from Berko 1958, morphophonological rule induction).
- **Observational learning** draws on Bandura's social learning theory adapted to computational observation — learning a process by watching its inputs and outputs.
- **Reinforcement learning** draws on explore-exploit tradeoffs, bandit problems, and hypothesis-testing under uncertainty.

We then asked: **what is the equivalent cognitive act for an LLM?** The answer is inference-time learning from novel examples in context. The model must do the same cognitive work (induce a rule, track associations, observe and generalize) but through text rather than embodied experience.

### Phase 2: Domain Coverage and Seed Design

For each sub-ability, we enumerated the cognitive facets that a complete evaluation should cover. Examples:

- **Language learning facets:** vowel harmony, consonant harmony, tone sandhi, reduplication, suppletion, ergativity, evidentiality, polysynthetic morphology, bleeding/feeding rule interactions, templatic morphology, switch reference, scope interactions.
- **Observational learning facets:** state machines, group operations, cipher decoding, physics simulations, automata, formal grammars, graph algorithms, cache policies, game-theoretic mechanisms.

For each facet we designed a **seed task** — a concrete, hand-crafted evaluation with a specific novel rule system, a set of training examples, and held-out test items. The seed embodies the cognitive challenge in a form that is:
- **Solvable** by a careful human reasoner
- **Novel** — uses invented names, fictional systems, synthetic rules that do not exist on the internet
- **Unambiguous** — exactly one correct solution derivable from the examples

### Phase 3: Principled Task Generation

From each seed, we generated task variants by:
1. Varying the underlying rule parameters (different harmony patterns, different state transition tables, different hidden variables)
2. Adjusting the number and composition of training examples
3. Ensuring ground truth is **always computed from the rule function** — never manually hardcoded. This is a critical integrity guarantee: the same function that generates examples also grades responses.

### Phase 4: Validation — Is This Task Actually Measuring Learning?

Every candidate task undergoes a five-point validation:

1. **Human feasibility.** A human works through the task from scratch to confirm the rule is discoverable from the given examples alone.
2. **Solution uniqueness.** We verify that exactly one rule is consistent with all examples. If multiple rules fit, we add disambiguating examples or discard the task.
3. **Logical consistency.** Every example is checked for contradictions, edge-case violations, or accidental patterns that could mislead.
4. **Anti-contamination.** We verify the task cannot be solved by recalling known puzzles, benchmark patterns, or standard reasoning templates.
5. **Appropriate difficulty.** The task should be challenging for frontier models yet solvable by a careful human. We target PhD-level difficulty.

### Phase 5: Empirical Calibration Across Model Scales

We run every validated task across 14 models spanning small (Gemma 4 26B, GPT-5.4 nano), mid-tier (GPT-5.4 mini, Gemini 2.5 Flash, Claude Haiku), and frontier (GPT-5.4, Claude Opus, Gemini 3.1 Pro) scales. This serves three purposes:

1. **Feasibility confirmation.** If NO model can solve a task, it may be broken or infeasible — we investigate and either fix or discard it.
2. **Discriminatory power.** If ALL models trivially solve a task, it provides no signal — we either increase difficulty or discard it. A useful task produces a gradient of performance.
3. **Failure mode analysis.** We study WHERE and WHY models fail — is it a genuine learning limitation, or a task design artifact (confusing formatting, ambiguous wording)? We fix design artifacts; genuine learning failures are the signal we want.

The vast majority of initial candidates (~90-95%) are rejected through this process — not because we are selecting for specific model outcomes, but because most randomly parameterized tasks either break validation (ambiguous rules) or lack discriminatory power (too easy or too hard for all models).

### Phase 6: Final Curation and Balance

The surviving tasks are assembled into a balanced benchmark:
- **Varied difficulty** across each category so both small and frontier models produce meaningful scores
- **No redundancy** — each task contributes a distinct signal
- **Cross-domain coverage** — the benchmark spans linguistics, mathematics, physics, CS theory, chemistry, game theory, etc.
- **Final end-to-end execution** — every task is run one more time on the platform to confirm correct behavior

### What This Methodology Is NOT

This is not a process of cherry-picking tasks to favor specific models. The methodology is blind to model identity during task design. Tasks are validated against human solvability and rule uniqueness BEFORE any model sees them. Model evaluation is a calibration step to ensure the measurement instrument works, not a filtering step to manufacture results.

---

## 6. Why Active-Retrieval Is a Learning Protocol

### The Concern

"Our task should be about learning, not retrieval."

### The Clarification

The term "active-retrieval" in our Concept Formation and Language Learning tasks does NOT mean information retrieval (like search or RAG). It means **active learning** in the cognitive science sense: the learner controls their own evidence-gathering process.

Here is exactly what happens:

1. The model receives a small set of initial examples (3-5 input-output pairs generated by a hidden rule).
2. The model must decide: "Do I understand the rule well enough to pass an exam, or do I need to see more examples?"
3. If it requests more, it gets another example — but this costs efficiency points.
4. When it declares it is ready, it takes an exam on novel inputs it has never seen.

This is a **learning protocol**, not a retrieval protocol. The cognitive acts being measured are:

- **Rule induction** — can the model discover the hidden pattern from examples?
- **Hypothesis formation under uncertainty** — does the model know when its understanding is sufficient?
- **Learning efficiency** — how many examples does it need before it can generalize?
- **Strategic metacognition** — does it over-study (request too many examples) or under-study (rush to the exam prematurely)?

This directly mirrors how human learning aptitude is measured. The LLAMA language aptitude test (Carroll 1981) measures exactly this: how quickly a learner can induce grammatical rules from a small number of examples. Our protocol is the LLM equivalent.

### Why This Design Is Better Than "Give All Examples at Once"

Giving all examples at once (as we do in Observational Learning) tests a different ability — working memory processing and pattern recognition from complete data. It does NOT test:
- Whether the model can identify the minimal sufficient evidence set
- Whether the model has metacognitive awareness of its own understanding
- How quickly the model learns (the efficiency dimension)

The interactive protocol isolates the **learning process itself**, not just the learning outcome.

---

## 7. Scoring Design

### 7.1 Category-Level Summary

| Category | Formula | Efficiency? | Key Cognitive Axis |
|---|---|---|---|
| Associative Learning | `correct / total_questions` | No | Causal inference from fixed observations |
| Concept Formation | `accuracy x (0.40 + 0.60 x efficiency)` | Yes | Inductive rule induction under evidence budget |
| Language Learning | `accuracy x (0.40 + 0.60 x efficiency)` | Yes | Multi-system morphophonological induction |
| Observational Learning | `correct_sequences / total_sequences` | No | Hidden process inference from complete demos |
| Runtime RL | `0.55 x solved + 0.25 x efficiency + 0.20 x progress` | Yes | Explore-exploit under hidden state |

### 7.2 Why Efficiency Scoring Matters

Binary accuracy hides a critical dimension. Consider two models that both score 100% accuracy on a concept formation task:

- **Model A** induces the rule from 5 examples and submits immediately.
- **Model B** requests all 20 examples before answering.

Pure accuracy says they are equal. But Model A demonstrates vastly superior learning ability — it abstracts rules from minimal evidence, which is the cognitive core of concept formation.

Our efficiency scoring captures this:
- **Free threshold:** First 3-5 examples are free (the minimum needed to see any pattern)
- **Budget pressure:** Each additional example degrades the efficiency multiplier
- **Floor:** Even a model that uses all examples still earns 40% of its accuracy score (correctness always matters)
- **Formula:** `Score = Accuracy x (0.40 + 0.60 x Efficiency)`

### 7.3 Common Design Principles

1. **Substring matching** for text answers — `re.search` (case-insensitive) so correct answers within reasoning are accepted
2. **Exact match** for numeric answers — `88` must be `88`
3. **Zero-accuracy guard** — zero correct answers always yields score 0.0 (no reward for being fast and wrong)
4. **Free exploration zone** — interactive tasks have penalty-free initial steps
5. **Computed ground truth** — all answers derived from rule functions, not hardcoded

---

## 8. Key Design Decisions

### 8.1 Inference-Time Only

All evaluation within a single conversation. No gradient updates. No fine-tuning. The model must learn from in-context examples.

### 8.2 Varied Difficulty

Tasks calibrated across difficulty levels. Small models should solve some tasks; frontier models should struggle on others. The benchmark produces a gradient, not a binary.

### 8.3 Epistemic Uncertainty Testing

Some tasks include questions where "UNKNOWN" is the correct answer — testing whether models recognize limits of evidence vs. always committing to a definitive answer.

### 8.4 Open to Task Addition/Removal

We are free to add new tasks or remove existing ones to maximize benchmark quality. No fixed task count is required. The goal is signal quality, not task quantity.

---

## 9. Task Quality Criteria

Every task evaluated on five dimensions before inclusion:

| Dimension | Requirement |
|---|---|
| Difficulty calibration | PhD-level. Frontier models may struggle. Careful human can solve. |
| Solution uniqueness | Exactly one rule consistent with all examples. No alternative interpretations. |
| Logical consistency | No contradictions, edge cases, or hidden ambiguities. Deterministic derivation. |
| Novelty / anti-contamination | Cannot be solved from training data or adapted known puzzles. |
| Human-model inversion | Prefer tasks natural for humans but adversarial for LLMs where possible. |

Additional checklist:
- Ground truth computed from rule functions, not hardcoded
- Sufficient examples for learning (not too few, not too many)
- Clean, readable code
- Robust I/O verification
- Demonstrated discriminatory power across model scales
- Human has manually solved the task end-to-end

### Task Feasibility Check: Tasks Where No Model Scores Perfectly

For any task where **no model achieves a perfect score (1.0)**, we perform an additional feasibility check:

> **The question:** Is it actually possible to score 1.0 on this task, or is the task itself flawed?

A task where no model scores perfectly could mean one of two things:
1. **The task is genuinely hard** — it represents a real learning frontier, the kind of task the benchmark is designed to include. These are valuable.
2. **The task has a defect** — ambiguous instructions, inconsistent examples, flawed ground truth, or an impossible rule. These must be fixed or removed.

**How we check:** A human expert (or the task author) manually works through the task from scratch. If a careful human can reach a score of 1.0, the task is feasible and any model failure is a genuine learning limitation. If the human also cannot score 1.0, the task is defective.

This check prevents us from inadvertently keeping broken tasks that look "hard" but are actually unsolvable by design. Tasks that survive this check — those where a human can score perfectly but no model can — are among the most valuable tasks in the benchmark, as they represent genuine upper bounds on current model learning ability.

---

## 10. Ablation Studies and Robustness Checks

The public notebook will include:

### Robustness
- **Test-retest reliability:** Consistency of model scores across repeated runs
- **Prompt-specific effects:** Sensitivity to prompt wording variations
- **Label accuracy:** Verification of ground truth against rule functions

### Diversity
- **Cross-task variance:** Performance distribution within each sub-ability
- **Domain coverage analysis:** Are all intended cognitive facets represented?

### Practicality
- **Example count vs. performance:** How does learning material quantity affect scores? (Key finding: models behave non-linearly — may suddenly drop to 0, unlike human gradual degradation)
- **Scaling curves:** Performance as a function of model scale
- **Initial example impact:** How the free threshold affects learning signal

### Specific Investigations
- Why Gemini 3.1 Pro dominates across categories
- Why larger models sometimes score less on concept formation (over-requesting examples?)
- Concept formation tasks with extreme bimodality (top 2 score 1, rest score 0)
- Observational learning: Gemini 2.5 Flash outperforming GPT-5.4, Claude Opus, and others
- UNKNOWN/epistemic uncertainty handling across model families

---

## 11. Results Summary

### Aggregate Scores by Category (14 models, 157 tasks)

**Concept Formation (19 tasks)**

| Model | Score |
|---|---|
| Gemini 3.1 Pro Preview | 0.724 |
| GLM-5 | 0.550 |
| Qwen 3 Next 80B Thinking | 0.541 |
| Gemini 2.5 Flash | 0.518 |
| Claude Sonnet 4.6 | 0.329 |
| Gemini 3.1 Flash-Lite | 0.323 |
| Gemma 4 26B | 0.278 |
| GPT-5.4 mini | 0.276 |
| GPT-5.4 | 0.268 |
| Claude Opus 4.6 | 0.262 |
| Claude Haiku 4.5 | 0.198 |
| DeepSeek V3.2 | 0.194 |
| Qwen 3 Next 80B Instruct | 0.191 |
| GPT-5.4 nano | 0.156 |

**Language Learning (26 tasks)**

| Model | Score |
|---|---|
| Gemini 3.1 Pro Preview | 0.756 |
| GLM-5 | 0.693 |
| Qwen 3 Next 80B Thinking | 0.623 |
| GPT-5.4 | 0.600 |
| Claude Opus 4.6 | 0.510 |
| Gemini 3.1 Flash-Lite | 0.492 |
| DeepSeek V3.2 | 0.473 |
| Gemini 2.5 Flash | 0.470 |
| Claude Sonnet 4.6 | 0.467 |
| GPT-5.4 mini | 0.463 |
| Qwen 3 Next 80B Instruct | 0.415 |
| Claude Haiku 4.5 | 0.349 |
| GPT-5.4 nano | 0.315 |
| Gemma 4 26B | 0.271 |

**Observational Learning (42 tasks)**

| Model | Score |
|---|---|
| Gemini 3.1 Pro Preview | 0.838 |
| Qwen 3 Next 80B Thinking | 0.580 |
| GLM-5 | 0.558 |
| DeepSeek V3.2 | 0.425 |
| Gemini 2.5 Flash | 0.350 |
| Claude Opus 4.6 | 0.315 |
| Qwen 3 Next 80B Instruct | 0.303 |
| GPT-5.4 | 0.291 |
| Gemini 3.1 Flash-Lite | 0.285 |
| Claude Sonnet 4.6 | 0.255 |
| Claude Haiku 4.5 | 0.241 |
| Gemma 4 26B | 0.207 |
| GPT-5.4 mini | 0.205 |
| GPT-5.4 nano | 0.144 |

**Reinforcement Learning (50 tasks)**

| Model | Score |
|---|---|
| Gemini 3.1 Pro Preview | 0.624 |
| GLM-5 | 0.527 |
| Gemini 2.5 Flash | 0.498 |
| Qwen 3 Next 80B Thinking | 0.474 |
| Gemma 4 26B | 0.448 |
| Claude Sonnet 4.6 | 0.420 |
| DeepSeek V3.2 | 0.415 |
| Gemini 3.1 Flash-Lite | 0.412 |
| GPT-5.4 | 0.411 |
| Claude Opus 4.6 | 0.409 |
| Claude Haiku 4.5 | 0.404 |
| GPT-5.4 mini | 0.288 |
| Qwen 3 Next 80B Instruct | 0.268 |
| GPT-5.4 nano | 0.202 |

**Associative Learning (20 tasks)** — JSON aggregate scores were missing/incorrect on the platform; computed from per-task means. See `analysis/PHASE_A_INSIGHTS.md` for full table.

| Model | Computed Score |
|---|---|
| Gemini 3.1 Pro Preview | 0.935 |
| GLM-5 | 0.792 |
| Claude Opus 4.6 | 0.662 |
| Claude Sonnet 4.6 | 0.651 |
| GPT-5.4 | 0.650 |
| Qwen 3 Next 80B Thinking | 0.628 |
| Gemini 2.5 Flash | 0.595 |
| Gemini 3.1 Flash-Lite Preview | 0.562 |
| DeepSeek V3.2 | 0.523 |
| Claude Haiku 4.5 | 0.502 |
| Gemma 4 26B A4B | 0.501 |
| GPT-5.4 mini | 0.487 |
| Qwen 3 Next 80B Instruct | 0.464 |
| GPT-5.4 nano | 0.437 |

---

## 12. Insights and Novel Hypotheses

### What This Benchmark Reveals That Existing Evaluations Cannot

1. **Learning efficiency is an invisible dimension.** Pure accuracy benchmarks cannot distinguish a model that learns from 5 examples vs. one that needs 20. Our efficiency scoring makes this visible for the first time across frontier models.

2. **"Learning" is not monolithic.** A model that excels at associative learning may fail at language learning. Our five-category decomposition reveals that learning is a family of distinct cognitive skills, not a single ability.

3. **Scale-performance inversions in concept formation.** Larger models sometimes score worse — potentially because they over-request examples (excessive caution, low confidence in their own hypothesis) rather than committing early. This suggests that model scale does not linearly improve learning strategy.

4. **Inference-time concept formation speed.** How many examples does a frontier model need to induce a rule? This is directly analogous to human IQ-test-style pattern recognition speed and has never been systematically measured.

5. **Unexpected model rankings.** Gemini 2.5 Flash outperforms GPT-5.4, Claude Opus 4.6, and others on observational learning. Gemma 4 26B (a small model) outperforms GPT-5.4 and Claude Opus on RL. These inversions suggest that learning ability does not track with general benchmark performance.

6. **Non-linear example count effects.** Unlike humans who degrade gradually, models can suddenly drop to 0% performance as task conditions change — revealing brittle learning that doesn't generalize the way human learning does.

### Dimensions of Learning Ability Quantified

- **Novel learning ability** — can the model learn something genuinely new?
- **Efficiency of learning** — how quickly does it learn?
- **Noise tolerance** — can it learn from imperfect or noisy examples?
- **Transfer** — can it apply learned rules to novel instances?
- **Strategic metacognition** — does it know when it has learned enough?

---

## 13. Deliverables

### Kaggle Benchmark
One unified benchmark containing all finalized tasks. Private until deadline, then auto-published.

### Kaggle Writeup (1500 words max)
Scientific paper style following the mandatory template:
- Project Name / Team / Problem Statement
- Task and benchmark construction
- **Dataset** (see Section 3 for how we address this)
- Technical details
- Results, insights, conclusions
- Affiliations / References

### Public Notebook
Full analysis: methodology, ablation studies, robustness checks, task-by-task results, scaling curves, efficiency analysis, statistical tests, cognitive profile visualizations.

### Cover Image
Required for submission.

---

## 14. Open Questions and Future Work

### Procedural Learning — NOW INCLUDED
Procedural learning (11 tasks) is fully included in the benchmark with scores, conversation logs, and task files across all 14 models. The benchmark now covers six sub-abilities (135 total tasks), not five. Scoring formulas are documented in SCORING.md.

### Tasks Potentially Needing Removal
- All-zero or all-perfect tasks (no discriminatory power)
- Extreme bimodality (only 2 models score non-zero)
- Tasks that skew category-level results
- Tasks that are redundant with other tasks

### We Are Open to Creating New Tasks
In any category, if analysis reveals gaps in coverage or if specific cognitive facets are underrepresented. No fixed task count is mandated.

### Areas We Could Expand
- Long-context learning (book-length novel fiction)
- STEM-specific tasks at PhD level where frontier models genuinely struggle
- Cross-task transfer (does learning in one domain improve learning in another?)
- Video/image modality learning

### Missing Frontier-Failure Tasks
We need more tasks where frontier models with PhD-level intelligence encounter genuine difficulty — especially in STEM domains where their pre-training knowledge is insufficient for completely novel problems.

---

## 15. Models Evaluated

14 models across all leaderboards:

| Model | Slug | Tier |
|---|---|---|
| Gemini 3.1 Pro Preview | gemini-3.1-pro-preview | Frontier |
| GLM-5 | glm-5 | Frontier |
| Claude Opus 4.6 | claude-opus-4.6 | Frontier |
| GPT-5.4 | gpt-5.4 | Frontier |
| Qwen 3 Next 80B Thinking | qwen-3-next-80b-thinking | Mid-tier |
| Gemini 3.1 Flash-Lite Preview | gemini-3.1-flash-lite-preview | Mid-tier |
| Gemini 2.5 Flash | gemini-2.5-flash | Mid-tier |
| Claude Sonnet 4.6 | claude-sonnet-4.6 | Mid-tier |
| DeepSeek V3.2 | deepseek-v3.2 | Mid-tier |
| GPT-5.4 mini | gpt-5.4-mini | Mid-tier |
| Claude Haiku 4.5 | claude-haiku-4.5 | Small |
| GPT-5.4 nano | gpt-5.4-nano | Small |
| Qwen 3 Next 80B Instruct | qwen-3-next-80b-instruct | Small |
| Gemma 4 26B A4B | gemma-4-26b-a4b | Small |


---

## 17. Phase Analysis Insights

Each analysis phase produces a dedicated insights file in `analysis/`. These files record findings, decisions, and questions to carry forward — so later phases can reference earlier work without re-running everything from scratch.

| Phase | File | Status | Summary |
|---|---|---|---|
| Phase A | `analysis/PHASE_A_INSIGHTS.md` | Complete | Data extraction, model tier corrections, per-task and per-model statistics, flagged tasks inventory, plain-language explanations of all key findings |
| Phase B | `analysis/PHASE_B_INSIGHTS.md` | Complete | Deep per-task and per-category analysis, classification of bimodal tasks, identification of tasks to remove or fix |
| Phase C | `analysis/PHASE_C_INSIGHTS.md` | Complete | Robustness + ablation: LOO stability (max rank change=1), random baselines (1.8–75× above random), ground truth spot-check (15/15 pass), epistemic UNKNOWN analysis (H5 inverted), final flagged task list for Phase D |
| Phase D | `analysis/PHASE_D_INSIGHTS.md` | Pending | Task curation decisions — final keep/fix/remove list with rationale |
| Phase E | `analysis/PHASE_E_INSIGHTS.md` | Pending | Writeup preparation — key charts, statistical claims, narrative structure |

**Convention:** Each insights file begins with the date it was produced, the script that generated it, and the output files it references. Findings that affect later phases are explicitly marked **→ Phase B carries this forward** (or similar).

---

## 16. References

### Benchmark Framework Papers (Cite in Writeup)

- Morris, J. et al. (Google DeepMind, 2026). "Measuring progress toward AGI: A cognitive framework." (Framework paper defining the five cognitive faculties: learning, metacognition, attention, executive functions, social cognition.) [PDF](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf) | [Blog post](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/)
- Hendrycks, D. et al. (2025). "A Definition of AGI." arXiv. https://arxiv.org/abs/2510.18212
- **Hendrycks, D. et al. (2020). "Measuring Massive Multitask Language Understanding." *ICLR 2021*. https://arxiv.org/abs/2009.03300** — The dominant existing benchmark. Cite to position LearningBench: MMLU measures *recalled* knowledge; LearningBench measures *acquired* knowledge. The distinction is the core contribution.
- **Srivastava, A. et al. (2022). "Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models." *TMLR 2023*. https://arxiv.org/abs/2206.04615** — BIG-Bench (204+ tasks). The closest large-scale predecessor. Position LearningBench as complementary: BIG-Bench tests capability breadth; LearningBench tests learning mechanism.
- **Brown, T. et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*. https://arxiv.org/abs/2005.14165** — Introduced in-context learning. This is what LearningBench evaluates: the quality of ICL, not just the existence of ICL. Cite in Problem Statement.
- **Chollet, F. (2019). "On the Measure of Intelligence." arXiv:1911.01547. https://arxiv.org/abs/1911.01547** — ARC benchmark. Closest conceptual predecessor: measures fluid intelligence (novel problem-solving), not crystallized knowledge. LearningBench extends this to multi-dimensional learning sub-abilities.
- **Zheng, L. et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023*. https://arxiv.org/abs/2306.05685** — Established that different evaluation methodologies produce different model rankings. Supports our finding that learning-specific evaluation reshuffles rankings compared to general benchmarks.
- **Lake, B.M., Ullman, T.D., Tenenbaum, J.B., & Gershman, S.J. (2017). "Building machines that learn and think like people." *Behavioral and Brain Sciences*, 40, e253.** — Foundational cognitive science argument for why machines should be evaluated on learning, not just knowledge. Already in writeup; ensure cited.

### In-Context Learning Foundation

- Min, S. et al. (2022). "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?" *EMNLP 2022*. https://arxiv.org/abs/2202.12837
- Dong, Q. et al. (2024). "A Survey on In-context Learning." *EMNLP 2024*. https://arxiv.org/abs/2301.00234

### Cognitive Science Foundations (Task Design)

- Carroll, J.B. (1981). "Twenty-five years of research on foreign language aptitude." In K. C. Diller (Ed.), *Individual differences and universals in language learning aptitude* (pp. 83–118). Rowley, MA: Newbury House.
- Berko, J. (1958). "The child's learning of English morphology." *WORD*, 14(2–3), 150–177. https://doi.org/10.1080/00437956.1958.11659661
- Ismayilzada, M. et al. (2024). "Evaluating Morphological Compositional Generalization in Large Language Models." arXiv:2410.12656. https://arxiv.org/abs/2410.12656
- Rescorla, R.A. and Wagner, A.R. (1972). "A theory of Pavlovian conditioning." In A. H. Black & W. F. Prokasy (Eds.), *Classical Conditioning II: Current Research and Theory* (pp. 64–99). Appleton-Century-Crofts.
- Bandura, A. (1977). *Social Learning Theory*. Englewood Cliffs, NJ: Prentice-Hall.

### Benchmark Design Methodology References

- **Olatunji, T. et al. (2025). "AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset." *ACL 2025 Long Papers*, pp. 1948–1973.** — ACL 2025 benchmark paper from Google Research + 13 institutions. 30 models evaluated, 15k questions, multi-metric evaluation axes, qualitative + quantitative methodology. Referenced for: benchmark comparison table design (Table 1 pattern), multi-metric evaluation design (Table 3 pattern), model taxonomy with Domain/Access/Size columns, limitations structure, and ethics section template.
- **Liang, P. et al. (2022). "Holistic Evaluation of Language Models." *NeurIPS 2022*. https://arxiv.org/abs/2211.09110** — HELM. Multi-dimensional evaluation framework. Position LearningBench as targeting a specific under-evaluated dimension (inference-time learning) rather than breadth.

---

## 18. Benchmark Comparison Table (for Writeup Section 2)

> **Design note:** Inspired by AfriMed-QA Table 1 and standard practice in top benchmark papers. Add this as the first table in the writeup to immediately establish novelty and signal field awareness.

| Feature | **LearningBench** | MMLU | BIG-Bench | ARC | HELM | Chatbot Arena |
|---|---|---|---|---|---|---|
| Measures inference-time learning | **✓** | ✗ | Partial | ✗ | ✗ | ✗ |
| Contamination-immune (fully synthetic) | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ |
| Interactive protocols (active evidence) | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ |
| Efficiency scoring (learning speed) | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ |
| Epistemic uncertainty axis (UNKNOWN) | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ |
| Multi-dimensional cognitive profile | **✓** | ✗ | Partial | ✗ | Partial | ✗ |
| Programmatically verified ground truth | **✓** | Partial | Partial | Partial | Partial | Human-judged |
| Models evaluated | 14 | Varies | 132 | Varies | 30 | 100s |
| Tasks | 134 | 15,908 | 204+ | 1,119 | — | Open-ended |
| Novel task instances (no memorization possible) | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## 19. Model Taxonomy (Full Columns for Leaderboard)

> **Design note:** AfriMed-QA Table 3 shows 30 models with columns: Domain, Access (Open/Closed), Size, Type. Adding Provider, Access, and Inference Mode columns to our leaderboard immediately enables the provider-level analysis findings without extra prose.

| Rank | Model | Score | Tier | Provider | Access | Inference Mode |
|---|---|---|---|---|---|---|
| 1 | Gemini 3.1 Pro Preview | 0.843 | Frontier | Google | Closed | Standard |
| 2 | GLM-5 | 0.672 | Frontier | Zhipu AI | Open | Standard |
| 3 | Qwen 3 Next 80B Thinking | 0.603 | Standard | Alibaba | Open | **Thinking** |
| 4 | GPT-5.4 | 0.486 | Frontier | OpenAI | Closed | Standard |
| 5 | Claude Opus 4.6 | 0.477 | Frontier | Anthropic | Closed | Standard |
| 6 | Gemini 2.5 Flash | 0.462 | Standard | Google | Closed | Standard |
| 7 | Claude Sonnet 4.6 | 0.450 | Standard | Anthropic | Closed | Standard |
| 8 | Gemini 3.1 Flash-Lite Preview | 0.436 | Efficient | Google | Closed | Standard |
| 9 | DeepSeek V3.2 | 0.434 | Standard | DeepSeek | Open | Standard |
| 10 | Claude Haiku 4.5 | 0.367 | Efficient | Anthropic | Closed | Standard |
| 11 | GPT-5.4 mini | 0.349 | Standard | OpenAI | Closed | Standard |
| 12 | Gemma 4 26B A4B | 0.347 | Efficient | Google | Open | Standard |
| 13 | Qwen 3 Next 80B Instruct | 0.337 | Standard | Alibaba | Open | Standard |
| 14 | GPT-5.4 nano | 0.244 | Efficient | OpenAI | Closed | Standard |
