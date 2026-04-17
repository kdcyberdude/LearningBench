# ConceptBench: Evidence-Calibrated Rule Induction
**18 novel tasks: can the model induce a hidden rule, and does it know when it has seen enough to commit? Efficiency penalizes over-querying.**

This benchmark measures **rule induction with minimal examples and early commitment** — not retrieval or analogy. Every task presents a hidden rule governing a fabricated system (alien grammar, invented cipher, Boolean circuit). Models must discover the rule from labeled examples and then classify novel instances. The efficiency component penalizes over-querying: requesting examples beyond the structural minimum is a sign of failure to generalize.

**18 tasks · Interactive · Score = `accuracy × (0.40 + 0.60 × efficiency)`**

Efficiency = `min_required / examples_requested`. A model that solves with exactly as many examples as the rule requires earns a maximum multiplier of 1.0; one that exhausts the full example budget with perfect accuracy earns at most 0.40.

| Task | What it tests |
|---|---|
| boolean-grid-rules | XOR/XNOR feature conjunction across 2D grid positions; 3 hidden binary attributes |
| boolean-matrix-rules | Layered Boolean formula (AND/OR/XOR/NOR) over 5 binary inputs inferred from a truth-table oracle |
| collatz-rule-induction | Odd/even branching recurrence with invented step constants; tests numeric pattern extraction |
| concept-from-examples | Classic feature-conjunction concept formation (shape × color × size over fabricated tokens) |
| counterfactual-classifier | Nested conditional classifier with 3 attributes and spurious feature; tests counterfactual attribution |
| distractor-rich-rule-induction | 10 surface features; only 3 are predictive; rule must be isolated from 7 irrelevant dimensions |
| feature-conjunction-induction | Two-of-three majority vote over invented attributes; deliberate near-miss traps |
| hidden-rule-extraction | Random morphological alternation rules on fabricated phoneme strings |
| modular-sequence-rule | Modular arithmetic recurrence with a hidden modulus inferred from a positional oracle |
| nested-exception-rule | Three-level exception hierarchy (base rule → exception → exception-to-exception) |
| overfit-detector | Exact memorization vs. generalization forced by held-out test items structurally unlike training |
| pda-rule-induction | Push-down automaton inference from accept/reject strings; hidden stack alphabet |
| phase-transition-induction | Abrupt regime shift at a hidden threshold; tests detection of non-linear transitions |
| recursive-feature-rule | Self-referential rule (output of rule applied to a subset feeds back into prediction) |
| rule-refinement | Multi-phase: initial rule, then counterexample forces revision; tests belief updating |
| sequential-rule-induction | Hidden state machine — output depends on current token plus hidden internal state |
| xor-feature-binding | Three hidden binary features; XOR conjunction; one test item provably UNKNOWN from any evidence |
| znot-rule-induction | Invented negation operator with precedence over AND/OR; symbolic logic induction |

Zero-accuracy guard: if the model cannot exceed chance accuracy, the efficiency multiplier provides no reward. Requesting all examples without learning anything scores 0.

---

## What this benchmark reveals

**Evidence efficiency is the sharpest separator.** Models with similar accuracy scores diverge dramatically on efficiency. The models that score highest are those that commit to a rule after the minimum revealing examples — not those that see everything and then decide.

**Multi-level exception rules expose brittle generalizers.** `nested-exception-rule` and `rule-refinement` require updating a working hypothesis when a counterexample appears. Most models re-describe the exception without modifying the base rule, producing an inconsistent set of conditions.

**Distractor richness exposes attention to relevance.** `distractor-rich-rule-induction` produces a 0.31-point gap between median and top models — the largest single-task gap in this sub-ability — indicating that irrelevant features mislead most models even when the predictive features are redundant.

**All-or-nothing on compositional transformations.** `grid-transform`, `interleave-reverse`, and `layered-transform` are solved fully (1.000) by the top 3–4 models and scored 0.000 by every other model. There is no partial understanding on these tasks. Either the compositional rule was induced correctly, or nothing was. This binary profile exposes an induction threshold, not a gradient of partial knowledge.


https://www.kaggle.com/benchmarks/kdcyberdude/conceptlearning/