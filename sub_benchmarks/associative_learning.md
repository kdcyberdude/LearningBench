# AssocBench: Causal Inference from Observations
**17 novel tasks — does the model distinguish genuine causes from spurious correlations, or does correlation always win?**

This benchmark measures **causal inference from fixed observations** — not pattern counting. Every task presents an invented conditioning experiment: fabricated stimuli, novel entity names, no web trace. Models must decide which cues genuinely cause an outcome, which are confounded, and which remain indeterminate given the evidence.

**17 tasks · Single-turn · Score = fraction of causal-inference questions correct**

| Task | What it tests |
|---|---|
| blocking-effect | Classical blocking: a co-present confirmed cause prevents inference about the co-paired cue |
| counterfactual-sequence-rewrite | Hidden cumulative state machine; tests non-local causal tracking through control tokens |
| inhibitory-summation | Quantitative inhibitory algebra: strong/weak excitors, partial/full inhibitors, modulators |
| latent-cross-binding | Two-dimensional conjunction rule (letter-count parity AND vowel count) on invented words |
| latent-set-binding | Hierarchical tier ordering with a fully held-out token requiring process-of-elimination |
| latent-set-variant | Three-way combinatorial set membership (one token from each of three groups) |
| learned-irrelevance | Learning-theoretic paradigm: pre-exposure history changes acquisition rate |
| occasion-setting | Hierarchical gate logic with 7 modulators and deliberate co-occurrence traps |
| overexpectation | Sub-additive compound conditioning: two confirmed causes together produce less than their sum |
| retrospective-revaluation | Retroactive credit reallocation across a multi-step match ledger |
| second-order-extinction | Multi-phase extinction with second-order chains and partial reinforcement |
| sensory-preconditioning | Multi-operator induction (4 novel infix operators on digits) with composition and inverse queries |
| serial-chain-reconstruction | Dual-phase symbol-to-integer bijection + hidden binary operation induction |
| spurious-hue-true-edge | Spurious 75%-reliable correlate vs. true XOR rule; all 4 test items have the spurious cue pointing wrong |
| temporal-pairing-kmp | Gated dual-signal binding with position-vs-rank selection gates |
| temporal-pairing-tnr | Dual formula switch (arithmetic + identity) triggered by a context token |
| xor-attribute-binding | Three hidden binary features with XOR/XNOR conjunction; one test item is genuinely UNKNOWN |

Scoring is a simple fraction correct across 6–14 questions per task. Questions span four types — confirmed attribution, causal insufficiency (UNKNOWN), compositional prediction, and meta-epistemic counting — so the aggregate score captures breadth of causal reasoning, not just one facet. 

---

## What this benchmark reveals

**Spurious-correlation resistance separates the field.** The hardest tasks (`spurious-hue-true-edge`, `overexpectation`, `counterfactual-sequence-rewrite`) require the model to override a statistically plausible but causally incorrect answer. Most models default to correlation; only models that apply genuine causal reasoning escape.

**Over-attribution is the dominant failure mode.** A model that says "UNKNOWN" too rarely misses confounded-attribution questions; one that says it too readily misses confirmed-attribution questions. Getting both right simultaneously requires understanding conditional independence, which most models do not demonstrate consistently.


https://www.kaggle.com/benchmarks/kdcyberdude/associativelearning/