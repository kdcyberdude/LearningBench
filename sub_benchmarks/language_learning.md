# LangBench: Phonological Rule Generalization
**26 invented language rules — can the model generalize a phonological pattern to unseen words, not just memorize the training pairs?**

This benchmark measures **phonological rule generalization across novel word forms** — not vocabulary recall. Every task invents a new natural-language-style morphophonological rule (vowel harmony, tone sandhi, reduplication, ergativity) and presents training pairs of (underlying form → surface form). Models must apply the induced rule to held-out words, with exact surface-form matching required. No partial credit, no synonyms, no approximations.

**26 tasks · Interactive · Score = `accuracy × (0.40 + 0.60 × efficiency)`**

Efficiency = `min_required / examples_requested`. Same formula as Concept Formation; the minimum is set per-task to the number of training pairs needed to fully disambiguate the rule.

| Task | What it tests |
|---|---|
| affix-ordering-rule | Two competing affix slots with position-sensitive precedence; 3-way interaction |
| ambiguous-trigger-resolution | Two plausible interpretations of the trigger remain viable until a specific context resolves them |
| conditional-vowel-harmony | Harmony that applies only if a lexical class condition is also met |
| counterintuitive-assimilation | Direction-reversed assimilation: rightmost segment harmonizes to leftmost, not the other way |
| cycle-rule-application | Cyclic application across morphological layers with bleeding order |
| discontinuous-harmony-rule | Non-adjacent harmony (applies across one opaque consonant class) |
| exception-class-induction | Rule applies to class A but not class B; class membership is not marked on the surface |
| feature-harmony-induction | Classic vowel harmony: backness agreement, with height harmony as a distractor |
| featural-underspecification | Underspecified tokens can surface as either value; model must tolerate ambiguity in output |
| implicational-universal | Application of one rule implies a second applies — absence of trigger means neither applies |
| morphological-reduplication | Total/partial reduplication with accent-sensitive truncation |
| multi-feature-harmony | Simultaneous harmony for two independent features (backness + rounding) |
| opacity-bleeding | Rule A bleeds rule B: A's output removes B's trigger; ordering must be inferred |
| opacity-counterfeeding | Rule A counterfeeds rule B: A would create triggers for B but applies too late |
| paradigm-uniformity | Output is determined by the base form in the paradigm, not the local surface context |
| partial-assimilation | Assimilation only to place feature, not to manner; requires featural decomposition |
| prosodic-tier-mapping | Two tonal tiers with different spreading rules; tests multi-tier tracking |
| reduplication-truncation | Reduplication target is a non-obvious prosodic unit (foot, mora) |
| rule-exception-induction | Rule with lexically listed exceptions; tests storage vs. computation trade-off |
| sandhi-cascade | Three ordered sandhi rules; tokens simultaneously trigger multiple rules |
| switch-reference-induction | Switch-reference marking: same vs. different subject across clauses |
| tonal-contour-induction | Contour tone simplification with environment-sensitive outputs |
| tone-sandhi-blocking | Sandhi blocked by a morpheme boundary; boundary is not marked explicitly |
| trigger-sensitive-harmony | Trigger status is itself a surface-level derivable property, not a lexical diacritic |
| velar-softening | Environment-triggered consonant mutation with gradation |
| vowel-raising-lowering | Height harmony with directional raising in front vowels and lowering in back vowels |

Exact surface-form matching is enforced by programmatic comparison against the generating function's output — not human transcription or approximate phonological equivalence.

---

## Domain coverage

Tasks span four categories, ensuring the benchmark is not reducible to a single linguistic modality:

| Domain | Tasks | Indicative examples |
|---|---|---|
| Morphophonology (natural language) | 16 | Vowel harmony, tone sandhi, reduplication, consonant gradation, allomorphy, paradigm uniformity |
| Formal language theory | 4 | CFG induction, PDA inference, regex intersection, syntax tree rewriting |
| Symbol systems & codes | 4 | Codon table inference, substitution ciphers, encoding with unknown endianness, numeral systems |
| Structured query / schema | 2 | SQL WHERE-clause inference, instruction-set reverse-engineering |

Natural language tasks dominate by design: morphophonology is the most direct test of rule generalization because it separates phonological competence (knowing sounds) from phonological *learning* (inducing a rule from examples). The non-natural-language categories are included to test whether the same inductive mechanism transfers, or whether models rely on prior linguistic structure.

---

## What this benchmark reveals

Models do not lack linguistic knowledge — they lack the ability to learn from linguistic evidence. All rules here are invented, so prior training on human languages provides no advantage. Even structurally simple rules, fully determined by three or four examples, are not reliably induced. The bottleneck is induction, not knowledge.

The clearest difficulty axis is rule interaction. Single rules are widely solved. When two rules interact — one removing the trigger of another, one sandhi blocked by a boundary the model never sees — performance collapses. Each added interaction hides an intermediate step, and models cannot reconstruct what they never observed.

Efficiency surfaces a subtler finding: models do not know when they have learned enough. They request the full example budget even on tasks where a few pairs fully determine the rule. This is not caution — it is the absence of any internal signal that generalization has been achieved.


https://www.kaggle.com/benchmarks/kdcyberdude/languagelearning/