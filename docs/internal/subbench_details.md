# Sub-Ability Benchmark Details

Each of LearningBench's six sub-abilities has its own independently structured benchmark. This file documents each one: what cognitive act it targets, how tasks are designed, the scoring formula and its rationale, per-task model scores, and key takeaways.

All 14 models were evaluated identically. Scores are deterministically computed by the same programmatic grader that generates each task — no LLM-as-judge, no hardcoded answer tables.

**Models evaluated (by overall rank):**

| Rank | Model | Overall Score |
|---|---|---|
| 1 | Gemini 3.1 Pro Preview | 0.851 |
| 2 | GLM-5 | 0.692 |
| 3 | Qwen 3 Next 80B Thinking | 0.606 |
| 4 | Claude Opus 4.6 | 0.508 |
| 5 | Gemini 2.5 Flash | 0.507 |
| 6 | GPT-5.4 | 0.493 |
| 7 | Claude Sonnet 4.6 | 0.466 |
| 8 | Gemini 3.1 Flash-Lite Preview | 0.458 |
| 9 | DeepSeek V3.2 | 0.432 |
| 10 | Claude Haiku 4.5 | 0.384 |
| 11 | GPT-5.4 mini | 0.362 |
| 12 | Gemma 4 26B A4B | 0.346 |
| 13 | Qwen 3 Next 80B Instruct | 0.340 |
| 14 | GPT-5.4 nano | 0.246 |

---

## 1. Associative Learning

**17 tasks · Single-turn · Overall mean scores: 0.43 – 0.95**

### What it measures

Associative learning is the capacity to infer causal structure from contingent observations — deciding which stimuli reliably predict an outcome, which are confounded by co-occurring causes, and which remain genuinely indeterminate. It maps onto classical conditioning frameworks but goes beyond simple pattern counting: the model must apply causal reasoning, recognise epistemic limits (when evidence is insufficient to decide), and compose known causes to predict novel compound situations.

Each task presents a structured observation log (e.g. an allergy trial record, a conditioning experiment, a co-occurrence table) and asks the model a multi-question battery about causation, blocking, confounding, and compositional prediction.

### Protocol

- **Format:** Single-turn. One prompt containing all observations; one structured response with answers to all questions.
- **Questions per task:** 6–14, spanning confirmed attributions, UNKNOWN/blocked inferences, compositional predictions, and meta-epistemic counting ("how many cues remain unresolved?").
- **Anti-recall:** All entity names are invented (QREL, MOVEN, THASP, etc.). No web trace exists for these experiments.

### Scoring formula

```
score = correct / total_questions
```

Each question is a binary match: the model's answer either contains the expected token (case-insensitive substring via `re.search`) or it does not. Integer-valued answers require exact integer equality.

**Why no efficiency component:** This is a single-shot reasoning task — the model cannot request more data. All evidence is provided up front. An efficiency penalty would be meaningless.

**Why partial credit (fraction, not pass/fail):** A model answering 7 of 8 questions correctly demonstrates substantially more causal knowledge than one answering 0 of 8. Fraction-correct gives a continuous and informative signal.

**The four question types and what they expose:**

| Question type | What it tests |
|---|---|
| Confirmed attribution | Can the model identify a cue that was solo-tested and confirmed? |
| Causal insufficiency / UNKNOWN | Does the model recognise when a co-present known cause blocks inference? |
| Compositional prediction | Can the model combine confirmed causes according to stated combination rules? |
| Meta-epistemic counting | Does the model know how many cues remain epistemically unresolved? |

A model that always answers "UNKNOWN" will miss confirmed-attribution questions. A model that over-attributes will miss confounded/blocking questions. Getting all types right requires genuine understanding of conditional independence and causal sufficiency.

### Sub-ability score by model

| Model | Associative Score |
|---|---|
| Gemini 3.1 Pro Preview | **0.948** |
| Claude Opus 4.6 | 0.678 |
| Claude Sonnet 4.6 | 0.664 |
| GPT-5.4 | 0.656 |
| Qwen 3 Next 80B Thinking | 0.649 |
| Gemini 2.5 Flash | 0.606 |
| GLM-5 | 0.775 |
| Gemini 3.1 Flash-Lite Preview | 0.595 |
| Gemma 4 26B A4B | 0.511 |
| Qwen 3 Next 80B Instruct | 0.484 |
| DeepSeek V3.2 | 0.523 |
| Claude Haiku 4.5 | 0.501 |
| GPT-5.4 mini | 0.477 |
| GPT-5.4 nano | 0.430 |

### Per-task scores (all 14 models)

| Task | Haiku | Opus | Sonnet | DeepSeek | GLM-5 | GPT-5.4 | GPT-mini | GPT-nano | G-Flash | G-Lite | G-Pro | Gemma | Qwen-I | Qwen-T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocking-effect | 0.625 | 1.000 | 1.000 | 0.750 | 0.500 | 1.000 | 0.750 | 0.875 | 0.500 | 0.625 | 0.500 | 0.875 | 0.875 | 0.500 |
| counterfactual-sequence-rewrite | 0.571 | 0.143 | 0.143 | 0.000 | 1.000 | 0.000 | 0.143 | 0.000 | 0.714 | 0.286 | 1.000 | 0.143 | 0.143 | 0.143 |
| inhibitory-summation | 0.250 | 0.500 | 0.750 | 0.625 | 0.750 | 0.500 | 0.375 | 0.625 | 0.625 | 0.375 | 1.000 | 0.500 | 0.625 | 0.625 |
| latent-cross-binding | 0.500 | 0.500 | 0.667 | 0.500 | 0.333 | 1.000 | 0.333 | 0.667 | 0.500 | 0.667 | 1.000 | 0.500 | 0.500 | 1.000 |
| latent-set-binding | 0.429 | 1.000 | 1.000 | 1.000 | 0.857 | 0.857 | 0.429 | 1.000 | 1.000 | 1.000 | 1.000 | 0.714 | 0.714 | 0.857 |
| latent-set-variant | 0.833 | 1.000 | 0.833 | 0.833 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 | 0.833 | 1.000 | 0.833 | 0.833 | 0.833 |
| learned-irrelevance | 0.750 | 0.875 | 0.875 | 0.750 | 0.875 | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.875 | 0.750 | 0.875 |
| occasion-setting | 0.857 | 1.000 | 1.000 | 0.786 | 0.929 | 1.000 | 0.929 | 0.500 | 0.643 | 1.000 | 0.786 | 0.857 | 0.857 | 0.786 |
| overexpectation | 0.250 | 0.250 | 0.250 | 0.250 | 0.625 | 0.500 | 0.500 | 0.000 | 0.250 | 0.250 | 1.000 | 0.875 | 0.250 | 0.250 |
| retrospective-revaluation | 0.429 | 0.714 | 0.714 | 0.429 | 0.714 | 0.571 | 0.429 | 0.286 | 1.000 | 0.571 | 1.000 | 0.429 | 0.429 | 1.000 |
| second-order-extinction | 0.800 | 1.000 | 1.000 | 0.700 | 1.000 | 1.000 | 0.800 | 0.500 | 0.700 | 0.900 | 1.000 | 0.600 | 0.600 | 0.900 |
| sensory-preconditioning | 0.125 | 0.375 | 0.375 | 0.250 | 1.000 | 0.250 | 0.125 | 0.250 | 0.250 | 0.500 | 1.000 | 0.125 | 0.125 | 0.125 |
| serial-chain-reconstruction | 0.100 | 1.000 | 0.100 | 0.100 | 0.000 | 0.300 | 0.300 | 0.100 | 0.200 | 0.100 | 1.000 | 0.200 | 0.200 | 0.300 |
| spurious-hue-true-edge | 0.000 | 0.000 | 0.250 | 0.000 | 0.750 | 0.000 | 0.000 | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 0.250 | 0.500 |
| temporal-pairing-kmp | 0.333 | 0.500 | 0.667 | 0.500 | 1.000 | 1.000 | 0.333 | 0.000 | 0.167 | 0.500 | 1.000 | 0.000 | 0.167 | 0.833 |
| temporal-pairing-tnr | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 0.500 | 0.750 | 1.000 | 1.000 | 0.500 | 0.750 | 1.000 |
| xor-attribute-binding | 0.667 | 0.667 | 0.667 | 0.667 | 0.833 | 0.167 | 0.167 | 0.167 | 0.667 | 0.500 | 0.833 | 0.667 | 0.167 | 0.500 |

### Key observations

- **Gemini Pro is the clear leader at 0.948**, scoring 1.000 on 10 of 17 tasks. The hardest tasks for every model are those requiring explicit resistance to spurious correlations (`spurious-hue-true-edge`, `counterfactual-sequence-rewrite`) and quantitative sub-additivity (`overexpectation`).
- **`overexpectation` is the hardest task** — only Gemini Pro and Gemma score above 0.5. It requires recognising that two individually confirmed causes together produce *less* than their independent sum, contradicting default linear-combination intuitions.
- **`spurious-hue-true-edge` exposes over-attribution** — most models score 0.0–0.25 because all 4 test items have the spurious cue pointing the wrong way. Models relying on correlation rather than the true XOR rule fail consistently.
- **`sensory-preconditioning` and `serial-chain-reconstruction`** are the other separators: only GLM-5 and Gemini Pro reach ≥ 0.75 on sensory-preconditioning; only Opus and Gemini Pro solve the chain-reconstruction task well.
- **GLM-5 notably outperforms its peers on several hard tasks** (`sensory-preconditioning` 1.000, `counterfactual-sequence-rewrite` 1.000), consistent with its strong overall #2 ranking.

---

## 2. Concept Formation

**18 tasks · Interactive multi-turn · Overall mean scores: 0.15 – 0.80**

### What it measures

Concept formation is the ability to actively induce a hidden rule from input/output examples, controlling your own evidence-gathering (how many examples to request) before committing to an answer. This is the "active learning" dimension of inductive reasoning: the model balances the cost of gathering more evidence against the risk of failing with an under-specified rule.

The critical cognitive act is **meta-calibration** — knowing when you have seen enough. A model that exhausts all 16 available examples before answering is not learning efficiently; a model that commits after 5 examples and gets it right has genuinely grasped the structure.

### Protocol

- **Format:** Multi-turn interactive. The model requests labeled examples one at a time (or in small batches), then declares readiness and answers 4 exam questions.
- **Initial examples:** 3 provided at the start before any requests.
- **Maximum examples:** 16 (13 additional beyond the initial set).
- **Free threshold:** The first 4–5 examples are "free" — no efficiency penalty for using up to the structural minimum needed to observe the pattern.
- **Exam:** 4 held-out test items, presented only after the model declares readiness.

### Scoring formula

```python
score = accuracy × (0.40 + 0.60 × efficiency)
```

Where:
- `accuracy = correct_count / 4` — fraction of exam questions answered correctly
- `efficiency = 1.0` if examples used ≤ free threshold; otherwise `max(0, 1 − paid_used / paid_budget)`
- **Zero-accuracy guard:** if accuracy == 0, score = 0.0 regardless of efficiency

**Boundary values (typical task):**

| Outcome | Score |
|---|---|
| 4/4 correct, ≤ free threshold examples | 1.00 |
| 4/4 correct, all 16 examples used | 0.40 |
| 2/4 correct, ≤ free threshold examples | 0.50 |
| 2/4 correct, all 16 examples used | 0.20 |
| 0/4 correct | 0.00 |

**Why multiplicative, not additive:** `accuracy × (0.40 + 0.60 × efficiency)` means a fast wrong answer earns 0. A model that gets 2/4 correct earns at most half the efficiency bonus of a perfect model.

### Sub-ability score by model

| Model | Concept Score |
|---|---|
| Gemini 3.1 Pro Preview | **0.799** |
| Gemini 2.5 Flash | 0.533 |
| Qwen 3 Next 80B Thinking | 0.563 |
| GLM-5 | 0.568 |
| Claude Opus 4.6 | 0.259 |
| GPT-5.4 | 0.285 |
| Claude Sonnet 4.6 | 0.332 |
| Gemini 3.1 Flash-Lite Preview | 0.313 |
| DeepSeek V3.2 | 0.194 |
| Claude Haiku 4.5 | 0.193 |
| GPT-5.4 mini | 0.224 |
| Gemma 4 26B A4B | 0.255 |
| Qwen 3 Next 80B Instruct | 0.186 |
| GPT-5.4 nano | 0.149 |

### Per-task scores (all 14 models)

| Task | Haiku | Opus | Sonnet | DeepSeek | GLM-5 | GPT-5.4 | GPT-mini | GPT-nano | G-Flash | G-Lite | G-Pro | Gemma | Qwen-I | Qwen-T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| consonant-clusters | 0.000 | 0.000 | 0.100 | 0.207 | 0.500 | 0.000 | 0.000 | 0.000 | 0.239 | 0.479 | 0.957 | 0.121 | 0.000 | 0.468 |
| digit-cipher | 0.000 | 0.400 | 0.000 | 0.400 | 1.000 | 0.000 | 0.000 | 0.000 | 0.160 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| disjunctive-noise | 0.400 | 0.400 | 0.460 | 0.000 | 0.700 | 0.435 | 0.525 | 0.100 | 0.300 | 0.380 | 0.525 | 0.435 | 0.320 | 0.750 |
| dual-recurrence | 0.300 | 0.400 | 0.400 | 0.200 | 0.000 | 0.400 | 0.220 | 0.220 | 0.000 | 0.000 | 0.200 | 0.400 | 0.000 | 0.000 |
| encoded-triple | 0.200 | 0.400 | 0.638 | 0.000 | 1.000 | 0.356 | 0.500 | 0.175 | 1.000 | 0.700 | 1.000 | 0.250 | 0.356 | 0.925 |
| grid-transform | 0.000 | 0.000 | 0.000 | 0.400 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| interleave-reverse | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| layered-transform | 0.000 | 0.000 | 0.000 | 0.000 | 0.963 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.944 |
| modular-subsequence | 0.100 | 0.100 | 0.000 | 0.140 | 0.000 | 0.240 | 0.000 | 0.170 | 0.000 | 0.240 | 0.220 | 0.200 | 0.000 | 0.240 |
| nested-logic | 0.200 | 0.520 | 0.500 | 0.480 | 0.750 | 0.700 | 0.500 | 0.350 | 0.750 | 0.750 | 1.000 | 0.525 | 0.640 | 0.250 |
| positional-encode | 0.200 | 0.100 | 0.220 | 0.100 | 0.250 | 0.600 | 0.000 | 0.130 | 0.750 | 0.250 | 0.960 | 0.440 | 0.190 | 0.250 |
| positional-mapping | 0.400 | 0.400 | 1.000 | 0.000 | 1.000 | 0.727 | 0.250 | 0.227 | 1.000 | 0.473 | 1.000 | 0.200 | 0.455 | 0.945 |
| relational-pairs | 0.420 | 0.400 | 0.560 | 0.510 | 0.600 | 0.660 | 0.660 | 0.450 | 0.720 | 0.750 | 0.600 | 0.560 | 0.450 | 0.750 |
| semantic-override | 0.750 | 0.750 | 0.800 | 0.300 | 0.750 | 0.338 | 0.200 | 0.338 | 0.950 | 0.638 | 0.850 | 0.500 | 0.338 | 0.900 |
| state-machine | 0.300 | 0.400 | 0.586 | 0.300 | 0.709 | 0.300 | 0.445 | 0.341 | 1.000 | 0.209 | 0.727 | 0.400 | 0.300 | 0.500 |
| triple-parity | 0.200 | 0.200 | 0.510 | 0.450 | 0.250 | 0.130 | 0.320 | 0.000 | 0.240 | 0.760 | 0.640 | 0.120 | 0.300 | 0.460 |
| violation-counter | 0.000 | 0.200 | 0.200 | 0.000 | 0.000 | 0.235 | 0.410 | 0.175 | 0.000 | 0.000 | 0.700 | 0.435 | 0.000 | 0.000 |
| vowel-rotation | 0.000 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.479 | 0.000 | 1.000 | 0.000 | 0.000 | 0.750 |

### Key observations

- **This is the most discriminating sub-ability.** Scores span 0.149 (GPT-5.4 nano) to 0.799 (Gemini Pro) — a 5× range. Most models cluster below 0.40.
- **Three tasks are near-universally hard:** `grid-transform`, `interleave-reverse`, and `layered-transform` are solved only by Gemini Pro, Gemini Flash, GLM-5, and Qwen Thinking. All require either exact spatial reasoning or multi-step string manipulation — tasks where imprecise rule induction fails immediately.
- **`vowel-rotation` and `digit-cipher` expose brittle concept induction.** Only Gemini Pro, GLM-5, and Qwen Thinking score above 0.5 — these require understanding rules that interact position and character class, which most models attempt to pattern-match rather than deduce.
- **`relational-pairs` and `disjunctive-noise` are the "easiest" tasks** — most models get partial credit — because the rules are more semantically approachable (modular quadratic residue, noisy disjunction), even if still hard.
- **GLM-5's #2 ranking on this sub-ability is notable:** it is the only non-Gemini model to reach ≥ 0.70 on concept formation, and it achieves 1.000 on `digit-cipher`, `encoded-triple`, `interleave-reverse`, and `positional-mapping`.
- **The Qwen Thinking vs. Instruct gap is visible here:** Thinking scores 0.563 vs. Instruct's 0.186 — a +0.38 uplift, consistent with the benchmark-wide finding that reasoning mode most strongly benefits induction-heavy sub-abilities.


---

## 3. Language Learning

**26 tasks · Interactive multi-turn · Overall mean scores: 0.30 – 0.78**

### What it measures

Language learning measures the capacity to induce morphophonological rules from a small set of labeled form–meaning pairs, then produce novel surface forms for unseen roots. The key cognitive acts are:

- **Phonological rule induction** from surface contrasts (e.g. vowel harmony, tone sandhi, consonant gradation)
- **Multi-system integration** — discovering that several independent rule systems co-apply simultaneously
- **Paradigm extension / wug-test** — generating novel forms for roots that never appeared in training
- **Efficiency under probe budget** — how few examples suffice before the model commits

All language names, morphemes, and phonological environments are purpose-built: DRELKOVAK, GWELTHAR, SKELTH, PRENTOVA, etc. No internet pre-training signal exists for these systems.

### Protocol

- **Format:** Multi-turn interactive. Same active-retrieval protocol as Concept Formation.
- **Exam:** 4 held-out novel root + suffix-template combinations, presented after the model declares readiness. Each form must match the expected surface string **exactly** (Unicode-NFC-normalised).
- **Why exact matching:** A single misapplied harmony feature or missing diacritic produces a wrong surface form. Partial-character credit would hide whether the model genuinely understood the rule.

### Scoring formula

Identical to Concept Formation:

```python
score = accuracy × (0.40 + 0.60 × efficiency)
```

Where accuracy is fraction of the 4 exam forms matched exactly, and efficiency follows the same free-threshold / linear-decay formula.

**Why exact surface form matching instead of substring:** The exam tests productive rule knowledge. A model that discovers only one of two simultaneous harmony systems will produce forms that are wrong in a systematic, predictable way. Partial credit would reward incomplete rule induction.

**Multi-system tasks are harder by design:** Tasks like `drelkovak-harmony` (vowel harmony × pharyngeal consonant harmony) or `gwelthar-mirative-evidential-tone` (tone × evidentiality × mirativity) require the model to simultaneously track independent rule systems. Discovering only one yields partial-correct forms — scored as 0 for those forms.

### Sub-ability score by model

| Model | Language Score |
|---|---|
| Gemini 3.1 Pro Preview | **0.782** |
| GLM-5 | 0.747 |
| Qwen 3 Next 80B Thinking | 0.641 |
| GPT-5.4 | 0.625 |
| Claude Opus 4.6 | 0.518 |
| Claude Sonnet 4.6 | 0.479 |
| Gemini 3.1 Flash-Lite Preview | 0.501 |
| Gemini 2.5 Flash | 0.502 |
| GPT-5.4 mini | 0.497 |
| DeepSeek V3.2 | 0.467 |
| Claude Haiku 4.5 | 0.355 |
| Qwen 3 Next 80B Instruct | 0.436 |
| Gemma 4 26B A4B | 0.248 |
| GPT-5.4 nano | 0.304 |

### Per-task scores (all 14 models)

| Task | Haiku | Opus | Sonnet | DeepSeek | GLM-5 | GPT-5.4 | GPT-mini | GPT-nano | G-Flash | G-Lite | G-Pro | Gemma | Qwen-I | Qwen-T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dimval-metathesis | 0.100 | 0.200 | 0.223 | 0.292 | 0.677 | 0.250 | 0.227 | 0.192 | 0.400 | 0.454 | 1.000 | 0.000 | 0.181 | 0.815 |
| drafnelt-switch-reference | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 | 0.400 | 0.950 | 1.000 | 1.000 | 0.338 | 0.800 | 1.000 |
| dralven-tone-sandhi | 0.225 | 0.400 | 0.200 | 0.200 | 1.000 | 0.375 | 0.675 | 0.188 | 0.300 | 0.638 | 0.300 | 0.100 | 0.250 | 0.450 |
| drelkovak-harmony | 0.210 | 1.000 | 0.500 | 0.660 | 0.720 | 0.750 | 0.230 | 0.000 | 0.750 | 0.250 | 0.750 | 0.280 | 0.220 | 0.750 |
| drelvak-reduplication | 0.300 | 0.400 | 0.500 | 0.461 | 0.589 | 0.250 | 0.196 | 0.307 | 0.700 | 0.500 | 0.614 | 0.300 | 0.218 | 0.436 |
| grelkan-suppletion | 0.360 | 0.400 | 0.300 | 0.600 | 0.640 | 0.300 | 0.230 | 0.260 | 0.300 | 0.240 | 0.480 | 0.280 | 0.440 | 0.000 |
| gwelthar-mirative-evidential-tone | 0.492 | 0.300 | 0.723 | 0.538 | 0.681 | 0.431 | 0.646 | 0.315 | 0.250 | 0.250 | 0.908 | 0.000 | 0.612 | 0.500 |
| kelstran-tone | 0.200 | 0.400 | 0.000 | 0.480 | 0.920 | 0.720 | 0.220 | 0.000 | 0.480 | 0.500 | 0.840 | 0.000 | 0.000 | 0.630 |
| kophar-quantity | 0.400 | 0.750 | 0.743 | 0.500 | 0.914 | 0.829 | 0.686 | 0.621 | 0.443 | 0.750 | 0.750 | 0.493 | 0.743 | 0.914 |
| mixed-radix-number | 0.238 | 0.400 | 1.000 | 0.700 | 0.250 | 0.475 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| norkvash-scalar | 0.460 | 0.580 | 0.000 | 0.520 | 0.820 | 0.820 | 0.570 | 0.000 | 0.400 | 0.250 | 0.580 | 0.260 | 0.760 | 0.500 |
| pelvan-agreement | 0.127 | 0.300 | 0.309 | 0.000 | 0.250 | 0.141 | 0.168 | 0.000 | 0.227 | 0.000 | 0.564 | 0.000 | 0.209 | 0.250 |
| prentova-allomorphy-wugtest | 0.400 | 0.400 | 0.300 | 0.300 | 0.614 | 0.443 | 0.479 | 0.493 | 0.614 | 0.654 | 0.743 | 0.457 | 0.186 | 1.000 |
| skelth-allomorph | 0.400 | 1.000 | 1.000 | 0.589 | 0.654 | 1.000 | 0.654 | 0.700 | 0.000 | 0.750 | 1.000 | 0.364 | 0.621 | 0.957 |
| sklonveth-root-pattern | 0.300 | 0.400 | 0.300 | 0.371 | 0.621 | 0.364 | 0.457 | 0.164 | 0.443 | 0.457 | 0.400 | 0.332 | 0.207 | 0.589 |
| skolvren-polysynthetic | 0.000 | 0.100 | 0.000 | 0.000 | 0.675 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 |
| skovar-deletion | 0.275 | 0.100 | 0.000 | 0.638 | 0.950 | 0.750 | 0.675 | 0.400 | 0.375 | 0.500 | 1.000 | 0.000 | 0.200 | 0.850 |
| strelkov-ergative | 0.600 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.650 | 0.200 | 0.500 | 1.000 | 1.000 | 0.000 | 0.867 | 1.000 |
| strevoklan-neg | 0.100 | 0.640 | 0.000 | 0.000 | 0.700 | 0.615 | 0.440 | 0.200 | 0.640 | 0.000 | 1.000 | 0.200 | 0.570 | 0.290 |
| telvari-evidentiality | 0.564 | 0.891 | 1.000 | 0.891 | 0.945 | 1.000 | 0.891 | 0.673 | 0.673 | 1.000 | 1.000 | 0.455 | 0.836 | 0.891 |
| threlkav-scope-ergativity | 0.782 | 1.000 | 1.000 | 0.891 | 1.000 | 1.000 | 0.945 | 0.782 | 1.000 | 1.000 | 1.000 | 0.727 | 0.836 | 1.000 |
| trenval-bleeding | 0.585 | 0.300 | 0.400 | 0.492 | 0.908 | 0.815 | 0.908 | 0.169 | 0.400 | 0.681 | 0.862 | 0.446 | 0.612 | 0.769 |
| trevkovan-gradation | 0.300 | 0.500 | 0.500 | 0.371 | 0.871 | 1.000 | 0.414 | 0.500 | 1.000 | 0.500 | 0.750 | 0.200 | 0.429 | 1.000 |
| vrelthan-rule-interaction | 0.400 | 0.400 | 0.450 | 0.450 | 0.250 | 0.413 | 0.425 | 0.300 | 0.250 | 0.425 | 0.750 | 0.400 | 0.450 | 0.250 |
| vrendel-templatic | 0.382 | 0.400 | 0.509 | 0.000 | 0.945 | 1.000 | 0.709 | 0.586 | 0.400 | 0.750 | 1.000 | 0.618 | 0.727 | 1.000 |
| wukal-tones | 0.341 | 0.200 | 0.500 | 0.445 | 0.836 | 0.500 | 0.473 | 0.464 | 0.545 | 0.473 | 0.500 | 0.209 | 0.364 | 0.836 |

### Key observations

- **This is the sub-ability where GLM-5 comes closest to matching Gemini Pro** (0.747 vs. 0.782). Open-source GLM-5 outranks every closed-source model except Google's on language induction.
- **`skolvren-polysynthetic` is the hardest task overall** — only GLM-5 (0.675) and Gemini Pro (0.550) score above 0. Six-slot polysynthetic verb template with noun incorporation and classifiers defeats every other model.
- **`threlkav-scope-ergativity` and `drafnelt-switch-reference` are near-solved** — most frontier models reach ≥ 0.80. These involve clear, learnable syntactic rules (switch-reference markers, scope-case interaction) where the pattern is structurally unambiguous.
- **`pelvan-agreement` remains hard even for top models:** Five-dimensional agreement grids (person × number × gender × animacy × case) with animacy override — even Gemini Pro only reaches 0.564.
- **GLM-5 achieves 1.000 on 4 tasks** (`drafnelt-switch-reference`, `dralven-tone-sandhi`, `skovar-deletion`, `strelkov-ergative`) — evidence of strong rule-induction capacity across very different phonological and syntactic domains.
- **Gemma 4 26B A4B is the weakest model** on language learning (0.248), scoring 0.000 on 6 tasks including `skolvren-polysynthetic`, `grelkan-suppletion`, and `gwelthar-mirative-evidential-tone`.

---

## 4. Observational Learning

**30 tasks · Single-shot · Overall mean scores: 0.22 – 0.85**

### What it measures

Observational learning measures the capacity to infer a hidden computational process (a rule, a machine, an operation) from a complete set of input/output demonstrations, then predict outputs for novel inputs. Unlike Concept Formation, the model receives **all demonstrations at once** — no active retrieval. The difficulty comes entirely from the complexity of the hidden process: aliased states in transducers, deceptive early patterns in group operations, interacting transformations in affine chains.

### Protocol

- **Format:** Single-shot. All training demonstrations provided in one prompt; 4 test queries answered in one response.
- **Grading:** Fraction of the 4 test cases for which the model's predicted output exactly matches the ground truth.
- **Anti-deception design:** Tasks are deliberately structured so shallow pattern matching fails. E.g. in `hidden-damping-physics`, the first 3 data points are consistent with a simpler model; the true damping constant only manifests across longer sequences.

### Scoring formula

```
score = correct_count / 4
```

(fraction of test cases fully correct)

**Why no efficiency component:** The model does not control data access — all demonstrations are provided up front. There is no "cost" to study time; the only dimension that matters is whether the hidden rule was correctly inferred.

**Why whole-sequence correctness (not positional partial credit):** For hidden state machines, a single error in state tracking cascades to wrong outputs for all subsequent steps. Positional partial credit would mask the core failure — the model did not fully infer the hidden structure.

### Sub-ability score by model

| Model | Observational Score |
|---|---|
| Gemini 3.1 Pro Preview | **0.853** |
| GLM-5 | 0.647 |
| Qwen 3 Next 80B Thinking | 0.668 |
| Gemini 2.5 Flash | 0.443 |
| Claude Opus 4.6 | 0.391 |
| Claude Sonnet 4.6 | 0.287 |
| GPT-5.4 | 0.309 |
| Gemini 3.1 Flash-Lite Preview | 0.329 |
| DeepSeek V3.2 | 0.428 |
| Claude Haiku 4.5 | 0.271 |
| GPT-5.4 mini | 0.215 |
| Gemma 4 26B A4B | 0.218 |
| Qwen 3 Next 80B Instruct | 0.241 |
| GPT-5.4 nano | 0.144 |

### Per-task scores (all 14 models)

| Task | Haiku | Opus | Sonnet | DeepSeek | GLM-5 | GPT-5.4 | GPT-mini | GPT-nano | G-Flash | G-Lite | G-Pro | Gemma | Qwen-I | Qwen-T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| affine-transform-chain | 0.000 | 0.750 | 0.000 | 0.500 | 0.000 | 0.500 | 0.250 | 0.000 | 0.500 | 0.000 | 0.500 | 0.000 | 0.000 | 0.500 |
| agglutinative-morphology | 0.250 | 0.500 | 0.250 | 0.000 | 0.500 | 0.000 | 0.250 | 0.250 | 0.250 | 0.250 | 1.000 | 0.000 | 0.000 | 0.250 |
| arithmetic-entropy-coding | 0.000 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 0.000 | 0.000 | 1.000 | 0.500 | 1.000 | 0.250 | 1.000 | 1.000 |
| auction-mechanism-second-price | 0.500 | 0.500 | 0.500 | 0.750 | 1.000 | 0.500 | 0.250 | 0.250 | 0.250 | 0.250 | 1.000 | 0.250 | 0.500 | 0.250 |
| codon-table-translation | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| feistel-cipher-round | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.750 |
| finite-state-transducer | 0.250 | 0.250 | 0.250 | 0.000 | 1.000 | 0.500 | 0.000 | 0.000 | 1.000 | 0.250 | 1.000 | 0.000 | 0.000 | 1.000 |
| flow-network-capacity | 0.750 | 0.500 | 0.250 | 0.500 | 0.000 | 0.500 | 0.250 | 0.000 | 0.250 | 0.250 | 0.750 | 0.500 | 0.000 | 0.500 |
| hidden-attribute-rule | 0.625 | 0.625 | 0.500 | 0.625 | 0.750 | 0.750 | 0.750 | 0.500 | 0.750 | 0.500 | 0.750 | 0.500 | 0.625 | 0.375 |
| hidden-damping-physics | 0.250 | 0.250 | 0.000 | 0.750 | 1.000 | 0.500 | 0.000 | 0.250 | 0.250 | 0.500 | 1.000 | 0.250 | 0.000 | 1.000 |
| hidden-matrix-fill | 0.000 | 0.000 | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| hidden-modal-logic-kripke | 0.375 | 0.625 | 0.500 | 0.500 | 0.000 | 0.375 | 0.500 | 0.750 | 0.375 | 0.625 | 0.375 | 0.250 | 0.000 | 0.000 |
| hidden-modal-logic-kripke2 | 0.500 | 0.625 | 0.500 | 0.375 | 0.000 | 0.500 | 0.250 | 0.375 | 0.625 | 0.625 | 1.000 | 0.250 | 0.500 | 0.500 |
| hidden-priority-order | 0.000 | 0.750 | 0.000 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 |
| hidden-token-filter | 0.250 | 0.250 | 0.250 | 0.250 | 0.750 | 0.250 | 0.000 | 0.000 | 0.750 | 0.250 | 1.000 | 0.000 | 0.000 | 0.750 |
| lattice-meet-join | 0.000 | 0.000 | 0.000 | 0.250 | 0.750 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.500 |
| lfu-cache-eviction | 0.250 | 0.500 | 0.500 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 | 1.000 | 0.000 | 0.500 | 0.750 |
| linear-feedback-shift-register | 0.000 | 0.250 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 1.000 | 0.250 | 0.000 | 0.250 |
| mealy-machine-output | 0.000 | 0.500 | 0.500 | 0.250 | 0.750 | 0.500 | 0.000 | 0.250 | 0.000 | 0.500 | 0.750 | 0.250 | 0.250 | 0.750 |
| phonological-alternation-harmony | 0.750 | 1.000 | 1.000 | 0.750 | 1.000 | 0.500 | 1.000 | 0.250 | 1.000 | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 |
| pipeline-hazard-stall-counting | 0.500 | 0.000 | 0.000 | 0.500 | 1.000 | 0.250 | 0.250 | 0.000 | 0.250 | 0.000 | 1.000 | 0.250 | 0.000 | 0.500 |
| pushdown-automaton-inference | 0.500 | 0.750 | 0.750 | 0.500 | 1.000 | 0.750 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 |
| regex-intersection-membership | 1.000 | 0.750 | 1.000 | 0.750 | 1.000 | 1.000 | 0.750 | 0.250 | 0.750 | 0.750 | 1.000 | 1.000 | 0.000 | 1.000 |
| ring-operations-hidden-carry | 0.000 | 0.000 | 0.250 | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.250 | 0.250 | 1.000 | 0.000 | 0.000 | 1.000 |
| shapley-values-cooperative-game | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| sigil-naming | 0.250 | 0.000 | 0.000 | 0.250 | 1.000 | 0.000 | 0.250 | 0.000 | 0.250 | 0.000 | 1.000 | 0.250 | 0.250 | 0.250 |
| singleton-gate-local-max | 0.250 | 0.500 | 0.250 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 | 0.500 | 0.250 | 0.000 | 0.000 |
| syntax-tree-rewrite | 0.500 | 1.000 | 0.250 | 0.750 | 1.000 | 0.500 | 0.500 | 0.250 | 0.250 | 0.500 | 1.000 | 0.250 | 0.750 | 0.750 |
| titration-curve-diprotic | 0.200 | 0.200 | 0.200 | 0.600 | 0.000 | 0.400 | 0.200 | 0.200 | 0.200 | 0.200 | 0.800 | 0.200 | 0.200 | 0.400 |
| two-counter-machine | 0.167 | 0.167 | 0.167 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 | 0.833 | 0.667 | 0.167 | 0.333 | 0.167 | 1.000 |

### Key observations

- **`shapley-values-cooperative-game`, `feistel-cipher-round`, and `lattice-meet-join`** are solved only by Gemini Pro, GLM-5, and Qwen Thinking. These require deep structural inference (cooperative game theory, 2-round Feistel reversal, abstract algebraic lattice operations) that eludes pattern matching.
- **`pushdown-automaton-inference` and `regex-intersection-membership`** are among the most broadly solvable tasks — most frontier models score ≥ 0.75. Automata inference from demonstrations appears more tractable when the model has strong formal language foundations.
- **`codon-table-translation`** shows a sharp split: 0.000 for 8 models, 1.000 for Gemini Pro/GLM-5/Qwen Thinking/DeepSeek — suggesting that this task either fully clicks or doesn't at all.
- **Qwen Thinking achieves 1.000 on 9 tasks** and scores 0.668 overall — the #3 position despite being a non-Google model. The Thinking vs. Instruct gap here is +0.43 (the largest gap across all sub-abilities), matching the paper's finding that observational tasks benefit most from extended reasoning.
- **`two-counter-machine` is the surprise:** Qwen Thinking (1.000) and Gemini Flash (0.833) outperform Gemini Pro (0.167) — one of the few tasks where chain-of-thought iteration appears to matter more than overall capability.
- **GLM-5 achieves 1.000 on 13 of 30 tasks**, the most perfect-scores of any non-Gemini model.

---

## 5. Reinforcement Learning

**30 tasks · Multi-turn interactive · Overall mean scores: 0.25 – 0.93**

### What it measures

Runtime RL measures the capacity to solve an unknown problem through sequential interaction — probing the environment with actions, receiving feedback, updating a hypothesis about hidden state, and converging on a solution within a step budget. The model must simultaneously:

1. **Explore** to identify hidden parameters (goal peg, secret code, forbidden rule, etc.)
2. **Exploit** its inferred model to reach the goal efficiently
3. **Track progress** under partial and sometimes veiled feedback

This is the closest LearningBench comes to measuring online learning: a closed loop of probe → observe → update → act.

### Protocol

- **Format:** Multi-turn. Each turn the model issues an action; the environment returns feedback.
- **Step budget:** Typically 20–50 turns depending on task complexity. A "free exploration zone" (usually 7 initial steps) incurs no efficiency penalty.
- **Hidden variables:** Every task hides one or more parameters the model must infer. Random action leads to slow convergence; systematic hypothesis updating earns both efficiency and success.

### Scoring formula

```python
score = 0.55 * solved + 0.25 * efficiency + 0.20 * progress
```

| Component | Weight | When active | What it measures |
|---|---|---|---|
| `solved` | 0.55 | Always | Binary: did the model complete the task? |
| `efficiency` | 0.25 | Only when solved | Speed relative to step budget (conditioned on success) |
| `progress` | 0.20 | Always | Partial completion even when not solved |

**Why success dominates (0.55):** The primary cognitive act in RL is completing the explore → infer → solve loop. A model that never solves any task scores near zero.

**Why efficiency is conditioned on success:** Efficiency without success is uninterpretable in RL — we cannot know whether incomplete steps were purposeful probes or random moves. The 0.10 floor prevents a last-step solution from scoring zero on efficiency.

**Why progress always contributes (0.20):** A model that consistently approaches the goal but runs out of budget still demonstrates partial learning. Progress is defined task-specifically: fraction of valid moves used, disks on goal peg, LOCK fraction in Mastermind, etc.

**The token-failure signal:** Across all RL runs, failed runs consume **4.3× more tokens** than solved runs (177K vs. 41K average). Many models, once their first hypothesis is wrong, cannot update at all — they repeat the same action or thrash. Token spend is a real-time failure indicator.

### Sub-ability score by model

| Model | RL Score |
|---|---|
| Gemini 3.1 Pro Preview | **0.929** |
| GLM-5 | 0.793 |
| Claude Opus 4.6 | 0.687 |
| Claude Haiku 4.5 | 0.566 |
| Gemma 4 26B A4B | 0.552 |
| Claude Sonnet 4.6 | 0.635 |
| GPT-5.4 | 0.663 |
| Gemini 2.5 Flash | 0.505 |
| Gemini 3.1 Flash-Lite Preview | 0.585 |
| DeepSeek V3.2 | 0.534 |
| GPT-5.4 mini | 0.454 |
| Qwen 3 Next 80B Thinking | 0.633 |
| Qwen 3 Next 80B Instruct | 0.332 |
| GPT-5.4 nano | 0.251 |

### Per-task scores (all 14 models)

| Task | Haiku | Opus | Sonnet | DeepSeek | GLM-5 | GPT-5.4 | GPT-mini | GPT-nano | G-Flash | G-Lite | G-Pro | Gemma | Qwen-I | Qwen-T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| affine-cipher-word | 0.040 | 0.040 | 0.040 | 0.000 | 0.787 | 0.800 | 0.000 | 0.800 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |
| arithmetic-next | 0.129 | 0.186 | 0.171 | 0.129 | 0.925 | 0.129 | 0.114 | 0.100 | 0.925 | 0.129 | 0.933 | 0.129 | 0.100 | 0.900 |
| base7-decode | 1.000 | 0.800 | 1.000 | 1.000 | 1.000 | 0.200 | 0.200 | 0.156 | 0.925 | 0.106 | 0.800 | 1.000 | 0.125 | 1.000 |
| battleship-1d | 0.940 | 0.828 | 0.000 | 0.940 | 0.907 | 0.934 | 0.974 | 0.000 | 0.788 | 0.000 | 1.000 | 0.960 | 0.000 | 0.000 |
| battleship-two-ships | 0.972 | 0.850 | 0.972 | 0.100 | 0.953 | 1.000 | 0.981 | 0.100 | 0.100 | 1.000 | 1.000 | 0.100 | 0.100 | 0.100 |
| bitstring-hamming | 0.143 | 0.143 | 0.186 | 0.114 | 0.171 | 0.186 | 0.129 | 0.071 | 0.114 | 0.785 | 0.949 | 0.171 | 0.000 | 0.171 |
| chebyshev-point | 0.186 | 0.991 | 0.939 | 0.983 | 1.000 | 1.000 | 0.186 | 0.157 | 1.000 | 1.000 | 1.000 | 0.164 | 0.164 | 1.000 |
| coin-balance | 1.000 | 1.000 | 0.802 | 0.960 | 1.000 | 1.000 | 0.960 | 0.200 | 0.881 | 0.800 | 1.000 | 1.000 | 0.841 | 1.000 |
| collatz-length | 0.836 | 0.928 | 0.044 | 0.102 | 0.060 | 0.020 | 0.014 | 0.012 | 0.888 | 0.044 | 0.888 | 0.019 | 0.025 | 0.046 |
| crt-unique | 0.200 | 1.000 | 0.200 | 0.809 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.809 | 1.000 | 1.000 | 0.933 | 0.978 |
| cubic-eval | 0.980 | 1.000 | 0.949 | 0.969 | 1.000 | 0.969 | 0.200 | 0.828 | 0.959 | 0.990 | 1.000 | 1.000 | 0.816 | 1.000 |
| digitwise-l1 | 0.129 | 0.162 | 0.153 | 0.166 | 0.154 | 0.173 | 0.162 | 0.160 | 0.136 | 0.170 | 1.000 | 0.160 | 0.154 | 0.148 |
| divisor-count | 0.790 | 0.800 | 0.698 | 0.759 | 0.728 | 0.728 | 0.800 | 0.000 | 0.800 | 0.800 | 0.800 | 0.800 | 0.000 | 0.728 |
| fib-like-next | 1.000 | 0.991 | 0.991 | 0.953 | 0.972 | 0.200 | 0.200 | 0.897 | 0.200 | 0.863 | 0.850 | 0.900 | 0.813 | 0.950 |
| graph-shortest-path | 0.800 | 0.844 | 0.879 | 0.844 | 0.874 | 0.120 | 0.860 | 0.080 | 0.810 | 0.888 | 0.894 | 0.844 | 0.800 | 0.800 |
| gray-hamming | 0.150 | 0.000 | 0.183 | 0.133 | 0.117 | 0.918 | 0.117 | 0.100 | 0.150 | 0.150 | 0.949 | 0.133 | 0.150 | 0.133 |
| hanoi-two | 0.200 | 1.000 | 0.857 | 0.200 | 1.000 | 1.000 | 0.993 | 1.000 | 0.090 | 0.898 | 1.000 | 1.000 | 0.067 | 0.966 |
| manhattan-point | 1.000 | 0.896 | 1.000 | 0.991 | 1.000 | 0.862 | 1.000 | 0.948 | 0.827 | 1.000 | 1.000 | 0.100 | 0.100 | 1.000 |
| mastermind-aggregate | 0.080 | 0.200 | 0.925 | 0.200 | 0.936 | 0.882 | 0.080 | 0.120 | 0.200 | 0.160 | 0.200 | 0.080 | 0.120 | 0.200 |
| perm-footrule | 0.000 | 0.000 | 0.946 | 0.067 | 0.979 | 0.946 | 0.067 | 0.067 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| product-hotcold | 1.000 | 0.897 | 0.831 | 0.146 | 0.897 | 1.000 | 1.000 | 0.164 | 0.191 | 0.963 | 1.000 | 0.182 | 0.182 | 0.953 |
| quadratic-root | 1.000 | 1.000 | 1.000 | 0.982 | 1.000 | 0.982 | 0.185 | 0.185 | 1.000 | 1.000 | 1.000 | 0.847 | 0.197 | 1.000 |
| recurrence-second-order | 0.200 | 0.983 | 0.983 | 0.948 | 1.000 | 0.965 | 0.200 | 0.200 | 0.200 | 0.900 | 0.974 | 0.950 | 0.050 | 0.900 |
| rule90-step | 0.888 | 0.933 | 0.854 | 0.921 | 1.000 | 0.899 | 0.910 | 0.133 | 0.156 | 1.000 | 1.000 | 0.000 | 1.000 | 0.089 |
| shift-cipher | 0.200 | 0.839 | 0.200 | 0.200 | 0.933 | 1.000 | 0.933 | 0.200 | 0.200 | 0.871 | 0.867 | 0.867 | 0.933 | 0.933 |
| sudoku-2x2 | 1.000 | 0.993 | 0.958 | 0.888 | 1.000 | 0.838 | 0.020 | 0.020 | 1.000 | 0.020 | 1.000 | 1.000 | 0.958 | 1.000 |
| sum-product-xy | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 0.200 | 0.933 | 1.000 | 0.991 | 0.200 | 1.000 |
| verbal-bandit | 0.885 | 0.200 | 0.180 | 0.200 | 0.200 | 0.060 | 0.850 | 0.200 | 1.000 | 0.900 | 1.000 | 0.200 | 0.900 | 0.900 |
| wordle-micro | 0.167 | 0.200 | 0.200 | 0.167 | 1.000 | 0.100 | 0.100 | 0.100 | 0.200 | 0.200 | 0.970 | 0.000 | 0.067 | 0.100 |
| xor-subset-hamming | 0.140 | 0.918 | 0.918 | 0.160 | 1.000 | 0.785 | 0.180 | 0.140 | 1.000 | 0.160 | 1.000 | 0.160 | 0.180 | 1.000 |

### Key observations

- **RL is the highest-scoring sub-ability for most models** — many tasks have clear algorithmic solutions (Battleship search, coin-balance weighing, quadratic root-finding) that reward systematic exploration without requiring deep structural induction.
- **`wordle-micro`, `digitwise-l1`, and `bitstring-hamming` are the hardest tasks** for most models. These involve veiled or indirect feedback that prevents straightforward hypothesis updating.
- **`digitwise-l1`** is uniquely difficult: only Gemini Pro (1.000) escapes the 0.13–0.17 floor. All other models appear to guess randomly, unable to interpret the L1-distance feedback signal.
- **Battleship tasks reveal a sharp competence split:** `battleship-1d` is solved (0.94+) by models with systematic search strategies; `battleship-two-ships` is harder — Claude Sonnet, DeepSeek, all Qwen variants score ≤ 0.10, suggesting they cannot coordinate a two-target search strategy.
- **`collatz-length`** shows an unusual pattern: Haiku (0.836) and Opus (0.928) strongly outperform GPT-5.4 (0.020) and GLM-5 (0.060). This task requires recognising a recursive sequence structure from indirect feedback, which appears to be model-family-specific.
- **The Qwen Thinking vs. Instruct gap is +0.30 on RL** (0.633 vs. 0.332), the second-largest sub-ability gap, consistent with the paper's finding that reasoning helps explore-exploit tasks.
- **`verbal-bandit`** is a notable outlier: Gemini Flash (1.000), Gemini Pro (1.000), and Qwen Instruct/Thinking (0.900) outperform Claude Opus (0.200) and GPT-5.4 (0.060). The task requires natural-language hypothesis updating from soft feedback, which favours models with stronger pragmatic inference.

---

## 6. Procedural Learning

**11 tasks · Multi-episode with transfer · Overall mean scores: 0.21 – 0.73**

### What it measures

Procedural learning is the capacity to acquire a skill, strategy, or action pattern **through repeated performance with corrective feedback** — and then apply that learned procedure to novel instances without assistance. It is the only sub-ability that explicitly measures the *learning trajectory* across practice trials, not just final accuracy.

The defining structure:

- **Practice phase (5 rounds):** The model interacts with the environment, receives feedback after each attempt, and must improve its strategy over trials.
- **Transfer phase (4 tests):** The model faces novel instances with *no corrective feedback whatsoever*, relying entirely on the procedure internalised during practice.

This two-phase design separates *learning* (did performance improve with practice?) from *transfer* (can the model apply what it learned cold?).

### Protocol

- **Format:** Multi-episode. Each of the 5 practice rounds is a complete interaction with the environment, followed by corrective feedback.
- **Transfer:** 4 held-out instances presented cold — no hints, no feedback during transfer.
- **Why this structure is unique:** No other major benchmark directly measures whether practice caused improvement. A model that scores 70% on round 1 and 70% on round 5 has not learned; a model that scores 30% on round 1 and 80% on round 5 has.

### Scoring formula

```python
score = 0.30 * transfer + 0.25 * asymptote + 0.25 * trajectory + 0.20 * consistency
```

| Component | Weight | What it captures |
|---|---|---|
| `transfer` | 0.30 | Fraction of cold transfer tests passed — the acid test of generalisation |
| `asymptote` | 0.25 | Mean efficiency of the *last half* of practice rounds — peak skill reached |
| `trajectory` | 0.25 | OLS slope of scores over rounds, normalised to [0,1] — did practice cause improvement? |
| `consistency` | 0.20 | Linearly-weighted mean across all practice rounds — steady reliable performance |

**Trajectory detail:** A slope of 1.0 = linear improvement from 0→1 across rounds. A slope of 0.5 = flat. A slope of 0.0 = linear deterioration. This component is **99% orthogonal to the asymptote** (Spearman ρ = −0.02, R² = 0.01 across 112 runs) — two models landing at the same final score can have radically different trajectories.

**Zero guard:** If asymptote < ε AND transfer < ε, return 0.0. A model that never solved any practice round and failed all transfer tests earns no spurious trajectory bonus.

**Score range profiles:**

| Profile | Approximate score |
|---|---|
| Perfect learner: linear improvement + all transfer passed | 0.90–1.00 |
| Strong learner: high asymptote + positive trajectory + good transfer | 0.65–0.85 |
| Pre-trained expert: flat-high practice + good transfer | 0.55–0.75 |
| Slow learner: low early + strong finish + moderate transfer | 0.45–0.65 |
| Inconsistent: high-variance practice + poor transfer | 0.25–0.45 |
| Non-learner: never solves practice + fails transfer | 0.00–0.15 |

### Sub-ability score by model

| Model | Procedural Score |
|---|---|
| Gemini 3.1 Pro Preview | **0.727** |
| Gemini 2.5 Flash | 0.485 |
| Claude Opus 4.6 | 0.454 |
| Qwen 3 Next 80B Instruct | 0.450 |
| Claude Sonnet 4.6 | 0.373 |
| GLM-5 | 0.536 |
| GPT-5.4 | 0.283 |
| Gemini 3.1 Flash-Lite Preview | 0.409 |
| DeepSeek V3.2 | 0.347 |
| Claude Haiku 4.5 | 0.352 |
| GPT-5.4 mini | 0.221 |
| Gemma 4 26B A4B | 0.237 |
| Qwen 3 Next 80B Thinking | 0.344 |
| GPT-5.4 nano | 0.202 |

### Per-task scores (all 14 models)

| Task | Haiku | Opus | Sonnet | DeepSeek | GLM-5 | GPT-5.4 | GPT-mini | GPT-nano | G-Flash | G-Lite | G-Pro | Gemma | Qwen-I | Qwen-T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive-sort-rule | 0.390 | 0.910 | 0.828 | 0.436 | 0.903 | 0.520 | 0.330 | 0.539 | 0.835 | 0.303 | 0.814 | 0.388 | 0.371 | 0.000 |
| boolean-circuit | 0.260 | 0.260 | 0.260 | 0.242 | 0.260 | 0.000 | 0.000 | 0.000 | 0.260 | 0.260 | 0.760 | 0.166 | 0.754 | 0.254 |
| dialect-morphology | 0.642 | 0.202 | 0.520 | 0.464 | 0.593 | 0.575 | 0.586 | 0.575 | 0.586 | 0.586 | 0.511 | 0.207 | 0.587 | 0.575 |
| grammar-induction | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.640 | 0.000 | 0.000 | 0.000 |
| lights-out-variant | 0.552 | 0.546 | 0.668 | 0.000 | 0.626 | 0.500 | 0.000 | 0.000 | 0.796 | 0.712 | 0.796 | 0.052 | 0.626 | 0.558 |
| nim-variant | 0.000 | 0.442 | 0.000 | 0.000 | 0.275 | 0.000 | 0.000 | 0.000 | 0.588 | 0.000 | 0.663 | 0.000 | 0.000 | 0.200 |
| opponent-strategy | 0.583 | 0.607 | 0.685 | 0.760 | 0.708 | 0.405 | 0.468 | 0.382 | 0.457 | 0.468 | 0.875 | 0.685 | 0.875 | 0.252 |
| packet-filter | 0.237 | 0.377 | 0.425 | 0.275 | 0.258 | 0.279 | 0.330 | 0.000 | 0.285 | 0.292 | 0.605 | 0.000 | 0.200 | 0.148 |
| sql-reverse-engineering | 0.397 | 0.508 | 0.247 | 0.306 | 0.800 | 0.257 | 0.309 | 0.252 | 0.602 | 0.433 | 0.800 | 0.529 | 0.115 | 0.230 |
| state-machine-password | 0.607 | 0.590 | 0.402 | 0.520 | 0.606 | 0.578 | 0.405 | 0.469 | 0.539 | 0.514 | 0.589 | 0.569 | 0.620 | 0.677 |
| voting-protocol | 0.210 | 0.050 | 0.070 | 0.810 | 0.870 | 0.000 | 0.000 | 0.000 | 0.390 | 0.930 | 0.940 | 0.010 | 0.800 | 0.890 |

### Key observations

- **This is the lowest-scoring sub-ability overall**, with most models in the 0.20–0.55 range. Skill acquisition across repeated episodes is genuinely hard — most models fail to show consistent improvement trajectories.
- **`grammar-induction`** is solved only by Gemini Pro (0.640) and Claude Opus (0.500). Every other model scores 0.000 — they cannot induce a productive grammar from examples even after 5 practice rounds with feedback.
- **`voting-protocol`** shows a dramatic split: GLM-5 (0.870), Gemini Pro (0.940), Gemini Lite (0.930), DeepSeek (0.810), Qwen Thinking (0.890) all score well, while GPT-5.4 (0.000), GPT-mini (0.000), Claude Opus (0.050), and Gemma (0.010) fail entirely. The task requires learning a hidden voting rule over episodes — some model families never discover it.
- **`nim-variant`** is solved only by Gemini Pro (0.663) and Gemini Flash (0.588) above 0.4. This adversarial game-theory task requires updating a strategy from loss/win feedback — a genuine trajectory-dependent learning task.
- **The Qwen Thinking vs. Instruct reversal:** On procedural tasks, Thinking (0.344) scores *lower* than Instruct (0.450). This is the only sub-ability where reasoning mode hurts performance — consistent with the paper's hypothesis that extended deliberation per round may blunt the rapid hypothesis-iteration loop needed for procedural learning. (Note: not statistically significant at p = 0.55.)
- **`adaptive-sort-rule`** is where Claude Opus excels (0.910), outperforming even Gemini Flash (0.835) — Opus appears well-suited to multi-episode strategy refinement tasks involving rule-based sorting.
- **`boolean-circuit`** and `state-machine-password` score consistently across models in the 0.25–0.65 range, suggesting these tasks have a difficulty floor that most models plateau at rather than learn through.

---

## Cross-Sub-Ability Summary

### Model scores across all six sub-abilities

| Model | Assoc | Concept | Language | Obs | RL | Proc | **Overall** |
|---|---|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | 0.948 | 0.799 | 0.782 | 0.853 | 0.929 | 0.727 | **0.851** |
| GLM-5 | 0.775 | 0.568 | 0.747 | 0.647 | 0.793 | 0.536 | **0.692** |
| Qwen 3 Next 80B Thinking | 0.649 | 0.563 | 0.641 | 0.668 | 0.633 | 0.344 | **0.606** |
| Claude Opus 4.6 | 0.678 | 0.259 | 0.518 | 0.391 | 0.687 | 0.454 | **0.508** |
| Gemini 2.5 Flash | 0.606 | 0.533 | 0.502 | 0.443 | 0.505 | 0.485 | **0.507** |
| GPT-5.4 | 0.656 | 0.285 | 0.625 | 0.309 | 0.663 | 0.283 | **0.493** |
| Claude Sonnet 4.6 | 0.664 | 0.332 | 0.479 | 0.287 | 0.635 | 0.373 | **0.466** |
| Gemini 3.1 Flash-Lite Preview | 0.595 | 0.313 | 0.501 | 0.329 | 0.585 | 0.409 | **0.458** |
| DeepSeek V3.2 | 0.523 | 0.194 | 0.467 | 0.428 | 0.534 | 0.347 | **0.432** |
| Claude Haiku 4.5 | 0.501 | 0.193 | 0.355 | 0.271 | 0.566 | 0.352 | **0.384** |
| GPT-5.4 mini | 0.477 | 0.224 | 0.497 | 0.215 | 0.454 | 0.221 | **0.362** |
| Gemma 4 26B A4B | 0.511 | 0.255 | 0.248 | 0.218 | 0.552 | 0.237 | **0.346** |
| Qwen 3 Next 80B Instruct | 0.484 | 0.186 | 0.436 | 0.241 | 0.332 | 0.450 | **0.340** |
| GPT-5.4 nano | 0.430 | 0.149 | 0.304 | 0.144 | 0.251 | 0.202 | **0.246** |

### What each sub-ability isolates

| Sub-ability | Core cognitive act | Why it is not measured elsewhere |
|---|---|---|
| **Associative** | Causal inference from fixed observations — blocking, confounding, epistemic limits | Most benchmarks test prediction, not causal attribution with explicit UNKNOWN states |
| **Concept Formation** | Active rule induction under evidence budget — knowing when to commit | Benchmark datasets provide all examples at once; no active retrieval pressure |
| **Language** | Morphophonological rule induction + wug-test generative transfer | Invented languages have no web trace; tests productive rule knowledge not memorisation |
| **Observational** | Hidden process inference from complete demonstrations | Emphasis on complex structural inference (Mealy machines, Feistel ciphers, lattices) not in standard reasoning benchmarks |
| **Reinforcement** | Explore-exploit loop under hidden state and step budget | LLM benchmarks rarely measure closed-loop hypothesis updating from environment feedback |
| **Procedural** | Skill acquisition trajectory + cold transfer | The only component measuring *whether learning occurred* (OLS slope) — no analogue exists |

### Key cross-cutting findings

1. **Gemini Pro's dominance is consistent across all six sub-abilities.** It is #1 on every sub-ability, with the closest competitors varying by sub-ability (GLM-5 on language/RL, Qwen Thinking on observational/concept).

2. **Reasoning mode (Qwen Thinking vs. Instruct) helps induction but may hurt procedural learning.** The Thinking variant scores +0.38 on Concept, +0.43 on Observational, +0.30 on RL — but −0.11 on Procedural (not significant). Deliberation depth appears to trade against iteration speed.

3. **Concept Formation is the most discriminating sub-ability** (5× score range). Observational is second. Both require genuine structural induction from limited evidence, which is rare among current models.

4. **RL scores are highest overall** — many RL tasks have algorithmic solutions that reward systematic search, even without deep rule induction. Models with strong tool-use and planning capacity do well here independently of their induction ability.

5. **GLM-5 (open-source) outperforms all closed-source models except Google on rule induction** (concept + observational sub-ability combined). This is statistically significant (Mann-Whitney p = 0.029, Cliff's δ = 0.71).

---

> **PS —** The task scripts can be dense to read cold. If you want to truly understand what a model experienced on any given task — the exact prompt it received, the examples it saw, and its raw responses turn by turn — the clearest way is to open `analysis_notebook.ipynb` and inspect the logged prompt traces directly. Reading a model conversation is almost always more illuminating than reading the generator function.
