# Benchmark Scoring Reference

This document explains how each cognitive sub-ability category is scored, the
formula used, and why each design choice reflects the underlying ability being
measured.

---

## Table of Contents

1. [Associative Learning](#associative-learning)
2. [Concept Formation](#concept-formation)
3. [Language Learning](#language-learning)
4. [Observational Learning](#observational-learning)
5. [Runtime RL](#runtime-rl)
6. [Procedural Learning](#procedural-learning)
7. [Cross-Category Summary](#cross-category-summary)

---

## Associative Learning

### What the sub-ability is

Associative learning is the capacity to infer causal structure from contingent
observations — deciding which stimuli reliably predict an outcome, which are
confounded by co-occurring causes, and which remain genuinely indeterminate.
It maps onto the classical CS/US conditioning framework but goes beyond simple
pattern counting: the model must apply causal reasoning, recognise epistemic
limits (when evidence is insufficient), and compose known causes to predict
novel compound situations.

### Scoring formula

```
score = correct / total_questions
```

Each task presents a fixed question set (typically 6–11 binary or categorical
questions). Every question is a binary match: the model's answer either
contains the expected token (matched case-insensitively via `re.search`) or it
does not. The final score is the fraction of questions answered correctly,
returning a float in `[0, 1]`.

Integer-valued questions (e.g. "how many of these have a confirmed effect?")
are compared with exact integer equality.

### Why this formula

**Fraction-correct directly measures causal inference quality.** Each question
probes a distinct facet of the same underlying ability:

| Question type | What it tests |
|---|---|
| Confirmed attribution | Can the model identify a cue that was solo-tested? |
| Causal insufficiency / UNKNOWN | Does the model recognise when a co-present known cause blocks inference? (blocking effect) |
| Compositional prediction | Can the model combine confirmed causes according to stated rules? |
| Meta-epistemic counting | Does the model know how many cues remain epistemically unresolved? |

Mixing question types within a single score means the aggregate float captures
breadth of associative reasoning, not just one facet. A model that always
defaults to "UNKNOWN" will miss confirmed-attribution questions; a model that
over-attributes will miss confounded/blocking questions. Getting all question
types right requires genuinely understanding conditional independence and causal
sufficiency.

**No efficiency component** is appropriate here. Associative learning from a
fixed trial log is a single-shot reasoning task, not an interactive learning
task. The model cannot request more data; it must reason from what it is given.
Adding a speed penalty would be meaningless.

**Partial credit via fraction** is fairer than pass/fail. A model that
correctly identifies 7 of 8 aspects demonstrates substantially more knowledge
than one that identifies 0 of 8.

### Matching rule

```python
def _str_match(expected: str, actual: str) -> bool:
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))
```

The model may embed reasoning alongside the answer ("Based on the blocking
effect, Plex is UNKNOWN"). Using `re.search` rather than strict equality
avoids penalising correct answers that happen to come with supporting
reasoning.

---

## Concept Formation

### What the sub-ability is

Concept formation is the ability to actively induce a hidden rule from
input/output examples, controlling your own evidence-gathering (how many
examples to request) before committing to an answer. This is the
"active learning" dimension of inductive reasoning: the model balances
the cost of gathering more evidence against the risk of failing the exam with
an under-specified rule.

### Scoring formula

```python
def _concept_score(correct_count, examples_used, max_examples, initial_examples):
    accuracy = correct_count / NUM_TEST_ITEMS      # 0..1
    if accuracy == 0:
        return 0.0
    effective_free = max(initial_examples, FREE_THRESHOLD)   # typically 4–5
    if examples_used <= effective_free:
        efficiency = 1.0
    else:
        paid_used   = examples_used - effective_free
        paid_budget = max_examples  - effective_free
        efficiency  = max(0.0, 1.0 - paid_used / paid_budget)
    return accuracy * (0.40 + 0.60 * efficiency)
```

The score lies in `[0, 1]`. Boundary values (typical task: INITIAL=3, MAX=16):

| Outcome | Score |
|---|---|
| 4/4 correct, ≤ free threshold examples | 1.00 |
| 4/4 correct, all 16 examples used | 0.40 |
| 2/4 correct, ≤ free threshold examples | 0.50 |
| 2/4 correct, all 16 examples used | 0.20 |
| 0/4 correct | 0.00 |

### Why this formula

**Accuracy is the dominant signal.** The 0.40 floor on the efficiency
multiplier means that even a model that exhausts all available examples still
scores 40% of its raw accuracy. Correctness is always worth chasing; the
efficiency component is a *bonus*, not a penalty that can erase correct
answers.

**Efficiency measures rule induction speed.** The first `FREE_THRESHOLD`
examples are free — they model the "necessary minimum" to see the rule at all.
Beyond that free zone, each additional example drawn from the budget linearly
reduces the efficiency component toward zero. This captures the cognitive
reality that a model which needs 16 examples to identify a rule with two
parameters has not grasped the rule as cleanly as one that needed 5.

**Multiplicative rather than additive combination.** The formula is
`accuracy × (0.40 + 0.60 × efficiency)` rather than `0.5 × accuracy +
0.5 × efficiency`. Multiplication means that efficiency is proportionally
meaningful: a model that gets 2/4 correct earns at most half the efficiency
bonus of a model that gets 4/4. A fast wrong answer is not rewarded.

**Zero-accuracy guard.** If `accuracy == 0`, the score is immediately 0.0
regardless of efficiency. There is no reward for being fast at being wrong.

**Why an active-retrieval protocol (not all examples at once)?** Giving all
20 examples at once would test working-memory processing, not learning
strategy. The active protocol forces the model to commit to a hypothesis under
uncertainty and tests whether it can identify the minimal sufficient evidence
set. This is the cognitive core of concept formation.

---

## Language Learning

### What the sub-ability is

Language learning measures the capacity to induct morphophonological rules
from a small set of labeled form–meaning pairs, and then produce novel surface
forms for unseen roots. The key cognitive acts are:

- **Phonological rule induction** from surface contrasts
- **Multi-system integration** (discovering that several independent rule
  systems co-apply simultaneously)
- **Paradigm extension / wug-test** (generating novel forms for roots that
  never appeared in training)
- **Efficiency under probe budget** (how few examples suffice)

All language names, morphemes, and phonological environments are
purpose-built (no internet pre-training signal exists for "DRELKOVAK" or
"GWELTHAR").

### Scoring formula

Identical to concept formation:

```python
def _concept_score(correct_count, examples_used, max_examples, initial_examples):
    accuracy = correct_count / NUM_TEST_ITEMS
    if accuracy == 0:
        return 0.0
    effective_free = max(initial_examples, FREE_THRESHOLD)
    if examples_used <= effective_free:
        efficiency = 1.0
    else:
        efficiency = max(0.0, 1.0 - (examples_used - effective_free) /
                                     (max_examples - effective_free))
    return accuracy * (0.40 + 0.60 * efficiency)
```

The examination phase is held-out: 4 novel root + suffix-template combinations
are presented after the model declares it is ready. Each form must match the
surface string exactly (via Unicode-NFC-normalised equality for phonological
diacritics).

### Why this formula (and its particular challenges)

**Exact surface form matching** is required rather than substring matching,
because a single misapplied harmony feature or missing diacritic (e.g. `nˤ`
vs `n`) produces a wrong surface form. Partial-character credit would hide
whether the model understood the rule. Correct forms are computed
programmatically from the rule functions — the model cannot memorise from
context.

**Efficiency pressure reflects real language acquisition.** Human language
aptitude research (Carroll 1981, LLAMA tests) consistently shows that speed of
acquisition from minimal input is the discriminating cognitive variable. A
model that requires all 20 training examples to discover vowel harmony vs.
front/back distinction is qualitatively worse than one that identifies the
pattern in 5 examples. The efficiency multiplier translates this into a
continuous gradient.

**Multi-system tasks are harder by design.** Tasks like DRELKOVAK
(vowel harmony × pharyngeal consonant harmony) or GWELTHAR (tone ×
evidentiality × mirativity) require the model to simultaneously manage
independent rule systems. A model that discovers only one system will produce
partially correct forms; the exam scoring does not give partial credit per form
(the form is right or wrong), which means partial rule knowledge is only
rewarded to the extent that the undiscovered system does not apply to that
particular test item.

**Wug-test items test productive rule knowledge.** Test roots are never in the
training set. A model that merely memorised training pairs will fail all 4 test
items; only genuine rule induction transfers.

---

## Observational Learning

### What the sub-ability is

Observational learning measures the capacity to infer a hidden computational
process (a rule, a machine, an operation) from a complete set of
input/output demonstrations, then predict outputs for novel inputs. Unlike
concept formation, the model is given **all demonstrations at once** — no
active retrieval. The difficulty comes entirely from the complexity of the
hidden process: aliased states in transducers, deceptive early patterns in
group operations, interacting transformations in affine chains, etc.

### Scoring formula

Most tasks use **fraction of test cases fully correct**:

```python
score = correct_count / len(test_cases)
```

where `correct_count` is the number of test cases for which the model's
predicted output exactly matches the ground truth.

Some tasks (e.g. `hidden_state_machine_obs_learning`) additionally compute a
**per-sequence positional score** as a diagnostic, but the primary task score
is still the fraction of fully correct predictions.

```python
# Primary score: whole-sequence correctness
score = correct_count / len(_TEST_CASES)

# Diagnostic (not used in primary score):
pos_score = sum(g == e for g, e in zip(got, expected)) / len(expected)
```

### Why this formula

**Whole-sequence correctness rewards complete inference.** For a hidden state
machine, a model that follows the correct path for 4 of 5 steps but makes a
single error in state tracking may produce wrong outputs for all subsequent
steps. Positional partial credit would mask the core failure: the model did not
fully infer the hidden structure. Fraction-of-sequences is the right unit.

**No efficiency component** because the model does not control data access —
all demonstrations are provided up front. There is no "cost" to study time.
The only dimension that matters is whether the hidden rule was correctly
inferred.

**Aliased/deceptive design forces deep inference.** Tasks are deliberately
structured so that shallow pattern matching fails. In `hidden_state_machine`,
two pairs of states produce identical single-step output signatures; the model
must reason across multi-step paths to distinguish them. In
`hidden_group_operation`, the first 3 demonstrations are "deceptive" (using
the identity element, making the operation look like ordinary addition).
Fraction-correct across 4 held-out test cases measures whether the model
successfully resolved these indistinguishabilities.

**Varied domains cover breadth of the sub-ability.** Observational learning
tasks span Mealy machines, abstract algebra (hidden group operations), cipher
decoding, physics simulations, linguistic transducers, automata, and more.
Each task uses the same scoring skeleton so aggregate scores across tasks are
directly comparable.

---

## Runtime RL

### What the sub-ability is

Runtime RL (reinforcement learning) measures the capacity to solve an unknown
problem through sequential interaction — probing the environment with actions,
receiving feedback, updating a hypothesis about hidden state, and converging on
a solution within a step budget. The model must simultaneously:

1. **Explore** to identify hidden parameters (goal, forbidden rule, secret
   code, etc.)
2. **Exploit** its inferred model to reach the goal efficiently
3. **Track progress** under partial and sometimes veiled feedback

### Scoring formula

```python
def _composite_score(solved, step_y, budget_n, min_explore, progress, *, floor=0.10):
    """
    success   (0.55) — did the model solve the task?
    efficiency (0.25) — how quickly (only when solved)?
    progress  (0.20) — how close did it get (always defined)?
    """
    progress = max(0.0, min(1.0, float(progress)))
    if solved:
        step_y = max(1, min(step_y, budget_n))
        if step_y <= min_explore:
            eff = 1.0
        else:
            paid_used   = step_y - min_explore
            paid_budget = budget_n - min_explore
            eff = max(floor, 1.0 - (1.0 - floor) * (paid_used / paid_budget))
    else:
        eff = 0.0
    return round(0.55 * float(solved) + 0.25 * eff + 0.20 * progress, 4)
```

Component weights and their meanings:

| Component | Weight | When active | What it measures |
|---|---|---|---|
| `success` | 0.55 | Always | Binary: did the model solve the task? |
| `efficiency` | 0.25 | Only when solved | Speed relative to the step budget |
| `progress` | 0.20 | Always | Partial completion even if not solved |

### Why this formula

**Success dominates (0.55).** The primary cognitive act in RL is to
successfully identify and reach the goal. A model that never solves any task
should score near zero regardless of how well it tracks partial progress.
Assigning more than half the weight to success ensures the score reflects
whether the model can actually complete the loop of explore → infer → solve.

**Efficiency is a bonus (0.25, only when solved).** Efficiency without success
is meaningless in RL — if the model didn't solve the task, we don't know
whether it was making purposeful progress or random moves. Conditioning
efficiency on `solved=True` means this component measures quality of the
solution path, not quality of an incomplete attempt.

The `min_explore` parameter defines a "free zone" of initial probes (typically
7) that are necessary to observe enough feedback to constrain the hidden
variables. Moves inside this zone incur no efficiency penalty. Beyond it,
efficiency decays linearly to a floor of 0.10 at the last step of the budget.
The floor prevents a model that solved the task on the last possible step from
scoring zero on efficiency — it still demonstrated it could solve the task.

**Progress always contributes (0.20).** This component rewards models that
make genuine headway even when they fail to reach the goal. Progress is defined
task-specifically: for Hanoi, it is `max(fraction of valid moves used, fraction
of disks on the goal peg)`; for Mastermind, it is the best LOCK fraction
achieved across all guesses. This 0.20 weight means a model that consistently
approaches the solution but exhausts its budget still scores around 0.20,
giving a non-trivial signal for model comparison.

**Why a composite (not just win-rate)?** Pure win-rate creates a binary signal
with high variance. The composite score is differentiating: two models that
both fail to solve Hanoi-Two might score 0.20 vs 0.04 based on how many disks
they moved to the right peg, allowing much finer-grained model ranking.

**Hidden variables create RL difficulty.** The runtime tasks hide one or more
parameters that the model must infer through interaction. Examples:
- `hanoi_two`: hidden goal peg (3 choices) × hidden forbidden disk-peg pair (9
  choices) = 27 configurations
- `mastermind_classic`: hidden slot-wise symbol permutation on one index
- `wordle_micro`: secret code with veiled LOCK/DRIFT telemetry

This means random action leads to slow convergence, and a model that
systematically probes and updates its hypothesis earns both higher efficiency
(fewer steps) and a higher chance of success.

---

## Procedural Learning

### What the sub-ability is

Procedural learning is the capacity to acquire a skill, strategy, or action
pattern **through repeated performance with corrective feedback** — and then to
apply that learned procedure to novel instances without assistance. It is the
only category in the benchmark that explicitly measures the *learning
trajectory* across practice trials, not just final accuracy.

The defining structure of every task in this category is:

- **Practice phase** (5 rounds): the model interacts with the environment,
  receives feedback after each attempt, and must improve its strategy over
  trials.
- **Transfer phase** (4 tests): the model faces novel instances of the same
  task with *no corrective feedback* whatsoever, relying entirely on the
  procedure it internalised during practice.

This two-phase design cleanly separates *learning* (did the model get better
with practice?) from *transfer* (can it apply what it learned cold?).

### Scoring formula

```python
def procedural_composite_score(
    round_scores: list[float],   # per-round efficiency score from practice
    test_score: float,           # fraction of transfer tests passed [0, 1]
) -> float:
    k = max(1, len(round_scores) // 2)
    asymptote   = mean(round_scores[-k:])          # last half of practice
    consistency = weighted_learning_mean(round_scores)
    trajectory  = learning_curve_slope(round_scores)
    return (
        0.30 * test_score   # transfer
      + 0.25 * asymptote    # peak skill
      + 0.25 * trajectory   # learning improvement
      + 0.20 * consistency  # overall practice quality
    )
```

Zero guard: if `asymptote < ε` **and** `test_score < ε`, return `0.0` —
a model that never solved any practice round and failed all transfer tests
should not earn a spurious trajectory bonus.

#### `learning_curve_slope` in detail

```python
def learning_curve_slope(round_scores):
    # OLS slope of scores over round indices
    slope = ols_slope(round_scores)
    # Normalise to [-1, +1] by the max achievable slope for scores in [0,1]
    max_slope = 1.0 / (len(round_scores) - 1)
    normalised = clip(slope / max_slope, -1, +1)
    return (normalised + 1.0) / 2.0   # rescale to [0, 1]
```

| Trajectory value | Meaning |
|---|---|
| 1.0 | Scores rise linearly from 0 → 1 over practice rounds (textbook learning curve) |
| 0.5 | Flat — no measurable change across rounds |
| 0.0 | Scores fall linearly from 1 → 0 (systematic deterioration) |

### Component weights and their meanings

| Component | Weight | What it captures |
|---|---|---|
| `transfer` | **0.30** | Can the model apply its learned procedure to unseen instances without any hints? This is the ultimate test of whether procedural learning generalised beyond the training instances. |
| `asymptote` | **0.25** | The mean efficiency of the *last half* of practice rounds. Measures the *ceiling* of skill the model reached — how good it became, not just whether it improved. |
| `trajectory` | **0.25** | The OLS slope of efficiency scores over rounds, normalised to [0, 1]. This is the most distinctive signal in this category: it directly measures whether practice caused improvement, is neutral for flat performers, and penalises declining performance. |
| `consistency` | **0.20** | Linearly-weighted mean across all practice rounds (later rounds weighted higher). Rewards models that performed well throughout practice, not just in one lucky round. |

### Why this formula

**Why four components instead of the existing 50/50 split?**

The previous formula `weighted_mean × 0.5 + test_score × 0.5` collapses two
distinct cognitive events — learning and transfer — into a single composite
without any signal about *whether learning actually occurred*. Consider two
models that both finish with `weighted_mean = 0.65`:

| Model | Round scores | Old score | New score |
|---|---|---|---|
| **Learner** | `[0.1, 0.3, 0.5, 0.8, 0.9]` | 0.58 | **0.72** |
| **Static**  | `[0.8, 0.7, 0.6, 0.6, 0.5]` | 0.48 | **0.37** |
| **Plateau** | `[0.7, 0.7, 0.7, 0.7, 0.7]` | 0.60 | **0.59** |

*(test_score = 0.5 in all three examples)*

The old formula assigns nearly identical scores to the Learner and the Plateau,
and even rewards the Static model more than the Learner. The new formula
correctly ranks them: Learner > Plateau > Static.

**Why is `trajectory` weighted equally with `asymptote`?**

A model that arrived at high performance before the first practice round (i.e.,
it already "knew" the task from pre-training) will show a flat trajectory
despite high asymptote. A model that genuinely *acquires* skill through the
practice interaction shows a positive slope. Since this category explicitly
evaluates procedural learning — "skill acquisition through performance" — the
*act of learning* (slope) deserves equal weight with the *level learned*
(asymptote). A pre-trained expert earns a high asymptote but only a neutral
trajectory; a genuine learner earns both.

**Why is `transfer` the single largest component (0.30)?**

Transfer is the acid test. It proves that whatever was acquired during practice
encoded a *general procedure*, not just instance-specific answers. All task
transfer items use inputs never seen in practice; the model cannot succeed
through memorisation. A model that improves in practice but then fails all
transfer tests has not learned a transferable procedure — exactly the cognitive
failure this category is designed to detect.

**Why a `consistency` component at all?**

Consistency (the linearly-weighted mean) distinguishes a model that solves one
practice round brilliantly and fails the rest from one that shows steady,
reliable performance throughout. A single lucky round inflates asymptote
slightly but contributes much less to consistency, which averages across all
rounds. This prevents over-rewarding spiked, high-variance learners.

**Separating this category from Runtime RL**

Both categories involve multi-turn interaction with feedback. The distinction
is structural: Runtime RL presents *one task with hidden state* that the model
must solve within a budget (explore → infer → exploit loop). Procedural
Learning presents *five separate instances of the same task* so that the model
can progressively refine its strategy *across tasks* — the cognitive act of
skill generalisation, not just within-task inference. The trajectory component
has no analogue in the Runtime RL formula precisely because Runtime RL measures
a single learn-and-solve arc, not a multi-episode learning curve.

### Score range and model discrimination

| Profile | Score (approx.) |
|---|---|
| Perfect learner: improves linearly in practice, passes all transfer tests | 0.90–1.00 |
| Strong learner: high asymptote, positive trajectory, good transfer | 0.65–0.85 |
| Pre-trained expert: flat-high practice (already knows task), good transfer | 0.55–0.75 |
| Slow learner: low early rounds but strong finish, moderate transfer | 0.45–0.65 |
| Inconsistent: high-variance practice, poor transfer | 0.25–0.45 |
| Non-learner: never solves practice, fails transfer | 0.00–0.15 |

---

## Cross-Category Summary

| Category | Scoring type | Efficiency component? | Partial credit? | Key cognitive axis |
|---|---|---|---|---|
| Associative Learning | Fraction correct (Q&A) | No | Yes (per question) | Causal inference from fixed observations |
| Concept Formation | `accuracy × (0.40 + 0.60 × eff)` | Yes (active retrieval) | Yes (fraction correct × efficiency) | Inductive rule induction under evidence budget |
| Language Learning | `accuracy × (0.40 + 0.60 × eff)` | Yes (active retrieval) | Yes | Multi-system morphophonological induction |
| Observational Learning | Fraction of sequences fully correct | No | Limited (fraction of sequences) | Hidden process inference from complete demonstrations |
| Runtime RL | `0.55×solved + 0.25×eff + 0.20×progress` | Yes (multi-turn) | Yes (progress always active) | Explore-exploit under hidden state |
| Procedural Learning | `0.30×transfer + 0.25×asymptote + 0.25×trajectory + 0.20×consistency` | Yes (within each practice round) | Yes (four components always active) | Skill acquisition through repeated practice → transfer |

### Common design principles across all categories

1. **No exact-equality string matching.** All string comparisons use
   `re.search` (case-insensitive substring) so that correct answers embedded
   in reasoning are not penalised.

2. **Integer fields use exact equality.** Numeric answers that require
   arithmetic cannot be granted substring leniency — `88` must be `88`.

3. **Zero-accuracy / zero-success guard.** Tasks that have an efficiency or
   progress component always return `0.0` if the model produced zero correct
   answers, preventing nonsensical scores from fast-but-wrong behaviour.

4. **Free zone for exploration.** Both interactive categories (concept
   formation / lang learning and runtime RL) define a minimum number of steps
   that are free from efficiency penalty. This acknowledges that some initial
   probing is necessary and unavoidable — the efficiency component only measures
   steps *beyond* what is structurally required.

5. **Ground truth is computed, not hardcoded.** All expected answers are
   derived by running the same mathematical rule function used for grading.
   This eliminates silent errors from hardcoded answer tables.
