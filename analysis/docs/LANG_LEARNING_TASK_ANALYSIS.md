# Language Learning Task Analysis

**Date:** April 2026  
**Scope:** 12 language learning tasks, 14 models, fresh run logs  
**Purpose:** Document why models fail to achieve full scores; classify root causes; record fixes applied.

---

## Overview

Each task is scored as `accuracy × efficiency`, where:
- **accuracy** = fraction of 4 test items answered correctly
- **efficiency** = `0.40 + 0.60 × E`, where `E` ∈ [0, 1] decays as examples beyond `FREE_THRESHOLD` are used

A model can lose score two ways:
1. **Wrong answers** — genuine reasoning failures
2. **Efficiency penalty** — correct answers but too many examples requested

The analysis below separates these two modes and records the best observed score per task across all 14 models.

---

## Part 1 — Tasks Fixed (Bugs or Design Flaws)

These tasks had **no model achieving 4/4 correct answers**, due to bugs in task code or design flaws that made correct answers impossible or misleading.

### 1. PELVAN — 5-Dimensional Split Agreement

**File:** `pelvan-agreement-lang-learning.py`

**Root cause:** `_S_AGR` dictionary was missing 4 entries for the `inanim × plural` cells of persons 1 and 2. When test item 4 (`S=1.pl.f.inanim`) was evaluated, `_pelvan_form` called `_S_AGR.get(("1","pl","f","inanim"), "??")` and returned `"??"`, making the expected answer `"??drunlov"` — a string no model could ever produce.

**Symptom:** All models scored 0 on item 4. The `??` did not appear in printed output, so the bug was invisible unless you printed `_TEST_EXPECTED_PV` directly.

**Fix applied:**
```python
# Added 4 missing entries following the inanim-sg consonant + asto/este pattern:
("1","pl","m","inanim"): "tasto",  # 't' from inanim-sg 1m='ta'
("1","pl","f","inanim"): "teste",  # 't' from inanim-sg 1f='te'
("2","pl","m","inanim"): "pasto",  # 'p' from inanim-sg 2m='pa'
("2","pl","f","inanim"): "peste",  # 'p' from inanim-sg 2f='pe'
```

After fix, test item 4 expected answer: `"testedrunlov"`.

---

### 2. SKOLVREN — 6-Slot Polysynthetic Template

**File:** `skolvren-polysynthetic-lang-learning.py`

**Root cause:** The S6 (evidentiality+mood) slot has an irregular fusion form: `(infer, inter) → "mekra"`. The 20 training examples covered `(infer, decl)` and `(direct, inter)` in isolation but **never** showed `(infer, inter)` together. Models systematically produced `"mekka"` by naively concatenating the `infer` token `"mek"` with the `inter` token `"venka"`, or by applying the regular `inter` suffix `-ka` to `"mek"`. This failed test item 2.

**Symptom:** Item 2 consistently wrong across all 14 models. Error pattern: `"mekka"` instead of `"mekra"`.

**Fix applied:** Replaced example 20 (index 19) — `("3","du","inanim","pres","make","tool","direct","decl")` — with a direct demonstration of the `(infer, inter)` fusion:
```python
("3","sg","anim","pres","carry",None,"infer","inter")
# → produces 'nolenolantrelmekra', teaching infer+inter → mekra
```

---

### 3. MIXED-RADIX (KROMATH) — Compositional Verb Morphology

**File:** `mixed-radix-number-lang-learning.py`

**Root cause:** Despite all 14 models achieving 4/4 correct answers, scores were capped well below 1.0. The task involves three independent rules (PAST gemination, PASS `va-` prefix, INFER `-nk` suffix) that interact in a fixed order. To be confident of the order, models reasonably request ~6–8 examples, but `FREE_THRESHOLD=3` meant any usage above 5 examples incurred efficiency penalties. At 8 examples (a reasonable verification depth for triple-rule chaining), the score was only `0.836`.

**Symptom:** Every model scored 4/4 correct but no model achieved score > ~0.95. The task was graded correct in accuracy but heavily penalized for normal exploratory behavior.

**Fix applied:**
```python
# Before:
FREE_THRESHOLD = 3
INITIAL_EXAMPLES = 5

# After:
FREE_THRESHOLD = 8
INITIAL_EXAMPLES = 8
```
Now models that submit immediately score 1.0; those requesting up to 8 examples for verification still score 1.0. Scores only drop for usage beyond 8.

---

### 4. DRALVEN — Tonal Grammar + Clash Resolution

**File:** `dralven-tone-sandhi-lang-learning.py`

**Root cause:** The first 4 initial examples were all drawn from the same root `"trák"` (all cycling through TAM values for one lexeme). Models saw:
- `trák + PRES` → no clash (baseline)
- `trák + PAST` → H+H clash → LH (caron)
- `trák + FUT` → H+L → gram dominates → L
- `trák + PERF` → H+H clash → LH (caron again)

They never saw an L+L clash (→ HL, circumflex) or an HL+H clash (→ HH, macron) in the free initial examples. These are the exact combinations tested in items 2 and 3 of the exam.

**Symptom:** Models consistently produced wrong diacritics for items 2 (`gèl+FUT`) and 3 (`prân+IMP`), confusing HL-circumflex with acute/grave and macron with other marks.

**Fix applied:** Reordered `_ALL_PAIRS` so the **first 4 initial examples** each demonstrate a different clash case, then increased `INITIAL_EXAMPLES` from 4 to 6:
```python
_PRIORITY_PAIRS = [
    ("trák", "PRES"),   # H root, no gram-tone — baseline
    ("trák", "PAST"),   # H+H clash → LH (caron)
    ("gèl",  "FUT"),    # L+L clash → HL (circumflex)
    ("prân", "IMP"),    # HL+H clash → HH (macron)
]
```
After fix: initial 6 examples cover all meaningful clash patterns before any exploration is needed.

---

### 5. STREVOKLAN — Negation + Polarity Items + Neg-Raising

**File:** `strevoklan-neg-lang-learning.py`

**Root cause (primary):** Two of the four test items were **identical** to training examples:
- Test item 1: `("warrior","think",True,None,"leave")` = training example 3
- Test item 3: `("hunter","run",False,"still",None)` = training example 5

This means models that memorize training examples score 2/4 "for free" without generalization, while models that attempt to generalize may get unexpected mismatches.

**Root cause (secondary):** Even for the non-leaked items, exact string matching for the `"still"` in positive context (`"'still' in positive: expected continuation"`) caused failures because models paraphrased the semantic reading rather than reproducing the exact canonical string.

**Fix applied:** Replaced the two leaked test specs with novel, non-training examples:
```python
# Before (items 1 and 3 were training leaks):
("warrior","think",True,None,"leave"),   # = training ex 3
("hunter","run",False,"still",None),     # = training ex 5

# After (fresh, non-training):
("elder","believe",False,None,"run"),    # attitude verb, positive → no neg-raising
("warrior","run",False,"still",None),    # 'still' in positive, novel subject
```

---

### 6. KOPHAR — Mass/Count Nouns + Measure Words + Base-6

**File:** `kophar-quantity-lang-learning.py`

**Root cause:** The 72 training examples are randomly shuffled with seed 13. Under this seed, the noun `"cloth"` (KOPHAR: `felthar`, flat class, measure word `bren`) first appeared at example index 11 (example 12). Since `INITIAL_EXAMPLES=6`, models never saw `cloth` in the free initial set. Test item 3 asked for `cloth×3 = "sketh bren felthar"`. Models that submitted early produced `"sketh cloth"` (using English) or omitted the measure word entirely.

**Symptom:** Item 3 wrong for all models that submitted with ≤ 8 examples. Correct models spent 10–14 examples to encounter `cloth` naturally.

**Fix applied:** After shuffle, promote the first `cloth` entry into position 5 (last initial example) if it falls outside the initial window:
```python
_cloth_idx = next(i for i, (n, _) in enumerate(_ALL_NP_SPECS) if n == "cloth")
if _cloth_idx >= 6:
    _ALL_NP_SPECS.insert(5, _ALL_NP_SPECS.pop(_cloth_idx))
```
After fix: example 6 is always a `cloth` NP (`cloth×7 → "trelvak-ven bren felthar"`), teaching the flat measure word before the exam.

---

## Part 2 — Efficiency-Limited Tasks (At Least One Model 4/4 Correct)

These tasks are **functionally correct** — a human expert or a sufficiently capable model can achieve 4/4. However, most models score below their accuracy potential because they request more examples than the `FREE_THRESHOLD` allows for free.

### Scoring Formula Reminder

```
score = accuracy × (0.40 + 0.60 × max(0, 1 − (used − free) / (max − free)))
```

If `used ≤ free`: efficiency = 1.0, score = accuracy.  
If `used = max`: efficiency = 0.0, score = accuracy × 0.40 (minimum for non-zero accuracy).

---

### 7. DRELVAK — Reduplication

**File:** `drelvak-reduplication-lang-learning.py`  
**MAX_EXAMPLES:** 18 | **INITIAL_EXAMPLES:** 5 | **FREE_THRESHOLD:** 5

| Models 4/4 correct | Typical examples used | Typical score |
|---|---|---|
| ~3–4 of 14 | 5 (submit immediately) | 1.00 |
| Majority | 8–12 | 0.55–0.84 |

**Why models request more examples:** The reduplication rule (`last CV of stem`) is easy to observe in simple cases but the nasal-final variant (adds `-m`) requires seeing at least one example. Models verify edge cases before submitting.

**Note on earlier `_last_cv` bug:** The `_last_cv` function had a multi-consonant cluster extraction bug. The training examples and test items were consistent with each other (the bug applied uniformly), so while the implementation differs from the stated description, the grader is internally consistent. A human expert working purely from examples (not the description) would learn the actual function. This is a documentation quality issue but does not produce wrong expected answers.

**Recommendation:** Increase `FREE_THRESHOLD` from 5 to 7 to allow models to verify the nasal variant without penalty.

---

### 8. GRELKAN — Suppletion

**File:** `grelkan-suppletion-lang-learning.py`  
**MAX_EXAMPLES:** 20 | **INITIAL_EXAMPLES:** 6 | **FREE_THRESHOLD:** 6

| Models 4/4 correct | Typical examples used | Typical score |
|---|---|---|
| ~2–3 of 14 | 6 (submit immediately) | 1.00 |
| Majority | 10–16 | 0.40–0.76 |

**Why models fail accuracy or efficiency:** The verb `brentor` has Class B suppletion where the `1st-person IMPF` stem equals the infinitive label itself (`"brentor"`). Models that query this specific cell learn it; those that submit early either guess the wrong stem or output `UNKNOWN`. The task is deliberately designed with this "trap" — it is a valid design choice testing whether the model queries strategically.

**Why efficiency suffers:** Class B has 3 distinct stems vs Class A's 2, requiring more targeted queries per verb to fully map the paradigm.

**No fix required:** The task is working as intended. Efficiency scores reflect genuine difficulty.

---

### 9. NORKVASH — Scalar Implicature

**File:** `norkvash-scalar-lang-learning.py`  
**MAX_EXAMPLES:** 16 | **INITIAL_EXAMPLES:** 5 | **FREE_THRESHOLD:** 5

| Models 4/4 correct | Typical examples used | Typical score |
|---|---|---|
| ~2–3 of 14 | 5–7 | 0.85–1.00 |
| Majority | 8–14 | 0.40–0.76 |

**Known design concern (does not prevent correct answers):** The task description text says `'some' → implies 'not all'`, while the code implements `'some' → implies 'not most'`. The exam hint says `'some' → 'not all'`. Models that follow the description/hint rather than examples may get a different answer than expected.

**However:** The grader uses the code's output as ground truth. Models that learn inductively from examples (as intended) will learn the code's rule. Only models that over-rely on the English meaning of "scalar implicature" are confused.

**Why efficiency suffers:** Models verify by requesting examples that directly show `'some'` vs `'most'` contrast, which typically requires 8–10 examples.

**Recommendation (optional):** Align the task description with the code: change `"'some' → implies 'not all'"` to `"'some' → implies 'not most'"` to eliminate the contradiction.

---

### 10. SKLONVETH — Root-and-Pattern Morphology

**File:** `sklonveth-root-pattern-lang-learning.py`  
**MAX_EXAMPLES:** 18 | **INITIAL_EXAMPLES:** 5 | **FREE_THRESHOLD:** 5

| Models 4/4 correct | Typical examples used | Typical score |
|---|---|---|
| ~4–5 of 14 | 5 | 1.00 |
| Majority | 7–12 | 0.65–0.89 |

**Known typo (low severity):** The task description's example for negation shows `dren-saakr` but the actual negated form is `dren-saakar`. Models that follow the examples (not the description typo) learn correctly. The typo may cause confusion on first read.

**Why efficiency suffers:** Semitic-style root insertion into vowel patterns requires seeing enough consonant-root + pattern combinations to generalize confidently. Models verify 2–3 extra examples before submitting.

**Recommendation:** Fix the typo in the description: `"dren-saakr"` → `"dren-saakar"`.

---

### 11. VRELTHAN — Rule Interaction (Opacity)

**File:** `vrelthan-rule-interaction-lang-learning.py`  
**MAX_EXAMPLES:** 20 | **INITIAL_EXAMPLES:** 6 | **FREE_THRESHOLD:** 6

| Models 4/4 correct | Typical examples used | Typical score |
|---|---|---|
| ~5–6 of 14 | 6 | 1.00 |
| Majority | 10–18 | 0.40–0.73 |

**Known design issue:** All 4 test items are present in the training list, so a model that exhausts all 20 examples will have memorized each test answer. This makes the task an interpolation exercise rather than a true generalization test.

**Why efficiency suffers:** Three ordered opaque rules (R1: vowel harmony, R2: final devoicing, R3: nasal assimilation, applied in that order) create non-obvious surface forms. Models that do not enumerate all 20 training examples before submitting must reason correctly about rule interaction — a hard inference task that many models fail.

**Recommendation:** Replace the 4 test items with genuinely novel stem+environment combinations not present in the 20-item training list.

---

### 12. WUKAL — Tonal System

**File:** `wukal-tones-lang-learning.py`  
**MAX_EXAMPLES:** 18 | **INITIAL_EXAMPLES:** 5 | **FREE_THRESHOLD:** 5

| Models 4/4 correct | Typical examples used | Typical score |
|---|---|---|
| ~3–4 of 14 | 5–8 | 0.84–1.00 |
| Majority | 10–16 | 0.40–0.73 |

**Known design concern:** The task description mentions an OCP (Obligatory Contour Principle) constraint that merges adjacent identical tones. However, this rule **never applies** in any of the 18 training examples or 4 test items. Models learn H-spreading and nasal blocking correctly but waste examples trying to find an OCP trigger.

**Why efficiency suffers:** Models request many examples searching for the OCP edge case that never appears, over-querying relative to what is actually tested.

**Recommendation:** Either (a) add training/test examples that trigger the OCP rule, or (b) remove the OCP from the description if it is not tested.

---

## Summary Table

| Task | Category | Max score achievable | Root cause of failures |
|---|---|---|---|
| **PELVAN** | Fixed | 1.00 ✓ | Bug: missing `_S_AGR` entries → `??` in expected answer |
| **SKOLVREN** | Fixed | 1.00 ✓ | Gap: `(infer,inter)` fusion never in training |
| **MIXED-RADIX** | Fixed | 1.00 ✓ | `FREE_THRESHOLD` too low for triple-rule task |
| **DRALVEN** | Fixed | 1.00 ✓ | Initial examples all same root, missing clash patterns |
| **STREVOKLAN** | Fixed | 1.00 ✓ | 2 of 4 test items leaked from training |
| **KOPHAR** | Fixed | 1.00 ✓ | Test noun `cloth` never in initial examples |
| **DRELVAK** | Efficiency-limited | 1.00 (a few) | Models over-verify nasal variant |
| **GRELKAN** | Efficiency-limited | 1.00 (a few) | Class B 3-stem paradigm requires more queries |
| **NORKVASH** | Efficiency-limited | 1.00 (a few) | Description/code contradiction; over-verification |
| **SKLONVETH** | Efficiency-limited | 1.00 (several) | Description typo; moderate over-verification |
| **VRELTHAN** | Efficiency-limited | 1.00 (several) | All test items in training; easy with full lookup |
| **WUKAL** | Efficiency-limited | 1.00 (a few) | Phantom OCP rule causes over-querying |

---

## Scoring Mechanics Reference

All tasks share the same efficiency formula:

```python
def _concept_score(correct_count, examples_used, max_examples, initial_examples):
    accuracy = correct_count / NUM_TEST_ITEMS
    if accuracy == 0:
        return 0.0
    eff_free = max(initial_examples, FREE_THRESHOLD)
    if max_examples <= eff_free or examples_used <= eff_free:
        efficiency = 1.0
    else:
        efficiency = max(0.0, 1.0 - (examples_used - eff_free) / (max_examples - eff_free))
    return accuracy * (0.40 + 0.60 * efficiency)
```

Key thresholds across tasks (after fixes):

| Task | INITIAL | FREE | MAX | Score at INITIAL | Score at MAX |
|---|---|---|---|---|---|
| PELVAN | 6 | 7 | 18 | 1.00 | 0.40 |
| SKOLVREN | 5 | 6 | 18 | 1.00 | 0.40 |
| MIXED-RADIX | **8** | **8** | 16 | 1.00 | 0.40 |
| DRALVEN | **6** | 3→**6** | 18 | 1.00 | 0.40 |
| STREVOKLAN | 4 | 4 | 14 | 1.00 | 0.40 |
| KOPHAR | 6 | 5→6 | 20 | 1.00 | 0.40 |
| DRELVAK | 5 | 5 | 18 | 1.00 | 0.40 |
| GRELKAN | 6 | 6 | 20 | 1.00 | 0.40 |
| NORKVASH | 5 | 5 | 16 | 1.00 | 0.40 |
| SKLONVETH | 5 | 5 | 18 | 1.00 | 0.40 |
| VRELTHAN | 6 | 6 | 20 | 1.00 | 0.40 |
| WUKAL | 5 | 5 | 18 | 1.00 | 0.40 |

Bold = changed by fixes in this session.

---

*Generated from fresh run logs downloaded via `19_fetch_task_runs.py` + `20_fetch_notebook_logs.py`.*  
*All fixes applied to `downloaded_tasks/language_learning/` in April 2026.*
