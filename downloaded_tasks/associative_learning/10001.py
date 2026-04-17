#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests inference-time induction of a three-component symbol-transformation rule "
    "from labeled examples alone — no rule is stated. "
    "Three symbol types (%, $, #) surround words; the model must discover independently "
    "that: (1) % contributes a flip when min(left-count, right-count) is odd; "
    "(2) $ contributes a flip when (right-count minus left-count) > 0; "
    "(3) # is a neutral distractor with no effect. "
    "The two active flip contributions compose via XOR: an even total of flips restores "
    "the original meaning; an odd total inverts it. "
    "The 12 training observations are crafted so that each alternative single-symbol "
    "hypothesis is falsified by at least one example, forcing joint inference of all "
    "three components. Test items interleave all three symbol types in novel configurations "
    "requiring full compositional application of the induced rule."
)

def _log_trace(task: str, description: str, prompt: str, answers: dict, expected: dict, score: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    print(f"\n  RESPONSES:")
    for key in expected:
        actual = answers.get(key, "?")
        exp = expected[key]
        match = "✓" if str(actual).lower() == str(exp).lower() else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")

def _str_match(expected: str, actual: str) -> bool:
    """Return True if expected appears anywhere in actual (case-insensitive)."""
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))

@dataclass
class ContextualFlipAnswer:
    word_1: str
    word_2: str
    word_3: str
    word_4: str
    word_5: str
    word_6: str
    word_7: str
    word_8: str

# ─── Hidden rule (never stated in prompt) ─────────────────────────────────────
#
#  Let L_x and R_x denote the count of symbol x to the LEFT and RIGHT of
#  the enclosed word, respectively.
#
#  % rule:   flip_% = (min(L_%, R_%) % 2 == 1)
#               — matched pairs of %; an unmatched % on either side is ignored
#               — one pair flips, two pairs restore, three pairs flip, etc.
#
#  $ rule:   flip_$ = ((R_$ - L_$) > 0)
#               — net rightward excess of $ triggers a flip
#               — balanced or left-heavy $ does NOT flip
#               — any strictly positive surplus on the right flips
#
#  # rule:   always False (pure distractor; appears in examples to block naive
#               "count all symbols" shortcuts)
#
#  Composition: result_flip = flip_% XOR flip_$
#               even total (both False or both True) -> original word
#               odd total (exactly one True)         -> opposite word
#
# ─── Training derivations ─────────────────────────────────────────────────────
#
#  Obs  1: %hot%        L%=1, R%=1; min=1 odd  -> flip_%=T | L$=0, R$=0; net=0  -> flip_$=F | T XOR F = T -> flip -> cold
#  Obs  2: %%fast%%     L%=2, R%=2; min=2 even -> flip_%=F | L$=0, R$=0; net=0  -> flip_$=F | F XOR F = F -> no flip -> fast
#  Obs  3: tall         L%=0, R%=0; min=0 even -> flip_%=F | L$=0, R$=0; net=0  -> flip_$=F | F -> no flip -> tall
#  Obs  4: bright$$     L%=0, R%=0; min=0 even -> flip_%=F | L$=0, R$=2; net=2  -> flip_$=T | F XOR T = T -> flip -> dark
#  Obs  5: $loud        L%=0, R%=0; min=0 even -> flip_%=F | L$=1, R$=0; net=-1 -> flip_$=F | F -> no flip -> loud
#  Obs  6: $hot$        L%=0, R%=0; min=0 even -> flip_%=F | L$=1, R$=1; net=0  -> flip_$=F | F -> no flip -> hot
#             (critical: balanced $ = no flip; eliminates "any $ = flip" hypothesis)
#  Obs  7: #fast#       L%=0, R%=0; min=0 even -> flip_%=F | L$=0, R$=0; net=0  -> flip_$=F | F -> no flip -> fast
#             (critical: # does not flip; eliminates "# = flip" hypothesis)
#  Obs  8: ##bright##   L%=0, R%=0; min=0 even -> flip_%=F | L$=0, R$=0; net=0  -> flip_$=F | F -> no flip -> bright
#             (confirms #-neutrality with two pairs; eliminates "paired # = flip")
#  Obs  9: %#tall#%     L%=1, R%=1; min=1 odd  -> flip_%=T | L$=0, R$=0; net=0  -> flip_$=F | T -> flip -> short
#             (critical: # interspersed with %; only % contributes)
#  Obs 10: %hot%$       L%=1, R%=1; min=1 odd  -> flip_%=T | L$=0, R$=1; net=1  -> flip_$=T | T XOR T = F -> no flip -> hot
#             (critical: both flip mechanisms cancel; eliminates independent % or $ rule)
#  Obs 11: %%tall%$%    L%=2, R%=2; min=2 even -> flip_%=F | L$=0, R$=1; net=1  -> flip_$=T | F XOR T = T -> flip -> short
#             (right side contains mixed % and $; tests parsing of interleaved symbols)
#  Obs 12: $%%loud%%    L%=2, R%=2; min=2 even -> flip_%=F | L$=1, R$=0; net=-1 -> flip_$=F | F -> no flip -> loud
#             (left side has $; left-heavy $ does not flip; reinforces net-rightward rule)
#
# ─── Test question derivations ────────────────────────────────────────────────
#
#  Q1: %%%fast$%%      L%=3, R%=2; min=2 even -> flip_%=F | L$=0, R$=1; net=1  -> flip_$=T | F XOR T = T -> flip -> slow
#  Q2: #$bright$#      L%=0, R%=0; min=0 even -> flip_%=F | L$=1, R$=1; net=0  -> flip_$=F | F -> no flip -> bright
#  Q3: %$%hot%%$       L%=2, R%=2; min=2 even -> flip_%=F | L$=1, R$=1; net=0  -> flip_$=F | F -> no flip -> hot
#  Q4: $$#tall#        L%=0, R%=0; min=0 even -> flip_%=F | L$=2, R$=0; net=-2 -> flip_$=F | F -> no flip -> tall
#  Q5: %$loud$$%       L%=1, R%=1; min=1 odd  -> flip_%=T | L$=1, R$=2; net=1  -> flip_$=T | T XOR T = F -> no flip -> loud
#  Q6: ##%$fast#%      L%=1, R%=1; min=1 odd  -> flip_%=T | L$=1, R$=0; net=-1 -> flip_$=F | T XOR F = T -> flip -> slow
#  Q7: $%%#loud#%$$%   L%=2, R%=2; min=2 even -> flip_%=F | L$=1, R$=2; net=1  -> flip_$=T | F XOR T = T -> flip -> quiet
#  Q8: %%%$hot$%%      L%=3, R%=2; min=2 even -> flip_%=F | L$=1, R$=1; net=0  -> flip_$=F | F -> no flip -> hot
#
# ──────────────────────────────────────────────────────────────────────────────

_CONTEXTUAL_FLIP_EXPECTED = {
    "word_1": "slow",
    "word_2": "bright",
    "word_3": "hot",
    "word_4": "tall",
    "word_5": "loud",
    "word_6": "slow",
    "word_7": "quiet",
    "word_8": "hot",
}

@kbench.task(
    name="contextual_flip_assoc_learning",
    description=(
        "Induce a hidden three-symbol transformation rule from 12 labeled examples "
        "(%/$/# with no stated semantics), then apply it to 8 novel symbol-wrapped words."
    ),
)
def contextual_flip_assoc_learning(llm) -> float:
    """Induce a hidden symbol-based transformation rule; return fraction of test queries correct."""

    prompt = "\n".join([
        "Each entry below shows a symbol-wrapped token and its meaning.",
        "Your task: discover the hidden rule governing how the symbols affect meaning,",
        "then apply that rule to determine the meaning of each query token.",
        "All meanings are single English words.",
        "",
        "Observations:",
        "  %hot%        ->  cold",
        "  %%fast%%     ->  fast",
        "  tall         ->  tall",
        "  bright$$     ->  dark",
        "  $loud        ->  loud",
        "  $hot$        ->  hot",
        "  #fast#       ->  fast",
        "  ##bright##   ->  bright",
        "  %#tall#%     ->  short",
        "  %hot%$       ->  hot",
        "  %%tall%$%    ->  short",
        "  $%%loud%%    ->  loud",
        "",
        "What is the meaning of each of the following?",
        "  Word 1:  %%%fast$%%",
        "  Word 2:  #$bright$#",
        "  Word 3:  %$%hot%%$",
        "  Word 4:  $$#tall#",
        "  Word 5:  %$loud$$%",
        "  Word 6:  ##%$fast#%",
        "  Word 7:  $%%#loud#%$$%",
        "  Word 8:  %%%$hot$%%",
        "",
    ])

    result = llm.prompt(prompt, schema=ContextualFlipAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_CONTEXTUAL_FLIP_EXPECTED)
    for key, exp_val in _CONTEXTUAL_FLIP_EXPECTED.items():
        act = str(getattr(result, key)).strip().lower()
        expn = str(exp_val).strip().lower()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _CONTEXTUAL_FLIP_EXPECTED}
    _log_trace("contextual_flip", _TASK_DESCRIPTION, prompt, answers, _CONTEXTUAL_FLIP_EXPECTED, score)
    return score

if __name__ == "__main__":
    contextual_flip_assoc_learning.run(kbench.llm)

