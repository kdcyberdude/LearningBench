#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests latent two-dimensional cross-domain binding. "
    "Objects associate with attributes only when BOTH of two hidden structural "
    "properties align: (1) letter-count parity (odd vs even) must match, AND "
    "(2) exact vowel count must match. "
    "The rule is never stated; the model must induce both constraints from ten "
    "training examples that include carefully chosen distractors — pairs sharing "
    "exactly one of the two required properties — to prevent single-variable "
    "shortcuts. Query pairs include novel words and cross-context reuse to test "
    "genuine generalisation rather than surface pattern-matching."
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
        match = "✓" if _str_match(str(exp), str(actual)) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")

def _str_match(expected: str, actual: str) -> bool:
    """Return True if expected appears anywhere in actual (case-insensitive)."""
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))

@dataclass
class LatentCrossBindingAnswer:
    q_1: str
    q_2: str
    q_3: str
    q_4: str
    q_5: str
    q_6: str

_LATENT_CROSS_EXPECTED = {
    "q_1": "YES",
    "q_2": "NO",
    "q_3": "NO",
    "q_4": "YES",
    "q_5": "YES",
    "q_6": "NO",
}

# Hidden rule:
#   An object associates with an attribute if and only if:
#     (a) len(object) % 2 == len(attribute) % 2   [same letter-count parity]
#     AND
#     (b) vowel_count(object) == vowel_count(attribute)  [exact vowel count equality]
#
# Word properties (LP = letter parity, V = vowel count):
#   Training objects:  Vrelm(ODD,1) Buxa(EVEN,2) Trond(ODD,1) Skove(ODD,2) Brox(EVEN,1) Prubax(EVEN,2)
#   Training attrs:    Strev(ODD,1) Frua(EVEN,2) Glire(ODD,2) Splint(EVEN,1) Kwend(ODD,1) Droib(ODD,2)
#   Query objects:     Drevs(ODD,1) Claubi(EVEN,3) Fluon(ODD,2) Sprixe(EVEN,2) Kruvox(EVEN,2)
#   Query attrs:       Prend(ODD,1) Grux(EVEN,1) [also reuses Frua, Kwend, Glire, Splint]
#
# Distractor coverage:
#   Same LP, diff VP -> NO: (Vrelm+Glire) (Buxa+Splint) (Trond+Droib) (Skove+Strev)
#   Diff LP, same VP -> NO: (Brox+Strev)  <- eliminates vowel-count-alone rule

@kbench.task(
    name="latent_cross_binding_assoc_learning",
    description=(
        "Induce a hidden two-property binding rule (letter-count parity + exact vowel "
        "count must both match) from ten labeled examples that include single-property "
        "distractors, then apply it to six novel query pairs."
    ),
)
def latent_cross_binding_assoc_learning(llm) -> float:
    """Latent two-dimensional cross-domain binding; return fraction of queries correct."""

    prompt = "\n".join([
        "Study the following associations between objects and attributes.",
        "Some pairs associate; others do not. Your task is to discover the hidden",
        "rule governing which pairs associate, then answer the queries below.",
        "",
        "Observations:",
        "  Vrelm  associates    with  Strev",
        "  Vrelm  does NOT associate  with  Glire",
        "  Buxa   associates    with  Frua",
        "  Buxa   does NOT associate  with  Splint",
        "  Trond  associates    with  Kwend",
        "  Trond  does NOT associate  with  Droib",
        "  Skove  associates    with  Glire",
        "  Skove  does NOT associate  with  Strev",
        "  Brox   does NOT associate  with  Strev",
        "  Prubax associates    with  Frua",
        "",
        "Queries:",
        "  Question 1: Does Drevs  associate with Prend?",
        "  Question 2: Does Claubi associate with Frua?",
        "  Question 3: Does Fluon  associate with Kwend?",
        "  Question 4: Does Sprixe associate with Frua?",
        "  Question 5: Does Fluon  associate with Glire?",
        "  Question 6: Does Kruvox associate with Splint?",
        "",
    ])

    result = llm.prompt(prompt, schema=LatentCrossBindingAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_LATENT_CROSS_EXPECTED)
    for key, exp_val in _LATENT_CROSS_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        expn = str(exp_val).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _LATENT_CROSS_EXPECTED}
    _log_trace("latent_cross_binding", _TASK_DESCRIPTION, prompt, answers, _LATENT_CROSS_EXPECTED, score)
    return score

if __name__ == "__main__":
    latent_cross_binding_assoc_learning.run(kbench.llm)

