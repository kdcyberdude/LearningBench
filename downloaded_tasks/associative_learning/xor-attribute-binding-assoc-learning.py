#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests three-dimensional latent-feature binding under a mixed XOR/XNOR conjunction. "
    "Each invented word has three hidden properties: vowel-count parity (P), letter-count "
    "parity (Q), and whether it ends in a vowel (R). Two words BOND iff P differs AND Q "
    "matches AND R matches — a 3D conjunction where one axis is XOR and two are XNOR. "
    "The rule is never named; the model must induce all three properties and their asymmetric "
    "interaction from 13 labeled examples. Distractors cover every single-dimension failure "
    "mode and a 13th example specifically rules out the family of 3-way parity (XOR-of-three) "
    "alternatives that otherwise fit the first 12. "
    "Query 6 introduces a word containing 'y', making vowel-count ambiguous and the "
    "outcome deterministically unresolvable — testing epistemic calibration alongside "
    "rule induction."
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
        match = "✓" if _label_match(str(exp), str(actual)) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _label_match(expected: str, actual: str) -> bool:
    """Whole-word match, guarding against negation prefixes."""
    token = re.escape(expected.strip())
    match = re.search(rf"\b{token}\b", actual.strip(), re.IGNORECASE)
    if not match:
        return False
    prefix = actual[: match.start()].strip()
    last_words = re.findall(r"\w+", prefix)[-3:]
    negations = {"not", "isn't", "isnt", "never", "no", "cannot", "can't", "cant", "neither", "without"}
    return not any(w.lower() in negations for w in last_words)


@dataclass
class XorAttributeAnswer:
    q_1: str
    q_2: str
    q_3: str
    q_4: str
    q_5: str
    q_6: str


# Hidden rule (never stated in prompt):
#   Words have three latent binary properties:
#     P = vowel-count parity  (1 if odd number of vowels {a,e,i,o,u}, 0 if even)
#     Q = letter-count parity (1 if odd total letters, 0 if even)
#     R = ends-in-vowel flag  (1 if last letter is a vowel, 0 otherwise)
#
#   BOND iff  (P_A != P_B)  AND  (Q_A == Q_B)  AND  (R_A == R_B)
#   i.e. vowel-parity XORs while letter-parity and end-type XNOR.
#
# Word properties  (P, Q, R):
#   Training:  Brox(1,0,0) Roeg(0,0,0) Doue(1,0,1) Froe(0,0,1)
#              Klind(1,1,0) Groen(0,1,0) Drund(1,1,0) Stuve(0,1,1) Liuva(1,1,1) Noex(0,0,0)
#   Queries:   Vrox(1,0,0) Voek(0,0,0) Bluxe(0,1,1) Kluva(0,1,1)
#              Roea(1,0,1) Snue(0,0,1) Dreun(0,1,0) Glint(1,1,0) Previ(0,1,1)
#              Crypt(?,1,0) — 'y' creates P-ambiguity → UNKNOWN
#
# Distractor coverage:
#   P same, Q match, R match  → NO BOND: (Roeg+Noex), (Klind+Drund)
#   P differ, Q differ, R match → NO BOND: (Brox+Groen), (Froe+Liuva)
#   P differ, Q match, R differ → NO BOND: (Brox+Froe), (Groen+Liuva), (Drund+Stuve)
#   P same, Q differ, R differ  → NO BOND: (Brox+Liuva)
#   P differ, Q differ, R differ → NO BOND: (Brox+Stuve) ← rules out all 3-way XOR-parity alternatives
#
# Uniqueness verification (exhaustive search over all 2- and 3-feature AND/OR/XOR rules):
#   Only 'P_ne AND Q_eq AND R_eq' fits all 13 training examples.
#   No 2-feature conjunction, no OR rule, and no XOR-parity rule survives.

_EXPECTED = {
    "q_1": "BOND",     # Vrox(1,0,0) + Voek(0,0,0): P differ, Q match, R match → BOND
    "q_2": "NO BOND",  # Bluxe(0,1,1) + Kluva(0,1,1): P same → NO BOND
    "q_3": "BOND",     # Roea(1,0,1) + Snue(0,0,1): P differ, Q match, R match → BOND
    "q_4": "NO BOND",  # Vrox(1,0,0) + Dreun(0,1,0): P differ, Q differ, R differ → NO BOND
    "q_5": "NO BOND",  # Glint(1,1,0) + Previ(0,1,1): P differ, Q match, R differ → NO BOND
    "q_6": "UNKNOWN",  # Crypt(?,1,0) + Groen(0,1,0): 'y' makes P indeterminate → UNKNOWN
}


@kbench.task(
    name="xor_attribute_binding_assoc_learning",
    description=(
        "Induce a hidden 3D mixed XOR/XNOR conjunction (vowel-parity XOR, "
        "letter-parity XNOR, end-type XNOR) from 13 labeled word-pair examples, "
        "then classify five novel pairs and identify one unresolvable pair."
    ),
)
def xor_attribute_binding_assoc_learning(llm) -> float:
    """Three-dimensional latent-feature binding with UNKNOWN calibration; return fraction correct."""

    prompt = "\n".join([
        "Study the following word pairs. Each pair either BONDS or does NOT bond.",
        "Your task is to discover the hidden rule and classify the six query pairs.",
        "",
        "Observations:",
        "  Brox  and Roeg   -> BOND",
        "  Doue  and Froe   -> BOND",
        "  Klind and Groen  -> BOND",
        "  Stuve and Liuva  -> BOND",
        "  Roeg  and Noex   -> NO BOND",
        "  Klind and Drund  -> NO BOND",
        "  Brox  and Groen  -> NO BOND",
        "  Froe  and Liuva  -> NO BOND",
        "  Brox  and Froe   -> NO BOND",
        "  Groen and Liuva  -> NO BOND",
        "  Brox  and Liuva  -> NO BOND",
        "  Drund and Stuve  -> NO BOND",
        "  Brox  and Stuve  -> NO BOND",
        "",
        "For each query, respond with EXACTLY one of the tokens shown in brackets.",
        "",
        "  Q1: Vrox  and Voek   [BOND / NO BOND / UNKNOWN]",
        "  Q2: Bluxe and Kluva  [BOND / NO BOND / UNKNOWN]",
        "  Q3: Roea  and Snue   [BOND / NO BOND / UNKNOWN]",
        "  Q4: Vrox  and Dreun  [BOND / NO BOND / UNKNOWN]",
        "  Q5: Glint and Previ  [BOND / NO BOND / UNKNOWN]",
        "  Q6: Crypt and Groen  [BOND / NO BOND / UNKNOWN]",
        "",
    ])

    result = llm.prompt(prompt, schema=XorAttributeAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_EXPECTED)

    for key, expn in _EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        if _label_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must be {expn}.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _EXPECTED}
    _log_trace("xor_attribute_binding", _TASK_DESCRIPTION, prompt, answers, _EXPECTED, score)
    return score


if __name__ == "__main__":
    xor_attribute_binding_assoc_learning.run(kbench.llm)

