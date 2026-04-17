#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests directed latent tripartite set binding with a novel nonsense vocabulary. "
    "Nine tokens are secretly partitioned into three ordered tiers (T1, T2, T3). "
    "A pair (X, Y) is VALID iff X belongs to a strictly lower tier than Y "
    "(i.e. tier(X) < tier(Y)); all other pairs are INVALID — including same-tier "
    "and reverse-order pairs. "
    "The model sees twelve labelled examples drawn to prevent single-tier-pair shortcuts "
    "and must classify seven unseen pairs. One query token (Zhovek) is entirely absent "
    "from the training examples, so the model must infer its tier from structural "
    "constraints alone (process of elimination across the other eight tokens). "
    "No semantic cues exist: all tokens are invented nonsense words. "
    "Success requires: (1) inferring there are three tiers, not two; "
    "(2) inferring the direction constraint; (3) placing all nine tokens correctly; "
    "(4) generalising to the fully held-out token."
)

# ---------------------------------------------------------------------------
# Hidden partition (NEVER shown in the prompt):
#
#   T1 (lowest):  Breln, Stovak, Quimp
#   T2 (middle):  Draxel, Fovnis, Yelkor
#   T3 (highest): Crundl, Hwaspe, Zhovek   ← Zhovek is NEVER in any training example
#
# Rule: VALID iff tier(X) < tier(Y) strictly.
#
# Training example construction:
#   — Every ordered (Ti, Tj) pair with i < j must appear at least once
#     to establish both directions.
#   — Every (Ti, Ti) and (Ti, Tj) with i > j must appear at least once
#     to block shortcuts from assuming "any cross-tier = VALID".
#   — No single token appears in only one role to prevent per-token memorisation.
#
# Pair coverage in training:
#   T1→T2 VALID:   (Breln, Draxel), (Stovak, Fovnis), (Quimp, Yelkor)
#   T1→T3 VALID:   (Breln, Crundl), (Quimp, Hwaspe)
#   T2→T3 VALID:   (Draxel, Hwaspe), (Fovnis, Crundl)
#   Same-tier INVALID: (Stovak, Quimp), (Fovnis, Draxel)
#   Reverse T2→T1 INVALID: (Yelkor, Stovak)
#   Reverse T3→T2 INVALID: (Crundl, Yelkor)
#   Reverse T3→T1 INVALID: (Hwaspe, Breln)
#
# 12 training examples total; all 8 non-Zhovek tokens appear in both positions.
#
# Query answers:
#   Pair 1: (Stovak, Crundl)   T1→T3  VALID
#   Pair 2: (Yelkor, Fovnis)   T2→T2  INVALID  (same tier)
#   Pair 3: (Hwaspe, Quimp)    T3→T1  INVALID  (reverse)
#   Pair 4: (Draxel, Zhovek)   T2→T3  VALID    (Zhovek inferred T3)
#   Pair 5: (Zhovek, Breln)    T3→T1  INVALID  (reverse)
#   Pair 6: (Quimp, Fovnis)    T1→T2  VALID
#   Pair 7: (Crundl, Draxel)   T3→T2  INVALID  (reverse)
# ---------------------------------------------------------------------------

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
class LatentSetAnswer:
    pair_1: str
    pair_2: str
    pair_3: str
    pair_4: str
    pair_5: str
    pair_6: str
    pair_7: str

_LATENT_SET_EXPECTED = {
    "pair_1": "VALID",
    "pair_2": "INVALID",
    "pair_3": "INVALID",
    "pair_4": "VALID",
    "pair_5": "INVALID",
    "pair_6": "VALID",
    "pair_7": "INVALID",
}

@kbench.task(
    name="latent_set_binding_assoc_learning",
    description=(
        "Infer a hidden directed tripartite tier structure from VALID/INVALID pair "
        "examples using a novel nonsense vocabulary; apply it to seven unseen pairs "
        "including one token (Zhovek) absent from all training examples."
    ),
)
def latent_set_binding_assoc_learning(llm) -> float:
    """Directed tripartite latent set binding; return fraction of pairs classified correctly."""

    prompt = "\n".join([
        "A hidden rule determines whether a pair of tokens is VALID or INVALID.",
        "Study the examples carefully. Discover the rule, then classify the new pairs.",
        "",
        "Examples:",
        "  (Breln,  Draxel) -> VALID",
        "  (Stovak, Fovnis) -> VALID",
        "  (Quimp,  Yelkor) -> VALID",
        "  (Breln,  Crundl) -> VALID",
        "  (Quimp,  Hwaspe) -> VALID",
        "  (Draxel, Hwaspe) -> VALID",
        "  (Fovnis, Crundl) -> VALID",
        "  (Stovak, Quimp)  -> INVALID",
        "  (Fovnis, Draxel) -> INVALID",
        "  (Yelkor, Stovak) -> INVALID",
        "  (Crundl, Yelkor) -> INVALID",
        "  (Hwaspe, Breln)  -> INVALID",
        "",
        "Classify these new pairs:",
        "  Pair 1: (Stovak, Crundl)",
        "  Pair 2: (Yelkor, Fovnis)",
        "  Pair 3: (Hwaspe, Quimp)",
        "  Pair 4: (Draxel, Zhovek)",
        "  Pair 5: (Zhovek, Breln)",
        "  Pair 6: (Quimp,  Fovnis)",
        "  Pair 7: (Crundl, Draxel)",
        "",
    ])

    result = llm.prompt(prompt, schema=LatentSetAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_LATENT_SET_EXPECTED)
    for key, exp_val in _LATENT_SET_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        expn = str(exp_val).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _LATENT_SET_EXPECTED}
    _log_trace("latent_set_binding", _TASK_DESCRIPTION, prompt, answers, _LATENT_SET_EXPECTED, score)
    return score

if __name__ == "__main__":
    latent_set_binding_assoc_learning.run(kbench.llm)

