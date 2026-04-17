#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests latent three-way set binding with novel vocabulary. "
    "Nine words (Vorn/Kleth/Spaud, Briw/Trusk/Plaven, Druze/Skant/Flob) are secretly "
    "partitioned into three groups of three. A triple is VALID iff its three words come "
    "from three different groups (one from each); otherwise INVALID. "
    "Eleven labeled triples — including mixed distractors — uniquely determine the "
    "partition (up to label permutation). No surface feature (word length, vowel count, "
    "first/last letter, alphabetical order) correlates with group membership. "
    "The model must perform combinatorial latent-group inference across three categories "
    "rather than the simpler bipartite case, requiring significantly more structured "
    "reasoning from a small example set."
)

# Hidden partition (labels are arbitrary; only the partition matters):
#   Group α: Vorn, Kleth, Spaud
#   Group β: Briw, Trusk, Plaven
#   Group γ: Druze, Skant, Flob
#
# Rule: a triple (X, Y, Z) is VALID iff {group(X), group(Y), group(Z)} = {α, β, γ}
#
# Anti-shortcut coverage in training set:
#   (Spaud, Briw, Skant) VALID  -> repeated first letter 'S' yet VALID  (kills "all-distinct first letters")
#   (Vorn, Plaven, Skant) VALID -> Vorn/Plaven both end 'n' yet VALID   (kills "all-distinct last letters")
#   (Vorn, Trusk, Flob) VALID   -> vowel_sum=3, same as (Vorn,Kleth,Briw) INVALID  (kills vowel-sum shortcut)
#   (Briw, Trusk, Plaven) INVALID -> all same-group, kills "three-way distractor from three known groups"
#   (Druze, Skant, Trusk) INVALID -> two from γ + one from β, covers partial-match distractor

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
class LatentSetVariantAnswer:
    triple_1: str
    triple_2: str
    triple_3: str
    triple_4: str
    triple_5: str
    triple_6: str

_LATENT_SET_VARIANT_EXPECTED = {
    "triple_1": "VALID",
    "triple_2": "VALID",
    "triple_3": "INVALID",
    "triple_4": "INVALID",
    "triple_5": "VALID",
    "triple_6": "VALID",
}

@kbench.task(
    name="latent_set_variant_assoc_learning",
    description=(
        "Latent three-way set binding: infer a hidden three-group partition from "
        "labeled triples, then classify six unseen triples as VALID or INVALID."
    ),
)
def latent_set_variant_assoc_learning(llm) -> float:
    """Three-group latent set binding (ternary variant); return fraction of queries correct."""

    prompt = "\n".join([
        "A hidden rule determines whether a triple of words is VALID or INVALID.",
        "Study the labeled examples below and discover the rule.",
        "",
        "Labeled examples:",
        "  (Vorn,  Briw,   Druze)  -> VALID",
        "  (Vorn,  Trusk,  Flob)   -> VALID",
        "  (Kleth, Plaven, Skant)  -> VALID",
        "  (Spaud, Briw,   Skant)  -> VALID",
        "  (Vorn,  Plaven, Skant)  -> VALID",
        "  (Trusk, Druze,  Kleth)  -> VALID",
        "  (Flob,  Plaven, Spaud)  -> VALID",
        "  (Vorn,  Kleth,  Spaud)  -> INVALID",
        "  (Briw,  Trusk,  Plaven) -> INVALID",
        "  (Vorn,  Kleth,  Briw)   -> INVALID",
        "  (Druze, Skant,  Trusk)  -> INVALID",
        "",
        "Classify each of the following triples:",
        "  Triple 1: (Kleth, Briw,  Druze)",
        "  Triple 2: (Spaud, Trusk, Druze)",
        "  Triple 3: (Kleth, Trusk, Plaven)",
        "  Triple 4: (Vorn,  Druze, Skant)",
        "  Triple 5: (Spaud, Plaven, Flob)",
        "  Triple 6: (Flob,  Briw,  Kleth)",
        "",
    ])

    result = llm.prompt(prompt, schema=LatentSetVariantAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_LATENT_SET_VARIANT_EXPECTED)
    for key, exp_val in _LATENT_SET_VARIANT_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        expn = str(exp_val).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _LATENT_SET_VARIANT_EXPECTED}
    _log_trace("latent_set_variant", _TASK_DESCRIPTION, prompt, answers, _LATENT_SET_VARIANT_EXPECTED, score)
    return score

if __name__ == "__main__":
    latent_set_variant_assoc_learning.run(kbench.llm)

