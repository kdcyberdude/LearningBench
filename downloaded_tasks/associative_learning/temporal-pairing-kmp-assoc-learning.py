#!/usr/bin/env python
# coding: utf-8

"""
Inference-time associative learning: gated selection among two signal tokens.

Hidden structure (never stated in the prompt):
  Six signal tokens V N T G Z J map to ranks 1..6 and output letters A..F.
  Each line has exactly one gate in {^, *, =} at the start, then fillers,
  then the two distinct signals in left-to-right order.

  ^  → output letter for the FIRST signal (by position).
  *  → output letter for the SECOND signal.
  =  → output letter for whichever signal has the HIGHER rank (order irrelevant).

The = regime is the diagnostic: reversed token order must still yield the max-rank letter.
"""

from dataclasses import dataclass

import random
import re

import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Gated dual-signal binding: six invented uppercase signal tokens each map to a hidden rank "
    "1–6 (output letters A–F). Each training line begins with one of three gate symbols; "
    "exactly two signal tokens appear among lowercase fillers. "
    "Gate ^ selects the first-occurring signal's letter; * selects the second; = selects the "
    "letter of the higher-ranked signal regardless of left–right order. "
    "The model must infer all six ranks and all three gates from 500 examples, then answer "
    "six queries including = cases where the higher-rank token appears second (order trap)."
)

_SIGNALS = ("V", "N", "T", "G", "Z", "J")
# Arbitrary bijection rank 1..6 → letter A..F (not alphabetical by token name).
_RANK: dict[str, int] = {"V": 1, "N": 2, "T": 3, "G": 4, "Z": 5, "J": 6}
_LETTER = {s: chr(64 + _RANK[s]) for s in _SIGNALS}

_FILLERS = [c for c in "abcdefghijklmnopqrsuvwxyz"]


def _outcome(gate: str, first: str, second: str) -> str:
    if gate == "^":
        return _LETTER[first]
    if gate == "*":
        return _LETTER[second]
    if gate == "=":
        hi = first if _RANK[first] > _RANK[second] else second
        return _LETTER[hi]
    raise ValueError(f"unknown gate {gate!r}")


def _build_line(gate: str, first: str, second: str, rng: random.Random) -> tuple[str, str]:
    """Return (line_without_arrow, expected_letter)."""
    parts: list[str] = [gate]
    n_pre = rng.randint(1, 4)
    n_mid = rng.randint(1, 4)
    n_post = rng.randint(1, 4)
    parts.extend(rng.sample(_FILLERS, k=n_pre))
    parts.append(first)
    parts.extend(rng.sample(_FILLERS, k=n_mid))
    parts.append(second)
    parts.extend(rng.sample(_FILLERS, k=n_post))
    letter = _outcome(gate, first, second)
    return " ".join(parts), letter


def _generate_examples(n: int, rng: random.Random) -> list[str]:
    lines: list[str] = []
    gates = ["^", "*", "="]
    for _ in range(n):
        a, b = rng.sample(list(_SIGNALS), k=2)
        gate = rng.choice(gates)
        # randomize which signal is left vs right
        if rng.random() < 0.5:
            first, second = a, b
        else:
            first, second = b, a
        body, letter = _build_line(gate, first, second, rng)
        lines.append(f"  - {body} -> {letter}")
    return lines


# Fixed test queries (body only; answer derived from _outcome).
_TEST_SPEC = [
    # ^ first signal wins by position
    ("^", "V", "T", "order"),
    # * second signal
    ("*", "V", "T", "order"),
    # = max rank: T(3) > V(1), V first
    ("=", "V", "T", "order"),
    # = same pair reversed: still T
    ("=", "T", "V", "order"),
    # ^ G before Z → G → D
    ("^", "G", "Z", "order"),
    # * Z before G → second is G → D
    ("*", "Z", "G", "order"),
]

_TEST_RNG = random.Random(42)


def _build_tests() -> tuple[list[str], dict[str, str]]:
    rng = _TEST_RNG
    prompts: list[str] = []
    expected: dict[str, str] = {}
    for i, (gate, f, s, _) in enumerate(_TEST_SPEC, start=1):
        body, letter = _build_line(gate, f, s, rng)
        prompts.append(f"  Query {i}: {body} -> ?")
        expected[f"q_{i}"] = letter
    return prompts, expected


_TEST_PROMPTS, _EXPECTED = _build_tests()


def _log_trace(task: str, description: str, prompt: str, answers: dict, expected: dict, score: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    print(f"\n  RESPONSES:")
    for key in expected:
        actual = answers.get(key, "?")
        exp = expected[key]
        match = "✓" if _letter_match(exp, str(actual)) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _letter_match(expected: str, actual: str) -> bool:
    exp = expected.strip().upper()
    act = actual.strip().upper()
    if len(exp) != 1:
        return False
    return bool(re.search(rf"\b{re.escape(exp)}\b", act)) or (len(act) == 1 and act == exp)


@dataclass
class GatedDualSignalAnswer:
    q_1: str
    q_2: str
    q_3: str
    q_4: str
    q_5: str
    q_6: str


@kbench.task(
    name="gated_dual_signal_binding_assoc_learning",
    description=(
        "Six signal tokens with hidden ranks; gate ^/* /= selects first, second, or max-rank "
        "signal's output letter. 500 in-context examples; six queries test order vs value for '='."
    ),
)
def gated_dual_signal_binding_assoc_learning(llm) -> float:
    rng = random.Random(0)
    example_lines = _generate_examples(500, rng)

    prompt = "\n".join([
        "Each line below is a space-separated list of tokens.",
        "Every line with a label ends with \"->\" and a single uppercase letter A–F.",
        "Infer the rule from the examples, then answer each query with one letter A–F.",
        "",
        *example_lines,
        "",
        *_TEST_PROMPTS,
        "",
    ])

    result = llm.prompt(prompt, schema=GatedDualSignalAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_EXPECTED)
    for key, expn in _EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        if _letter_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must be {expn}.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _EXPECTED}
    _log_trace("gated_dual_signal_binding", _TASK_DESCRIPTION, prompt, answers, _EXPECTED, score)
    return score


# --- Sanity: expected answers match hidden rule ---
for i, (g, f, s, _) in enumerate(_TEST_SPEC, start=1):
    assert _EXPECTED[f"q_{i}"] == _outcome(g, f, s), (_TEST_SPEC[i - 1], _EXPECTED[f"q_{i}"])

if __name__ == "__main__":
    gated_dual_signal_binding_assoc_learning.run(kbench.llm)

