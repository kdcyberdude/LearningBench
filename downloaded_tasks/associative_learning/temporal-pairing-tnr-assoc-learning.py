#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import random
import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests associative temporal binding with a multiplicative-vs-additive delay rule and a "
    "two-dimensional identity switch. Each sequence begins with a two-digit code 'd1d2'. "
    "An '@' token is the anchor. Without a leading '*' token the target uppercase letter "
    "is placed exactly (d1+d2) token-positions after '@', and the letter is the d1-th letter "
    "of the alphabet (A=1, B=2, …). When '*' precedes '@', the delay becomes d1×d2 positions "
    "and the predicted letter is the d2-th letter of the alphabet. Both the delay formula "
    "AND the letter identity change simultaneously when '*' is present. "
    "The model sees 500 examples and must predict the missing letter in four test sequences. "
    "Success requires inferring that '*' switches BOTH the arithmetic operation (add→multiply) "
    "AND which digit indexes the predicted letter (d1→d2), from evidence alone."
)

# ── Hidden structure ──────────────────────────────────────────────────────────
#
#  Two-digit leading code: d1 (tens digit), d2 (ones digit).
#  '@' is the temporal anchor token.
#  Optional '*' token immediately after the leading code.
#
#  WITHOUT '*':
#    delay  = d1 + d2           (additive; target is this many positions after '@')
#    letter = chr(64 + d1)      (d1-th letter of alphabet: 1→A, 2→B, 3→C, 4→D …)
#
#  WITH '*' (immediately following the leading code):
#    delay  = d1 × d2           (multiplicative; distinct from additive for d1≠1,d2≠1)
#    letter = chr(64 + d2)      (d2-th letter of alphabet)
#
#  Fillers: lowercase consonants drawn without replacement per sequence.
#  Predicted uppercase letter (A–D) never appears as a filler.
#  One trailing filler follows the target letter in every example.
#
# ─────────────────────────────────────────────────────────────────────────────

_FILLER_POOL = list("bcdfghjklmnpqrstvwxyz")


def _apply_rule(d1: int, d2: int, star: bool) -> tuple[int, str]:
    """Return (delay, letter) for the given digits and star flag."""
    if star:
        delay = d1 * d2
        letter = chr(64 + d2)
    else:
        delay = d1 + d2
        letter = chr(64 + d1)
    return delay, letter


def _build_sequence(d1: int, d2: int, star: bool, rng: random.Random, mask: bool = False) -> tuple[str, str]:
    """
    Build one complete sequence string and return (sequence, answer_letter).

    When mask=True the target letter is replaced with '_' (test mode).
    The '@' anchor is always preceded by (star ? 1 : 0) prefix fillers before
    the anchor so that the anchor position varies slightly.
    """
    delay, letter = _apply_rule(d1, d2, star)

    forbidden = {letter.lower()}
    pool = [c for c in _FILLER_POOL if c not in forbidden]
    fillers = rng.sample(pool, k=delay + 3)

    pre_anchor_count = 1 if star else rng.randint(0, 2)
    pre_anchor = fillers[:pre_anchor_count]
    post_anchor = fillers[pre_anchor_count: pre_anchor_count + (delay - 1)]
    trailing = [fillers[pre_anchor_count + (delay - 1)]]

    target_token = "_" if mask else letter

    code = f"{d1}{d2}"
    parts: list[str] = [code]
    if star:
        parts.append("*")
    parts.extend(pre_anchor)
    parts.append("@")
    parts.extend(post_anchor)
    parts.append(target_token)
    parts.extend(trailing)

    return " ".join(parts), letter


def _generate_examples(n: int, rng: random.Random) -> list[str]:
    """
    Generate n labelled example sequences using the hidden rule.
    d1, d2 ∈ {1,2,3,4}; star is sampled 50/50.
    """
    examples: list[str] = []
    for _ in range(n):
        d1 = rng.randint(1, 4)
        d2 = rng.randint(1, 4)
        star = rng.random() < 0.5
        seq, _ = _build_sequence(d1, d2, star, rng, mask=False)
        examples.append(f"  - {seq}")
    return examples


# ── Fixed test items ──────────────────────────────────────────────────────────
#
#  Seq 1: d1=1, d2=3, no *  → delay=4, letter=A
#  Seq 2: d1=2, d2=4, WITH * → delay=8, letter=D
#  Seq 3: d1=3, d2=1, no *  → delay=4, letter=C
#  Seq 4: d1=2, d2=4, no *  → delay=6, letter=B
#         (same digits as Seq 2 but no '*': tests both dimensions simultaneously)
#
# ─────────────────────────────────────────────────────────────────────────────

_TEST_PARAMS = [
    (1, 3, False),
    (2, 4, True),
    (3, 1, False),
    (2, 4, False),
]

_TEST_RNG_SEED = 42


def _build_test_items() -> tuple[list[str], dict[str, str]]:
    """Build the four fixed test sequences and their ground-truth answers."""
    rng = random.Random(_TEST_RNG_SEED)
    seqs: list[str] = []
    expected: dict[str, str] = {}
    for i, (d1, d2, star) in enumerate(_TEST_PARAMS, start=1):
        seq, letter = _build_sequence(d1, d2, star, rng, mask=True)
        seqs.append(f"  Seq {i}: {seq}")
        expected[f"seq_{i}"] = letter
    return seqs, expected


_TEST_SEQUENCES, _TNR_EXPECTED = _build_test_items()


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
class TemporalPairingTnrAnswer:
    seq_1: str
    seq_2: str
    seq_3: str
    seq_4: str


@kbench.task(
    name="temporal_pairing_tnr_assoc_learning",
    description=(
        "Test additive vs multiplicative temporal rules: '@' anchor, optional '*' switch. "
        "No '*': delay=d1+d2, letter=d1-th; with '*': delay=d1*d2, letter=d2-th. "
        "Both rules flip with '*'. Includes tricky test cases."
    ),
)
def temporal_pairing_tnr_assoc_learning(llm) -> float:
    """Dual-rule temporal binding (@-anchor, *-switch); return fraction correct."""

    rng = random.Random(0)
    example_lines = _generate_examples(5000, rng)

    prompt = "\n".join([
        "Each sequence below begins with a two-digit code and contains a special token '@'.",
        "The '@' is a temporal anchor. Somewhere after '@', an uppercase letter appears.",
        "An optional token '*' may follow the two-digit code and changes the hidden rules.",
        "All other tokens are irrelevant lowercase fillers.",
        "",
        "Study these 5000 examples to discover the hidden rules:",
        *example_lines,
        "",
        "Predict the single uppercase letter that replaces '_' in each sequence below.",
        *_TEST_SEQUENCES,
        "",
    ])

    result = llm.prompt(prompt, schema=TemporalPairingTnrAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_TNR_EXPECTED)
    for key, exp_val in _TNR_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        expn = str(exp_val).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _TNR_EXPECTED}
    _log_trace("temporal_pairing_tnr", _TASK_DESCRIPTION, prompt, answers, _TNR_EXPECTED, score)
    return score


if __name__ == "__main__":
    temporal_pairing_tnr_assoc_learning.run(kbench.llm)

