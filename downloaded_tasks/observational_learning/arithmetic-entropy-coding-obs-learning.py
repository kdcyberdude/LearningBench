#!/usr/bin/env python
# coding: utf-8

import math
from dataclasses import dataclass

import kaggle_benchmarks as kbench


def _log_trace(task, description, prompt, test_results, score, reasoning=""):
    sep = "=" * 70
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    if reasoning:
        print(f"\n  REASONING:\n{reasoning}")
    print(f"\n  TEST RESULTS:")
    for r in test_results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"    [{status}] Q{r['q']}: expected={r['expected']!r}  got={r['got']!r}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


_TASK_DESCRIPTION = (
    "The model observes symbol-sequence to encoded-bit-length pairs from an entropy coder "
    "with five hidden non-uniform symbol probabilities over alphabet {A,B,C,D,E}. "
    "The probabilities are all distinct. The hidden rule is ceil(-Σ log₂(P(x))) summed "
    "over every symbol occurrence. Early demos have equal counts and only pin the sum of "
    "all h-values; structured demos with one dominant symbol allow isolating individual "
    "h-values through inequality reasoning. All five h-values are uniquely determined from "
    "the 13 demonstrations. The model must then compute the encoded length for four new sequences."
)

# Hidden distribution: P(A)=10/27, P(B)=7/27, P(C)=5/27, P(D)=3/27, P(E)=2/27
# All five probabilities are strictly distinct.
# Uniqueness has been verified: no alternative integer-weighted distribution
# (with total ≤ 49) produces identical bit lengths for all 13 demos.
_PROBS = {
    "A": 10 / 27,
    "B": 7 / 27,
    "C": 5 / 27,
    "D": 3 / 27,
    "E": 2 / 27,
}


def _entropy_length(seq: str, probs: dict) -> int:
    return math.ceil(sum(-math.log2(probs[s]) for s in seq if probs.get(s, 0) > 0))


# fmt: off
# Demonstrations are ordered to guide inference:
#   Phase 1 (demos 1-3):  equal counts of each symbol → pins sum S = Σ h_x
#   Phase 2 (demos 4-8):  one symbol dominates (7 copies) + 1 of each other →
#                          each demo pins one h-value relative to S
#   Phase 3 (demos 9-13): varied mixes → resolve residual ambiguity and
#                          confirm the relative ordering of all five h-values
_DEMO_SEQS = [
    "AABBCCDDEE",      # 2 each: ceil(2S) = 26 → S ∈ (12.5, 13]
    "AABCDDEEBC",      # 2 each
    "BBAAECDCDE",      # 2 each
    "AAAAAAAABCDE",    # 8A+1B+1C+1D+1E: 7h_A+S → 23 → h_A ∈ (1.32, 1.47]
    "BBBBBBBACDE",     # 1A+7B+1C+1D+1E: 7h_B+S → 25
    "CCCCCCCABDE",     # 1A+1B+7C+1D+1E: 7h_C+S → 28
    "DDDDDDDABCE",     # 1A+1B+1C+7D+1E: 7h_D+S → 32
    "EEEEEEEABCD",     # 1A+1B+1C+1D+7E: 7h_E+S → 36
    "AAABBBCDE",       # 3A+3B: tests h_A+h_B separation
    "AACCCDE",         # 2A+3C+1D+1E: cross-checks h_C
    "AADDDDE",         # 2A+4D+1E: cross-checks h_D
    "AABBBBCDE",       # 2A+4B+1C+1D+1E: refines h_B
    "CCDDDDDE",        # 2C+5D+1E: cross-checks h_D vs h_C
]
# fmt: on

_DEMOS = [(seq, _entropy_length(seq, _PROBS)) for seq in _DEMO_SEQS]

# Test sequences are designed to probe different combinations of the distribution:
#   Q1: A-dominant mix (5A + 3B + singles) — tests common-symbol region
#   Q2: mid-range mix (4B + 3C + 2D + 1E) — no A, tests middle and rare symbols
#   Q3: rare-symbol only (3C + 3D + 3E)   — stresses accurate rare h-values
#   Q4: one of each (A+B+C+D+E)           — baseline cross-check on full S
_TEST_CASES = [
    ("AAAAABBBCDE", _entropy_length("AAAAABBBCDE", _PROBS)),  # → 23
    ("BBBBCCCDDE",  _entropy_length("BBBBCCCDDE",  _PROBS)),  # → 26
    ("CCCDDDEEE",   _entropy_length("CCCDDDEEE",   _PROBS)),  # → 29
    ("ABCDE",       _entropy_length("ABCDE",        _PROBS)),  # → 13
]


def _build_prompt(demos: list, test_seqs: list) -> str:
    lines = [
        "You are observing a data compression system that encodes symbol sequences.",
        "The alphabet is {A, B, C, D, E}. Each symbol has a fixed but hidden probability.",
        "Encoded length = ceil( sum of -log2(P(x)) over every symbol occurrence ).",
        "",
        "Observations (symbol sequence → encoded bit length):",
    ]
    for i, (seq, length) in enumerate(demos, 1):
        counts = {s: seq.count(s) for s in "ABCDE" if seq.count(s) > 0}
        lines.append(f"  {i:2d}. {seq}  (counts: {counts}) → {length} bits")
    lines.append("")
    lines.append("Now solve these 4 test questions:")
    for i, seq in enumerate(test_seqs, 1):
        counts = {s: seq.count(s) for s in "ABCDE" if seq.count(s) > 0}
        lines.append(f"  Q{i}: {seq}  (counts: {counts}) → ? bits")
    lines.append("")
    lines.append("Submit answer_1 through answer_4 as integer bit lengths.")
    return "\n".join(lines)


def _prepare():
    prompt = _build_prompt(_DEMOS, [tc[0] for tc in _TEST_CASES])

    def grade_fn(response):
        results = []
        correct = 0
        for i, (seq, expected) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"answer_{i}", None)
            got = None
            ok = False
            try:
                got = int(raw)
                ok = got == expected
            except (TypeError, ValueError):
                pass
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: int
    answer_2: int
    answer_3: int
    answer_4: int


@kbench.task(
    name="arithmetic_entropy_coding_obs_learning",
    description=(
        "Observe 13 sequence→bit-length pairs for an entropy coder with five hidden "
        "non-uniform probabilities over {A,B,C,D,E} (all distinct). Structured demos "
        "allow isolating each symbol's self-information. Compute encoded lengths for "
        "4 new sequences."
    ),
)
def arithmetic_entropy_coding_obs_learning(llm) -> float:
    """Infer five hidden symbol probabilities from 13 coding examples; compute bit lengths for 4 new sequences."""
    prompt, grade_fn = _prepare()
    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None

    if response is not None:
        score, test_results = grade_fn(response)
    else:
        score = 0.0
        test_results = [
            {"q": i, "expected": _TEST_CASES[i - 1][1], "got": None, "correct": False}
            for i in range(1, 5)
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="arithmetic_entropy_coding_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    arithmetic_entropy_coding_obs_learning.run(kbench.llm)

