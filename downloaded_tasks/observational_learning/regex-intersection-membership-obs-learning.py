#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
from typing import List, Tuple

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can infer THREE independent hidden structural constraints "
    "from membership observations over alphabet {a, b, c}. "
    "The intersection is: (same first and last character) ∩ (no two adjacent characters identical) "
    "∩ (contains the letter 'c' at least once). "
    "All rules are purely structural — no counting or arithmetic required. "
    "Early positive examples satisfy all three simultaneously, risking anchoring on a single rule. "
    "Demos progressively expose each constraint independently as the sole failure mode. "
    "Uniqueness is enforced: (a) IN demos include non-palindromes with matching endpoints to rule out "
    "palindrome as L1; (b) fails-L2 demos cover 'aa' and 'cc' runs (not just 'bb') to rule out "
    "'no bb' as L2; (c) fails-L3 demos have no 'c' but do contain 'b' to rule out 'contains b' "
    "as L3, and include both odd and even lengths to rule out length-parity as L3. "
    "No pairwise constraint subset (L1∩L2, L1∩L3, L2∩L3) fits all 14 demos."
)

_FIXED_SEED = 0


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


def _in_l1(s: str) -> bool:
    """L1: first and last characters are identical."""
    return len(s) >= 1 and s[0] == s[-1]


def _in_l2(s: str) -> bool:
    """L2: no two adjacent characters are the same (no 'aa', 'bb', or 'cc')."""
    return all(s[i] != s[i + 1] for i in range(len(s) - 1))


def _in_l3(s: str) -> bool:
    """L3: string contains the letter 'c' at least once."""
    return "c" in s


def _in_intersection(s: str) -> bool:
    return _in_l1(s) and _in_l2(s) and _in_l3(s)


# 14 demos structured in phases:
# Phase 1: 5 IN strings — all 3 constraints satisfied simultaneously.
#   Demo 1 ("abca")  is NOT a palindrome yet shares endpoints → rules out palindrome as L1.
#   Demo 3 ("cbac")  has even length 4 → rules out "length odd" for future constraint guessing.
#   Demo 4 ("acabca") has length 6 → rules out "length ≤ 5" for L2.
#   Demo 2 ("bcb") starts with 'b' → rules out "starts with 'a'" as L1.
# Phase 2: 3 NOT IN strings that fail L1 only (first ≠ last, no adjacent dups, has 'c')
# Phase 3: 3 NOT IN strings that fail L2 only (first == last, has adjacent dup, has 'c')
#   Covers 'aa' run ("caac") and 'cc' run ("acca") → rules out "no bb" as full L2 rule.
# Phase 4: 3 NOT IN strings that fail L3 only (first == last, no adjacent dups, no 'c')
#   "aba" has 'b' but no 'c' → rules out "contains 'b'" as L3.
#   "babab" (odd length) and "abba"... wait. Let me keep: "aba" (len 3) and "babab" (len 5)
#   are both NOT IN (fail L3), while "bcb" (len 3) is IN → rules out "length odd" as L3.
#
# Uniqueness: removing any single constraint causes ≥2 demo mismatches.
_DEMOS: List[Tuple[str, bool]] = [
    # IN: all 3 satisfied
    ("abca", True),     # L1: a==a ✓, L2: a-b-c-a no adj dups ✓, L3: has 'c' ✓; NOT palindrome
    ("bcb", True),      # L1: b==b ✓, L2: b-c-b ✓, L3: has 'c' ✓; starts with 'b'
    ("cbac", True),     # L1: c==c ✓, L2: c-b-a-c ✓, L3: has 'c' ✓; even length 4
    ("acabca", True),   # L1: a==a ✓, L2: a-c-a-b-c-a ✓, L3: has 'c' ✓; length 6
    ("bcacb", True),    # L1: b==b ✓, L2: b-c-a-c-b ✓, L3: has 'c' ✓
    # NOT IN: fails L1 only (first ≠ last, no adjacent dups, has 'c')
    ("abc", False),     # a≠c; L2 ✓, L3 ✓
    ("bca", False),     # b≠a; L2 ✓, L3 ✓
    ("acb", False),     # a≠b; L2 ✓, L3 ✓
    # NOT IN: fails L2 only (first == last, has adjacent dup, has 'c')
    ("acca", False),    # a==a ✓; 'cc' adj dup fails L2; has 'c' ✓
    ("cbbc", False),    # c==c ✓; 'bb' adj dup fails L2; has 'c' ✓
    ("caac", False),    # c==c ✓; 'aa' adj dup fails L2; has 'c' ✓
    # NOT IN: fails L3 only (first == last, no adjacent dups, no 'c')
    ("aba", False),     # a==a ✓, L2 ✓; no 'c' fails L3. Has 'b' → rules out "contains b"
    ("babab", False),   # b==b ✓, L2 ✓; no 'c'. Length 5 (odd). "bcb" (odd) is IN → L3 ≠ "odd len"
    ("ababa", False),   # a==a ✓, L2 ✓; no 'c'. Length 5 (odd). Reinforces L3.
]

# Four test cases — each probes a distinct outcome; all strings are fresh (not in demos):
# Q1: "cabac" — c==c ✓, no adj dups ✓, has 'c' ✓          → IN  (all 3 pass)
# Q2: "bac"   — b≠c (first≠last), no adj dups ✓, has 'c' ✓ → NOT IN (fails L1 only)
# Q3: "bccb"  — b==b ✓, 'cc' adj dup fails L2, has 'c' ✓   → NOT IN (fails L2 only)
# Q4: "ababa" is a demo; use "abab"? No: a≠b fails L1.
#     Use "bab"   — b==b ✓, no adj dups ✓, no 'c'            → NOT IN (fails L3 only)
_TEST_STRINGS = ["cabac", "bac", "bccb", "bab"]
_TEST_CASES = [(s, _in_intersection(s)) for s in _TEST_STRINGS]


def _prepare():
    lines = [
        "You are observing membership tests in a hidden language over the alphabet {a, b, c}.",
        "Each string is either IN or NOT IN the language.",
        "",
        "Observations:",
    ]
    for i, (s, m) in enumerate(_DEMOS, 1):
        label = "IN" if m else "NOT IN"
        lines.append(f'  {i:2d}. "{s}" → {label}')
    lines += [
        "",
        "Now solve these 4 test questions:",
    ]
    for q, (s, _) in enumerate(_TEST_CASES, 1):
        lines.append(f'  Q{q}: "{s}" → IN or NOT IN?')
    lines += [
        "",
        "Submit as member_1, member_2, member_3, member_4 (true=IN, false=NOT IN).",
    ]
    prompt = "\n".join(lines)

    def grade_fn(response):
        results = []
        correct = 0
        for q_idx, (s, exp) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"member_{q_idx}", None)
            if isinstance(raw, bool):
                got = raw
            elif isinstance(raw, str):
                got = raw.strip().lower() in ("true", "yes", "in", "1")
            else:
                got = None
            is_correct = (got == exp) if got is not None else False
            label = "IN" if exp else "NOT IN"
            got_label = ("IN" if got else "NOT IN") if got is not None else None
            results.append(
                {
                    "q": q_idx,
                    "expected": label,
                    "got": got_label,
                    "correct": is_correct,
                }
            )
            if is_correct:
                correct += 1
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    member_1: bool
    member_2: bool
    member_3: bool
    member_4: bool


@kbench.task(
    name="regex_intersection_membership_obs_learning",
    description=(
        "Given {a,b,c}, classify if a string satisfies all: (1) first=last, (2) no double letters, (3) contains 'c'. Each failure mode shown. 4 test Qs: Q1=IN, Q2=fails-L1, Q3=fails-L2, Q4=fails-L3."
    ),
)
def regex_intersection_membership_obs_learning(llm) -> float:
    """
    3 hidden rules over {a,b,c}:
      L1: first=last,
      L2: no double letters,
      L3: must contain 'c'.
    IN if all 3 hold. 4 test Qs: Q1=IN, Q2=fails-L1, Q3=fails-L2, Q4=fails-L3.
    Returns fraction correct.
    """
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
            {
                "q": i,
                "expected": "IN" if _TEST_CASES[i - 1][1] else "NOT IN",
                "got": None,
                "correct": False,
            }
            for i in range(1, 5)
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="regex_intersection_membership_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )

    return score


if __name__ == "__main__":
    regex_intersection_membership_obs_learning.run(kbench.llm)

