#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can infer the hidden CFL a^n b^(n+k) c^k from accept/reject "
    "examples. Early demos use k=0 (just a^n b^n), making it look like a simple equal-count "
    "language. Later demos reveal the generalized rule: #b = #a + #c. Success requires "
    "classifying 4 test strings that span different combinations of n, b-count, and c-count."
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


def _accepts(s: str) -> bool:
    """Accept iff s = a^n b^(n+k) c^k, n≥1, k≥0 (equivalently: #b = #a + #c, all a's before b's before c's)."""
    if not s:
        return False
    i = 0
    n = 0
    while i < len(s) and s[i] == "a":
        n += 1
        i += 1
    if n == 0:
        return False
    m = 0
    while i < len(s) and s[i] == "b":
        m += 1
        i += 1
    k = 0
    while i < len(s) and s[i] == "c":
        k += 1
        i += 1
    if i != len(s):
        return False  # unexpected characters or wrong order
    return m == n + k


# --- 13 DEMO STRINGS ---
# First 4: k=0 demos (a^n b^n) — looks like simple equal-count language
# Next 9: k>0 demos reveal the generalized rule #b = #a + #c

_DEMO_STRINGS = [
    # k=0 demos (ACCEPT)
    "ab",  # n=1, b=1, c=0: 1 = 1+0 ✓
    "aabb",  # n=2, b=2, c=0: 2 = 2+0 ✓
    "aaabbb",  # n=3, b=3, c=0: 3 = 3+0 ✓
    "aaaabbbb",  # n=4, b=4, c=0: 4 = 4+0 ✓
    # k=0 reject (wrong b count)
    "aab",  # n=2, b=1, c=0: 1 ≠ 2+0 → REJECT
    "abbb",  # n=1, b=3, c=0: 3 ≠ 1+0 → REJECT
    # k>0 accepts (reveal the generalization)
    "abbc",  # n=1, b=2, c=1: 2 = 1+1 ✓
    "aabbbc",  # n=2, b=3, c=1: 3 = 2+1 ✓
    "abbcc",  # n=1, b=3, c=2: 3 = 1+2 ✓
    "aabbbcc",  # n=2, b=4, c=2: 4 = 2+2 ✓
    "aaabbbc",  # n=3, b=4, c=1: 4 = 3+1 ✓
    "aaabbbbbcc",  # n=3, b=5, c=2: 5 = 3+2 ✓
    # k>0 reject (off by one — tricky)
    "abbbc",  # n=1, b=3, c=1: 3 ≠ 1+1=2 → REJECT
]

_DEMO_DATA = [(s, _accepts(s)) for s in _DEMO_STRINGS]

# 4 Test cases — varied n/k combinations, probing the full constraint
_TEST_STRINGS = [
    "aaabbbbbcc",  # n=3, b=5, c=2: 5=3+2 → ACCEPT
    "aaabbbbcc",  # n=3, b=4, c=2: 4≠3+2=5 → REJECT
    "aabbbbccc",  # n=2, b=4, c=3: 4≠2+3=5 → REJECT  wait: 4≠5 → REJECT
    "aabbbbc",  # n=2, b=4, c=1: 4≠2+1=3 → REJECT  hmm too many rejects
]
# Let me rebalance: 2 accept, 2 reject
_TEST_STRINGS = [
    "aaabbbbbcc",  # n=3, b=5, c=2: 5=3+2 → ACCEPT
    "aabbbbccc",  # n=2, b=5, c=3: 5=2+3 ✓ → ACCEPT
    "aaabbbbcc",  # n=3, b=4, c=2: 4≠5 → REJECT
    "aabbbbbcc",  # n=2, b=5, c=2: 5≠4 → REJECT
]

_TEST_CASES = [(s, _accepts(s)) for s in _TEST_STRINGS]


def _prepare():
    lines = [
        "You are observing a language recognizer over the alphabet {a, b, c}.",
        "Each string is either accepted or rejected by a hidden rule.",
        "",
        "Observations:",
    ]
    for i, (s, acc) in enumerate(_DEMO_DATA, 1):
        label = "ACCEPT" if acc else "REJECT"
        lines.append(f'  {i:2d}. "{s}" → {label}')
    lines += [
        "",
        "Now solve these 4 test questions:",
    ]
    for q, (s, _) in enumerate(_TEST_CASES, 1):
        lines.append(f'  Q{q}: "{s}" → ACCEPT or REJECT?')
    lines += [
        "",
        "Submit as accepted_1, accepted_2, accepted_3, accepted_4 (true=ACCEPT, false=REJECT).",
    ]
    prompt = "\n".join(lines)

    def grade_fn(response):
        results = []
        correct = 0
        for q_idx, (s, exp) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"accepted_{q_idx}", None)
            if isinstance(raw, bool):
                got = raw
            elif isinstance(raw, str):
                got = raw.strip().lower() in ("true", "yes", "accept", "1")
            else:
                got = None
            is_correct = (got == exp) if got is not None else False
            label = "ACCEPT" if exp else "REJECT"
            got_label = ("ACCEPT" if got else "REJECT") if got is not None else None
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
    accepted_1: bool
    accepted_2: bool
    accepted_3: bool
    accepted_4: bool


@kbench.task(
    name="pushdown_automaton_inference_obs_learning",
    description="Observe accept/reject examples for a hidden language over {a,b,c}. Early demos look like a^n b^n; later ones reveal the generalised rule a^n b^(n+k) c^k (i.e. #b = #a + #c). Classify 4 test strings probing different n/k.",
)
def pushdown_automaton_inference_obs_learning(llm) -> float:
    """
    Hidden language: a^n b^(n+k) c^k  where n≥1, k≥0.
    Equivalently: strings of form a...ab...bc...c where #b = #a + #c.

    Deception: first 6 demos use k=0 so rule looks like a^n b^n (equal counts).
    Later demos with k>0 reveal that b-count = a-count + c-count.

    Four test strings: 2 accepted (n=3,k=2 and n=2,k=3) and 2 rejected (off by 1 each).

    Returns fraction of 4 test questions answered correctly (0.0 to 1.0).
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
                "expected": "ACCEPT" if _TEST_CASES[i - 1][1] else "REJECT",
                "got": None,
                "correct": False,
            }
            for i in range(1, 5)
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="pushdown_automaton_inference_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )

    return score


if __name__ == "__main__":
    pushdown_automaton_inference_obs_learning.run(kbench.llm)

