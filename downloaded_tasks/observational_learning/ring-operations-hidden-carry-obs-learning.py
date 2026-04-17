#!/usr/bin/env python
# coding: utf-8

import random
from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can infer a ring with hidden addition carry from operation examples. "
    "The ring uses a+ᵣb=(a+b+K) mod M where K is hidden. Early addition demos use small values "
    "where a+b+K < M so the carry is invisible and results look like standard addition. "
    "Later demos use large values causing the carry to fire, revealing K. Success requires "
    "computing 4 compound expressions (a+ᵣb)×ᵣc using the inferred operations."
)

_FIXED_SEED = 0
_PRIMES = [11, 13, 17, 19, 23]


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


def _add_r(a: int, b: int, K: int, M: int) -> int:
    return (a + b + K) % M


def _mul_r(a: int, b: int, M: int) -> int:
    return (a * b) % M


def _build_prompt(add_demos: list, mul_demos: list, test_triples: list, M: int) -> str:
    lines = [
        f"You are observing a RING on the set {{0, 1, ..., {M - 1}}}.",
        "The ring has two hidden operations: +ᵣ (addition) and ×ᵣ (multiplication).",
        "",
        "Observations for +ᵣ (a +ᵣ b = c):",
    ]
    for x, y, z in add_demos:
        lines.append(f"  {x} +ᵣ {y} = {z}")
    lines.append("")
    lines.append("Observations for ×ᵣ (a ×ᵣ b = c):")
    for x, y, z in mul_demos:
        lines.append(f"  {x} ×ᵣ {y} = {z}")
    lines += [
        "",
        "For each test below, compute the expression using the EXACT hidden operations.",
        "Submit each result as an integer in {0, ..., " + str(M - 1) + "}.",
        "",
    ]
    for i, (a, b, c) in enumerate(test_triples, 1):
        lines.append(f"  Test {i}: ({a} +ᵣ {b}) ×ᵣ {c} = ?")
    lines += ["", "Submit answers as answer_1 through answer_4."]
    return "\n".join(lines)


def _prepare():
    rng = random.Random(_FIXED_SEED)

    M = rng.choice(_PRIMES)
    K = rng.randint(2, M // 3)

    # Build add demos: first 3 deceptive (no carry), then enough to reveal K
    add_demos: list = []
    used: set = set()

    # Deceptive: small a, b so a+b+K < M (carry invisible)
    for _ in range(300):
        if len(add_demos) >= 3:
            break
        a = rng.randint(0, M // 5)
        b = rng.randint(0, M // 5)
        if (a, b) not in used and (a + b + K) < M:
            used.add((a, b))
            add_demos.append((a, b, _add_r(a, b, K, M)))

    # Revealing: large a, b so carry fires, revealing K
    for _ in range(500):
        if len(add_demos) >= 10:
            break
        a = rng.randint(M // 2, M - 1)
        b = rng.randint(M // 2, M - 1)
        if (a, b) not in used and (a + b + K) >= M:
            used.add((a, b))
            add_demos.append((a, b, _add_r(a, b, K, M)))

    # Fill remaining with any
    for _ in range(300):
        if len(add_demos) >= 12:
            break
        a, b = rng.randint(0, M - 1), rng.randint(0, M - 1)
        if (a, b) not in used:
            used.add((a, b))
            add_demos.append((a, b, _add_r(a, b, K, M)))

    # Mul demos: standard mod-M multiplication
    mul_demos: list = []
    used_m: set = set()
    while len(mul_demos) < 8:
        a, b = rng.randint(1, M - 1), rng.randint(1, M - 1)
        if (a, b) not in used_m:
            used_m.add((a, b))
            mul_demos.append((a, b, _mul_r(a, b, M)))

    # Build 4 test triples – all require carry to fire (a+b+K >= M)
    test_triples = []
    test_gts = []
    used_t: set = set()
    for _ in range(4):
        for _attempt in range(500):
            ta = rng.randint(M // 3, M - 1)
            tb = rng.randint(M // 3, M - 1)
            tc = rng.randint(2, M - 1)
            if (ta, tb, tc) not in used_t and (ta + tb + K) >= M:
                used_t.add((ta, tb, tc))
                sum_ab = _add_r(ta, tb, K, M)
                gt = _mul_r(sum_ab, tc, M)
                test_triples.append((ta, tb, tc))
                test_gts.append(gt)
                break
        else:
            # fallback
            ta, tb, tc = M - 1, M - 1, 2
            sum_ab = _add_r(ta, tb, K, M)
            test_triples.append((ta, tb, tc))
            test_gts.append(_mul_r(sum_ab, tc, M))

    prompt = _build_prompt(add_demos, mul_demos, test_triples, M)

    def grade_fn(response):
        test_results = []
        correct = 0
        for i in range(4):
            field = f"answer_{i + 1}"
            raw = getattr(response, field, None)
            expected = str(test_gts[i])
            try:
                got_val = int(raw)
                is_correct = got_val == test_gts[i]
            except (TypeError, ValueError):
                is_correct = False
            got = str(raw)
            if is_correct:
                correct += 1
            test_results.append(
                {"q": i + 1, "expected": expected, "got": got, "correct": is_correct}
            )
        return correct / 4, test_results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: int
    answer_2: int
    answer_3: int
    answer_4: int


@kbench.task(
    name="ring_operations_hidden_carry_obs_learning",
    description=(
        "Ring over Z_M: a+ᵣb=(a+b+K) mod M, a×ᵣb=(a*b) mod M. Demos first hide, then reveal carry K. Answer 4 (a+ᵣb)×ᵣc cases."
    ),
)
def ring_operations_hidden_carry_obs_learning(llm) -> float:
    """
    Ring on Z_M (M prime):
      a +ᵣ b = (a + b + K) mod M  (hidden carry K)
      a ×ᵣ b = (a × b) mod M

    First 3 addition demos: no carry (looks standard). Later: carry reveals K.
    12 add demos, 8 mul demos, 4 test cases (all require carry).
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
            {"q": i + 1, "expected": "?", "got": None, "correct": False}
            for i in range(4)
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        "ring_operations_hidden_carry_obs_learning",
        _TASK_DESCRIPTION,
        prompt,
        test_results,
        score,
        str(reasoning),
    )
    return score


if __name__ == "__main__":
    ring_operations_hidden_carry_obs_learning.run(kbench.llm)

