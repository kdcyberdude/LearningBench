#!/usr/bin/env python
# coding: utf-8

import random
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
    "The model observes plaintext→ciphertext pairs from a 2-round Feistel cipher with a hidden "
    "round function F(x) = (a·x + b) mod P where P is a hidden prime. Sufficient pairs allow "
    "recovering a, b, P. Success requires decrypting 4 test ciphertexts by reversing both "
    "Feistel rounds to recover the original (L, R) plaintext pairs."
)

_FIXED_SEED = 0

_PRIMES = [17, 19, 23, 29, 31, 37, 41, 43]

# Fixed cipher parameters — derived from seed, never change
_P = 29
_A = 7
_B = 11


def _feistel_encrypt(L: int, R: int, a: int, b: int, P: int) -> tuple:
    def F(x: int) -> int:
        return (a * x + b) % P

    L1 = R
    R1 = (L ^ F(R)) % P
    L2 = R1
    R2 = (L1 ^ F(R1)) % P
    return L2, R2


def _feistel_decrypt(C0: int, C1: int, a: int, b: int, P: int) -> tuple:
    def F(x: int) -> int:
        return (a * x + b) % P

    L2, R2 = C0, C1
    R1 = L2
    L1 = (R2 ^ F(L2)) % P
    R = L1
    L = (R1 ^ F(L1)) % P
    return L, R


def _make_test_cases():
    P, a, b = _P, _A, _B
    rng = random.Random(_FIXED_SEED + 99)
    used: set = set()
    cases = []
    while len(cases) < 4:
        gt_L = rng.randint(0, P - 1)
        gt_R = rng.randint(0, P - 1)
        if (gt_L, gt_R) in used:
            continue
        used.add((gt_L, gt_R))
        ct = _feistel_encrypt(gt_L, gt_R, a, b, P)
        cases.append((ct, (gt_L, gt_R)))
    return cases


_TEST_CASES = _make_test_cases()


def _build_prompt(demos: list, test_cases: list, P: int) -> str:
    lines = [
        "You are observing a FEISTEL CIPHER encrypting pairs of integers.",
        f"Each plaintext and ciphertext is a pair of integers in [0, {P - 1}].",
        "The cipher structure and round function are NOT described — deduce them from the examples.",
        "",
        "Observations (plaintext L,R → ciphertext C0,C1):",
    ]
    for i, ((pl, pr), (cl, cr)) in enumerate(demos, 1):
        lines.append(f"  {i}. ({pl}, {pr}) → ({cl}, {cr})")
    lines += [
        "",
        "Decrypt each of the following ciphertexts to recover the original (L, R):",
    ]
    for i, ((c0, c1), _) in enumerate(test_cases, 1):
        lines.append(f"  Q{i}: ciphertext = ({c0}, {c1})")
    lines += [
        "",
        "Submit your answers as left_1/right_1, left_2/right_2, left_3/right_3, left_4/right_4.",
    ]
    return "\n".join(lines)


def _prepare():
    P, a, b = _P, _A, _B
    rng = random.Random(_FIXED_SEED)

    used_plains: set = set()
    # Reserve test case plaintexts
    for _, (gt_L, gt_R) in _TEST_CASES:
        used_plains.add((gt_L, gt_R))

    demos: list = []
    while len(demos) < 14:
        L = rng.randint(0, P - 1)
        R = rng.randint(0, P - 1)
        if (L, R) in used_plains:
            continue
        used_plains.add((L, R))
        C0, C1 = _feistel_encrypt(L, R, a, b, P)
        demos.append(((L, R), (C0, C1)))

    prompt = _build_prompt(demos, _TEST_CASES, P)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (ct, (gt_L, gt_R)) in enumerate(_TEST_CASES, 1):
            got_L = getattr(response, f"left_{i}", None)
            got_R = getattr(response, f"right_{i}", None)
            ok = False
            try:
                ok = int(got_L) == gt_L and int(got_R) == gt_R
            except (TypeError, ValueError):
                ok = False
            if ok:
                correct += 1
            expected = (gt_L, gt_R)
            got = (got_L, got_R)
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    left_1: int
    right_1: int
    left_2: int
    right_2: int
    left_3: int
    right_3: int
    left_4: int
    right_4: int


@kbench.task(
    name="feistel_cipher_round_obs_learning",
    description=(
        "Observe 14 plaintext→ciphertext pairs encrypted by a 2-round Feistel cipher with hidden "
        "round function F(x)=(ax+b) mod P. Infer the structure and coefficients; "
        "submit decrypted plaintexts (L, R) for 4 new ciphertexts."
    ),
)
def feistel_cipher_round_obs_learning(llm) -> float:
    """Infer a 2-round Feistel cipher's hidden F(x)=(ax+b) mod P from 14 pairs; decrypt 4 ciphertexts."""
    prompt, grade_fn = _prepare()
    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None
    score, test_results = (
        grade_fn(response)
        if response is not None
        else (
            0.0,
            [
                {
                    "q": i,
                    "expected": _TEST_CASES[i - 1][1],
                    "got": None,
                    "correct": False,
                }
                for i in range(1, 5)
            ],
        )
    )
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="feistel_cipher_round_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    feistel_cipher_round_obs_learning.run(kbench.llm)

