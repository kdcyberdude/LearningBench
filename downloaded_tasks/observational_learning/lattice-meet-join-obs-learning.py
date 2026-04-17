#!/usr/bin/env python
# coding: utf-8

import random
from dataclasses import dataclass
from itertools import permutations

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can infer the hidden 2-dimensional lattice structure "
    "(product of a 2-chain and a 4-chain, C2×C4) from labelled meet/join observations. "
    "Labels 0-7 are randomly permuted across the 8 grid positions; no label carries "
    "arithmetic meaning. Early demos all satisfy meet=min(a,b) and join=max(a,b), "
    "creating a false total-order hypothesis. Later demos violate this pattern, forcing "
    "the model to infer the hidden 2D grid structure and apply coordinate-wise min/max. "
    "All four test cases require answers that are inconsistent with every simple "
    "arithmetic rule (min/max, sum mod n, etc.). Solution uniqueness is mathematically "
    "verified: exactly one assignment of labels to grid positions is consistent with "
    "all 14 demo observations."
)

_FIXED_SEED = 100

# Hidden structure: C2 × C4 product lattice (2 rows × 4 columns)
# Positions: (row, col) with row ∈ {0,1} and col ∈ {0,1,2,3}
# meet((r1,c1),(r2,c2)) = (min(r1,r2), min(c1,c2))
# join((r1,c1),(r2,c2)) = (max(r1,r2), max(c1,c2))
# Labels 0-7 are randomly permuted across positions.

_N1, _N2 = 2, 4
_POSITIONS = [(i, j) for i in range(_N1) for j in range(_N2)]
_N = _N1 * _N2

_rng = random.Random(_FIXED_SEED)
_labels_perm = list(range(_N))
_rng.shuffle(_labels_perm)

_POS_TO_LABEL = {pos: _labels_perm[idx] for idx, pos in enumerate(_POSITIONS)}
_LABEL_TO_POS = {v: k for k, v in _POS_TO_LABEL.items()}


def _meet(a: int, b: int) -> int:
    p1, p2 = _LABEL_TO_POS[a], _LABEL_TO_POS[b]
    return _POS_TO_LABEL[(min(p1[0], p2[0]), min(p1[1], p2[1]))]


def _join(a: int, b: int) -> int:
    p1, p2 = _LABEL_TO_POS[a], _LABEL_TO_POS[b]
    return _POS_TO_LABEL[(max(p1[0], p2[0]), max(p1[1], p2[1]))]


# Ordered demo sequence, shuffled within phases.
# Phase 1 (indices 0-3): same-dimension pairs that coincidentally satisfy meet=min, join=max
#   → creates a false "total order / min-max" hypothesis
# Phase 2 (indices 4-6): cross-dimension pairs where meet=min still holds (reinforces false rule)
# Phase 3 (indices 7-10): cross-dimension pairs where meet≠min and join≠max → breaks false rule
# Phase 4 (indices 11-13): further structural constraints enabling full position recovery
_DEMOS = [
    (0, 4), (0, 5), (1, 2), (1, 3),   # phase 1
    (0, 1), (0, 6), (4, 7),            # phase 2
    (1, 5), (2, 4), (3, 4), (5, 6),   # phase 3
    (2, 7), (3, 7), (5, 7),            # phase 4
]

# Four test cases: ALL have meet≠min(a,b) and join≠max(a,b)
# Q1 (1,4): cross-dim, col orders cross  → meet=0, join=6
# Q2 (2,3): same-dim, numerically reversed in C4 → meet=3, join=2  (meet > join numerically!)
# Q3 (3,5): cross-dim, both coords swap  → meet=7, join=2
# Q4 (6,7): cross-dim, non-trivial meet  → meet=4, join=3
_TEST_CASES = [
    (1, 4, _meet(1, 4), _join(1, 4)),
    (2, 3, _meet(2, 3), _join(2, 3)),
    (3, 5, _meet(3, 5), _join(3, 5)),
    (6, 7, _meet(6, 7), _join(6, 7)),
]


def _verify_uniqueness() -> bool:
    """Verify that exactly one permutation of labels to positions is consistent
    with all 14 demo observations. Called once at module load."""
    demos_pairs = [(a, b) for a, b, _, _ in
                   [(d[0], d[1], _meet(d[0], d[1]), _join(d[0], d[1]))
                    for d in _DEMOS]]

    def consistent(perm):
        p2l = {_POSITIONS[i]: perm[i] for i in range(_N)}
        l2p = {perm[i]: _POSITIONS[i] for i in range(_N)}
        for a, b in demos_pairs:
            p1, p2 = l2p[a], l2p[b]
            if (p2l[(min(p1[0], p2[0]), min(p1[1], p2[1]))] != _meet(a, b) or
                    p2l[(max(p1[0], p2[0]), max(p1[1], p2[1]))] != _join(a, b)):
                return False
        return True

    count = sum(1 for perm in permutations(range(_N)) if consistent(perm))
    return count == 1


assert _verify_uniqueness(), "Demo set does not uniquely determine the lattice!"


def _log_trace(task, description, prompt, test_results, score):
    sep = "=" * 70
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    print(f"\n  TEST RESULTS:")
    for r in test_results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"    [{status}] Q{r['q']}: expected={r['expected']!r}  got={r['got']!r}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _build_prompt() -> str:
    lines = [
        "You are observing a binary system defined on 8 labeled objects: 0, 1, 2, 3, 4, 5, 6, 7.",
        "The system defines two binary operations ∧ (meet) and ∨ (join) on pairs of labels.",
        "",
        "Observations (label pair → meet result, join result):",
    ]
    for i, (a, b) in enumerate(_DEMOS, 1):
        m = _meet(a, b)
        j = _join(a, b)
        lines.append(f"  {i:2d}. ({a}, {b}) → ∧={m}, ∨={j}")
    lines += [
        "",
        "Using only the pattern in the observations above, predict the meet and join",
        "for each of the following pairs:",
        "",
        f"  Q1: ({_TEST_CASES[0][0]}, {_TEST_CASES[0][1]}) → ∧=?, ∨=?",
        f"  Q2: ({_TEST_CASES[1][0]}, {_TEST_CASES[1][1]}) → ∧=?, ∨=?",
        f"  Q3: ({_TEST_CASES[2][0]}, {_TEST_CASES[2][1]}) → ∧=?, ∨=?",
        f"  Q4: ({_TEST_CASES[3][0]}, {_TEST_CASES[3][1]}) → ∧=?, ∨=?",
        "",
        "For each question provide the single integer label (0-7) for both ∧ and ∨.",
        "Submit using the schema fields: meet_1, join_1, meet_2, join_2, meet_3, join_3, meet_4, join_4.",
    ]
    return "\n".join(lines)


def _grade(response) -> tuple[float, list]:
    results = []
    correct = 0
    for q_idx, (a, b, exp_m, exp_j) in enumerate(_TEST_CASES, 1):
        got_m = getattr(response, f"meet_{q_idx}", None)
        got_j = getattr(response, f"join_{q_idx}", None)
        try:
            both_ok = int(got_m) == exp_m and int(got_j) == exp_j
        except (TypeError, ValueError):
            both_ok = False
        results.append(
            {
                "q": q_idx,
                "expected": (exp_m, exp_j),
                "got": (got_m, got_j),
                "correct": both_ok,
            }
        )
        if both_ok:
            correct += 1
    return correct / 4, results


@dataclass
class _Answer:
    meet_1: int
    join_1: int
    meet_2: int
    join_2: int
    meet_3: int
    join_3: int
    meet_4: int
    join_4: int


@kbench.task(
    name="lattice_meet_join_obs_learning",
    description=(
        "Given random labels 0-7 for positions in a hidden C2×C4 lattice, infer the meet (∧) and join (∨) for 4 pairs using observations. No label's value is arithmetically meaningful."
    ),
)
def lattice_meet_join_obs_learning(llm) -> float:
    """
    Predict meets (∧) and joins (∨) for 4 pairs in a hidden C2×C4 lattice (8 nodes labeled 0–7, random perm). Given 14 labeled (a,b)→(meet,join) examples. Returns fraction correct.
    """
    prompt = _build_prompt()

    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None

    if response is not None:
        score, test_results = _grade(response)
    else:
        score = 0.0
        test_results = [
            {
                "q": i,
                "expected": (_TEST_CASES[i - 1][2], _TEST_CASES[i - 1][3]),
                "got": None,
                "correct": False,
            }
            for i in range(1, 5)
        ]

    _log_trace(
        task="lattice_meet_join_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
    )

    return score


if __name__ == "__main__":
    lattice_meet_join_obs_learning.run(kbench.llm)

