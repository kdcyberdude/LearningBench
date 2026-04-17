#!/usr/bin/env python
# coding: utf-8

import math
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
    "The model observes (input, output) point pairs produced by a hidden chain of 3 affine "
    "transforms (rotation at multiples of 30°, uniform scale 1.5 or 2.0, integer translation). "
    "Early demos use the origin and axis-aligned points where rotation is hard to disentangle "
    "from scale. Success requires inferring all three transform parameters and predicting the "
    "output for novel off-axis test points."
)

_FIXED_SEED = 0


def _apply_affine(x: float, y: float, theta_deg: float, s: float, tx: int, ty: int) -> tuple:
    theta = math.radians(theta_deg)
    x2 = s * (x * math.cos(theta) - y * math.sin(theta)) + tx
    y2 = s * (x * math.sin(theta) + y * math.cos(theta)) + ty
    return (round(x2), round(y2))


def _apply_chain(x: float, y: float, transforms: list) -> tuple:
    for theta, s, tx, ty in transforms:
        xi, yi = _apply_affine(x, y, theta, s, tx, ty)
        x, y = float(xi), float(yi)
    return (round(x), round(y))


def _make_test_cases():
    rng = random.Random(_FIXED_SEED + 99)
    transforms = []
    for _ in range(3):
        theta = float(rng.choice([30, 60, 90, 120, 150]))
        s = rng.choice([1.5, 2.0])
        tx = rng.randint(-8, 8)
        ty = rng.randint(-8, 8)
        transforms.append((theta, s, tx, ty))

    used = set()
    cases = []
    candidate_pts = [
        (7, -3), (-5, 6), (4, 9), (-8, -4), (6, 5),
        (3, -7), (-6, 2), (9, 1), (-4, -9), (8, -6),
    ]
    for pt in candidate_pts:
        if pt not in used and len(cases) < 4:
            used.add(pt)
            out = _apply_chain(pt[0], pt[1], transforms)
            cases.append((pt, out))
    return transforms, cases


_TC_TRANSFORMS, _TEST_CASES = _make_test_cases()


def _build_prompt(demos: list, test_pts: list) -> str:
    lines = [
        "You are observing a sequence of (input, output) pairs where a hidden 2D transform",
        "chain maps each input point to an output point.",
        "",
        "Observations (input → output):",
    ]
    for i, ((xi, yi), (xo, yo)) in enumerate(demos, 1):
        lines.append(f"  {i:2d}. ({xi:4d}, {yi:4d}) → ({xo:4d}, {yo:4d})")
    lines.append("")
    lines.append("Now solve these 4 test questions:")
    for i, (pt, _) in enumerate(test_pts, 1):
        lines.append(f"  Q{i}: Input ({pt[0]}, {pt[1]}) → ?")
    lines.append("")
    lines.append("Submit answer_1 through answer_4 as 'x,y' strings (e.g. '12,-7').")
    return "\n".join(lines)


def _prepare():
    rng = random.Random(_FIXED_SEED)
    transforms = _TC_TRANSFORMS

    # Build 14 demos using a separate point set
    demos = []
    anchor_pts = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    used = set(anchor_pts) | {tc[0] for tc in _TEST_CASES}
    for pt in anchor_pts:
        out = _apply_chain(pt[0], pt[1], transforms)
        demos.append((pt, out))

    extra_pts = [
        (2, 1), (-2, 3), (3, -2), (-3, -1), (4, 2),
        (-4, 3), (5, -1), (-5, 4), (2, -4),
    ]
    for pt in extra_pts:
        if pt not in used and len(demos) < 14:
            used.add(pt)
            out = _apply_chain(pt[0], pt[1], transforms)
            demos.append((pt, out))

    while len(demos) < 14:
        pt = (rng.randint(-6, 6), rng.randint(-6, 6))
        if pt not in used:
            used.add(pt)
            out = _apply_chain(pt[0], pt[1], transforms)
            demos.append((pt, out))

    prompt = _build_prompt(demos, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (inp, expected) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"answer_{i}", None)
            got = None
            ok = False
            if isinstance(raw, str):
                try:
                    parts = [p.strip() for p in raw.replace("(", "").replace(")", "").split(",")]
                    got = (int(parts[0]), int(parts[1]))
                    ok = (got == expected)
                except Exception:
                    pass
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str


@kbench.task(
    name="affine_transform_chain_obs_learning",
    description=(
        "Observe (input→output) point pairs for a hidden chain of 3 affine transforms "
        "(rotation at multiples of 30°, scale 1.5 or 2.0, integer translation). "
        "14 demos reveal all transform parameters. Predict outputs for 4 novel test points."
    ),
)
def affine_transform_chain_obs_learning(llm) -> float:
    """Infer a 3-step affine transform chain from 14 I/O point pairs; predict outputs for 4 test points."""
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

    reasoning = getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    _log_trace(
        task="affine_transform_chain_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    affine_transform_chain_obs_learning.run(kbench.llm)

