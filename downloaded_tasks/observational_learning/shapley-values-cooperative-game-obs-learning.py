#!/usr/bin/env python
# coding: utf-8

import math
import random
from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can infer a novel, unpublished allocation rule for "
    "3-player cooperative games purely from input-output observations. The rule "
    "is NOT the Shapley value, Nash bargaining, or any other standard formula — "
    "it is a bespoke 'inverse-solo-weighted' allocation that the model must "
    "reconstruct from scratch by observing game→allocation pairs. The rule has "
    "a clear and unique interpretation that a careful human reasoner can extract, "
    "but it is unknown to any model's training data. Difficulty is PhD-level "
    "inference under deliberate anti-contamination design."
)

_FIXED_SEED = 42


# ---------------------------------------------------------------------------
# The hidden allocation rule (never exposed to the evaluated model):
#
#   For each pair (i,j) with pairwise surplus X_ij = v({i,j}) - v({i}) - v({j}):
#       player i receives  X_ij * a_j / (a_i + a_j)
#       player j receives  X_ij * a_i / (a_i + a_j)
#   where a_k = v({k}) is player k's solo value.
#   (Weaker player gets proportionally more of each pairwise surplus.)
#
#   The grand-coalition surplus W = v({1,2,3}) - v({1,2}) - v({1,3}) - v({2,3})
#                                   + v({1}) + v({2}) + v({3})
#   is split equally: W / 3 to each player.
#
#   Final allocation:
#       ψ_1 = a_1 + X_12 · a_2/(a_1+a_2) + X_13 · a_3/(a_1+a_3) + W/3
#       ψ_2 = a_2 + X_12 · a_1/(a_1+a_2) + X_23 · a_3/(a_2+a_3) + W/3
#       ψ_3 = a_3 + X_13 · a_1/(a_1+a_3) + X_23 · a_2/(a_2+a_3) + W/3
#
#   This rule is efficient: ψ_1 + ψ_2 + ψ_3 = v({1,2,3}).
# ---------------------------------------------------------------------------


def _alloc(a1: float, a2: float, a3: float,
           X12: float, X13: float, X23: float, W: float) -> tuple:
    psi1 = a1 + X12 * a2 / (a1 + a2) + X13 * a3 / (a1 + a3) + W / 3
    psi2 = a2 + X12 * a1 / (a1 + a2) + X23 * a3 / (a2 + a3) + W / 3
    psi3 = a3 + X13 * a1 / (a1 + a3) + X23 * a2 / (a2 + a3) + W / 3
    return psi1, psi2, psi3


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


def _fmt_game(a1, a2, a3, X12, X13, X23, W) -> str:
    v12 = a1 + a2 + X12
    v13 = a1 + a3 + X13
    v23 = a2 + a3 + X23
    v123 = a1 + a2 + a3 + X12 + X13 + X23 + W
    return (
        f"v({{1}})={a1}, v({{2}})={a2}, v({{3}})={a3}, "
        f"v({{1,2}})={v12}, v({{1,3}})={v13}, v({{2,3}})={v23}, "
        f"v({{1,2,3}})={v123}"
    )


def _fmt_alloc(psi1, psi2, psi3) -> str:
    return f"ψ1={psi1:.4f}, ψ2={psi2:.4f}, ψ3={psi3:.4f}"


def _build_prompt(demos: list, test_games: list) -> str:
    lines = [
        "You are studying a cooperative game involving three players {1, 2, 3}.",
        "Each game is described by a characteristic function v(S) giving the worth of each coalition.",
        "An allocation rule assigns each player a real-valued payoff (ψ1, ψ2, ψ3).",
        "The allocation always sums to v({1,2,3}) (efficiency).",
        "",
        "Study the observations below carefully. Infer the allocation rule precisely.",
        "",
        "Observations (input game → output allocation):",
        "",
    ]

    # Phase 1: single pair bonus (label is opaque to model)
    for i, (params, psi) in enumerate(demos, 1):
        a1, a2, a3, X12, X13, X23, W = params
        psi1, psi2, psi3 = psi
        lines.append(f"  [{i}]  {_fmt_game(a1, a2, a3, X12, X13, X23, W)}")
        lines.append(f"        → {_fmt_alloc(psi1, psi2, psi3)}")
        lines.append("")

    lines += [
        "Apply the rule you inferred to the following test games.",
        "For each, compute the allocation (ψ1, ψ2, ψ3).",
        "",
        "Test games:",
        "",
    ]
    for i, params in enumerate(test_games, 1):
        a1, a2, a3, X12, X13, X23, W = params
        lines.append(f"  Test {i}: {_fmt_game(a1, a2, a3, X12, X13, X23, W)}")

    lines += [
        "",
        "For each test game submit the answer as 'X.XXXX,Y.YYYY,Z.ZZZZ' "
        "representing ψ1,ψ2,ψ3 rounded to 4 decimal places.",
        "Format: answer_1 through answer_4.",
    ]
    return "\n".join(lines)


def _parse_answer(s) -> tuple:
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return None
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) >= 3:
        try:
            return float(nums[0]), float(nums[1]), float(nums[2])
        except ValueError:
            return None
    return None


def _prepare():
    # ---------------------------------------------------------------------------
    # Curated demo games — progressively disclose the rule structure:
    #
    # Phase 1 (demos 1-3): exactly ONE nonzero pair bonus, W=0.
    #   The model can directly read off the pair-bonus split ratio.
    #   Each demo uses a different active pair so all three pairings are covered.
    #
    # Phase 2 (demos 4-6): exactly TWO nonzero pair bonuses, W=0.
    #   The model verifies the additive structure across two pairs.
    #
    # Phase 3 (demos 7-10): ALL bonuses active including W > 0.
    #   The model must infer how W is handled (equal thirds).
    #
    # Anti-contamination: all solo values are distinct (no ties), psi values are
    # all distinct per game, and no pair surplus equals (a_i + a_j) which would
    # cause the 'equal-split' illusion (psi_i = psi_j despite unequal solos).
    # ---------------------------------------------------------------------------
    # (a1, a2, a3, X12, X13, X23, W)
    demo_params = [
        # Phase 1 — single pair bonus
        (1, 4, 6,  6, 0, 0, 0),   # X12 only; split 4.8 to p1, 1.2 to p2 (ratio a2/sum = 4/5)
        (5, 2, 7,  0, 8, 0, 0),   # X13 only; split to p1 and p3 by 7/12 and 5/12
        (4, 7, 1,  0, 0, 6, 0),   # X23 only; split to p2 and p3 by 1/8 and 7/8
        # Phase 2 — two pair bonuses
        (2, 5, 3,  5, 4, 0, 0),   # X12 + X13
        (4, 1, 6,  3, 0, 5, 0),   # X12 + X23
        (3, 6, 2,  0, 4, 7, 0),   # X13 + X23
        # Phase 3 — all bonuses + W
        (2, 5, 3,  5, 4, 3, 3),
        (4, 1, 6,  3, 5, 4, 6),
        (3, 7, 2,  4, 2, 6, 3),
        (1, 5, 4,  6, 3, 4, 6),
    ]

    demos = []
    for params in demo_params:
        a1, a2, a3, X12, X13, X23, W = params
        psi = _alloc(a1, a2, a3, X12, X13, X23, W)
        demos.append((params, psi))

    # ---------------------------------------------------------------------------
    # Test games: all bonuses active, strongly asymmetric solo values,
    # all three psi values distinct, outputs not confusable with equal-split rule.
    # Generated deterministically from FIXED_SEED.
    # ---------------------------------------------------------------------------
    rng = random.Random(_FIXED_SEED)
    test_params_list = []
    test_psi_list = []
    seen: set = set()

    while len(test_params_list) < 4:
        a1 = rng.randint(1, 8)
        a2 = rng.randint(1, 8)
        a3 = rng.randint(1, 8)
        if len({a1, a2, a3}) < 3:
            continue
        X12 = rng.randint(2, 8)
        X13 = rng.randint(2, 8)
        X23 = rng.randint(2, 8)
        W = rng.randint(1, 6)

        # Avoid X = a_i + a_j for any active pair (prevents equal-split illusion)
        if X12 == a1 + a2 or X13 == a1 + a3 or X23 == a2 + a3:
            continue

        psi1, psi2, psi3 = _alloc(a1, a2, a3, X12, X13, X23, W)
        rounded = tuple(round(p, 4) for p in [psi1, psi2, psi3])

        # All three allocations must be distinct
        if len(set(rounded)) < 3:
            continue

        key = (a1, a2, a3, X12, X13, X23, W)
        if key in seen:
            continue
        seen.add(key)

        test_params_list.append(key)
        test_psi_list.append((psi1, psi2, psi3))

    prompt = _build_prompt(demos, test_params_list)

    def grade_fn(response):
        test_results = []
        correct = 0
        for i in range(4):
            field = f"answer_{i + 1}"
            raw = getattr(response, field, None)
            gt1, gt2, gt3 = test_psi_list[i]
            expected = f"{gt1:.4f},{gt2:.4f},{gt3:.4f}"
            parsed = _parse_answer(raw)
            if parsed is not None:
                p1, p2, p3 = parsed
                is_correct = (
                    abs(p1 - gt1) < 0.05
                    and abs(p2 - gt2) < 0.05
                    and abs(p3 - gt3) < 0.05
                )
            else:
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
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str


@kbench.task(
    name="shapley_values_cooperative_game_obs_learning",
    description=(
        "Infer a novel 3-player cooperative game allocation rule (not standard, e.g. not Shapley) from observed examples and apply it to 4 new games. Must use only observed data; accuracy ±0.05 per player."
    ),
)
def shapley_values_cooperative_game_obs_learning(llm) -> float:
    """
    Hidden rule: Pair (i,j) surplus X_ij: player i gets X_ij * a_j/(a_i+a_j). Grand surplus split equally. Returns fraction of 4 test games solved (±0.05 tolerance).
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
        "shapley_values_cooperative_game_obs_learning",
        _TASK_DESCRIPTION,
        prompt,
        test_results,
        score,
        str(reasoning),
    )
    return score


if __name__ == "__main__":
    shapley_values_cooperative_game_obs_learning.run(kbench.llm)

