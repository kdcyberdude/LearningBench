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
    "The model observes sealed-bid auction results and must deduce three hidden rules: "
    "a reserve price R below which no sale occurs, a buyer's premium B% applied to the "
    "second-highest bid, and a seller's discount S% deducted when the top bid is >= 2*R. "
    "The demos are carefully sequenced to pin all three parameters uniquely: "
    "R is pinned by a boundary pair (top=R → sale; top=R-1 → no sale), "
    "B is determined from sale prices where no discount applies, "
    "and S is determined from discount-branch prices. "
    "Parameters are R=100, B=20%, S=10%."
)

_FIXED_SEED = 0


def _winning_price(bids: list, R: int, B: int, S: int) -> int:
    """
    Rules:
    1. If max(bids) < R: no sale, price = 0
    2. Otherwise: base = second_highest_bid
    3. Apply buyer's premium: price = round(base * (1 + B/100))
    4. If max(bids) >= 2*R: apply seller's discount — price = round(price * (1 - S/100))
    """
    if not bids or max(bids) < R:
        return 0
    sorted_bids = sorted(bids, reverse=True)
    second = sorted_bids[1] if len(sorted_bids) > 1 else sorted_bids[0]
    price = round(second * (1 + B / 100))
    if sorted_bids[0] >= 2 * R:
        price = round(price * (1 - S / 100))
    return price


def _make_test_cases():
    R, B, S = 100, 20, 10
    test_bids_list = [
        [250, 220, 190, 160, 130],  # top >= 2R: both premium and discount
        [90, 75, 60, 45, 30],       # top < R: no sale
        [160, 140, 120, 100, 80],   # R <= top < 2R: premium only
        [200, 175, 150, 125, 100],  # top == 2R: boundary, discount applies
    ]
    cases = [(bids, _winning_price(bids, R, B, S)) for bids in test_bids_list]
    return R, B, S, cases


_TC_R, _TC_B, _TC_S, _TEST_CASES = _make_test_cases()


def _build_prompt(demos: list, test_bids_list: list) -> str:
    lines = [
        "You are observing a sealed-bid auction market.",
        "Each auction shows the submitted bids (sorted descending) and the resulting winning price.",
        "A price of 0 means no sale occurred.",
        "",
        "Observations (bids → winning price):",
    ]
    for i, (bids, price) in enumerate(demos, 1):
        lines.append(f"  {i:2d}. bids={sorted(bids, reverse=True)} → {price}")
    lines.append("")
    lines.append("Now solve these 4 test questions:")
    for i, (bids, _) in enumerate(test_bids_list, 1):
        lines.append(f"  Q{i}: bids={sorted(bids, reverse=True)} → ?")
    lines.append("")
    lines.append("Submit answer_1 through answer_4 as integer winning prices.")
    return "\n".join(lines)


def _prepare():
    R, B, S = _TC_R, _TC_B, _TC_S

    # Phase 1: all bids well above R=100, top < 2R=200 → only second-highest + premium visible.
    # Four distinct second-highest values give four distinct prices, pinning B=20%.
    phase1 = [
        [170, 150, 130, 110, 95],   # price = round(150*1.2) = 180
        [180, 160, 140, 120, 105],  # price = round(160*1.2) = 192
        [175, 155, 135, 115, 100],  # price = round(155*1.2) = 186
        [165, 145, 125, 105, 90],   # price = round(145*1.2) = 174
    ]

    # Phase 2: boundary pair that pins R=100 exactly.
    # top=100 → sale (price=round(85*1.2)=102); top=99 → no sale (price=0).
    # Also a clearly-below-reserve case to confirm the no-sale branch.
    phase2 = [
        [100, 85, 70, 55, 40],  # top=100=R → sale
        [99, 85, 70, 55, 40],   # top=99<R  → no sale
        [80, 70, 60, 50, 40],   # top=80<R  → no sale (reinforces reserve)
    ]

    # Phase 3: top >= 2R=200 → discount applies.
    # Three discount-branch examples with varied second-highest values pin S=10%.
    # Also interspersed premium-only examples so the discount is not trivially
    # the dominant pattern.
    phase3 = [
        [210, 185, 160, 135, 110],  # top>=2R: price=round(round(185*1.2)*0.9)=200
        [220, 195, 170, 145, 120],  # top>=2R: price=round(round(195*1.2)*0.9)=211
        [200, 180, 160, 140, 120],  # top=200=2R: price=round(round(180*1.2)*0.9)=194
        [150, 130, 110, 90, 75],    # top<2R: price=round(130*1.2)=156
        [230, 205, 180, 155, 130],  # top>=2R: price=round(round(205*1.2)*0.9)=221
        [140, 120, 100, 80, 65],    # top<2R: price=round(120*1.2)=144
        [125, 110, 95, 80, 65],     # top<2R: price=round(110*1.2)=132
    ]

    demos = []
    for bids in phase1 + phase2 + phase3:
        price = _winning_price(bids, R, B, S)
        demos.append((bids, price))

    prompt = _build_prompt(demos, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (bids, expected) in enumerate(_TEST_CASES, 1):
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
    name="auction_mechanism_second_price_obs_learning",
    description=(
        "Observe 14 sealed-bid auction examples, infer reserve price R, buyer premium B%, and seller discount S%. Then predict winning prices for 4 new cases, covering all rule branches."
    ),
)
def auction_mechanism_second_price_obs_learning(llm) -> float:
    """Infer hidden reserve price, buyer's premium, and threshold discount from 14 auction examples; predict 4 prices."""
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
        task="auction_mechanism_second_price_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    auction_mechanism_second_price_obs_learning.run(kbench.llm)

