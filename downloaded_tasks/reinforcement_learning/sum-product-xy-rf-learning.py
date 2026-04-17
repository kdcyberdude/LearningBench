#!/usr/bin/env python
# coding: utf-8

import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import kaggle_benchmarks as kbench


class TurnLLM(Protocol):
    def __call__(self, user_message: str) -> str: ...


@dataclass
class RuntimeTaskResult:
    task_id: str
    solved: bool
    num_steps: int
    max_steps: int
    detail: dict[str, Any] = field(default_factory=dict)
    conversation: list = field(default_factory=list)
    progress: float = 0.0


def _composite_score(
    solved: bool,
    step_y: int,
    budget_n: int,
    min_explore: int,
    progress: float,
    *,
    floor: float = 0.10,
) -> float:
    """
    Graded RL cognitive ability score in [0, 1].
      success   (0.55) — did the model solve the task?
      efficiency (0.25) — how quickly (only when solved)?
      progress  (0.20) — how close did it get (always defined)?
    A model that never engages scores 0.0; partial progress is always rewarded.
    """
    progress = max(0.0, min(1.0, float(progress)))
    if solved:
        step_y = max(1, min(step_y, budget_n))
        if step_y <= min_explore:
            eff = 1.0
        else:
            paid_used = step_y - min_explore
            paid_budget = budget_n - min_explore
            eff = max(floor, 1.0 - (1.0 - floor) * (paid_used / paid_budget)) if paid_budget > 0 else 1.0
    else:
        eff = 0.0
    return round(0.55 * float(solved) + 0.25 * eff + 0.20 * progress, 4)


BUDGET_N = 30
MIN_EXPLORE = 6  # random linear functionals on (x,y) require ≥6 diverse probes

_TASK_DESCRIPTION = (
    "Hidden (x,y); each turn reveals a fresh random linear functional a·x+b·y "
    "(small integer coeffs, 12% additive noise on LINE values)."
)


def _log_trace(
    task: str,
    description: str,
    conversation: list,
    solved: bool,
    num_steps: int,
    budget: int,
    final_score: float,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    for entry in conversation:
        print(f"\n[USER — Turn {entry['turn']}]\n{entry['user']}")
        print(f"\n[ASSISTANT — Turn {entry['turn']}]\n{entry['response']}")
    status = "PASS ✓" if solved else "FAIL ✗"
    print(f"\n  RESULT: {status}  steps={num_steps}/{budget}  score={final_score:.4f}")
    print(f"{sep}\n")


MAX_STEPS = 24
NOISE_RATE = 0.12  # 12% probability of adding noise to LINE values

_INTRO = (
    "Hidden positive integers **x** and **y** (each roughly mid-range, undisclosed bounds).\n"
    "Each turn the environment samples small coefficients α,β ∈ {-3,…,3} (not both zero) and reports "
    "**LINE = α·x + β·y** (integer, with 12% additive noise ±[0,2]).\n"
    "Alternatively send `PAIR u v` to claim (x,y).\n"
    "Win only on a correct PAIR.\n"
    f"Budget {MAX_STEPS} turns."
)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    x = rng.randint(6, 38)
    y = rng.randint(6, 38)
    last_fb = ""
    conversation: list = []
    line_probes_used = 0

    for t in range(cap):
        user = _INTRO if t == 0 else f"{last_fb}\n\nLINE reading or PAIR?"
        raw = llm(user).upper()
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m_p = re.search(r"PAIR\s+(-?\d+)\s+(-?\d+)", raw)
        if m_p:
            u, v = int(m_p.group(1)), int(m_p.group(2))
            progress = min(1.0, line_probes_used / 3.0)
            if u == x and v == y:
                return RuntimeTaskResult(
                    task_id="sum_product_xy",
                    solved=True,
                    num_steps=t + 1,
                    max_steps=cap,
                    detail={"family": "linear_functional_probes"},
                    conversation=conversation,
                    progress=progress,
                )
            last_fb = "PAIR rejected."
            continue
        while True:
            a = rng.randint(-3, 3)
            b = rng.randint(-3, 3)
            if a != 0 or b != 0:
                break
        val = a * x + b * y
        if rng.random() < NOISE_RATE:
            val = val + rng.randint(-2, 2)
        line_probes_used += 1
        last_fb = f"LINE sample: α={a}, β={b} → **{val}**."

    progress = min(1.0, line_probes_used / 3.0)
    return RuntimeTaskResult(
        task_id="sum_product_xy",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        detail={"family": "linear_functional_probes"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="sum_product_xy_rf_learning",
    description="Hidden (x,y); each turn reveals a fresh random linear functional a·x+b·y (small integer coeffs, 12% additive noise). Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 30 steps.",
)
def sum_product_xy_rf_learning(llm) -> float:
    """Hidden integers (x,y); each turn reveals a noisy random linear functional a*x+b*y. Infer the pair from probes; returns composite RL score in [0,1]. seed=0."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "sum_product_xy_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    sum_product_xy_rf_learning.run(kbench.llm)

