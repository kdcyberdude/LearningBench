#!/usr/bin/env python
# coding: utf-8

import re
import random
import kaggle_benchmarks as kbench
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class TurnLLM(Protocol):
    def __call__(self, user_message: str) -> str: ...


@dataclass
class RuntimeTaskResult:
    task_id: str
    solved: bool
    num_steps: int
    max_steps: int
    intro: str = ""
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


_TASK_DESCRIPTION = (
    "A modified Collatz-like map with a hidden odd perturbation δ: n→3n+δ for odd n; n→n//2 for even n. "
    "The model is shown the starting value and must find the number of steps to reach 1 "
    "using HIGHER/LOWER feedback on guesses, without knowing δ. "
    "Success means naming the exact step count within the budget."
)

BUDGET_N = 28
MIN_EXPLORE = 6  # free exploration turns; no efficiency penalty within this zone


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


def first_int(raw: str) -> Optional[int]:
    nums = [int(x) for x in re.findall(r"-?\d+", raw)]
    return nums[0] if nums else None


MAX_STEPS = 22


def _chain_length(n: int, delta: int) -> int:
    """Count steps to reach 1 under n→3n+delta (odd) / n→n//2 (even). Cap at 500."""
    c = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + delta
        c += 1
        if c > 500:
            return 500
    return c


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    delta = rng.choice([1, 3, 5, 7])
    n0 = rng.randint(14, 48)
    target = _chain_length(n0, delta)
    best_proximity = 0.0
    last_fb = ""
    intro = (
        "A modified Collatz-like map on positive integers:\n"
        "  • if n is even: n ← n // 2\n"
        "  • if n is odd:  n ← 3n + δ   (δ is odd but hidden)\n"
        "Repeat until you first reach **1**. Each application counts as one **step**.\n"
        f"Starting from **n = {n0}**, how many steps until termination?\n"
        "Wrong integers receive HIGHER/LOWER relative to the true count.\n"
        f"Budget {cap}."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nStep count?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        g = first_int(raw)
        if g is None:
            last_fb = "Send one integer."
            continue
        proximity = max(0.0, 1.0 - abs(g - target) / max(target, 1))
        best_proximity = max(best_proximity, proximity)
        if g == target:
            return RuntimeTaskResult(
                task_id="collatz_length",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"n0": n0, "delta": delta, "family": "synthetic_dynamics"},
                conversation=conversation,
                progress=1.0,
            )
        last_fb = "HIGHER." if g < target else "LOWER."

    return RuntimeTaskResult(
        task_id="collatz_length",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"n0": n0, "delta": delta, "family": "synthetic_dynamics"},
        conversation=conversation,
        progress=best_proximity,
    )


@dataclass
class _TurnOneInt:
    value: int


@kbench.task(
    name="collatz_length_rf_learning",
    description="Modified Collatz map with hidden odd perturbation δ (δ∈{1,3,5,7}); count steps to reach 1. Multi-turn RL: model only sees HIGHER/LOWER feedback; return float in [0,1], cap 28 steps.",
)
def collatz_length_rf_learning(llm) -> float:
    """Modified Collatz step count with hidden odd delta; HIGHER/LOWER search; composite [0,1]."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnOneInt)
            return str(int(r.value))
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "collatz_length_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    collatz_length_rf_learning.run(kbench.llm)

