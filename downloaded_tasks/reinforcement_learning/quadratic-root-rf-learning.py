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


BUDGET_N = 32
MIN_EXPLORE = 7  # free exploration turns; no efficiency penalty within this zone

_TASK_DESCRIPTION = (
    "Piecewise linear hidden function with unique zero; probe with integer x values; "
    "commit with final integer guess."
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


def first_int(raw: str) -> Optional[int]:
    nums = [int(x) for x in re.findall(r"-?\d+", raw)]
    return nums[0] if nums else None


def first_two_ints(raw: str) -> Optional[tuple[int, int]]:
    nums = [int(x) for x in re.findall(r"-?\d+", raw)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


MAX_STEPS = 26
DOMAIN_LO, DOMAIN_HI = 5, 80

_INTRO = (
    "Hidden scalar **k** in [5,80]. A black-box returns **STRESS(x) ≈ |f(x)|** where f is piecewise-linear "
    "with a unique zero at x=k (slopes undisclosed and may differ left/right).\n"
    "Readings have ±1 noise 10% of the time.\n"
    "Each turn guess an integer x; you receive STRESS(x). Win by guessing **k** exactly.\n"
    f"Budget {MAX_STEPS} turns."
)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    k0 = rng.randint(12, 55)
    slope_left = rng.randint(2, 5)
    slope_right = rng.randint(2, 5)
    last_fb = ""
    conversation: list = []

    def f(x: int) -> int:
        if x < k0:
            return slope_left * (k0 - x)
        if x > k0:
            return slope_right * (x - k0)
        return 0

    initial_max_stress = max(f(DOMAIN_LO), f(DOMAIN_HI))
    min_true_stress = initial_max_stress

    for t in range(cap):
        user = _INTRO if t == 0 else f"{last_fb}\n\nProbe x or final answer on k?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        g = first_int(raw)
        if g is None:
            last_fb = "Send one integer."
            continue
        true_stress = f(g)
        min_true_stress = min(min_true_stress, true_stress)
        progress = (
            1.0 - min_true_stress / initial_max_stress if initial_max_stress > 0 else 0.0
        )
        if g == k0:
            return RuntimeTaskResult(
                task_id="quadratic_root",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                detail={"family": "piecewise_black_box"},
                conversation=conversation,
                progress=progress,
            )
        y = true_stress
        if rng.random() < 0.10:
            y = max(0, y + rng.choice([-1, 1]))
        last_fb = f"STRESS({g}) = **{y}**."

    progress = 1.0 - min_true_stress / initial_max_stress if initial_max_stress > 0 else 0.0
    return RuntimeTaskResult(
        task_id="quadratic_root",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        detail={"family": "piecewise_black_box"},
        conversation=conversation,
        progress=progress,
    )


@dataclass
class _TurnOneInt:
    value: int


@kbench.task(
    name="quadratic_root_rf_learning",
    description="Piecewise linear hidden function with unique zero; probe with integer x values; commit with final integer guess. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 32 steps.",
)
def quadratic_root_rf_learning(llm) -> float:
    """Piecewise-linear black box: find integer k with f(k)=0 using noisy |f(x)| probes. Multi-turn RL via llm.prompt; returns composite score in [0,1]. seed=0 fixes the instance."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnOneInt)
            return str(int(r.value))
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "quadratic_root_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    quadratic_root_rf_learning.run(kbench.llm)

