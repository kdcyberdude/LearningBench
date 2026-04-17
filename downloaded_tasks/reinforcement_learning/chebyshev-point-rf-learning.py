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
    "A hidden 2D landmark must be found by probing with (x, y) coordinate pairs. "
    "Feedback is a weighted L∞-style SHELL distance where the axis weights (1 or 2) are hidden and fixed. "
    "The model must infer both the weights and the landmark location through trial-and-error within the step budget."
)

BUDGET_N = 34
MIN_EXPLORE = 8  # free exploration turns; no efficiency penalty within this zone


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


HI = 14
MAX_STEPS = 28


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    sx, sy = rng.randint(0, HI), rng.randint(0, HI)
    wx = rng.choice([1, 2])
    wy = rng.choice([1, 2])
    last_fb = ""
    intro = (
        f"Hidden landmark (x,y) with 0≤x,y≤{HI}.\n"
        "Sensors report **SHELL** = max(α·|Δx|, β·|Δy|) for unknown positive integers α,β ∈ {1,2} "
        "(fixed for the episode, undisclosed).\n"
        "Infer geometry online; exact coordinate match wins.\n"
        f"Budget {cap} probes."
    )
    conversation: list = []
    best_proximity = 0.0
    max_shell = max(wx * HI, wy * HI, 1)
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nTwo integers?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        pair = first_two_ints(raw)
        if pair is None:
            last_fb = "Need two integers."
            continue
        gx, gy = pair
        if not (0 <= gx <= HI and 0 <= gy <= HI):
            last_fb = f"Stay inside 0..{HI}."
            continue
        true_shell = max(wx * abs(gx - sx), wy * abs(gy - sy))
        best_proximity = max(best_proximity, 1.0 - true_shell / max_shell)
        if (gx, gy) == (sx, sy):
            return RuntimeTaskResult(
                task_id="chebyshev_point",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "weighted_linf"},
                conversation=conversation,
                progress=best_proximity,
            )
        d = max(wx * abs(gx - sx), wy * abs(gy - sy))
        if rng.random() < 0.11:
            d = max(0, d + rng.choice([-1, 1]))
        last_fb = f"SHELL readout: **{d}**."

    return RuntimeTaskResult(
        task_id="chebyshev_point",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "weighted_linf"},
        conversation=conversation,
        progress=best_proximity,
    )


@dataclass
class _TurnTwoInts:
    first: int
    second: int


@kbench.task(
    name="chebyshev_point_rf_learning",
    description="2D point; feedback is weighted L∞-style shell with hidden axis weights. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 34 steps.",
)
def chebyshev_point_rf_learning(llm) -> float:
    """2D landmark search with hidden weighted L-infinity shell distance; composite score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnTwoInts)
            return f"{int(r.first)} {int(r.second)}"
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "chebyshev_point_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    chebyshev_point_rf_learning.run(kbench.llm)

