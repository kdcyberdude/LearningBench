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
MIN_EXPLORE = 8  # free exploration turns; no efficiency penalty within this zone

_TASK_DESCRIPTION = "Hidden factors (a,b); feedback is L1 distance in factor space, product never stated."


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


LO, HI = 6, 17
MAX_STEPS = 26
MAX_TAXI = (HI - LO) + (HI - LO)  # = 22

_INTRO = (
    "Two hidden integers **α** and **β** each lie in a fixed closed interval (undisclosed bounds, both in low teens).\n"
    "Each turn submit **two** integers `u v` interpreted as a candidate pair.\n"
    "Telemetry: **TAXI** = |u−α| + |v−β| (no directional split per coordinate).\n"
    "You are **not** graded on the product α·β; only an exact (α,β) match wins.\n"
    f"Budget {MAX_STEPS} pairs."
)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    alpha = rng.randint(LO, HI)
    beta = rng.randint(LO, HI)
    last_fb = ""
    conversation: list = []
    best_proximity = 0.0

    for t in range(cap):
        user = _INTRO if t == 0 else f"Readout:\n{last_fb}\n\nNext pair `u v`?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        nums = [int(x) for x in re.findall(r"-?\d+", raw)]
        if len(nums) < 2:
            last_fb = "Need two integers."
            continue
        u, v = nums[0], nums[1]
        true_taxi = abs(u - alpha) + abs(v - beta)
        best_proximity = max(best_proximity, 1.0 - true_taxi / MAX_TAXI)
        if u == alpha and v == beta:
            return RuntimeTaskResult(
                task_id="product_hotcold",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                detail={"family": "latent_factor_l1"},
                conversation=conversation,
                progress=best_proximity,
            )
        last_fb = f"TAXI distance: **{true_taxi}**."

    return RuntimeTaskResult(
        task_id="product_hotcold",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        detail={"family": "latent_factor_l1"},
        conversation=conversation,
        progress=best_proximity,
    )


@kbench.task(
    name="product_hotcold_rf_learning",
    description="Hidden factors (a,b); feedback is L1 distance in factor space, product never stated. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 32 steps.",
)
def product_hotcold_rf_learning(llm) -> float:
    """Infer hidden factors (a,b) from L1 distance in factor space (product never stated). Multi-turn episode via llm.prompt; returns composite score in [0,1] (success/efficiency/progress). seed=0 fixes the instance."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "product_hotcold_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    product_hotcold_rf_learning.run(kbench.llm)

