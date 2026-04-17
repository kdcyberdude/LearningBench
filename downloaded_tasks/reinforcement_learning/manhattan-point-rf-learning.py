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
    conversation: list = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
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


BUDGET_N = 34
MIN_EXPLORE = 8  # free exploration turns; no efficiency penalty within this zone

_TASK_DESCRIPTION = "Hidden 2D point; feedback is a noisy tier of a non-Euclidean score (L1 + weighted Linf)."


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


HIGH = 12
MAX_STEPS = 28

_INTRO = (
    f"A hidden landmark has integer coordinates (x, y) with 0 ≤ x,y ≤ {HIGH}.\n"
    "Each probe returns only a **coarse telemetry tier** of a non-linear distance functional "
    "(details are proprietary to the environment — infer its geometry online).\n"
    "Tiers are noisy: with 12% probability the reported tier may be off by one level.\n"
    "Send two integers per turn. Win by an exact coordinate match.\n"
    f"Budget: {MAX_STEPS} probes."
)


def _score(gx: int, gy: int, sx: int, sy: int, w: float) -> float:
    l1 = abs(gx - sx) + abs(gy - sy)
    linf = max(abs(gx - sx), abs(gy - sy))
    return l1 + w * linf


def _tier(x: float) -> str:
    if x <= 1.5:
        return "TIER_0 — co-located"
    if x <= 4.5:
        return "TIER_1 — adjacent shell"
    if x <= 9.0:
        return "TIER_2 — mid field"
    if x <= 16.0:
        return "TIER_3 — far field"
    return "TIER_4 — extreme separation"


def _tier_value(tier_str: str) -> float:
    """Map tier string to progress value (higher = closer to goal)."""
    if "TIER_0" in tier_str:
        return 1.0
    if "TIER_1" in tier_str:
        return 0.75
    if "TIER_2" in tier_str:
        return 0.50
    if "TIER_3" in tier_str:
        return 0.25
    return 0.10  # TIER_4


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    sx = rng.randint(0, HIGH)
    sy = rng.randint(0, HIGH)
    w = rng.choice([0.35, 0.5, 0.65, 0.8])  # hidden from agent
    last_fb = ""
    conversation: list = []
    best_tier_progress = 0.0
    for t in range(cap):
        user = (
            _INTRO
            if t == 0
            else f"Last telemetry:\n{last_fb}\n\nNext probe (two integers)?"
        )
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        pair = first_two_ints(raw)
        if pair is None:
            last_fb = "Parse failure — need two integers."
            continue
        gx, gy = pair
        if not (0 <= gx <= HIGH and 0 <= gy <= HIGH):
            last_fb = f"Coordinates must lie in 0..{HIGH}."
            continue
        if (gx, gy) == (sx, sy):
            best_tier_progress = 1.0
            return RuntimeTaskResult(
                task_id="manhattan_point",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                conversation=conversation,
                detail={"secret": (sx, sy), "family": "noisy_metric_shells"},
                progress=best_tier_progress,
            )
        s = _score(gx, gy, sx, sy, w)
        tier = _tier(s)
        if rng.random() < 0.12:
            tier = (
                "TIER_1 — adjacent shell"
                if "TIER_0" in tier
                else "TIER_2 — mid field"
                if "TIER_1" in tier
                else "TIER_3 — far field"
                if "TIER_2" in tier
                else tier
            )
        best_tier_progress = max(best_tier_progress, _tier_value(tier))
        last_fb = f"Telemetry readout: **{tier}**."

    return RuntimeTaskResult(
        task_id="manhattan_point",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        conversation=conversation,
        detail={"secret": (sx, sy), "family": "noisy_metric_shells"},
        progress=best_tier_progress,
    )


@dataclass
class _TurnTwoInts:
    first: int
    second: int


@kbench.task(
    name="manhattan_point_rf_learning",
    description="Hidden 2D point; feedback is a **noisy tier** of a non-Euclidean score (L1 + weighted Linf). Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 34 steps.",
)
def manhattan_point_rf_learning(llm) -> float:
    """Learn hidden 2D point from noisy L1+Linf-tier feedback; returns composite [0,1] for up to 34 turns."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnTwoInts)
            return f"{int(r.first)} {int(r.second)}"
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "manhattan_point_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    manhattan_point_rf_learning.run(kbench.llm)

