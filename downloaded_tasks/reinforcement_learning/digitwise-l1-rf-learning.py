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
    "A secret 4-digit lock code must be cracked; each guess returns PRESSURE — "
    "a weighted L1 distance where the per-position weights (1–4) are fixed but hidden. "
    "The model must infer both the weights and the secret code through adaptive guessing. "
    "Success means submitting the exact 4-digit code within the step budget."
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


MAX_STEPS = 22


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    secret = "".join(str(rng.randint(0, 9)) for _ in range(4))
    w = [rng.randint(1, 4) for _ in range(4)]
    # Worst-case pressure: each digit off by 9
    max_pressure = sum(w[i] * 9 for i in range(4))
    min_pressure_seen = float('inf')
    last_fb = ""
    intro = (
        "Secret lock: four decimal digits (0–9), leading zeros allowed.\n"
        "Each guess returns **PRESSURE** = a weighted sum of absolute digit errors.\n"
        "Position weights are **fixed but undisclosed** integers in {1,2,3,4} — learn them online.\n"
        f"Exact match opens the lock. Budget {cap} tries."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"Gauge:\n{last_fb}\n\nNext 4-digit guess?"
        raw_response = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw_response})
        raw = re.sub(r"\D", "", raw_response)
        if len(raw) < 4:
            last_fb = "Need four digits."
            continue
        g = raw[:4]
        if g == secret:
            progress = 1.0
            return RuntimeTaskResult(
                task_id="digitwise_l1",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "weighted_digit_metric"},
                conversation=conversation,
                progress=progress,
            )
        s = sum(w[i] * abs(int(g[i]) - int(secret[i])) for i in range(4))
        min_pressure_seen = min(min_pressure_seen, s)
        last_fb = f"PRESSURE reading: **{s}**."

    if min_pressure_seen != float('inf') and max_pressure > 0:
        progress = max(0.0, 1.0 - min_pressure_seen / max_pressure)
    else:
        progress = 0.0
    return RuntimeTaskResult(
        task_id="digitwise_l1",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "weighted_digit_metric"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="digitwise_l1_rf_learning",
    description="Four-digit code; feedback is a **hidden-weighted** digit error sum (weights unknown). Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 28 steps.",
)
def digitwise_l1_rf_learning(llm) -> float:
    """4-digit code; PRESSURE is weighted L1 with hidden per-position weights in 1..4. Multi-turn RL score in [0,1]; BUDGET_N=28, MIN_EXPLORE=6."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "digitwise_l1_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    digitwise_l1_rf_learning.run(kbench.llm)

