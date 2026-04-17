#!/usr/bin/env python
# coding: utf-8

import re
import math
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
    "A hidden integer n must be characterized: the model can issue up to 8 GCD queries "
    "to learn gcd(n, q) for chosen q, then must guess the divisor count τ(n). "
    "Learning happens by probing strategically with GCD queries to narrow down n's factorization. "
    "Success means guessing the exact number of divisors within the step budget."
)

BUDGET_N = 30
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


MAX_STEPS = 24
N_MIN, N_MAX = 28, 120
GCD_CAP = 8


def _tau(n: int) -> int:
    c = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            c += 1
            j = n // i
            if j != i:
                c += 1
        i += 1
    return c


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    n = rng.randint(N_MIN, N_MAX)
    target = _tau(n)
    gcd_used = 0
    last_fb = ""
    intro = (
        f"A hidden integer **n** with {N_MIN} ≤ n ≤ {N_MAX}.\n"
        f"You may ask up to **eight** questions `GCD q` with integer **q** in 2..{N_MAX}; "
        "the reply is **gcd(n, q)**.\n"
        "Otherwise send a single integer guess for **τ(n)** (number of positive divisors of n).\n"
        f"Total budget: {cap} messages."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nGCD q or τ guess?"
        raw = llm(user).strip()
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        gq = re.match(r"^\s*GCD\s+(\d+)\s*$", raw, re.I)
        if gq:
            q = int(gq.group(1))
            if gcd_used >= GCD_CAP:
                last_fb = "No GCD queries left; guess τ(n) with one integer."
                continue
            if q < 2 or q > N_MAX:
                last_fb = f"q must be in 2..{N_MAX}."
                continue
            gcd_used += 1
            last_fb = f"gcd(n, {q}) = **{math.gcd(n, q)}**."
            continue
        g = first_int(raw)
        if g is None:
            last_fb = "Use `GCD q` or one integer for τ(n)."
            continue
        if g == target:
            progress = min(1.0, gcd_used / 8.0)
            return RuntimeTaskResult(
                task_id="divisor_count",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"gcd_queries": gcd_used, "family": "hidden_tau"},
                conversation=conversation,
                progress=progress,
            )
        last_fb = "HIGHER." if g < target else "LOWER."

    progress = min(1.0, gcd_used / 8.0)
    return RuntimeTaskResult(
        task_id="divisor_count",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"gcd_queries": gcd_used, "family": "hidden_tau"},
        conversation=conversation,
        progress=progress,
    )


@dataclass
class _TurnOneInt:
    value: int


@kbench.task(
    name="divisor_count_rf_learning",
    description="Hidden n; oracle returns gcd(n, q); guess divisor count τ(n). Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 30 steps.",
)
def divisor_count_rf_learning(llm) -> float:
    """Hidden n; gcd(n,q) oracle queries; guess divisor count tau(n). Multi-turn RL score in [0,1]; BUDGET_N=30, MIN_EXPLORE=8."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnOneInt)
            return str(int(r.value))
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "divisor_count_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    divisor_count_rf_learning.run(kbench.llm)

