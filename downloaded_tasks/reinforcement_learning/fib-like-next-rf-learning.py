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
    "Only S[0] and S[1] of a non-Fibonacci linear recurrence mod 97 with unknown coefficients α, β are shown. "
    "The model must issue QUERY k (1 ≤ k ≤ 5) commands to reveal S[k], gather enough data to infer α and β, "
    "then guess S[6] exactly. Wrong guesses receive only the parity of (guess − S[6]). "
    "Success means guessing S[6] exactly within the step budget."
)

BUDGET_N = 30
MIN_EXPLORE = 6  # hidden recurrence parameters need ≥6 sequence probes


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


MOD = 97
MAX_STEPS = 24


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    p = rng.randint(2, 6)
    q = rng.randint(1, 5)
    s0 = rng.randint(1, 20)
    s1 = rng.randint(1, 20)
    seq = [s0, s1]
    for _ in range(8):
        seq.append((p * seq[-1] + q * seq[-2]) % MOD)
    target = seq[6]
    query_count = 0
    last_fb = ""
    # Hardening: only show S[0] and S[1]; model must use QUERY to discover more terms
    intro = (
        f"A sequence satisfies **S[k] = (α·S[k−1] + β·S[k−2]) mod {MOD}** with unknown positive integers α,β (each ≤ 6).\n"
        f"You observe S[0]={seq[0]}, S[1]={seq[1]}.\n"
        "Commands:\n"
        "  `QUERY k` (1 ≤ k ≤ 5) → reveals S[k]\n"
        "  An integer guess → claims the value of S[6]\n"
        "After each wrong integer guess you only learn the **parity** of (guess − S[6]) (EVEN/ODD gap).\n"
        f"Budget {cap} turns."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"Oracle:\n{last_fb}\n\nQUERY k or guess for S[6]?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        # Check for QUERY command first
        m_q = re.search(r"\bQUERY\s+([1-5])\b", raw, re.I)
        if m_q:
            k = int(m_q.group(1))
            query_count += 1
            last_fb = f"S[{k}] = {seq[k]}."
            continue
        # Otherwise treat as an integer guess
        g = first_int(raw)
        if g is None:
            last_fb = "Send `QUERY k` (k in 1..5) or an integer guess for S[6]."
            continue
        if (g % MOD) == (target % MOD):
            progress = min(1.0, query_count / 4.0)
            return RuntimeTaskResult(
                task_id="fib_like_next",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "modular_recurrence_parity"},
                conversation=conversation,
                progress=progress,
            )
        par = (g - target) % 2
        last_fb = f"Parity channel: gap is **{'ODD' if par else 'EVEN'}** (mod 2 only)."

    progress = min(1.0, query_count / 4.0)
    return RuntimeTaskResult(
        task_id="fib_like_next",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "modular_recurrence_parity"},
        conversation=conversation,
        progress=progress,
    )


@dataclass
class _TurnOneInt:
    value: int


@kbench.task(
    name="fib_like_next_rf_learning",
    description="Non-Fibonacci linear recurrence modulo M; infer next term with parity oracle. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 30 steps.",
)
def fib_like_next_rf_learning(llm) -> float:
    """Linear recurrence mod 97 with unknown coeffs; QUERY k reveals S[k]; guess S[6] with parity error hints. Score in [0,1]; BUDGET_N=30, MIN_EXPLORE=6."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "fib_like_next_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    fib_like_next_rf_learning.run(kbench.llm)

