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
    "A hidden integer in [0, 210) must be identified using at most three modular remainder queries "
    "with primes 7, 11, and 13, followed by a GUESS command. "
    "The oracle is noisy (15% flip probability): each returned remainder may be off by ±1 mod p. "
    "The model must combine the residue information (effectively Chinese Remainder Theorem) "
    "without the theorem being named, then commit to the exact answer."
)

BUDGET_N = 26
MIN_EXPLORE = 6  # CRT remainder lattice needs ≥6 calibration probes


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


HI = 210
PRIMES = (7, 11, 13)
MAX_STEPS = 20
NOISE_RATE = 0.15


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    secret = rng.randint(0, HI - 1)
    rem_uses = 0
    remainders_collected = 0
    last_fb = ""
    intro = (
        f"A hidden integer **Z** satisfies 0 ≤ Z < {HI}.\n"
        "You may issue at most **three** commands `REM p` where p ∈ {7,11,13} and learn **Z mod p**.\n"
        f"Warning: the oracle is noisy — each remainder has a {int(NOISE_RATE*100)}% chance of being off by ±1 (mod p).\n"
        "Alternatively send `GUESS z` for an integer z.\n"
        "Win only with a correct GUESS. (No theorem names — combine evidence yourself.)\n"
        f"Budget {cap} commands."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nREM or GUESS?"
        raw = llm(user).upper()
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m_g = re.search(r"GUESS\s+(-?\d+)", raw)
        if m_g:
            z = int(m_g.group(1))
            if z == secret:
                progress = remainders_collected / 3.0
                return RuntimeTaskResult(
                    task_id="crt_unique",
                    solved=True,
                    num_steps=t + 1,
                    max_steps=cap,
                    intro=intro,
                    detail={"family": "modular_probes"},
                    conversation=conversation,
                    progress=progress,
                )
            last_fb = "GUESS incorrect."
            continue
        m_r = re.search(r"REM\s+(7|11|13)\b", raw)
        if not m_r:
            last_fb = "Use `REM p` with p∈{7,11,13} or `GUESS z`."
            continue
        if rem_uses >= 3:
            last_fb = "No REM budget left."
            continue
        p = int(m_r.group(1))
        rem_uses += 1
        remainders_collected += 1
        actual_rem = secret % p
        if rng.random() < NOISE_RATE:
            offset = rng.choice([-1, 1])
            actual_rem = (actual_rem + offset) % p
        last_fb = f"Z mod {p} = **{actual_rem}**."

    progress = remainders_collected / 3.0
    return RuntimeTaskResult(
        task_id="crt_unique",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "modular_probes"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="crt_unique_rf_learning",
    description="Hidden integer; limited modular glimpses (no CRT naming); then exact guess. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 26 steps.",
)
def crt_unique_rf_learning(llm) -> float:
    """Noisy modular remainder queries then exact integer guess; composite score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "crt_unique_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    crt_unique_rf_learning.run(kbench.llm)

