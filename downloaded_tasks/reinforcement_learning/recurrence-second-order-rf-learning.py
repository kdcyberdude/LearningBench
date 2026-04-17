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
MIN_EXPLORE = 6  # free exploration turns; no efficiency penalty within this zone

_TASK_DESCRIPTION = (
    "Hidden order-2 linear recurrence modulo M; query terms or claim coefficients. "
    "Noisy oracle (15% single-step flip)."
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


MOD = 256
MAX_STEPS = 26
NOISE_RATE = 0.15  # 15% probability of ±1 flip on AT oracle responses

_INTRO = (
    f"Sequence defined by **U[k] = (α·U[k−1] + β·U[k−2]) mod {MOD}** with unknown positive α,β.\n"
    "Commands:\n"
    "  `AT k` with 0≤k≤20 → returns U[k] (noisy oracle: 15% chance of ±1 single-step flip)\n"
    "  `COEFF p q` → claim α=p, β=q\n"
    "Win only via correct COEFF.\n"
    f"Budget {MAX_STEPS}."
)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    a = rng.randint(1, 7)
    b = rng.randint(1, 7)
    u0 = rng.randint(0, 40)
    u1 = rng.randint(0, 40)
    seq = [u0, u1]
    for _ in range(32):
        seq.append((a * seq[-1] + b * seq[-2]) % MOD)
    last_fb = ""
    intro = (
        f"Sequence defined by **U[k] = (α·U[k−1] + β·U[k−2]) mod {MOD}** with unknown positive α,β.\n"
        f"Seeded with U[0]={u0}, U[1]={u1}.\n"
        "Commands:\n"
        "  `AT k` with 0≤k≤20 → returns U[k] (noisy oracle: 15% chance of ±1 single-step flip)\n"
        "  `COEFF p q` → claim α=p, β=q\n"
        "Win only via correct COEFF.\n"
        f"Budget {cap}."
    )
    conversation: list = []
    at_probes_made = 0

    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nAT or COEFF?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        raw = raw.upper()
        m_c = re.search(r"COEFF\s+(\d+)\s+(\d+)", raw)
        if m_c:
            p, q = int(m_c.group(1)), int(m_c.group(2))
            progress = min(1.0, at_probes_made / 4.0)
            if p == a and q == b:
                return RuntimeTaskResult(
                    task_id="recurrence_second_order",
                    solved=True,
                    num_steps=t + 1,
                    max_steps=cap,
                    detail={"family": "mod_recurrence_ident"},
                    conversation=conversation,
                    progress=progress,
                )
            last_fb = "COEFF rejected."
            continue
        m_a = re.search(r"AT\s+(\d+)", raw)
        if not m_a:
            last_fb = "Use `AT k` or `COEFF p q`."
            continue
        k = int(m_a.group(1))
        if not (0 <= k <= 20):
            last_fb = "k must be 0..20."
            continue
        at_probes_made += 1
        val = seq[k]
        if rng.random() < NOISE_RATE:
            val = max(0, min(MOD - 1, val + rng.choice([-1, 1])))
        last_fb = f"U[{k}] = **{val}**."

    progress = min(1.0, at_probes_made / 4.0)
    return RuntimeTaskResult(
        task_id="recurrence_second_order",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        detail={"family": "mod_recurrence_ident"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="recurrence_second_order_rf_learning",
    description="Hidden order-2 linear recurrence modulo M; query terms or claim coefficients. Noisy oracle (15% single-step flip). Multi-turn RL; return float in [0,1], cap 32 steps.",
)
def recurrence_second_order_rf_learning(llm) -> float:
    """Hidden order-2 linear recurrence modulo M; query terms or claim coefficients with a 15% noisy oracle. Multi-turn RL; composite score in [0,1]. seed=0."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "recurrence_second_order_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    recurrence_second_order_rf_learning.run(kbench.llm)

