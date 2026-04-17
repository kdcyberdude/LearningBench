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
    "A hidden integer cubic polynomial P(x) with small coefficients must be identified. "
    "The model can issue VAL x queries to evaluate P at chosen points, then must answer ANS y = P(7). "
    "Warning: one oracle query (among the first four) may be corrupted — detect and handle outliers. "
    "Learning happens by probing strategically to reconstruct the polynomial and compute the target value."
)

BUDGET_N = 28
MIN_EXPLORE = 6  # cubic black-box curvature needs ≥6 exploratory samples


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


def _eval(c: list[int], x: int) -> int:
    return c[0] + c[1] * x + c[2] * x * x + c[3] * x * x * x


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    d, c2, b, a = [rng.randint(-3, 3) for _ in range(4)]
    if a == 0:
        a = 1
    coeff = [d, c2, b, a]
    target = _eval(coeff, 7)
    # Corruption: one of the first 4 probes is off by a hidden offset
    corrupt_probe_idx = rng.randint(0, 3)
    corrupt_offset = rng.choice([-3, -2, -1, 1, 2, 3])
    val_count = 0  # counts VAL queries issued
    distinct_probes: set[int] = set()
    last_fb = ""
    intro = (
        "Hidden degree-≤3 integer polynomial **P**.\n"
        "Commands:\n"
        "  `VAL x` → returns P(x)  (one of the first four responses may be corrupted — detect outliers)\n"
        "  `ANS y` → claim P(7)=y\n"
        "Win only via correct ANS.\n"
        f"Budget {cap} commands."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nVAL or ANS?"
        raw = llm(user).upper()
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m_a = re.search(r"ANS\s+(-?\d+)", raw)
        if m_a:
            y = int(m_a.group(1))
            if y == target:
                progress = min(1.0, len(distinct_probes) / 4.0)
                return RuntimeTaskResult(
                    task_id="cubic_eval",
                    solved=True,
                    num_steps=t + 1,
                    max_steps=cap,
                    intro=intro,
                    detail={"family": "cubic_black_box"},
                    conversation=conversation,
                    progress=progress,
                )
            last_fb = "ANS rejected."
            continue
        m_v = re.search(r"VAL\s+(-?\d+)", raw)
        if not m_v:
            last_fb = "Use `VAL x` or `ANS y`."
            continue
        x = int(m_v.group(1))
        if abs(x) > 25:
            last_fb = "VAL domain [-25,25] for this episode."
            continue
        distinct_probes.add(x)
        true_val = _eval(coeff, x)
        # Apply corruption to the designated probe among first 4
        if val_count < 4 and val_count == corrupt_probe_idx:
            reported_val = true_val + corrupt_offset
        else:
            reported_val = true_val
        val_count += 1
        last_fb = f"P({x}) = **{reported_val}**."

    progress = min(1.0, len(distinct_probes) / 4.0)
    return RuntimeTaskResult(
        task_id="cubic_eval",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "cubic_black_box"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="cubic_eval_rf_learning",
    description="Integer cubic black-box: probe x → P(x); infer P(7). Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 28 steps.",
)
def cubic_eval_rf_learning(llm) -> float:
    """Integer cubic P(x): VAL probes evaluate P; answer ANS P(7); one early oracle may lie. Multi-turn RL score in [0,1]; BUDGET_N=28, MIN_EXPLORE=6."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "cubic_eval_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    cubic_eval_rf_learning.run(kbench.llm)

