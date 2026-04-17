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
    "A hidden 12-bit integer must be found by guessing integers and receiving MISMATCH feedback — "
    "the Hamming distance after an undisclosed REFLECT (bitwise XOR-mix) transform, with 9% dither. "
    "The model must reason about a hidden bit transformation while searching the integer space. "
    "Success means guessing the exact secret integer within the step budget."
)

BUDGET_N = 30
MIN_EXPLORE = 8  # REFLECT-mixed Hamming channel needs extra free probes


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


BITS = 12
MAX_STEPS = 24


def _reflect(x: int) -> int:
    """Undisclosed to the agent: REFLECT mixing (x XOR (x>>1))."""
    return x ^ (x >> 1)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    secret = rng.randint(0, (1 << BITS) - 1)
    mask = (1 << BITS) - 1
    r_sec = _reflect(secret) & mask
    last_fb = ""
    best_bit_accuracy = 0.0
    intro = (
        f"Hidden integer S in [0, 2^{BITS}).\n"
        f"Environment applies a fixed **REFLECT** mixing map to the {BITS}-bit pattern (details proprietary).\n"
        "Each guess T returns **MISMATCH** = Hamming distance between REFLECT(T) and REFLECT(S).\n"
        "Readings dither ±1 with 9% probability.\n"
        f"Win when T=S. Budget {cap}."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nT?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m = re.search(r"\b(\d+)\b", raw)
        if not m:
            last_fb = "Send one integer."
            continue
        tt = int(m.group(1)) & mask
        # Track true bit accuracy (un-noised) toward secret
        actual_h = bin(tt ^ secret).count("1")
        best_bit_accuracy = max(best_bit_accuracy, 1.0 - actual_h / BITS)
        if tt == secret:
            return RuntimeTaskResult(
                task_id="gray_hamming",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "reflect_hamming"},
                conversation=conversation,
                progress=best_bit_accuracy,
            )
        h = bin((_reflect(tt) & mask) ^ r_sec).count("1")
        if rng.random() < 0.09:
            h = max(0, min(BITS, h + rng.choice([-1, 1])))
        last_fb = f"MISMATCH = **{h}**."

    return RuntimeTaskResult(
        task_id="gray_hamming",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "reflect_hamming"},
        conversation=conversation,
        progress=best_bit_accuracy,
    )


@dataclass
class _TurnOneInt:
    value: int


@kbench.task(
    name="gray_hamming_rf_learning",
    description="Search in integer space; feedback is Hamming distance after unknown REFLECT bitmasking. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 30 steps.",
)
def gray_hamming_rf_learning(llm) -> float:
    """12-bit secret; MISMATCH is Hamming distance after hidden REFLECT XOR plus dither. Score in [0,1]; BUDGET_N=30, MIN_EXPLORE=8."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnOneInt)
            return str(int(r.value))
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "gray_hamming_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    gray_hamming_rf_learning.run(kbench.llm)

