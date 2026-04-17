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


BUDGET_N = 28
MIN_EXPLORE = 6  # free exploration turns; no efficiency penalty within this zone

_TASK_DESCRIPTION = (
    "Four registers are shown; the environment XORs a hidden non-empty subset into a 10-bit target S. "
    "Each integer guess is masked to 10 bits; feedback is Hamming distance to S (BIT_DIVERGENCE), "
    "sometimes dithered so the search is noisy. The model must discover S through trial and error. "
    "Success is an exact masked match within the step budget."
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


MASK_BITS = 10
MAX_STEPS = 22

_INTRO = (
    "Four source registers (values revealed per instance).\n"
    "The environment XOR-combines a **hidden non-empty subset** of them into a 10-bit **bundle** S.\n"
    "Each turn propose a non-negative integer G (masked to 10 bits). Telemetry: **BIT_DIVERGENCE** "
    "(count of differing bits between bundles) — occasionally dithered.\n"
    f"Win when your masked G equals S within {MAX_STEPS} guesses."
)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    nums = [rng.randint(15, 220) for _ in range(4)]
    if len(set(nums)) < 4:
        nums = [17, 41, 89, 133]
    subset_mask = rng.randint(1, 15)
    secret_xor = 0
    for i, v in enumerate(nums):
        if subset_mask & (1 << i):
            secret_xor ^= v
    s = secret_xor & ((1 << MASK_BITS) - 1)
    best_bit_accuracy = 0.0
    last_fb = ""
    pool = ", ".join(str(x) for x in nums)
    intro = (
        f"Four source registers: {pool}.\n"
        "The environment XOR-combines a **hidden non-empty subset** of them into a 10-bit **bundle** S.\n"
        "Each turn propose a non-negative integer G (masked to 10 bits). Telemetry: **BIT_DIVERGENCE** "
        "(count of differing bits between bundles) — occasionally dithered.\n"
        f"Win when your masked G equals S within {cap} guesses."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nInteger guess G?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m = re.search(r"\b(\d+)\b", raw)
        if not m:
            last_fb = "Send one non-negative integer."
            continue
        g = int(m.group(1)) & ((1 << MASK_BITS) - 1)
        if g == s:
            return RuntimeTaskResult(
                task_id="xor_subset_hamming",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                detail={"family": "xor_search"},
                conversation=conversation,
                progress=1.0,
            )
        true_h = bin(g ^ s).count("1")
        best_bit_accuracy = max(best_bit_accuracy, 1.0 - true_h / MASK_BITS)
        h = true_h
        if rng.random() < 0.09:
            h = max(0, min(MASK_BITS, h + rng.choice([-1, 1])))
        last_fb = f"BIT_DIVERGENCE: **{h}**."

    return RuntimeTaskResult(
        task_id="xor_subset_hamming",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        detail={"family": "xor_search"},
        conversation=conversation,
        progress=best_bit_accuracy,
    )


@dataclass
class _TurnOneInt:
    value: int


@kbench.task(
    name="xor_subset_hamming_rf_learning",
    description="XOR of a hidden subset of four integers; Hamming distance on 10-bit pattern. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 28 steps.",
)
def xor_subset_hamming_rf_learning(llm) -> float:
    """Registers XOR a hidden subset into a 10-bit target; guesses get Hamming distance (BIT_DIVERGENCE), sometimes dithered. Returns composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnOneInt)
            return str(int(r.value))
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "xor_subset_hamming_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    xor_subset_hamming_rf_learning.run(kbench.llm)

