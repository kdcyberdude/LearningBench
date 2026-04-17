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
    "A hidden integer is fixed and the model must find it by probing with integer guesses. "
    "Each probe returns BIT_OVERLAP — the popcount of the XOR between the guess and the secret — "
    "a noisy bitwise distance oracle rather than a simple numeric comparison. "
    "Success means naming the exact secret integer within the step budget."
)

BUDGET_N = 34
MIN_EXPLORE = 7  # free exploration turns; no efficiency penalty within this zone


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


MAX_STEPS = 28


def popcount(x: int) -> int:
    return x.bit_count()


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    secret = rng.randint(50, 9_000)
    last_fb = ""
    intro = (
        "A hidden 14-bit-class integer is fixed for this episode.\n"
        "Each probe is an integer guess g. Telemetry returns **BIT_OVERLAP** = popcount(g XOR secret) "
        "(number of differing bits in binary — leading zeros count).\n"
        "This is *not* a simple arithmetic sequence puzzle; infer the scalar via bitwise search.\n"
        f"Win by naming the exact secret. Budget {cap} probes."
    )
    conversation: list = []
    best_hamming_progress = 0.0
    for t in range(cap):
        user = intro if t == 0 else f"Channel:\n{last_fb}\n\nInteger probe?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        g = first_int(raw)
        if g is None:
            last_fb = "Send one integer."
            continue
        if g < 0:
            last_fb = "Non-negative integers only."
            continue
        actual_hamming = bin(g ^ secret).count('1')
        best_hamming_progress = max(best_hamming_progress, 1.0 - actual_hamming / 14.0)
        if g == secret:
            return RuntimeTaskResult(
                task_id="arithmetic_next",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "xor_popcount_search"},
                conversation=conversation,
                progress=best_hamming_progress,
            )
        ov = popcount(g ^ secret)
        if rng.random() < 0.07:
            ov = max(0, ov + rng.choice([-1, 1]))
        last_fb = f"BIT_OVERLAP = **{ov}** (noisy sensor)."

    return RuntimeTaskResult(
        task_id="arithmetic_next",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "xor_popcount_search"},
        conversation=conversation,
        progress=best_hamming_progress,
    )


@dataclass
class _TurnOneInt:
    value: int


@kbench.task(
    name="arithmetic_next_rf_learning",
    description="Hidden integer; feedback is **bit overlap** (popcount of XOR) — not a classic sequence. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 34 steps.",
)
def arithmetic_next_rf_learning(llm) -> float:
    """
    Goal: Hidden integer; feedback is **bit overlap** (popcount of XOR) — not a classic sequence.

    What runs: `run` implements the episode. Each turn the environment sends a user message;
    the model responds via `llm.prompt` (wrapped as `turn` below). Success means satisfying
    the hidden criterion using only that feedback—there is no shortcut label in the prompt.

    Per-turn replies may use `llm.prompt(..., schema=...)` (Kaggle Benchmarks cookbook structured
    output) when one clear commitment fits a schema; otherwise plain text is used for messy formats.

    Score returned here: 0.0 if unsolved after BUDGET_N=34 turns. If solved on step Y:
    turns ≤ MIN_EXPLORE score 1.0 (free exploration zone); beyond that, linear decay from
    1.0 to 0.1 over the remaining budget. Same `seed` fixes the hidden instance.
    """

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnOneInt)
            return str(int(r.value))
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "arithmetic_next_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    arithmetic_next_rf_learning.run(kbench.llm)

