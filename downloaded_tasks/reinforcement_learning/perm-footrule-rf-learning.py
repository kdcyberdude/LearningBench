#!/usr/bin/env python
# coding: utf-8

import random
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
MIN_EXPLORE = 7  # free exploration turns; no efficiency penalty within this zone

_TASK_DESCRIPTION = "Permutation; feedback is FLOW cost (footrule) with sensor dither."


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


ALPH = "ABCDE"
MAX_STEPS = 22
MAX_FOOTRULE = 6  # approximate max footrule distance for a length-5 permutation

_INTRO = (
    f"Hidden ordering of {list(ALPH)} (each symbol once).\n"
    "Submit a candidate ordering. Telemetry reports **FLOW** = sum of absolute index mismatches "
    "between your ordering and the hidden one (0 only when exact).\n"
    "Readings jitter ±1 with 12% probability.\n"
    f"Budget {MAX_STEPS}."
)


def _flow(secret: str, guess: str) -> int:
    pos_s = {c: i for i, c in enumerate(secret)}
    pos_g = {c: i for i, c in enumerate(guess)}
    return sum(abs(pos_s[c] - pos_g[c]) for c in ALPH)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    xs = list(ALPH)
    rng.shuffle(xs)
    secret = "".join(xs)
    last_fb = ""
    conversation: list = []
    best_footrule_reduction = 0.0

    for t in range(cap):
        user = _INTRO if t == 0 else f"{last_fb}\n\nOrdering?"
        raw = "".join(c for c in llm(user).upper() if c in ALPH)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        if len(raw) < 5:
            last_fb = "Need five distinct letters from A–E."
            continue
        guess = raw[:5]
        if len(set(guess)) != 5:
            last_fb = "Symbols must be distinct."
            continue
        true_footrule = _flow(secret, guess)
        best_footrule_reduction = max(best_footrule_reduction, 1.0 - true_footrule / MAX_FOOTRULE)
        if guess == secret:
            return RuntimeTaskResult(
                task_id="perm_footrule",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                detail={"family": "flow_metric_perm"},
                conversation=conversation,
                progress=best_footrule_reduction,
            )
        d = true_footrule
        if rng.random() < 0.12:
            d = max(0, d + rng.choice([-1, 1]))
        last_fb = f"FLOW: **{d}**."

    return RuntimeTaskResult(
        task_id="perm_footrule",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        detail={"family": "flow_metric_perm"},
        conversation=conversation,
        progress=best_footrule_reduction,
    )


@kbench.task(
    name="perm_footrule_rf_learning",
    description="Permutation; feedback is FLOW cost (footrule) with sensor dither. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 28 steps.",
)
def perm_footrule_rf_learning(llm) -> float:
    """Permutation search with noisy footrule FLOW; returns composite [0,1] for up to 28 turns."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "perm_footrule_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    perm_footrule_rf_learning.run(kbench.llm)

