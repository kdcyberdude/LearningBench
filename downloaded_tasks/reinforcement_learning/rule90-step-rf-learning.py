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


BUDGET_N = 26
MIN_EXPLORE = 6

_TASK_DESCRIPTION = (
    "Discover a hidden 1D cellular automaton rule (one of four Wolfram rules: 30, 90, 110, 150) "
    "by probing it with TEST queries, then predict the next step of a challenge string with ANSWER."
)

_CA_RULES: dict[int, dict[tuple[int, int, int], int]] = {
    30:  {(0,0,0):0,(0,0,1):1,(0,1,0):1,(0,1,1):1,(1,0,0):1,(1,0,1):0,(1,1,0):0,(1,1,1):0},
    90:  {(0,0,0):0,(0,0,1):1,(0,1,0):0,(0,1,1):1,(1,0,0):1,(1,0,1):0,(1,1,0):1,(1,1,1):0},
    110: {(0,0,0):0,(0,0,1):1,(0,1,0):1,(0,1,1):1,(1,0,0):0,(1,0,1):1,(1,1,0):1,(1,1,1):0},
    150: {(0,0,0):0,(0,0,1):1,(0,1,0):1,(0,1,1):0,(1,0,0):1,(1,0,1):0,(1,1,0):0,(1,1,1):1},
}


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


def _apply_rule(rule: dict[tuple[int, int, int], int], bits: str) -> str:
    """Apply a CA rule to a 9-bit string with periodic boundary."""
    n = len(bits)
    result = []
    for i in range(n):
        left = int(bits[(i - 1) % n])
        mid = int(bits[i])
        right = int(bits[(i + 1) % n])
        result.append(str(rule[(left, mid, right)]))
    return "".join(result)


def _parse_bits9(text: str) -> Optional[str]:
    """Extract the first run of 9 binary digits from text."""
    bits = re.sub(r"[^01]", "", text)
    return bits[:9] if len(bits) >= 9 else None


MAX_STEPS = 20


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    hidden_rule = rng.choice([30, 90, 110, 150])
    challenge = "".join(rng.choice("01") for _ in range(9))
    answer = _apply_rule(_CA_RULES[hidden_rule], challenge)

    best_mismatch = 9
    last_fb = ""
    intro = (
        "A hidden one-dimensional cellular automaton rule governs the evolution of 9-cell binary strings.\n"
        "Each cell's next state depends on its left neighbor, itself, and its right neighbor (periodic boundary).\n"
        "The rule is one of four known rules but its identity is NOT disclosed.\n\n"
        "You may:\n"
        "  TEST: <bits>   — apply the hidden rule to any 9-bit string and observe the output\n"
        "  ANSWER: <bits> — submit your prediction for the next step of the challenge string\n\n"
        f"Challenge string: {challenge}\n"
        "Predict its next state under the hidden rule.\n"
        f"Budget: {cap} turns."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nYour next action?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})

        raw_upper = raw.upper()
        ans_m = re.search(r"ANSWER\s*:\s*([01\s]+)", raw_upper)
        test_m = re.search(r"TEST\s*:\s*([01\s]+)", raw_upper)

        if ans_m:
            bits = _parse_bits9(ans_m.group(1))
            if bits is None:
                last_fb = "ANSWER needs exactly 9 bits (0s and 1s)."
            elif bits == answer:
                progress = 1.0
                return RuntimeTaskResult(
                    task_id="rule90_step",
                    solved=True,
                    num_steps=t + 1,
                    max_steps=cap,
                    detail={"hidden_rule": hidden_rule, "family": "ca_discovery"},
                    conversation=conversation,
                    progress=progress,
                )
            else:
                h = sum(a != b for a, b in zip(bits, answer))
                best_mismatch = min(best_mismatch, h)
                last_fb = f"Mismatch count: {h}"
        elif test_m:
            bits = _parse_bits9(test_m.group(1))
            if bits is None:
                last_fb = "TEST needs exactly 9 bits (0s and 1s)."
            else:
                result_bits = _apply_rule(_CA_RULES[hidden_rule], bits)
                last_fb = f"After one step: {result_bits}"
        else:
            last_fb = (
                "Use TEST: <9 bits> to probe the hidden rule, "
                "or ANSWER: <9 bits> to submit your prediction for the challenge string."
            )

    progress = 1.0 - best_mismatch / 9.0
    return RuntimeTaskResult(
        task_id="rule90_step",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        detail={"hidden_rule": hidden_rule, "family": "ca_discovery"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="rule90_step_rf_learning",
    description=(
        "Discover a hidden 1D CA rule (one of four Wolfram rules: 30, 90, 110, 150) via TEST probes, "
        "then predict the challenge string's next step with ANSWER. Multi-turn RL; return float in [0,1], cap 26 steps."
    ),
)
def rule90_step_rf_learning(llm) -> float:
    """Discover which Wolfram rule (30/90/110/150) is active via TEST probes, then ANSWER the one-step evolution of a challenge bit string. Returns composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "rule90_step_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    rule90_step_rf_learning.run(kbench.llm)

