#!/usr/bin/env python
# coding: utf-8

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
    "A hidden tape of 14 binary glyphs (λ or ρ) must be discovered by submitting full-length guesses. "
    "Feedback is a DIVERGENCE count (Hamming distance under obfuscated naming) with occasional sensor dither. "
    "The model must infer the exact tape by adaptive search within the step budget."
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


SYM = ("λ", "ρ")
BITS = 14
MAX_STEPS = 24


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    secret = "".join(rng.choice(SYM) for _ in range(BITS))
    last_fb = ""
    intro = (
        f"A hidden tape has length {BITS}. Each cell is either **λ** or **ρ**.\n"
        "Submit a full tape of those glyphs (non-glyph characters stripped).\n"
        "Telemetry returns **DIVERGENCE** — the number of slots where your glyph disagrees with the tape.\n"
        "(This is *not* named Hamming; treat it as an unknown distance oracle.)\n"
        f"Budget: {cap} submissions."
    )
    conversation: list = []
    best_bit_accuracy = 0.0
    for t in range(cap):
        user = intro if t == 0 else f"Telemetry:\n{last_fb}\n\nNext tape?"
        raw_in = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw_in})
        raw = "".join(c for c in raw_in if c in SYM)
        if len(raw) != BITS:
            last_fb = f"Need exactly {BITS} glyphs from {{{SYM[0]}, {SYM[1]}}}; got {len(raw)}."
            continue
        real_h = sum(g != t for g, t in zip(raw, secret))
        best_bit_accuracy = max(best_bit_accuracy, 1.0 - real_h / len(secret))
        if raw == secret:
            return RuntimeTaskResult(
                task_id="bitstring_hamming",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "glyph_hamming"},
                conversation=conversation,
                progress=best_bit_accuracy,
            )
        h = sum(a != b for a, b in zip(raw, secret))
        flip = rng.random() < 0.08
        rep = h + (1 if flip and h < BITS else -1 if flip and h > 0 else 0)
        last_fb = f"DIVERGENCE ≈ **{rep}** (sensor dither)."

    return RuntimeTaskResult(
        task_id="bitstring_hamming",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "glyph_hamming"},
        conversation=conversation,
        progress=best_bit_accuracy,
    )


@kbench.task(
    name="bitstring_hamming_rf_learning",
    description="Hidden string over a binary glyph alphabet; feedback is DIVERGENCE count (Hamming under the hood). Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 30 steps.",
)
def bitstring_hamming_rf_learning(llm) -> float:
    """14-bit binary tape with Hamming-style DIVERGENCE feedback; composite score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "bitstring_hamming_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    bitstring_hamming_rf_learning.run(kbench.llm)

