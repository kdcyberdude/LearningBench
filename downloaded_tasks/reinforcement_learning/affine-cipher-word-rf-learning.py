#!/usr/bin/env python
# coding: utf-8

import random
import string
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
    "A random monoalphabetic substitution permutes all 26 capital letters. "
    "The model sees ciphertext for a plaintext drawn from a closed pool and must "
    "recover the word using **graded** cipher-consistent position telemetry (not a single binary bit)."
)

BUDGET_N = 24
MIN_EXPLORE = 7  # pool + substitution need multi-probe alignment signal


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


WORDS = ("CRANE", "GLOBE", "STORM", "QUART", "ZEBRA", "JUMPS", "VEXED", "FLINT")
MAX_STEPS = 16


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    plain = rng.choice(WORDS)
    letters = list(string.ascii_uppercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    sub = dict(zip(letters, shuffled))
    cipher = "".join(sub[c] for c in plain)
    pool = ", ".join(WORDS)
    last_fb = ""
    intro = (
        "A fixed **symbol substitution** permutes the 26 capital Latin letters (bijection unknown).\n"
        f"The plaintext is **exactly one** of: {pool}\n"
        f"Ciphertext: **{cipher}**\n"
        "Recover the plaintext word (uppercase A–Z). Wrong guesses may retry.\n"
        f"Budget {cap}."
    )
    conversation: list = []
    best_aligned = 0
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nPlaintext?"
        raw_response = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw_response})
        guess = "".join(c for c in raw_response.upper() if c in string.ascii_uppercase)
        if len(guess) < len(plain):
            last_fb = f"Need length {len(plain)} from the pool."
            continue
        guess = guess[: len(plain)]
        if guess not in WORDS:
            last_fb = "ORBIT_REJECT — candidate must be one of the listed pool entries."
            continue
        if guess != plain:
            aligned = sum(1 for i in range(len(plain)) if sub[guess[i]] == plain[i])
            best_aligned = max(best_aligned, aligned)
            if aligned >= 3:
                last_fb = (
                    f"CIPHER_LOCK — {aligned} positions carry consistent cipher mappings."
                )
            elif aligned == 2:
                last_fb = "CIPHER_PARTIAL — 2 positions consistent."
            elif aligned == 1:
                last_fb = "CIPHER_TRACE — 1 position consistent."
            else:
                last_fb = "CIPHER_VOID — no cipher-consistent positions."
            continue
        progress = best_aligned / len(plain)
        return RuntimeTaskResult(
            task_id="affine_cipher_word",
            solved=True,
            num_steps=t + 1,
            max_steps=cap,
            intro=intro,
            detail={"family": "monoalphabetic_pool"},
            conversation=conversation,
            progress=progress,
        )

    progress = best_aligned / len(plain)
    return RuntimeTaskResult(
        task_id="affine_cipher_word",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "monoalphabetic_pool"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="affine_cipher_word_rf_learning",
    description="Monoalphabetic substitution; closed word pool; graded CIPHER_* alignment telemetry. Multi-turn RL; return float in [0,1], cap 24 steps.",
)
def affine_cipher_word_rf_learning(llm) -> float:
    """Monoalphabetic ciphertext recovery; multi-turn run/llm loop; composite score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "affine_cipher_word_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    affine_cipher_word_rf_learning.run(kbench.llm)

