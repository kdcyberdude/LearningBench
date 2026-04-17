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
    "Among 12 labeled coins one is heavier; identify it with 3-vs-3 weighings. "
    "A hidden integer **drift bias** in {-1,0,+1} shifts the tilt outcome near parity, "
    "so textbook memorized decision trees misfire until inferred. "
    "The scale also **jams 8%** of the time with random telemetry. "
    "Feedback uses ALPHA_TILT / BETA_TILT / NEUTRAL_LOCK / PHANTOM_READING."
)

BUDGET_N = 24
MIN_EXPLORE = 7  # infer drift + survive jam noise


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


N = 12
MAX_STEPS = 14


def _parse_pans(raw: str) -> Optional[tuple[list[int], list[int]]]:
    s = raw.replace(" ", "").upper()
    m = re.search(r"([\d,]+)\|([\d,]+)", s)
    if not m:
        m = re.search(r"LEFT\s*([\d,]+)\s*RIGHT\s*([\d,]+)", s)
    if not m:
        return None

    def ids(part: str) -> list[int]:
        return [int(x) for x in part.split(",") if x.strip().isdigit()]

    L, R = ids(m.group(1)), ids(m.group(2))
    return (L, R)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    heavy = rng.randint(0, N - 1)
    drift = rng.choice([-1, 0, 1])
    last_fb = ""
    intro = (
        f"{N} coins labeled 0..{N - 1}; exactly one is **heavy**.\n"
        "Each weighing must place **exactly three** coins on each pan: `a,b,c|d,e,f`.\n"
        "Telemetry uses **ALPHA_TILT** (left pan heavier net), **BETA_TILT** (right pan heavier net), "
        "or **NEUTRAL_LOCK** (balanced net).\n"
        "A hidden **drift bias** (fixed for the episode, not enumerated here) can nudge near-ties — "
        "infer it from repeated consistent weighings.\n"
        "The scale **jams 8%** of the time and prints **PHANTOM_READING** with a random tilt token.\n"
        "Finish with `ANSWER: k`.\n"
        f"Budget {cap} turns."
    )
    conversation: list = []
    valid_weighings_made = 0
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nWeigh or answer?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m_ans = re.search(r"ANSWER\s*:\s*(\d+)", raw.upper())
        if m_ans:
            k = int(m_ans.group(1))
            if k == heavy:
                progress = min(1.0, valid_weighings_made / 3.0)
                return RuntimeTaskResult(
                    task_id="coin_balance",
                    solved=True,
                    num_steps=t + 1,
                    max_steps=cap,
                    intro=intro,
                    detail={"heavy": heavy, "drift": drift, "family": "noisy_weighing"},
                    conversation=conversation,
                    progress=progress,
                )
            last_fb = "Wrong index."
            continue
        parsed = _parse_pans(raw)
        if parsed is None:
            last_fb = "Need `a,b,c|d,e,f` with six distinct indices in range."
            continue
        L, R = parsed
        if (
            len(L) != 3
            or len(R) != 3
            or set(L) & set(R)
            or any(x < 0 or x >= N for x in L + R)
        ):
            last_fb = "Illegal pans — need 3 distinct per side, all in 0..11."
            continue
        if rng.random() < 0.08:
            last_fb = (
                "PHANTOM_READING — "
                f"{rng.choice(['ALPHA_TILT', 'BETA_TILT', 'NEUTRAL_LOCK'])} (unreliable)."
            )
            continue
        hl, hr = sum(1 for x in L if x == heavy), sum(1 for x in R if x == heavy)
        if hl and hr:
            last_fb = "ERROR — heavy cannot be on both pans."
            continue
        valid_weighings_made += 1
        edge = (hl - hr) + drift
        if edge > 0:
            last_fb = "ALPHA_TILT."
        elif edge < 0:
            last_fb = "BETA_TILT."
        else:
            last_fb = "NEUTRAL_LOCK."

    progress = min(1.0, valid_weighings_made / 3.0)
    return RuntimeTaskResult(
        task_id="coin_balance",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"heavy": heavy, "drift": drift, "family": "noisy_weighing"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="coin_balance_rf_learning",
    description="12 coins, one heavy; 3-vs-3 weighings; hidden drift bias + 8% PHANTOM_READING jam. Multi-turn RL; return float in [0,1], cap 24 steps.",
)
def coin_balance_rf_learning(llm) -> float:
    """12-coin heavy-coin ID with biased and jammed weighings; composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "coin_balance_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    coin_balance_rf_learning.run(kbench.llm)

