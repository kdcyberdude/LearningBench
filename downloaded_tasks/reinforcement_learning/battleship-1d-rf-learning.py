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
    "A hidden rigid object spans 4–5 consecutive slots on a 20-slot line. "
    "Sensor feedback is one-step delayed (echo channel). "
    "Synthetic raster vocabulary (PREAMBLE_VOID / RASTER_NULL / RASTER_TOUCH / RASTER_CLEAR)."
)

BUDGET_N = 44
MIN_EXPLORE = 10  # delayed echo + larger line needs more free probes


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


LINE_LEN = 20
MAX_STEPS = 36


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    ship_len = rng.choice([4, 5])
    start = rng.randint(0, LINE_LEN - ship_len)
    cells = frozenset(range(start, start + ship_len))
    hits: set[int] = set()
    last_fb = ""
    prev_idx: Optional[int] = None
    prev_result: Optional[str] = None
    intro = (
        f"A rigid object spans **{ship_len} consecutive** slots indexed 0..{LINE_LEN - 1}.\n"
        "Sensors are broken: each turn you learn the **previous** raster ping's outcome, not the current one.\n"
        "First ping returns **PREAMBLE_VOID** only.\n"
        "Echo vocabulary: **RASTER_NULL** (no overlap), **RASTER_TOUCH** (overlap, object intact), "
        "**RASTER_CLEAR** (all segments have been pinged at least once).\n"
        f"Budget: {cap} pings."
    )
    conversation: list = []
    cells_hit = 0
    for t in range(cap):
        user = intro if t == 0 else f"Downlink:\n{last_fb}\n\nNext slot index?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m = re.search(r"\b(\d+)\b", raw)
        if not m:
            last_fb = "Need one integer index."
            continue
        g = int(m.group(1))
        if not (0 <= g < LINE_LEN):
            last_fb = f"Index in 0..{LINE_LEN - 1}."
            continue

        if prev_idx is None:
            last_fb = "PREAMBLE_VOID — no prior ping to echo."
        else:
            if prev_result == "RASTER_NULL":
                last_fb = f"Echo for slot {prev_idx}: RASTER_NULL."
            elif prev_result == "RASTER_TOUCH":
                last_fb = f"Echo for slot {prev_idx}: RASTER_TOUCH (overlap, object intact)."
            else:
                last_fb = f"Echo for slot {prev_idx}: {prev_result}"

        if g not in cells:
            prev_result = "RASTER_NULL"
        else:
            if g not in hits:
                hits.add(g)
                cells_hit = len(hits)
            if hits == set(cells):
                prev_result = "RASTER_CLEAR"
            else:
                prev_result = "RASTER_TOUCH"
        prev_idx = g

        if hits == set(cells):
            progress = cells_hit / ship_len
            return RuntimeTaskResult(
                task_id="battleship_1d",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"ship": sorted(cells), "family": "delayed_sensor_search"},
                conversation=conversation,
                progress=progress,
            )

    progress = cells_hit / ship_len
    return RuntimeTaskResult(
        task_id="battleship_1d",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "delayed_sensor_search"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="battleship_1d_rf_learning",
    description="20-slot line; hidden length-4/5 object; one-step delayed RASTER_* echo telemetry. Multi-turn RL; return float in [0,1], cap 44 steps.",
)
def battleship_1d_rf_learning(llm) -> float:
    """One-step delayed raster feedback on a 20-slot line; composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "battleship_1d_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    battleship_1d_rf_learning.run(kbench.llm)

