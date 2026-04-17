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
    "Two disjoint length-2 craft on a 24-slot line. Each salvo pings two slots; telemetry is "
    "**AGG_STRIKE** (aggregate hits only, 0–2) without per-beam attribution. "
    "Hidden configuration space ≫ 100 placements."
)

BUDGET_N = 34
MIN_EXPLORE = 10  # aggregate-only feedback needs more exploration


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


N = 24
SHIP = 2
MAX_STEPS = 28


def _place_two(rng: random.Random) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    for _ in range(200):
        a = rng.randint(0, N - SHIP)
        ship_a = frozenset({a, a + 1})
        b = rng.randint(0, N - SHIP)
        ship_b = frozenset({b, b + 1})
        if ship_a.isdisjoint(ship_b):
            return frozenset(ship_a | ship_b), ship_a, ship_b
    return frozenset({0, 1, 4, 5}), frozenset({0, 1}), frozenset({4, 5})


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    cells, ship_a_cells, ship_b_cells = _place_two(rng)
    hits: set[int] = set()
    last_fb = ""
    intro = (
        f"Field 0..{N - 1}. Two disjoint **{SHIP}-segment** craft.\n"
        "Each turn submit **two distinct indices** `i j` (order free).\n"
        "Telemetry: **AGG_STRIKE** = number of indices that hit any craft (0–2). "
        "You do **not** learn which beam caused a strike beyond the aggregate count.\n"
        f"Destroy all {len(cells)} occupied cells. Budget {cap} salvos."
    )
    conversation: list = []
    ships_sunk = 0
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nTwo indices?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        nums = [int(x) for x in re.findall(r"\b(\d+)\b", raw)]
        if len(nums) < 2:
            last_fb = "Need two integers."
            continue
        i, j = nums[0], nums[1]
        if i == j or not (0 <= i < N and 0 <= j < N):
            last_fb = "Distinct indices in range."
            continue
        band = int(i in cells) + int(j in cells)
        hits.update({i, j} & set(cells))
        ships_sunk = int(ship_a_cells.issubset(hits)) + int(ship_b_cells.issubset(hits))
        if hits == set(cells):
            progress = ships_sunk / 2.0
            return RuntimeTaskResult(
                task_id="battleship_two_ships",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "salvo_search"},
                conversation=conversation,
                progress=progress,
            )
        last_fb = f"AGG_STRIKE={band}."

    progress = ships_sunk / 2.0
    return RuntimeTaskResult(
        task_id="battleship_two_ships",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "salvo_search"},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="battleship_two_ships_rf_learning",
    description="Two disjoint length-2 craft on 24 slots; salvo telemetry AGG_STRIKE (0–2 aggregate). Multi-turn RL; return float in [0,1], cap 34 steps.",
)
def battleship_two_ships_rf_learning(llm) -> float:
    """Two length-2 ships with aggregate salvo strikes only; composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "battleship_two_ships_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    battleship_two_ships_rf_learning.run(kbench.llm)

