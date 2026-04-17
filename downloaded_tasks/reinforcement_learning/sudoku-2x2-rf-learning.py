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


BUDGET_N = 40
MIN_EXPLORE = 8  # 4×4 Latin completion needs multi-turn probing / revision

_TASK_DESCRIPTION = (
    "4×4 Latin lattice (symbols 1–4): each row and column is a permutation. "
    "Partial clues are revealed; the hidden completion is **unique** among Latin squares "
    "consistent with those clues. Feedback uses synthetic bands (NO_MATCH / PARTIAL / EXACT)."
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


N = 4
MAX_STEPS = 32


def _is_latin_grid(g: list[list[int]]) -> bool:
    sym = list(range(1, N + 1))
    for row in g:
        if sorted(row) != sym:
            return False
    for c in range(N):
        if sorted(g[r][c] for r in range(N)) != sym:
            return False
    return True


def _shuffle_latin(rng: random.Random) -> list[list[int]]:
    base = [[(i + j) % N + 1 for j in range(N)] for i in range(N)]
    syms = list(range(1, N + 1))
    rng.shuffle(syms)
    grid = [[syms[v - 1] for v in row] for row in base]
    rng.shuffle(grid)
    cols = list(range(N))
    rng.shuffle(cols)
    return [[row[c] for c in cols] for row in grid]


def _count_latin_extensions(
    fixed: dict[tuple[int, int], int], *, limit: int = 2
) -> int:
    """Count Latin completions consistent with fixed clues (cap at `limit`)."""
    g = [[0] * N for _ in range(N)]
    for (r, c), v in fixed.items():
        g[r][c] = v

    count = 0

    def row_ok(r: int, v: int, c_skip: int) -> bool:
        for c in range(N):
            if c == c_skip:
                continue
            if g[r][c] == v:
                return False
        return True

    def col_ok(c: int, v: int, r_skip: int) -> bool:
        for r in range(N):
            if r == r_skip:
                continue
            if g[r][c] == v:
                return False
        return True

    def dfs(idx: int) -> None:
        nonlocal count
        if count >= limit:
            return
        if idx == N * N:
            count += 1
            return
        r, c = divmod(idx, N)
        if g[r][c] != 0:
            dfs(idx + 1)
            return
        for v in range(1, N + 1):
            if not row_ok(r, v, c) or not col_ok(c, v, r):
                continue
            g[r][c] = v
            dfs(idx + 1)
            g[r][c] = 0

    dfs(0)
    return count


def _make_unique_clues(rng: random.Random, grid: list[list[int]]) -> dict[tuple[int, int], int]:
    cells = [(r, c) for r in range(N) for c in range(N)]
    rng.shuffle(cells)
    k_start = 6
    for extra in range(11):
        k = min(N * N, k_start + extra)
        reveal = cells[:k]
        fixed = {(r, c): grid[r][c] for r, c in reveal}
        if _count_latin_extensions(fixed, limit=2) == 1:
            return fixed
    return {(r, c): grid[r][c] for r in range(N) for c in range(N)}


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    grid = _shuffle_latin(rng)
    clues = _make_unique_clues(rng, grid)
    clue_lines = "\n".join(f"  ({r},{c}) = {clues[(r, c)]}" for (r, c) in sorted(clues.keys()))
    last_fb = ""
    intro = (
        f"{N}×{N} **Latin lattice** using symbols {{1,2,3,4}}: each **row** and **column** "
        "must contain each symbol exactly once.\n"
        "Clues (fixed cells):\n"
        f"{clue_lines}\n"
        "There is **exactly one** Latin completion consistent with these clues.\n"
        "Reply with `GRID` followed by **sixteen** digits in **row-major** order (r0c0..r3c3).\n"
        "Feedback bands after each attempt:\n"
        "  • **EXACT** — matches the hidden completion.\n"
        "  • **PARTIAL** — valid Latin grid and respects all clues, but not the hidden one.\n"
        "  • **NO_MATCH** — breaks Latin rules or violates a clue.\n"
        f"Budget: **{cap}** attempts."
    )

    best_partial_score = 0.0
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nGRID?"
        raw_llm = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw_llm})
        nums = [int(x) for x in re.findall(r"[1-4]", raw_llm)]
        if len(nums) < N * N:
            last_fb = "NO_MATCH — need sixteen digits from {1,2,3,4}."
            best_partial_score = max(best_partial_score, 0.1)
            continue
        cand = [nums[i * N : (i + 1) * N] for i in range(N)]

        if not _is_latin_grid(cand):
            last_fb = "NO_MATCH — each row/column must be a permutation of {1,2,3,4}."
            best_partial_score = max(best_partial_score, 0.1)
            continue
        if any(cand[r][c] != clues[(r, c)] for (r, c) in clues):
            last_fb = "NO_MATCH — violates a fixed clue."
            best_partial_score = max(best_partial_score, 0.1)
            continue
        if cand == grid:
            return RuntimeTaskResult(
                task_id="sudoku_2x2",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "latin_square_4"},
                conversation=conversation,
                progress=1.0,
            )
        last_fb = "PARTIAL — valid Latin + clues, but not the hidden completion."
        best_partial_score = max(best_partial_score, 0.5)

    return RuntimeTaskResult(
        task_id="sudoku_2x2",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "latin_square_4"},
        conversation=conversation,
        progress=best_partial_score,
    )


@kbench.task(
    name="sudoku_2x2_rf_learning",
    description="4×4 Latin lattice with partial clues; unique completion; synthetic NO_MATCH/PARTIAL/EXACT feedback. Multi-turn RL; return float in [0,1], cap 40 steps.",
)
def sudoku_2x2_rf_learning(llm) -> float:
    """Fill the unique 4x4 Latin-square completion consistent with partial clues; feedback tiers NO_MATCH/PARTIAL/EXACT. Returns composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "sudoku_2x2_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    sudoku_2x2_rf_learning.run(kbench.llm)

