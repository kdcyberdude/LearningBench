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
    "Three-disk Hanoi on pegs NORTH/EAST/SOUTH. The goal peg and a hidden forbidden "
    "disk-peg placement rule are unknown at episode start; the model must infer both "
    "through move attempts and consequence feedback. Hidden configuration space: 27. "
    "Success means stacking all three disks on the hidden goal peg."
)

BUDGET_N = 40
MIN_EXPLORE = 7  # free exploration turns; enough to probe forbidden rule + goal identity


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


MAX_STEPS = 32
PEGS = ("NORTH", "EAST", "SOUTH")
NUM_DISKS = 3

# Hidden configuration: goal peg (one of 3) × forbidden (disk_size, peg) pair (one of 9)
# total hidden space = 3 × 9 = 27


def _parse_move(raw: str) -> Optional[tuple[str, str]]:
    u = raw.upper().strip()
    peg_pat = r"(NORTH|EAST|SOUTH)"
    m = re.search(rf"MOVE\s+{peg_pat}\s+(?:TO\s+)?{peg_pat}", u)
    if m:
        a, b = m.group(1), m.group(2)
        return (a, b) if a != b else None
    m = re.search(rf"{peg_pat}\s*-+>\s*{peg_pat}", u)
    if m:
        a, b = m.group(1), m.group(2)
        return (a, b) if a != b else None
    return None


def _can_move(st: dict[str, list[int]], src: str, dst: str) -> bool:
    if not st[src]:
        return False
    top = st[src][-1]
    if not st[dst]:
        return True
    return st[dst][-1] > top


def _apply_move(st: dict[str, list[int]], src: str, dst: str) -> None:
    st[dst].append(st[src].pop())


def _state_str(st: dict[str, list[int]]) -> str:
    parts = [f"{p}: {st[p]}" for p in PEGS]
    return "  |  ".join(parts) + "  (top = last element)"


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)

    # Hidden variables
    goal_peg: str = rng.choice(PEGS)
    # forbidden: placing disk of size `forbidden_disk` on peg `forbidden_peg` is disallowed
    forbidden_disk: int = rng.choice([1, 2, 3])
    forbidden_peg: str = rng.choice(PEGS)

    # Initial state: all disks on NORTH (opposite of any possible goal)
    start_peg = "NORTH"
    stacks: dict[str, list[int]] = {p: [] for p in PEGS}
    stacks[start_peg] = [3, 2, 1]  # [3] on bottom, [1] on top

    valid_moves_made = 0

    intro = (
        "Three disks **1** (small), **2** (medium), **3** (large) are stacked on peg **NORTH** "
        "(large at bottom). Pegs: **NORTH**, **EAST**, **SOUTH**.\n"
        "Standard Hanoi rules: move one top disk at a time; never place a larger disk on a smaller one.\n"
        "Two unknowns you must discover through interaction:\n"
        "  1. The **goal peg** — which peg you must stack all three disks on (not necessarily SOUTH).\n"
        "  2. A **hidden restriction** — one specific (disk_size, peg) combination is forbidden; "
        "attempting it yields **RESTRICTED** feedback so you can learn to avoid it.\n"
        "Move syntax: `MOVE NORTH EAST` or `NORTH -> EAST`.\n"
        f"Completing the puzzle on the correct goal peg within {cap} moves wins."
    )

    last_fb = ""
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else (
            f"Current state: {_state_str(stacks)}\n{last_fb}\n\nYour move?"
        )
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        mv = _parse_move(raw)
        if mv is None:
            last_fb = "Unreadable. Use `MOVE NORTH EAST` or `NORTH->EAST`."
            continue
        src, dst = mv
        if src not in PEGS or dst not in PEGS:
            last_fb = f"Pegs must be NORTH, EAST, or SOUTH. Got: {src}, {dst}."
            continue
        if not _can_move(stacks, src, dst):
            last_fb = "BLOCKED — that move violates standard Hanoi stacking rules."
            continue
        moving_disk = stacks[src][-1]
        if moving_disk == forbidden_disk and dst == forbidden_peg:
            last_fb = (
                f"RESTRICTED — disk {moving_disk} cannot be placed on {dst}. "
                "This is the hidden restriction for this episode."
            )
            continue
        _apply_move(stacks, src, dst)
        valid_moves_made += 1
        disks_on_goal_peg = len(stacks[goal_peg])
        progress_moves = min(1.0, valid_moves_made / (BUDGET_N * 0.5))
        progress_disks = disks_on_goal_peg / NUM_DISKS
        current_progress = max(progress_moves, progress_disks)
        # Check win condition
        if stacks[goal_peg] == [3, 2, 1]:
            return RuntimeTaskResult(
                task_id="hanoi_two",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={
                    "goal_peg": goal_peg,
                    "forbidden_disk": forbidden_disk,
                    "forbidden_peg": forbidden_peg,
                    "family": "hidden_goal_constrained_hanoi",
                },
                conversation=conversation,
                progress=current_progress,
            )
        # Check if wrong goal peg was completed (inform model)
        for p in PEGS:
            if p != goal_peg and stacks[p] == [3, 2, 1]:
                last_fb = (
                    f"Applied {src}->{dst}. All disks on {p}, but that is NOT the goal peg. "
                    "Keep going — find the correct goal peg."
                )
                break
        else:
            last_fb = f"Applied {src}->{dst}."

    disks_on_goal_peg = len(stacks[goal_peg])
    progress_moves = min(1.0, valid_moves_made / (BUDGET_N * 0.5))
    progress_disks = disks_on_goal_peg / NUM_DISKS
    final_progress = max(progress_moves, progress_disks)
    return RuntimeTaskResult(
        task_id="hanoi_two",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={
            "goal_peg": goal_peg,
            "forbidden_disk": forbidden_disk,
            "forbidden_peg": forbidden_peg,
            "family": "hidden_goal_constrained_hanoi",
        },
        conversation=conversation,
        progress=final_progress,
    )


@kbench.task(
    name="hanoi_two_rf_learning",
    description="Three-disk Hanoi with hidden goal peg and hidden forbidden disk-peg rule (27 configurations). Multi-turn RL: model must infer both unknowns through move feedback; return float in [0,1], cap 40 steps.",
)
def hanoi_two_rf_learning(llm) -> float:
    """
    Three-disk Hanoi: infer hidden goal peg and forbidden (disk, peg) rule from move feedback;
    stack all disks on goal. Composite RL score.
    """

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "hanoi_two_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    hanoi_two_rf_learning.run(kbench.llm)

