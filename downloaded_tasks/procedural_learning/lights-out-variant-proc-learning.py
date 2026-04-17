#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether the model can discover a hidden L-shaped toggle mask on a 4×4 Lights-Out grid "
    "through 5 practice boards (up to 16 moves each), then apply the learned rule on a final board "
    "in a single move. The hidden rule is that toggling cell (r,c) also flips (r,c+1) and (r+1,c). "
    "What makes it hard is that no rule description is given — the model must infer the mask from "
    "feedback alone. Success requires solving all 5 practice boards and the final test board."
)


def _log_trace(task: str, phases: list[dict], final_score: float, initial_prompt: str = "") -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {_TASK_DESCRIPTION}")
    if initial_prompt:
        print(f"\n  INITIAL PROMPT:\n{initial_prompt}")
    for phase in phases:
        label = phase["label"]
        print(f"\n  [{label}]  correct={phase['correct']}")
        for t in phase["turns"]:
            print(f"    Turn {t['turn']}  submitted: {t['submitted']}", end="")
            if "feedback" in t:
                print(f"  →  {t['feedback']}", end="")
            print()
        status = "PASS ✓" if phase["solved"] else "FAIL ✗"
        print(f"    {status}  steps={phase['steps']}  score={phase['score']:.4f}")
    print(f"\n  Final score : {final_score:.4f}")
    print(f"{sep}\n")


def _efficiency_score(solved: bool, step_y: int, budget_n: int, floor: float = 0.1) -> float:
    if not solved:
        return 0.0
    if budget_n <= 1:
        return 1.0
    step_y = max(1, min(step_y, budget_n))
    return 1.0 - (1.0 - floor) * ((step_y - 1) / (budget_n - 1))


MASK = [(0, 0), (0, 1), (1, 0)]
GRID_SIZE = 4
BUDGET = 16

LEARNING_GRIDS = [
    [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
    [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
    [[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 1, 1]],
    [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 0]],
    [[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1]],
]
TEST_GRID = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]]


def _toggle(grid: list[list[int]], r: int, c: int, mask: list[tuple[int, int]], grid_size: int) -> list[list[int]]:
    new = [row[:] for row in grid]
    for dr, dc in mask:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            new[nr][nc] ^= 1
    return new


def _grid_to_str(grid: list[list[int]]) -> str:
    return " | ".join("".join(str(grid[r][c]) for c in range(len(grid[0]))) for r in range(len(grid)))


@dataclass
class _ToggleSubmission:
    row: int
    col: int


@kbench.task(
    name="lights_out_variant_proc_learning",
    description=(
        "Learn a hidden toggle-neighborhood rule on a 4×4 lights-out grid across 5 practice boards, "
        "then solve a final board in one move. Score = learning_efficiency×0.5 + test_pass×0.5."
    ),
)
def lights_out_variant_proc_learning(llm) -> float:
    """5 practice boards with hidden L-shaped toggle mask (up to 16 moves each), then 1 final board with single attempt. Score=learning_avg×0.5+test×0.5."""
    phases = []
    test_passed = False

    with kbench.chats.new("lights_out_variant"):
        learning_scores = []

        initial_prompt = ""
        for idx, initial_grid in enumerate(LEARNING_GRIDS):
            turns = []
            solved = False
            num_steps = 0
            grid = [row[:] for row in initial_grid]

            def all_off(g: list[list[int]]) -> bool:
                return all(g[r][c] == 0 for r in range(GRID_SIZE) for c in range(GRID_SIZE))

            intro = (
                "You are playing a Lights-Out variant on a 4×4 grid.\n"
                "All lights must be turned OFF (0=off, 1=on).\n"
                "Toggling cell (r,c) also flips a HIDDEN set of neighbouring cells.\n"
                "You must discover the hidden toggle rule through experimentation.\n"
                "After 5 practice boards you face a final board.\n\n"
            ) if idx == 0 else ""

            on_count = sum(grid[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE))
            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 — Initial board ({on_count} lights on):\n"
                f"  {_grid_to_str(grid)}\n"
                f"Rows are 0-indexed top to bottom, cols 0-indexed left to right.\n"
                f"Attempt 1 of {BUDGET}. Submit row and col to toggle."
            )

            if idx == 0:
                initial_prompt = next_prompt

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                try:
                    submission = llm.prompt(next_prompt, schema=_ToggleSubmission)
                except Exception:
                    entry = {"turn": turn, "submitted": "PARSE_ERROR", "feedback": "Failed to parse response — turn wasted."}
                    turns.append(entry)
                    next_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Submit your next toggle."
                    continue

                r, c = submission.row, submission.col
                entry = {"turn": turn, "submitted": f"({r},{c})"}

                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    grid = _toggle(grid, r, c, MASK, GRID_SIZE)

                if all_off(grid):
                    solved = True
                    turns.append(entry)
                    break

                on_count = sum(grid[rr][cc] for rr in range(GRID_SIZE) for cc in range(GRID_SIZE))
                new_display = _grid_to_str(grid)
                feedback = f"Toggled ({r},{c}). New board ({on_count} lights still on): {new_display}"
                entry["feedback"] = feedback
                turns.append(entry)
                next_prompt = (
                    f"{feedback}\n\n"
                    f"Attempt {turn + 1} of {BUDGET}. Submit your next toggle."
                )

            eff = _efficiency_score(solved, num_steps, BUDGET)
            learning_scores.append(eff)
            phases.append({
                "label": f"Practice {idx + 1}/5",
                "correct": "all zeros (solved)",
                "turns": turns,
                "solved": solved,
                "steps": num_steps,
                "score": eff,
            })

        test_grid = [row[:] for row in TEST_GRID]
        on_count = sum(test_grid[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE))
        test_prompt = (
            f"Final test — Board ({on_count} lights on):\n"
            f"  {_grid_to_str(test_grid)}\n"
            "This is your only attempt. No hints will be given.\n"
            "Submit your single toggle to try to turn all lights off."
        )
        try:
            test_submission = llm.prompt(test_prompt, schema=_ToggleSubmission)
        except Exception:
            test_submission = None

        if test_submission is not None:
            tr, tc = test_submission.row, test_submission.col
            if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                test_grid = _toggle(test_grid, tr, tc, MASK, GRID_SIZE)
        else:
            tr, tc = -1, -1
        test_passed = all(test_grid[r][c] == 0 for r in range(GRID_SIZE) for c in range(GRID_SIZE))

        phases.append({
            "label": "Final test",
            "correct": "all zeros (solved)",
            "turns": [{"turn": 1, "submitted": f"({tr},{tc})"}],
            "solved": test_passed,
            "steps": 1,
            "score": 1.0 if test_passed else 0.0,
        })

    final_score = sum(learning_scores) / 5 * 0.5 + (1.0 if test_passed else 0.0) * 0.5
    _log_trace("LIGHTS OUT VARIANT", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    lights_out_variant_proc_learning.run(kbench.llm)

