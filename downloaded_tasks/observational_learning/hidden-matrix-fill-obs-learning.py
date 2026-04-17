#!/usr/bin/env python
# coding: utf-8

import random
from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can infer a hidden ternary cellular-automaton rule from demo "
    "grids and apply it to fill a missing interior row in four test grids. Each grid row "
    "is produced from the previous row by a local, deterministic rule: the new symbol at "
    "column c depends only on the three symbols directly above it (columns c-1, c, c+1, "
    "toroidal wrap). The rule is a 27-entry lookup table (all possible (left, center, right) "
    "triples over alphabet {0,1,2}) shared across every demo and test grid. Seeds vary per "
    "grid. All 27 input triples appear across the demos, uniquely pinning the rule."
)

_FIXED_SEED = 0
_P = 3   # alphabet size {0, 1, 2}
_W = 9   # grid width (columns)
_H = 8   # grid height (rows)
_N_DEMOS = 12
_TEST_HIDE_ROWS = [3, 4, 5, 6]  # one per test case; always interior, row r-1 is always visible


# ---------------------------------------------------------------------------
# Core CA helpers
# ---------------------------------------------------------------------------

def _make_rule(rng: random.Random) -> dict:
    triples = [(l, c, r) for l in range(_P) for c in range(_P) for r in range(_P)]
    return {t: rng.randint(0, _P - 1) for t in triples}


def _ca_step(row: list, rule: dict) -> list:
    W = len(row)
    return [rule[(row[(i - 1) % W], row[i], row[(i + 1) % W])] for i in range(W)]


def _evolve(seed: list, rule: dict, H: int) -> list:
    rows = [seed]
    for _ in range(H - 1):
        rows.append(_ca_step(rows[-1], rule))
    return rows


def _is_good_rule(rule: dict, n_test: int = 30) -> bool:
    """Reject rules that converge to a fixed point or period-2 cycle in most seeds."""
    rng = random.Random(777)
    for _ in range(n_test):
        seed = [rng.randint(0, _P - 1) for _ in range(_W)]
        rows = _evolve(seed, rule, 25)
        tail = rows[10:]
        if all(r == tail[0] for r in tail):
            return False
        if all(r == tail[i % 2] for i, r in enumerate(tail)):
            return False
    return True


def _all_triples_covered(demos: list) -> bool:
    seen = set()
    for mat in demos:
        for r in range(1, _H):
            for c in range(_W):
                t = (mat[r - 1][(c - 1) % _W], mat[r - 1][c], mat[r - 1][(c + 1) % _W])
                seen.add(t)
    return len(seen) == _P ** 3


def _render_grid(mat: list, hide_row: int | None = None) -> str:
    lines = []
    for r, row in enumerate(mat):
        symbols = "  ".join("?" if r == hide_row else str(v) for v in row)
        lines.append(f"  row {r}: {symbols}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build everything at module load (fixed seed → deterministic)
# ---------------------------------------------------------------------------

def _build_all():
    rng = random.Random(_FIXED_SEED)

    # Find a rule with good mixing (no quick convergence)
    for _ in range(100_000):
        rule = _make_rule(rng)
        if _is_good_rule(rule):
            break

    # Generate demo grids (different seeds, same rule)
    demo_rng = random.Random(_FIXED_SEED + 1)
    demos = []
    for _ in range(_N_DEMOS):
        seed = [demo_rng.randint(0, _P - 1) for _ in range(_W)]
        demos.append(_evolve(seed, rule, _H))

    assert _all_triples_covered(demos), "Demos do not cover all 27 triples; increase _N_DEMOS."

    # Generate test grids
    test_rng = random.Random(_FIXED_SEED + 2)
    test_cases = []
    for hide_row in _TEST_HIDE_ROWS:
        seed = [test_rng.randint(0, _P - 1) for _ in range(_W)]
        mat = _evolve(seed, rule, _H)
        gt_row = mat[hide_row]
        test_cases.append((mat, hide_row, gt_row))

    return rule, demos, test_cases


_RULE, _DEMOS, _TEST_CASES = _build_all()
_GT_ROWS = [tc[2] for tc in _TEST_CASES]


# ---------------------------------------------------------------------------
# Prompt & grading
# ---------------------------------------------------------------------------

def _prepare():
    demo_block = "\n\n".join(
        f"Demo {i + 1}:\n{_render_grid(mat)}" for i, mat in enumerate(_DEMOS)
    )

    test_blocks = []
    for qi, (mat, hide_row, _) in enumerate(_TEST_CASES, 1):
        test_blocks.append(
            f"Q{qi} — Test grid (row {hide_row} is hidden, shown as ?):\n"
            f"{_render_grid(mat, hide_row)}"
        )

    prompt = (
        "You are watching an expert fill grids according to a HIDDEN LOCAL RULE.\n\n"
        "Grid layout: each grid has 8 rows (row 0 … row 7) and 9 columns (col 0 … col 8).\n"
        "Symbols are drawn from the alphabet {0, 1, 2}.\n\n"
        "How rows are related: every row (row 1 onward) is produced from the row directly "
        "above it by a deterministic local rule. The new symbol at column c depends only on "
        "the three symbols in the previous row at columns (c−1), c, and (c+1) "
        "(columns wrap around: column −1 = column 8, column 9 = column 0).\n\n"
        "The same hidden rule governs every grid. Only the first row (row 0) differs "
        "between grids.\n\n"
        f"{demo_block}\n\n"
        "Now apply what you have learned to these four test grids. "
        "Each has exactly one row replaced by '?'. Predict the missing row.\n\n"
        + "\n\n".join(test_blocks)
        + "\n\n"
        f"For each Q1–Q4, provide the {_W} symbols (left to right) of the hidden row. "
        "Use fields row_1, row_2, row_3, row_4."
    )

    def grade_fn(response):
        test_results = []
        correct_count = 0
        for qi, (mat, hide_row, gt_row) in enumerate(_TEST_CASES, 1):
            field = f"row_{qi}"
            raw = getattr(response, field, None)
            got_repr = str(raw)[:80] if raw is not None else None
            if not isinstance(raw, list):
                test_results.append(
                    {"q": qi, "expected": gt_row, "got": got_repr, "correct": False}
                )
                continue
            try:
                submitted = [int(x) for x in raw]
                correct = submitted == gt_row
            except (TypeError, ValueError):
                submitted = None
                correct = False
            test_results.append(
                {"q": qi, "expected": gt_row, "got": submitted, "correct": correct}
            )
            if correct:
                correct_count += 1
        score = correct_count / len(_TEST_CASES)
        return score, test_results

    return prompt, grade_fn


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_trace(task, description, prompt, test_results, score, reasoning=""):
    sep = "=" * 70
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    if reasoning:
        print(f"\n  REASONING:\n{reasoning}")
    print(f"\n  TEST RESULTS:")
    for r in test_results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"    [{status}] Q{r['q']}: expected={r['expected']!r}  got={r['got']!r}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Task entry point
# ---------------------------------------------------------------------------

@dataclass
class _Answer:
    row_1: list
    row_2: list
    row_3: list
    row_4: list


@kbench.task(
    name="hidden_matrix_fill_obs_learning",
    description=(
        "Infer a hidden ternary 1D cellular automaton (3-symbol, 27-rule, toroidal). "
        "See 12 demo grids. Predict one unseen interior row for each of 4 test grids."
    ),
)
def hidden_matrix_fill_obs_learning(llm) -> float:
    """
    Each grid evolves by a hidden rule: a 27-entry lookup table mapping (left, center, right) ∈ {0,1,2}³ to {0,1,2}. Infer the rule from demos; predict one hidden interior row (r ≥ 3) for each test grid.
    """
    prompt, grade_fn = _prepare()
    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None
    if response is not None:
        score, test_results = grade_fn(response)
    else:
        score = 0.0
        test_results = [
            {"q": i, "expected": _GT_ROWS[i - 1], "got": None, "correct": False}
            for i in range(1, len(_TEST_CASES) + 1)
        ]
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="hidden_matrix_fill_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    hidden_matrix_fill_obs_learning.run(kbench.llm)

