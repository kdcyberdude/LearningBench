#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Compositional grid transform
# Rule: transpose then horizontal flip on the 5x5 grid.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "The model must infer a fixed spatial transformation applied to 5x5 integer grids (values 0, 1, 2). "
    "The hidden rule is: transpose the grid then flip it horizontally. "
    "The model actively requests labeled input/output grid pairs and must submit the transformed test grid. "
    "What makes it hard is that transpose and horizontal flip are easy to confuse individually, and the "
    "composition is non-obvious from few examples. "
)


def _log_trace(task: str, turns: list[dict], exam_results: list[dict],
               final_score: float, examples_used: int,
               exam_prompt: str = "", exam_raw: list[str] = None) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {_TASK_DESCRIPTION}")
    print(f"\n{sep}\n  CONVERSATION\n{sep}")
    for t in turns:
        print(f"\n[USER \u2014 Turn {t['turn']}]")
        print(t.get("prompt", ""))
        print(f"\n[ASSISTANT \u2014 Turn {t['turn']}]")
        print(f"action: {t['action']}")
        response = t.get("response", "")
        print(f"answer: {response if response else '(none)'}")
    print(f"\n{sep}\n  EXAMINATION\n{sep}")
    if exam_prompt:
        print("\n[USER \u2014 Exam]")
        print(exam_prompt)
    if exam_raw:
        print("\n[ASSISTANT \u2014 Exam]")
        print("\n".join(exam_raw))
    print(f"\n{sep}\n  RESULTS\n{sep}")
    for r in exam_results:
        status = "CORRECT" if r["correct"] else "WRONG  "
        print(f"  Test {r['item']}: {status}   input={r['input']!r}   expected={r['expected']!r}   got={r['answer']!r}")
    correct = sum(1 for r in exam_results if r["correct"])
    print(f"\n  Examples used : {examples_used}/{MAX_EXAMPLES}")
    print(f"  Exam accuracy : {correct}/{len(exam_results)}")
    print(f"  Final score   : {final_score:.4f}")
    print(f"{sep}\n")

NUM_TEST_ITEMS = 1
FREE_THRESHOLD = 5


def _concept_score(correct_count: int, examples_used: int,
                   max_examples: int, initial_examples: int) -> float:
    accuracy = correct_count / NUM_TEST_ITEMS
    if accuracy == 0:
        return 0.0
    effective_free = max(initial_examples, FREE_THRESHOLD)
    if max_examples <= effective_free or examples_used <= effective_free:
        efficiency = 1.0
    else:
        paid_used   = examples_used - effective_free
        paid_budget = max_examples  - effective_free
        efficiency  = max(0.0, 1.0 - paid_used / paid_budget)
    return accuracy * (0.40 + 0.60 * efficiency)


def _transpose(g: list[list[int]]) -> list[list[int]]:
    n = len(g)
    return [[g[j][i] for j in range(n)] for i in range(n)]


def _hflip(g: list[list[int]]) -> list[list[int]]:
    return [row[::-1] for row in g]


def _grid_rule(g: list[list[int]]) -> list[list[int]]:
    return _hflip(_transpose(g))


_ALL_INPUTS = [
    [[1, 0, 2, 1, 0], [2, 1, 0, 2, 1], [0, 2, 1, 0, 2], [1, 1, 2, 0, 1], [0, 0, 1, 2, 2]],
    [[2, 2, 0, 1, 1], [0, 1, 1, 0, 2], [1, 0, 2, 2, 0], [2, 1, 0, 1, 2], [0, 2, 1, 0, 1]],
    [[0, 1, 2, 0, 1], [1, 2, 0, 1, 0], [2, 0, 1, 2, 1], [0, 1, 2, 0, 2], [1, 0, 1, 2, 0]],
    [[1, 2, 0, 2, 1], [0, 0, 1, 1, 2], [2, 1, 2, 0, 1], [1, 0, 0, 2, 0], [0, 2, 1, 1, 2]],
    [[0, 0, 2, 1, 2], [1, 2, 0, 0, 1], [2, 1, 1, 2, 0], [0, 2, 0, 1, 1], [1, 0, 2, 0, 2]],
    [[2, 1, 0, 0, 2], [0, 2, 1, 2, 0], [1, 0, 2, 1, 1], [2, 1, 0, 2, 0], [0, 2, 1, 0, 1]],
    [[1, 1, 2, 0, 2], [0, 2, 0, 1, 1], [2, 0, 1, 2, 0], [1, 2, 0, 0, 2], [0, 1, 2, 1, 0]],
    [[0, 2, 1, 0, 2], [2, 0, 2, 1, 0], [1, 1, 0, 2, 1], [0, 2, 1, 0, 2], [2, 0, 1, 2, 1]],
    [[1, 0, 0, 2, 1], [2, 1, 2, 0, 0], [0, 2, 1, 1, 2], [1, 0, 2, 0, 1], [2, 1, 0, 2, 0]],
    [[2, 0, 1, 2, 0], [1, 2, 0, 1, 2], [0, 1, 2, 0, 1], [2, 0, 1, 2, 0], [1, 2, 0, 1, 2]],
    [[0, 1, 1, 2, 0], [2, 0, 2, 1, 1], [1, 2, 0, 0, 2], [0, 1, 2, 1, 0], [2, 0, 1, 2, 1]],
    [[1, 2, 1, 0, 2], [0, 0, 2, 1, 1], [2, 1, 0, 2, 0], [1, 2, 1, 0, 2], [0, 1, 0, 2, 1]],
    [[2, 0, 2, 1, 0], [1, 1, 0, 2, 2], [0, 2, 1, 0, 1], [2, 0, 2, 1, 0], [1, 2, 0, 0, 2]],
    [[0, 2, 0, 1, 2], [1, 0, 2, 0, 1], [2, 1, 0, 2, 0], [0, 2, 1, 0, 2], [1, 0, 2, 1, 0]],
    [[1, 0, 1, 2, 1], [2, 1, 0, 0, 2], [0, 2, 1, 1, 0], [1, 0, 2, 0, 1], [2, 1, 0, 2, 1]],
    [[2, 1, 2, 0, 1], [0, 2, 0, 1, 2], [1, 0, 1, 2, 0], [2, 1, 2, 0, 1], [0, 2, 0, 1, 2]],
    [[0, 0, 1, 2, 0], [2, 1, 2, 0, 1], [1, 2, 0, 1, 2], [0, 1, 2, 0, 1], [2, 0, 1, 2, 0]],
    [[1, 2, 0, 1, 1], [0, 1, 2, 0, 2], [2, 0, 1, 2, 0], [1, 2, 0, 1, 1], [0, 1, 2, 0, 2]],
    [[2, 1, 1, 0, 2], [0, 2, 0, 1, 0], [1, 0, 2, 2, 1], [2, 1, 0, 0, 2], [0, 2, 1, 2, 0]],
    [[2, 0, 1, 1, 0], [1, 2, 0, 2, 1], [0, 1, 2, 0, 2], [1, 0, 1, 2, 0], [2, 1, 0, 1, 1]],
]
_ALL_OUTPUTS = [_grid_rule(g) for g in _ALL_INPUTS]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 15

_TEST_INPUTS = [
    [[1, 0, 2, 0, 1], [2, 1, 0, 1, 2], [0, 2, 1, 2, 0], [1, 0, 2, 0, 1], [2, 1, 0, 1, 2]],
    [[0, 1, 0, 2, 1], [1, 2, 1, 0, 2], [2, 0, 2, 1, 0], [0, 1, 0, 2, 1], [1, 2, 1, 0, 2]],
    [[2, 0, 1, 1, 0], [0, 1, 2, 0, 1], [1, 2, 0, 2, 2], [0, 0, 1, 1, 0], [2, 1, 0, 2, 1]],
    [[1, 2, 0, 1, 2], [0, 1, 2, 0, 1], [2, 0, 1, 2, 0], [1, 2, 0, 1, 2], [0, 1, 2, 0, 1]],
]
_TEST_EXPECTED = [_grid_rule(g) for g in _TEST_INPUTS]


def _fmt_grid(grid: list[list[int]]) -> str:
    return "\n".join("  " + " ".join(str(c) for c in row) for row in grid)


def _fmt_example(idx: int, inp: list[list[int]], out: list[list[int]]) -> str:
    return (
        f"Example {idx} INPUT:\n{_fmt_grid(inp)}\n"
        f"Example {idx} OUTPUT:\n{_fmt_grid(out)}"
    )


@dataclass
class _ConceptAction:
    action: str
    grid: list[list[int]]

@dataclass
class _ExamAnswers:
    grid_1: list[list[int]]
    grid_2: list[list[int]]
    grid_3: list[list[int]]
    grid_4: list[list[int]]


@kbench.task(
    name="grid_transform_concept_learning",
    description="Active concept formation: request examples to learn, then examine on 4 test grids. Score = accuracy * efficiency.",
)
def grid_transform_concept_learning(llm) -> float:
    """Active concept formation: infer compositional grid transform; request examples or enter 4-item examination. Score=accuracy*efficiency."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    initial_lines = [
        "You see input/output pairs of 5x5 grids with integers 0, 1, 2.",
        "One fixed transformation maps every INPUT to its OUTPUT.",
        "Infer the transformation from the examples.",
        "",
    ]
    for i in range(INITIAL_EXAMPLES):
        initial_lines.append(_fmt_example(i + 1, _ALL_INPUTS[i], _ALL_OUTPUTS[i]))
        initial_lines.append("")
    initial_lines += [
        "You have two actions:",
        "  action='request' — LEARN: get one more labeled example to study the rule (up to "
        f"{MAX_EXAMPLES} total)",
        "  action='submit'  — EXAMINE: enter examination mode where you will answer 4 test grids",
        "                     in a single response. No feedback, no going back.",
        "",
        f"You have seen {INITIAL_EXAMPLES} examples. {MAX_EXAMPLES - INITIAL_EXAMPLES} more are available.",
        "Your goal: study enough examples to confidently identify the transformation, then enter the examination.",
        "When you submit, you will answer 4 unseen test grids in a single response — make sure you have mastered the rule.",
        "Best scores go to models that need the fewest examples to answer all 4 correctly.",
    ]
    next_prompt = "\n".join(initial_lines)

    exam_results = []

    with kbench.chats.new("grid_transform"):
        for turn in range(1, MAX_EXAMPLES + 2):
            current_prompt = next_prompt
            try:
                sub = llm.prompt(current_prompt, schema=_ConceptAction)
            except Exception:
                entry = {"turn": turn, "action": "PARSE_ERROR", "prompt": current_prompt, "feedback": "Parse error — turn wasted."}
                turns.append(entry)
                next_prompt = "Parse error. Use action='request' or action='submit' with grid field."
                continue

            action = (sub.action or "").strip().lower()
            entry = {"turn": turn, "action": action, "prompt": current_prompt, "response": str(sub.grid)}

            if action == "request":
                if examples_shown >= MAX_EXAMPLES:
                    entry["feedback"] = "No more examples. You must submit to enter examination."
                    turns.append(entry)
                    next_prompt = (
                        "No more examples available. You must now enter examination mode.\n"
                        "action='submit' to begin the examination (grid field will be ignored for this action)."
                    )
                else:
                    new_ex = _fmt_example(examples_shown + 1, _ALL_INPUTS[examples_shown], _ALL_OUTPUTS[examples_shown])
                    examples_shown += 1
                    remaining = MAX_EXAMPLES - examples_shown
                    entry["feedback"] = f"Showed example {examples_shown}."
                    turns.append(entry)
                    next_prompt = (
                        f"{new_ex}\n\n"
                        f"You have seen {examples_shown} examples. {remaining} more available.\n\n"
                        "action='request' for another example or action='submit' to enter examination."
                    )
            elif action == "submit":
                entry["feedback"] = "Entering examination mode."
                turns.append(entry)
                break
            else:
                entry["feedback"] = "Unknown action."
                turns.append(entry)
                next_prompt = "Unknown action. Use action='request' or action='submit'."

        exam_lines = [
            "EXAMINATION — Apply the transformation to each of these 4 grids.",
            "Provide all output grids at once: grid_1 through grid_4 (each a 5x5 list of lists).",
            "",
        ]
        for i in range(NUM_TEST_ITEMS):
            exam_lines.append(f"Grid {i + 1} INPUT:\n{_fmt_grid(_TEST_INPUTS[i])}")
            exam_lines.append("")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_grids = [sub.grid_1, sub.grid_2, sub.grid_3, sub.grid_4]
        except Exception:
            raw_grids = [[], [], [], []]
        for test_idx in range(NUM_TEST_ITEMS):
            submitted = raw_grids[test_idx]
            correct = submitted == _TEST_EXPECTED[test_idx]
            exam_results.append({
                "item": test_idx + 1,
                "input": _TEST_INPUTS[test_idx],
                "expected": _TEST_EXPECTED[test_idx],
                "answer": submitted,
                "correct": correct,
            })

    exam_raw = [f"grid_{i+1}: {raw_grids[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("COMPOSITIONAL GRID TRANSFORM", turns, exam_results, final_score, examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    grid_transform_concept_learning.run(kbench.llm)

