#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Exact violation counter
# Rule: Each element after the first is valid if value > previous OR value == 1. Count violations.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "The model must infer a sequential rule with an exception and count violations in integer sequences. "
    "The rule is: each element after the first is valid if it is strictly greater than its predecessor OR equals 1; "
    "a violation occurs when neither condition holds. The output is the count of violations. "
    "The model actively requests labeled sequence examples and must submit the correct violation count for 4 test sequences. "
    "What makes it hard is the exception clause (value == 1 is always valid regardless of predecessor), "
    "which interferes with the naive 'must be increasing' hypothesis. "
    "Success means correctly counting violations for 4 test sequences in the examination phase."
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

NUM_TEST_ITEMS = 4
FREE_THRESHOLD = 8


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


def _rule(seq: list[int]) -> int:
    count = 0
    for i in range(1, len(seq)):
        ok = seq[i] > seq[i - 1] or seq[i] == 1
        if not ok:
            count += 1
    return count


_ALL_INPUTS = [
    [1, 3, 5, 2, 7],
    [1, 4, 4, 6, 8],
    [2, 5, 1, 3, 4],
    [3, 2, 4, 3, 5],
    [1, 5, 3, 7, 2, 9],
    [2, 4, 6, 1, 3, 5],
    [5, 3, 7, 2, 1, 8],
    [4, 2, 1, 6, 3],
    [7, 8, 1, 5, 2],
    [1, 1, 1, 1, 1],
    [6, 5, 4, 3, 2],
    [2, 3, 2, 3, 2],
    [10, 1, 5, 1, 8],
    [1, 9, 7, 1, 6],
    [3, 4, 5, 2, 6],
    [8, 1, 3, 1, 5],
    [2, 6, 4, 8, 3],
    [1, 2, 1, 3, 1],
    [5, 7, 3, 9, 2],
    [4, 6, 1, 8, 5],
]
_ALL_OUTPUTS = [_rule(s) for s in _ALL_INPUTS]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 10

_TEST_INPUTS = [[3, 5, 1, 4, 2, 7], [8, 1, 6, 3, 9, 2], [1, 2, 3, 1, 5, 4], [4, 7, 2, 1, 8, 3]]
_TEST_EXPECTED = [_rule(s) for s in _TEST_INPUTS]


@dataclass
class _ConceptAction:
    action: str
    answer: int

@dataclass
class _ExamAnswers:
    answer_1: int
    answer_2: int
    answer_3: int
    answer_4: int


@kbench.task(
    name="violation_counter_concept_learning",
    description="Active concept formation: request examples to learn, then examine on 4 test sequences. Score = accuracy * efficiency.",
)
def violation_counter_concept_learning(llm) -> float:
    """Active concept formation: infer sequential validity rule; request examples or enter 4-item examination. Score=accuracy*efficiency."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    initial_lines = [
        "Sequences of numbers follow a rule. Some elements VIOLATE the rule.",
        "Count the number of violations in each sequence.",
        "",
        "Labeled examples (sequence -> violation count):",
    ]
    for i in range(INITIAL_EXAMPLES):
        initial_lines.append(f"  Example {i + 1}: {_ALL_INPUTS[i]} -> {_ALL_OUTPUTS[i]} violation(s)")
    initial_lines += [
        "",
        "You have two actions:",
        "  action='request' — LEARN: get one more labeled example to study the rule (up to "
        f"{MAX_EXAMPLES} total)",
        "  action='submit'  — EXAMINE: enter examination mode where you will answer 4 test sequences",
        "                     in a single response. No feedback, no going back.",
        "",
        f"You have seen {INITIAL_EXAMPLES} examples. {MAX_EXAMPLES - INITIAL_EXAMPLES} more are available.",
        "Your goal: study enough examples to confidently identify the violation rule, then enter the examination.",
        "When you submit, you will answer 4 unseen test sequences in a single response — make sure you have mastered the rule.",
        "Best scores go to models that need the fewest examples to answer all 4 correctly.",
    ]
    next_prompt = "\n".join(initial_lines)

    exam_results = []

    with kbench.chats.new("violation_counter"):
        for turn in range(1, MAX_EXAMPLES + 2):
            current_prompt = next_prompt
            try:
                sub = llm.prompt(current_prompt, schema=_ConceptAction)
            except Exception:
                entry = {"turn": turn, "action": "PARSE_ERROR", "prompt": current_prompt, "feedback": "Parse error — turn wasted."}
                turns.append(entry)
                next_prompt = "Parse error. Use action='request' or action='submit' with integer answer field."
                continue

            action = (sub.action or "").strip().lower()
            entry = {"turn": turn, "action": action, "prompt": current_prompt, "response": str(sub.answer)}

            if action == "request":
                if examples_shown >= MAX_EXAMPLES:
                    entry["feedback"] = "No more examples. You must submit to enter examination."
                    turns.append(entry)
                    next_prompt = (
                        "No more examples available. You must now enter examination mode.\n"
                        "action='submit' to begin the examination (answer field will be ignored for this action)."
                    )
                else:
                    idx = examples_shown
                    ex_line = f"Example {idx + 1}: {_ALL_INPUTS[idx]} -> {_ALL_OUTPUTS[idx]} violation(s)"
                    examples_shown += 1
                    remaining = MAX_EXAMPLES - examples_shown
                    entry["feedback"] = f"Showed example {examples_shown}."
                    turns.append(entry)
                    next_prompt = (
                        f"{ex_line}\n\n"
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
            "EXAMINATION — Count violations for each of these 4 sequences.",
            "Provide all violation counts at once: answer_1 through answer_4 (integers).",
            "",
        ]
        for i in range(NUM_TEST_ITEMS):
            exam_lines.append(f"  Sequence {i + 1}: {_TEST_INPUTS[i]}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = [-1, -1, -1, -1]
        for test_idx in range(NUM_TEST_ITEMS):
            answer = raw_answers[test_idx]
            expected = _TEST_EXPECTED[test_idx]
            correct = answer == expected
            exam_results.append({
                "item": test_idx + 1,
                "input": _TEST_INPUTS[test_idx],
                "expected": expected,
                "answer": answer,
                "correct": correct,
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("EXACT VIOLATION COUNTER", turns, exam_results, final_score, examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    violation_counter_concept_learning.run(kbench.llm)

