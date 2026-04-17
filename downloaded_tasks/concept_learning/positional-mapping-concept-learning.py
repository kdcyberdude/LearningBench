#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Cross-modal positional mapping
# Rule: YES iff the 1-based index of the single uppercase letter equals the trailing integer.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "The model must infer a cross-modal rule relating letter case and a trailing number in a mixed string. "
    "The rule is: YES iff the 1-based index of the single uppercase letter among all letters equals the trailing integer. "
    "The model actively requests labeled string examples and must submit YES or NO for test strings. "
    "What makes it hard is that the model must simultaneously track letter positions (ignoring the number) "
    "and connect that positional fact to the number — a cross-modal relationship spanning two domains. "
    "Success means correctly classifying 4 test strings in the examination phase."
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

def _rule(seq: str) -> str:
    tokens = seq.split()
    letters = [t for t in tokens if t.isalpha()]
    number = int(tokens[-1])
    for idx, t in enumerate(letters, 1):
        if t.isupper():
            return "YES" if idx == number else "NO"
    return "NO"

_ALL_INPUTS = [
    "a b C d e 3",
    "x y z W 4",
    "A b c 2",
    "a b c D 3",
    "q r s t U v 5",
    "m N o p 2",
    "a b c d e f G h 8",
    "Z a b c 1",
    "a b c D e f g 4",
    "x y Z a b 2",
    "m n o p q R s t 6",
    "H i j k 1",
    "a B c d e 3",
    "p q R s t u 3",
    "a b c d E 5",
    "w x Y z 3",
    "A b c d e f 1",
    "i j k L m n 4",
    "a b c d e F g 6",
    "v w x y Z 5",
]
_ALL_OUTPUTS = [_rule(s) for s in _ALL_INPUTS]

MAX_EXAMPLES = 16
INITIAL_EXAMPLES = 0

_TEST_INPUTS = ["a b C d e f 3", "x Y z w 2", "m n o P q 5", "A b c d 1"]
_TEST_EXPECTED = [_rule(s) for s in _TEST_INPUTS]

@dataclass
class _ConceptAction:
    action: str
    answer: str


@dataclass
class _ExamAnswers:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str

def _str_match(expected: str, actual: str) -> bool:
    """Return True if expected appears anywhere in actual (case-insensitive)."""
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))

@kbench.task(
    name="positional_mapping_concept_learning",
    description="Active concept formation: request examples to learn, then examine on 4 test strings. Score = accuracy * efficiency.",
)
def positional_mapping_concept_learning(llm) -> float:
    """Active concept formation: infer YES/NO rule linking uppercase letter position and trailing number; request examples or enter 4-item examination. Score=accuracy*efficiency."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    initial_lines = [
        "A set of strings containing letters and one trailing number are classified as YES or NO.",
        "The rule involves a relationship between the properties of the letters and the number.",
        "",
        "Labeled examples:",
    ]
    for i in range(INITIAL_EXAMPLES):
        initial_lines.append(f"  Example {i + 1}: '{_ALL_INPUTS[i]}' -> {_ALL_OUTPUTS[i]}")
    initial_lines += [
        "",
        "You have two actions:",
        "  action='request' — LEARN: get one more labeled example to study the rule (up to "
        f"{MAX_EXAMPLES} total)",
        "  action='submit'  — EXAMINE: enter examination mode where you will answer 4 test strings",
        "                     in a single response. No feedback, no going back.",
        "",
        f"You have seen {INITIAL_EXAMPLES} examples. {MAX_EXAMPLES - INITIAL_EXAMPLES} more are available.",
        "Your goal: study enough examples to confidently identify the rule, then enter the examination.",
        "When you submit, you will answer 4 unseen test strings in a single response — make sure you have mastered the rule.",
        "Best scores go to models that need the fewest examples to answer all 4 correctly.",
    ]
    next_prompt = "\n".join(initial_lines)

    exam_results = []

    with kbench.chats.new("positional_mapping"):
        for turn in range(1, MAX_EXAMPLES + 2):
            current_prompt = next_prompt
            try:
                sub = llm.prompt(current_prompt, schema=_ConceptAction)
            except Exception:
                entry = {"turn": turn, "action": "PARSE_ERROR", "prompt": current_prompt, "feedback": "Parse error — turn wasted."}
                turns.append(entry)
                next_prompt = "Parse error. Use action='request' or action='submit' with answer field."
                continue

            action = (sub.action or "").strip().lower()
            entry = {"turn": turn, "action": action, "prompt": current_prompt, "response": (sub.answer or "").strip()}

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
                    ex_line = f"Example {idx + 1}: '{_ALL_INPUTS[idx]}' -> {_ALL_OUTPUTS[idx]}"
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
            "EXAMINATION — Classify each of these 4 strings as YES or NO.",
            "Provide all answers at once: answer_1 through answer_4.",
            "",
        ]
        for i in range(NUM_TEST_ITEMS):
            exam_lines.append(f"  String {i + 1}: '{_TEST_INPUTS[i]}'")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for test_idx in range(NUM_TEST_ITEMS):
            answer = (raw_answers[test_idx] or "").strip()
            correct = _str_match(_TEST_EXPECTED[test_idx], answer)
            exam_results.append({
                "item": test_idx + 1,
                "input": _TEST_INPUTS[test_idx],
                "expected": _TEST_EXPECTED[test_idx],
                "answer": answer,
                "correct": correct,
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("CROSS MODAL POSITIONAL MAPPING", turns, exam_results, final_score, examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score

if __name__ == "__main__":
    positional_mapping_concept_learning.run(kbench.llm)

