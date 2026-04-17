#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Positional affine digit cipher
# Rule: For digit d at 0-indexed position i (left-to-right),
#       letter = ALPHA[(d * 3 + i * 7) % 26]
#       Multi-digit integers are encoded position-by-position and concatenated.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

import re
import string
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "The model must crack a positional affine cipher applied to integers. "
    "The rule is: for digit d at 0-indexed position i (left to right), "
    "the encoded letter is ALPHA[(d * 3 + i * 7) % 26] where ALPHA is the standard "
    "26-letter alphabet (a=0, b=1, …, z=25). "
    "Multi-digit integers are encoded digit by digit and concatenated. "
    "Crucially, the same digit encodes to DIFFERENT letters depending on its position, "
    "so a simple digit→letter lookup table never works. "
    "The model must infer BOTH unknown coefficients (3 and 7) and the modular-alphabet "
    "structure from the examples. "
    "What makes it hard: (1) positional dependence must be discovered — not assumed; "
    "(2) aliasing (different digits at different positions can map to the same letter) "
    "creates false patterns; (3) two independent structural parameters must be jointly "
    "inferred from multi-digit examples where digit identity and position co-vary. "
    "Success means correctly encoding 4 test numbers in the examination phase."
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

_ALPHA = string.ascii_lowercase


def _rule(n: int) -> str:
    return "".join(
        _ALPHA[(int(d) * 3 + i * 7) % 26] for i, d in enumerate(str(n))
    )


_ALL_INPUTS = [
    17395,
    84026,
    53971,   # encodes to 'pqpqf' — digit 5 at pos 0 and digit 9 at pos 2 both → 'p'
    60248,
    20563,
    91748,
    35402,
    76819,
    42537,
    98304,
    10293,
    65481,
    27046,
    83915,
    49260,
    71832,
    56047,
    30289,
    14756,
    92038,
]
_ALL_OUTPUTS = [_rule(n) for n in _ALL_INPUTS]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 3

_TEST_INPUTS = [48163, 72905, 31684, 50297]
_TEST_EXPECTED = [_rule(n) for n in _TEST_INPUTS]


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
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


@kbench.task(
    name="digit_cipher_concept_learning",
    description="Active concept formation: request examples to learn a positional affine cipher, then examine on 4 test numbers. Score = accuracy * efficiency.",
)
def digit_cipher_concept_learning(llm) -> float:
    """Active concept formation: infer a positional affine cipher from examples; request more or enter 4-item examination. Score=accuracy*efficiency."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    initial_lines = [
        "An encoding rule converts integers into lowercase letter strings.",
        "Each digit of the integer is independently encoded to a single letter,",
        "and the letters are concatenated left to right.",
        "",
        "Labeled examples:",
    ]
    for i in range(INITIAL_EXAMPLES):
        initial_lines.append(
            f"  Example {i + 1}: {_ALL_INPUTS[i]} -> '{_ALL_OUTPUTS[i]}'"
        )
    initial_lines += [
        "",
        "You have two actions:",
        "  action='request' — LEARN: get one more labeled example (up to "
        f"{MAX_EXAMPLES} total)",
        "  action='submit'  — EXAMINE: enter examination mode where you will answer 4 test numbers",
        "                     in a single response. No feedback, no going back.",
        "",
        f"You have seen {INITIAL_EXAMPLES} examples. {MAX_EXAMPLES - INITIAL_EXAMPLES} more are available.",
        "Your goal: study enough examples to confidently identify the rule, then enter the examination.",
        "When you submit, you will answer 4 unseen test numbers in a single response — make sure you have mastered the rule.",
        "Best scores go to models that need the fewest examples to answer all 4 correctly.",
    ]
    next_prompt = "\n".join(initial_lines)

    exam_results = []

    with kbench.chats.new("digit_cipher"):
        for turn in range(1, MAX_EXAMPLES + 2):
            current_prompt = next_prompt
            try:
                sub = llm.prompt(current_prompt, schema=_ConceptAction)
            except Exception:
                entry = {
                    "turn": turn,
                    "action": "PARSE_ERROR",
                    "prompt": current_prompt,
                    "feedback": "Parse error — turn wasted.",
                }
                turns.append(entry)
                next_prompt = (
                    "Parse error. Use action='request' or action='submit' "
                    "with the answer field."
                )
                continue

            action = (sub.action or "").strip().lower()
            entry = {"turn": turn, "action": action, "prompt": current_prompt, "response": (sub.answer or "").strip()}

            if action == "request":
                if examples_shown >= MAX_EXAMPLES:
                    entry["feedback"] = "No more examples. You must submit to enter examination."
                    turns.append(entry)
                    next_prompt = (
                        "No more examples available. You must now enter examination mode.\n"
                        "Use action='submit' to begin "
                        "(the answer field will be ignored for this action)."
                    )
                else:
                    idx = examples_shown
                    ex_line = (
                        f"Example {idx + 1}: "
                        f"{_ALL_INPUTS[idx]} -> '{_ALL_OUTPUTS[idx]}'"
                    )
                    examples_shown += 1
                    remaining = MAX_EXAMPLES - examples_shown
                    entry["feedback"] = f"Showed example {examples_shown}."
                    turns.append(entry)
                    next_prompt = (
                        f"{ex_line}\n\n"
                        f"You have seen {examples_shown} examples. "
                        f"{remaining} more available.\n\n"
                        "action='request' for another example or "
                        "action='submit' to enter examination."
                    )
            elif action == "submit":
                entry["feedback"] = "Entering examination mode."
                turns.append(entry)
                break
            else:
                entry["feedback"] = "Unknown action."
                turns.append(entry)
                next_prompt = (
                    "Unknown action. Use action='request' or action='submit'."
                )

        exam_lines = [
            "EXAMINATION — Encode each of these 4 numbers using the rule you have learned.",
            "Provide all answers at once: answer_1 through answer_4.",
            "",
        ]
        for i in range(NUM_TEST_ITEMS):
            exam_lines.append(f"  Number {i + 1}: {_TEST_INPUTS[i]}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for test_idx in range(NUM_TEST_ITEMS):
            answer = (raw_answers[test_idx] or "").strip()
            correct = _str_match(_TEST_EXPECTED[test_idx], answer)
            exam_results.append(
                {
                    "item": test_idx + 1,
                    "input": _TEST_INPUTS[test_idx],
                    "expected": _TEST_EXPECTED[test_idx],
                    "answer": answer,
                    "correct": correct,
                }
            )

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace(
        "POSITIONAL AFFINE DIGIT CIPHER",
        turns,
        exam_results,
        final_score,
        examples_shown,
        exam_prompt=exam_prompt,
        exam_raw=exam_raw,
    )
    return final_score


if __name__ == "__main__":
    digit_cipher_concept_learning.run(kbench.llm)

