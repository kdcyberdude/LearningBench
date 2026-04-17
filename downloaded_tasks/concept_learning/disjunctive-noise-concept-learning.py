#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Relational disjunctive rule with structured noise
# Rule: IN iff (weight > height) OR (color_code % 3 == 0)
# The 'material' attribute is irrelevant.
# Two training labels are structurally noisy (indices 4 and 11), chosen so
# that each noisy example satisfies exactly one of the two disjuncts,
# making the noise non-obvious and resistant to majority-vote heuristics.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "The model must infer a disjunctive classification rule for objects with four properties: "
    "weight (integer 1-9), height (integer 1-9), color_code (integer 1-12), and material (string). "
    "The rule is: IN iff (weight > height) OR (color_code % 3 == 0). "
    "The 'material' attribute is a red herring. "
    "Two training labels are intentionally noisy (indices 4 and 11 are mislabeled). "
    "The noise examples each satisfy exactly one disjunct with true label IN but are labeled OUT, "
    "making simple majority-vote or single-feature heuristics misleading. "
    "The model must infer the full relational-arithmetic rule from the examples "
    "and correctly classify 4 test items. Score = accuracy * efficiency."
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
FREE_THRESHOLD = 4


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


def _rule(weight: int, height: int, color_code: int) -> str:
    return "IN" if (weight > height) or (color_code % 3 == 0) else "OUT"


def _fmt(weight: int, height: int, color_code: int, material: str) -> str:
    return f"weight={weight}, height={height}, color_code={color_code}, material={material}"


# Training pool: 20 objects.
# Properties chosen so that:
#   - Neither single feature (weight, height, color_code alone) nor any simple
#     monotone threshold on one attribute is sufficient to separate IN/OUT.
#   - The irrelevant 'material' feature co-varies with the label in ~60% of
#     examples (spurious correlation), tempting feature-selection shortcuts.
#   - The two noise examples (indices 4 and 11) each satisfy exactly ONE
#     disjunct, so they look plausibly IN but are labeled OUT.
_ALL_INPUTS: list[tuple[int, int, int, str]] = [
    # (weight, height, color_code, material)
    (7, 3, 5,  "wood"),    # 0: w>h=T, cc%3=2 => IN  (correct)
    (2, 6, 9,  "metal"),   # 1: w>h=F, cc%3=0 => IN  (correct)
    (1, 5, 7,  "wood"),    # 2: w>h=F, cc%3=1 => OUT (correct)
    (4, 2, 11, "metal"),   # 3: w>h=T, cc%3=2 => IN  (correct)
    (3, 1, 4,  "plastic"), # 4: w>h=T, cc%3=1 => IN  (NOISY: labeled OUT)
    (5, 8, 2,  "wood"),    # 5: w>h=F, cc%3=2 => OUT (correct)
    (6, 4, 6,  "metal"),   # 6: w>h=T, cc%3=0 => IN  (correct)
    (1, 7, 10, "wood"),    # 7: w>h=F, cc%3=1 => OUT (correct)
    (8, 2, 1,  "plastic"), # 8: w>h=T, cc%3=1 => IN  (correct)
    (2, 9, 8,  "wood"),    # 9: w>h=F, cc%3=2 => OUT (correct)
    (5, 3, 12, "metal"),   #10: w>h=T, cc%3=0 => IN  (correct)
    (1, 4, 3,  "plastic"), #11: w>h=F, cc%3=0 => IN  (NOISY: labeled OUT)
    (7, 9, 5,  "wood"),    #12: w>h=F, cc%3=2 => OUT (correct)
    (6, 2, 7,  "metal"),   #13: w>h=T, cc%3=1 => IN  (correct)
    (3, 8, 9,  "wood"),    #14: w>h=F, cc%3=0 => IN  (correct)
    (4, 6, 2,  "plastic"), #15: w>h=F, cc%3=2 => OUT (correct)
    (9, 1, 4,  "metal"),   #16: w>h=T, cc%3=1 => IN  (correct)
    (2, 5, 11, "wood"),    #17: w>h=F, cc%3=2 => OUT (correct)
    (6, 3, 6,  "plastic"), #18: w>h=T, cc%3=0 => IN  (correct)
    (1, 8, 5,  "metal"),   #19: w>h=F, cc%3=2 => OUT (correct)
]
_ALL_OUTPUTS = [_rule(w, h, c) for w, h, c, _ in _ALL_INPUTS]
# Inject two structured noise labels: each noisy example satisfies exactly
# one disjunct (true label = IN) but is mislabeled as OUT.
_ALL_OUTPUTS[4]  = "OUT"  # (3>1)=T but cc%3=1≠0 — only the relational disjunct fires
_ALL_OUTPUTS[11] = "OUT"  # (1<4)=F but cc%3=0 — only the modular disjunct fires

MAX_EXAMPLES = 14
INITIAL_EXAMPLES = 4

# Test items: none appear in training; each tests a distinct combination of
# the two disjuncts to prevent partial-rule guessing.
# Test 0: both disjuncts fire          → IN
# Test 1: only weight>height fires     → IN
# Test 2: only color_code%3==0 fires   → IN
# Test 3: neither disjunct fires       → OUT
_TEST_INPUTS: list[tuple[int, int, int, str]] = [
    (8, 2, 6,  "wood"),    # w>h=T, cc%3=0 => IN
    (7, 4, 5,  "plastic"), # w>h=T, cc%3=2 => IN
    (2, 9, 12, "metal"),   # w>h=F, cc%3=0 => IN
    (3, 6, 8,  "wood"),    # w>h=F, cc%3=2 => OUT
]
_TEST_EXPECTED = [_rule(w, h, c) for w, h, c, _ in _TEST_INPUTS]


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
    name="disjunctive_noise_concept_learning",
    description="Active concept formation: request examples to learn a relational-arithmetic disjunctive rule from noisy labels, then examine on 4 test items. Score = accuracy * efficiency.",
)
def disjunctive_noise_concept_learning(llm) -> float:
    """Active concept formation: infer relational-arithmetic disjunctive IN/OUT rule from noisy labeled objects; request examples or enter 4-item examination. Score=accuracy*efficiency."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    initial_lines = [
        "Objects have four properties: weight (integer 1-9), height (integer 1-9),",
        "color_code (integer 1-12), and material (stone/wood/metal/plastic).",
        "Some objects are IN a category and some are OUT.",
        "The labeling may contain a small number of errors (noise).",
        "Infer the most likely underlying rule despite possible noise.",
        "",
        "Labeled examples:",
    ]
    for i in range(INITIAL_EXAMPLES):
        w, h, c, m = _ALL_INPUTS[i]
        initial_lines.append(f"  Example {i + 1}: {_fmt(w, h, c, m)} -> {_ALL_OUTPUTS[i]}")
    initial_lines += [
        "",
        "You have two actions:",
        "  action='request' — LEARN: get one more labeled example (up to "
        f"{MAX_EXAMPLES} total)",
        "  action='submit'  — EXAMINE: enter examination mode where you will answer 4 test items",
        "                     in a single response. No feedback, no going back.",
        "",
        f"You have seen {INITIAL_EXAMPLES} examples. {MAX_EXAMPLES - INITIAL_EXAMPLES} more are available.",
        "Your goal: study enough examples to confidently identify the rule, then enter the examination.",
        "When you submit, you will answer 4 unseen test items in a single response — make sure you have mastered the rule.",
        "Best scores go to models that need the fewest examples to answer all 4 correctly.",
    ]
    next_prompt = "\n".join(initial_lines)

    exam_results = []

    with kbench.chats.new("disjunctive_noise"):
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
                    w, h, c, m = _ALL_INPUTS[idx]
                    ex_line = f"Example {idx + 1}: {_fmt(w, h, c, m)} -> {_ALL_OUTPUTS[idx]}"
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
            "EXAMINATION — Classify each of these 4 items as IN or OUT.",
            "Provide all answers at once: answer_1 through answer_4.",
            "",
        ]
        for i in range(NUM_TEST_ITEMS):
            w, h, c, m = _TEST_INPUTS[i]
            exam_lines.append(f"  Item {i + 1}: {_fmt(w, h, c, m)}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for test_idx in range(NUM_TEST_ITEMS):
            answer = (raw_answers[test_idx] or "").strip().upper()
            correct = _str_match(_TEST_EXPECTED[test_idx], answer)
            w, h, c, m = _TEST_INPUTS[test_idx]
            exam_results.append({
                "item": test_idx + 1,
                "input": _fmt(w, h, c, m),
                "expected": _TEST_EXPECTED[test_idx],
                "answer": answer,
                "correct": correct,
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("RELATIONAL DISJUNCTIVE RULE WITH STRUCTURED NOISE", turns, exam_results, final_score, examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    disjunctive_noise_concept_learning.run(kbench.llm)

