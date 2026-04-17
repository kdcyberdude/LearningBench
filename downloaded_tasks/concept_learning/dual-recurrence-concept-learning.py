#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Interleaved dual recurrence (mod 17)
# Rule: A(n) = (A(n-1) + B(n-1)) % 17; B(n) = (A(n-2) * B(n-2)) % 17; A(0)=3, A(1)=5, B(0)=2, B(1)=7.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "The model must infer two coupled recurrence sequences evolving together modulo 17. "
    "The rules are: A(n) = (A(n-1) + B(n-1)) mod 17, and B(n) = (A(n-2) * B(n-2)) mod 17, "
    "with seeds A(0)=3, A(1)=5, B(0)=2, B(1)=7. "
    "The model actively requests observed term pairs and must correctly answer 4 examination "
    "questions asking for specific A(n) and B(m) values as integers. "
    "What makes it hard is the coupled, interleaved nature: A depends on B and B depends on the prior A, "
    "with different lag offsets, all under mod-17 arithmetic. "
    "Success means correctly computing 4 test pairs in the examination phase."
)

NUM_TEST_ITEMS = 4
FREE_THRESHOLD = 5


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
        print(f"  Test {r['item']}: {status}   A({r['a_idx']})={r['expected_a']} got={r['answer_a']}   B({r['b_idx']})={r['expected_b']} got={r['answer_b']}")
    correct = sum(1 for r in exam_results if r["correct"])
    print(f"\n  Examples used : {examples_used}/{MAX_EXAMPLES}")
    print(f"  Exam accuracy : {correct}/{len(exam_results)}")
    print(f"  Final score   : {final_score:.4f}")
    print(f"{sep}\n")

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


def _build_sequences(n: int) -> tuple[list[int], list[int]]:
    a = [3, 5]
    b = [2, 7]
    for _ in range(2, n + 1):
        a.append((a[-1] + b[-1]) % 17)
        b.append((a[-2] * b[-2]) % 17)
    return a, b


_SEQ_A, _SEQ_B = _build_sequences(20)

_ALL_INPUTS = list(range(20))
_ALL_OUTPUTS = [(f"A({i})={_SEQ_A[i]}", f"B({i})={_SEQ_B[i]}") for i in range(20)]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 4

_TEST_INPUTS = [(13, 15), (10, 12), (8, 9), (14, 16)]
_TEST_EXPECTED = [(_SEQ_A[a_idx], _SEQ_B[b_idx]) for a_idx, b_idx in _TEST_INPUTS]


@dataclass
class _ConceptAction:
    action: str
    answer_a: int
    answer_b: int

@dataclass
class _ExamAnswers:
    answer_a_1: int
    answer_b_1: int
    answer_a_2: int
    answer_b_2: int
    answer_a_3: int
    answer_b_3: int
    answer_a_4: int
    answer_b_4: int


@kbench.task(
    name="dual_recurrence_concept_learning",
    description="Active concept formation: request term pairs to learn, then examine on 4 test pairs. Score = accuracy * efficiency.",
)
def dual_recurrence_concept_learning(llm) -> float:
    """Active concept formation: infer coupled mod-17 recurrence; request examples or enter 4-item examination. Score=accuracy*efficiency."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    def _fmt_ex(i: int) -> str:
        a_s, b_s = _ALL_OUTPUTS[i]
        return f"  Step {i + 1}: {a_s},  {b_s}"

    initial_lines = [
        "Two sequences A and B evolve together according to hidden coupled rules.",
        "A(0)=3, A(1)=5, B(0)=2, B(1)=7.",
        "",
        "Observed terms so far:",
    ]
    for i in range(INITIAL_EXAMPLES):
        initial_lines.append(_fmt_ex(i))
    initial_lines += [
        "",
        "You have two actions:",
        "  action='request' — LEARN: get the next pair of terms to study the rule (up to "
        f"{MAX_EXAMPLES} total)",
        "  action='submit'  — EXAMINE: enter examination mode where you will answer 4 test pairs",
        "                     in a single response. No feedback, no going back.",
        "",
        f"You have seen {INITIAL_EXAMPLES} term pairs. {MAX_EXAMPLES - INITIAL_EXAMPLES} more are available.",
        "Your goal: study enough terms to confidently identify the recurrence rules, then enter the examination.",
        "When you submit, you will answer 4 unseen test pairs in a single response — make sure you have mastered both rules.",
        "Best scores go to models that need the fewest examples to answer all 4 correctly.",
    ]
    next_prompt = "\n".join(initial_lines)

    exam_results = []

    with kbench.chats.new("dual_recurrence"):
        for turn in range(1, MAX_EXAMPLES + 2):
            current_prompt = next_prompt
            try:
                sub = llm.prompt(current_prompt, schema=_ConceptAction)
            except Exception:
                entry = {"turn": turn, "action": "PARSE_ERROR", "prompt": current_prompt, "feedback": "Parse error — turn wasted."}
                turns.append(entry)
                next_prompt = "Parse error. Use action='request' or action='submit' with answer_a/answer_b fields."
                continue

            action = (sub.action or "").strip().lower()
            entry = {"turn": turn, "action": action, "prompt": current_prompt, "response": f"a={sub.answer_a}, b={sub.answer_b}"}

            if action == "request":
                if examples_shown >= MAX_EXAMPLES:
                    entry["feedback"] = "No more terms. You must submit to enter examination."
                    turns.append(entry)
                    next_prompt = (
                        "No more terms available. You must now enter examination mode.\n"
                        "action='submit' to begin the examination (answer fields will be ignored for this action)."
                    )
                else:
                    idx = examples_shown
                    ex_line = _fmt_ex(idx)
                    examples_shown += 1
                    remaining = MAX_EXAMPLES - examples_shown
                    entry["feedback"] = f"Showed term pair {examples_shown}."
                    turns.append(entry)
                    next_prompt = (
                        f"{ex_line}\n\n"
                        f"You have seen {examples_shown} term pairs. {remaining} more available.\n\n"
                        "action='request' for another term pair or action='submit' to enter examination."
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
            "EXAMINATION — Compute each of these 4 requested values using the rules you have learned.",
            "Provide all answers at once: answer_a_1/answer_b_1 through answer_a_4/answer_b_4 (integers).",
            "",
        ]
        for i, (a_idx, b_idx) in enumerate(_TEST_INPUTS, 1):
            exam_lines.append(f"  Pair {i}: A({a_idx}) = ? and B({b_idx}) = ?")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw = [
                (sub.answer_a_1, sub.answer_b_1),
                (sub.answer_a_2, sub.answer_b_2),
                (sub.answer_a_3, sub.answer_b_3),
                (sub.answer_a_4, sub.answer_b_4),
            ]
        except Exception:
            raw = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
        for test_idx in range(NUM_TEST_ITEMS):
            a_idx, b_idx = _TEST_INPUTS[test_idx]
            expected_a, expected_b = _TEST_EXPECTED[test_idx]
            ans_a, ans_b = raw[test_idx]
            correct = ans_a == expected_a and ans_b == expected_b
            exam_results.append({
                "item": test_idx + 1,
                "a_idx": a_idx,
                "b_idx": b_idx,
                "expected_a": expected_a,
                "expected_b": expected_b,
                "answer_a": ans_a,
                "answer_b": ans_b,
                "correct": correct,
            })

    exam_raw = [f"answer_a_{i+1}: {raw[i][0]}  answer_b_{i+1}: {raw[i][1]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("INTERLEAVED DUAL RECURRENCE", turns, exam_results, final_score, examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    dual_recurrence_concept_learning.run(kbench.llm)

