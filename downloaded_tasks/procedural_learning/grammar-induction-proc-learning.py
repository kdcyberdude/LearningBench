#!/usr/bin/env python
# coding: utf-8

import itertools as _itertools
import re as _re
from dataclasses import dataclass

import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests procedural learning of hidden context-free grammars (palindromes over {a,b} or a^n b^n strings). "
    "The model tests strings for membership and must submit a regex that perfectly captures the language "
    "across 5 grammar instances with a limited probe budget. "
    "What makes it hard is that both grammars require structural reasoning beyond regular-expression power, "
    "forcing the model to discover the generative pattern from membership queries. "
    "Success = 50% learning efficiency + 50% correct final test."
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


BUDGET = 14


def _accepts_palindrome(s: str) -> bool:
    return all(c in "ab" for c in s) and s == s[::-1]


def _accepts_anbn(s: str) -> bool:
    if not s:
        return True
    if not all(c in "ab" for c in s):
        return False
    n = len(s)
    if n % 2 != 0:
        return False
    half = n // 2
    return s[:half] == "a" * half and s[half:] == "b" * half


LEARNING_GRAMMARS = [
    {"name": "palindromes over {a,b}", "accepts": _accepts_palindrome, "alphabet": "ab"},
    {"name": "a^n b^n strings", "accepts": _accepts_anbn, "alphabet": "ab"},
    {"name": "palindromes over {a,b}", "accepts": _accepts_palindrome, "alphabet": "ab"},
    {"name": "a^n b^n strings", "accepts": _accepts_anbn, "alphabet": "ab"},
    {"name": "palindromes over {a,b}", "accepts": _accepts_palindrome, "alphabet": "ab"},
]
TEST_GRAMMAR = {"name": "a^n b^n strings", "accepts": _accepts_anbn, "alphabet": "ab"}


def _gen_test_set(alphabet: str, max_len: int, accepts_fn) -> list[str]:
    strings = []
    for length in range(0, max_len + 1):
        for chars in _itertools.product(alphabet, repeat=length):
            strings.append("".join(chars))
    return strings[:100]


def _check_pattern(pattern: str, test_set: list[str], accepts_fn) -> tuple[bool, int, int]:
    try:
        if pattern.startswith("^"):
            compiled = _re.compile(pattern + "$" if not pattern.endswith("$") else pattern)
        else:
            compiled = _re.compile("^" + pattern + "$")
    except _re.error:
        return False, 0, len(test_set)
    correct = sum(
        1 for s in test_set
        if bool(accepts_fn(s)) == bool(compiled.match(s))
    )
    return correct == len(test_set), correct, len(test_set)


@dataclass
class _GrammarAction:
    action: str      # "test" or "submit"
    string: str      # string to test; "" when action="submit"
    pattern: str     # regex pattern; "" when action="test"


@kbench.task(
    name="grammar_induction_proc_learning",
    description=(
        "Test strings against a hidden context-free grammar and submit a regex pattern that captures "
        "the language, across 5 grammar instances. Score = learning_efficiency×0.5 + test_pass×0.5."
    ),
)
def grammar_induction_proc_learning(llm) -> float:
    """5 practice grammar instances (palindromes/anbn), test strings and submit regex, then 1 no-hint test. Score=learning_avg×0.5+test×0.5."""
    phases = []
    test_passed = False

    with kbench.chats.new("grammar_induction"):
        learning_scores = []
        initial_prompt = ""

        for idx, grammar in enumerate(LEARNING_GRAMMARS):
            grammar_name = grammar["name"]
            accepts_fn = grammar["accepts"]
            alphabet = grammar["alphabet"]
            test_set = _gen_test_set(alphabet, 6, accepts_fn)

            turns = []
            solved = False
            num_steps = 0

            intro = (
                "A hidden context-free grammar generates strings over an alphabet.\n"
                "Actions:\n"
                "  action='test',   string='ab'  → 'MEMBER' or 'NOT_MEMBER'\n"
                "  action='submit', pattern='regex'  → graded against 100 test strings\n"
                "Grading: your regex must agree with the hidden grammar on all test strings.\n"
                "After 5 practice instances you face a final instance.\n\n"
            ) if idx == 0 else ""

            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 — Hidden grammar over alphabet {{{', '.join(alphabet)}}}.\n"
                f"Budget: {BUDGET} actions. Attempt 1 of {BUDGET}.\n"
                "Test a string or submit your regex pattern."
            )

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                if idx == 0 and turn == 1:
                    initial_prompt = next_prompt
                try:
                    submission = llm.prompt(next_prompt, schema=_GrammarAction)
                except Exception:
                    entry = {"turn": turn, "submitted": "PARSE_ERROR", "feedback": "Failed to parse response — turn wasted."}
                    turns.append(entry)
                    next_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Test more strings or submit a revised pattern."
                    continue

                action = submission.action.strip().lower()
                entry = {"turn": turn, "submitted": action}

                if action == "test":
                    s = submission.string
                    if len(s) > 10:
                        s = s[:10]
                    result = "MEMBER" if accepts_fn(s) else "NOT_MEMBER"
                    entry["submitted"] = f"test '{s}'"
                    feedback = result
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Test another string or submit your pattern."
                    )
                elif action == "submit":
                    pattern = submission.pattern
                    entry["submitted"] = f"submit pattern='{pattern}'"
                    ok, correct_count, total = _check_pattern(pattern, test_set, accepts_fn)
                    if ok:
                        solved = True
                        turns.append(entry)
                        break
                    feedback = f"WRONG ({correct_count}/{total} correct)"
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Test more strings or submit a revised pattern."
                    )
                else:
                    entry["submitted"] = f"unknown action: {action}"
                    entry["feedback"] = "Use action='test' or action='submit'."
                    turns.append(entry)
                    next_prompt = (
                        f"Unknown action. Use 'test' or 'submit'.\n\n"
                        f"Attempt {turn + 1} of {BUDGET}."
                    )

            eff = _efficiency_score(solved, num_steps, BUDGET)
            learning_scores.append(eff)
            phases.append({
                "label": f"Practice {idx + 1}/5",
                "correct": grammar_name,
                "turns": turns,
                "solved": solved,
                "steps": num_steps,
                "score": eff,
            })

        test_grammar = TEST_GRAMMAR
        test_accepts = test_grammar["accepts"]
        test_alphabet = test_grammar["alphabet"]
        test_set = _gen_test_set(test_alphabet, 6, test_accepts)
        test_turns = []
        test_solved = False
        test_num_steps = 0

        test_prompt = (
            f"Final test — Hidden grammar over alphabet {{{', '.join(test_alphabet)}}}.\n"
            f"Budget: {BUDGET} actions. Attempt 1 of {BUDGET}.\n"
            "This is the final instance. Test strings or submit your regex pattern."
        )

        for turn in range(1, BUDGET + 1):
            test_num_steps = turn
            try:
                test_submission = llm.prompt(test_prompt, schema=_GrammarAction)
            except Exception:
                entry = {"turn": turn, "submitted": "PARSE_ERROR", "feedback": "Failed to parse response — turn wasted."}
                test_turns.append(entry)
                test_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Test more strings or submit a revised pattern."
                continue

            action = test_submission.action.strip().lower()
            entry = {"turn": turn, "submitted": action}

            if action == "test":
                s = test_submission.string
                if len(s) > 10:
                    s = s[:10]
                result = "MEMBER" if test_accepts(s) else "NOT_MEMBER"
                entry["submitted"] = f"test '{s}'"
                feedback = result
                entry["feedback"] = feedback
                test_turns.append(entry)
                test_prompt = (
                    f"{feedback}\n\n"
                    f"Attempt {turn + 1} of {BUDGET}. Test another string or submit your pattern."
                )
            elif action == "submit":
                pattern = test_submission.pattern
                entry["submitted"] = f"submit pattern='{pattern}'"
                ok, correct_count, total = _check_pattern(pattern, test_set, test_accepts)
                if ok:
                    test_solved = True
                    test_turns.append(entry)
                    break
                feedback = f"WRONG ({correct_count}/{total} correct)"
                entry["feedback"] = feedback
                test_turns.append(entry)
                test_prompt = (
                    f"{feedback}\n\n"
                    f"Attempt {turn + 1} of {BUDGET}. Submit a revised pattern."
                )
            else:
                entry["submitted"] = f"unknown action: {action}"
                test_turns.append(entry)
                test_prompt = f"Unknown action.\n\nAttempt {turn + 1} of {BUDGET}."

        test_passed = test_solved
        phases.append({
            "label": "Final test",
            "correct": test_grammar["name"],
            "turns": test_turns,
            "solved": test_passed,
            "steps": test_num_steps,
            "score": 1.0 if test_passed else 0.0,
        })

    final_score = sum(learning_scores) / 5 * 0.5 + (1.0 if test_passed else 0.0) * 0.5
    _log_trace("GRAMMAR INDUCTION", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    grammar_induction_proc_learning.run(kbench.llm)

