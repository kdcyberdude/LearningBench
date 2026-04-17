from dataclasses import dataclass
from typing import Optional

import kaggle_benchmarks as kbench

def efficiency_score(
    solved: bool,
    step_y: int,
    budget_n: int,
    min_necessary: int,
    floor: float = 0.1,
) -> float:
    """Efficiency credit for solving within a budget."""
    if not solved:
        return 0.0
    step_y = max(1, min(step_y, budget_n))
    if step_y <= min_necessary:
        return 1.0
    paid_used = step_y - min_necessary
    paid_budget = budget_n - min_necessary
    if paid_budget <= 0:
        return 1.0
    return max(floor, 1.0 - (1.0 - floor) * (paid_used / paid_budget))


def weighted_learning_mean(round_scores: list, weights: list = None) -> float:
    """Weighted mean emphasising later rounds."""
    n = len(round_scores)
    if n == 0:
        return 0.0
    if weights is None:
        weights = list(range(1, n + 1))
    denom = sum(weights)
    if denom == 0:
        return sum(round_scores) / n
    return sum(s * w for s, w in zip(round_scores, weights)) / denom


def _learning_curve_slope(round_scores: list) -> float:
    """Normalised OLS slope of round scores â [0,1] (0.5 = flat, 1.0 = perfect rise)."""
    n = len(round_scores)
    if n < 2:
        return 0.5
    x_mean = (n - 1) / 2.0
    y_mean = sum(round_scores) / n
    num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(round_scores))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.5
    slope = num / den
    max_slope = 1.0 / (n - 1)
    normalised = max(-1.0, min(1.0, slope / max_slope))
    return round((normalised + 1.0) / 2.0, 4)


def procedural_composite_score(round_scores: list, test_score: float) -> float:
    """Four-component procedural learning score in [0, 1].

    transfer    0.30  â test_score: transfer to novel instances without feedback
    asymptote   0.25  â mean of latter half of round_scores: peak skill reached
    trajectory  0.25  â learning_curve_slope: evidence of genuine improvement
    consistency 0.20  â weighted_learning_mean: overall quality, later rounds weighted more
    """
    n = len(round_scores)
    if n == 0:
        return 0.0
    k = max(1, n // 2)
    asymptote = sum(round_scores[-k:]) / k
    if asymptote < 1e-9 and float(test_score) < 1e-9:
        return 0.0
    consistency = weighted_learning_mean(round_scores)
    trajectory = _learning_curve_slope(round_scores)
    raw = (
        0.30 * float(test_score)
        + 0.25 * asymptote
        + 0.25 * trajectory
        + 0.20 * consistency
    )
    return round(raw, 4)

# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

_TASK_DESCRIPTION = (
    "Tests procedural learning of a hidden three-key sorting rule: odd-last-digit numbers first, "
    "then by descending digit-sum, then by value. The model receives swap-position hints after each "
    "wrong attempt across 5 practice rounds before four no-hint final tests. What makes it hard is that "
    "the rule combines parity, digit-sum, and value in a non-obvious priority order. "
    "Success = 50% weighted learning efficiency + 50% mean score on four final tests."
)


def _log_trace(
    task: str, phases: list[dict], final_score: float, initial_prompt: str = ""
) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {_TASK_DESCRIPTION}")
    if initial_prompt:
        print(f"\n  INITIAL PROMPT:\n{initial_prompt}")
    for phase in phases:
        label = phase["label"]
        print(f"\n  [{label}]  correct={phase['correct']}")
        for t in phase["turns"]:
            reasoning = t.get("reasoning", "")
            if reasoning:
                snippet = reasoning
                print(f"    Turn {t['turn']}  reasoning: {snippet}")
            print(f"    Turn {t['turn']}  submitted: {t['submitted']}", end="")
            if "feedback" in t:
                print(f"  â  {t['feedback']}", end="")
            print()
        status = "PASS â" if phase["solved"] else "FAIL â"
        print(f"    {status}  steps={phase['steps']}  score={phase['score']:.4f}")
    print(f"\n  Final score : {final_score:.4f}")
    print(f"{sep}\n")


BUDGET = 14
# Minimum attempts to pin a 3-level composite order from pairwise swap hints alone (no prior).
MIN_NECESSARY = 3

LEARNING_INPUTS = [
    [73, 42, 81, 56, 29, 64],
    [35, 88, 47, 62, 91, 23],
    [54, 17, 76, 43, 89, 31],
    [66, 39, 52, 85, 14, 77],
    [28, 93, 41, 60, 37, 74],
]
TEST_INPUTS = [
    [67, 24, 95, 48, 33, 72],
    [19, 58, 3, 86, 41, 70],
    [12, 55, 90, 27, 63, 8],
    [44, 71, 15, 99, 22, 50],
]


def _digit_sum(n: int) -> int:
    return sum(int(d) for d in str(abs(n)))


def _sort_key(n: int) -> tuple:
    last_digit = abs(n) % 10
    parity = 0 if last_digit % 2 == 1 else 1
    return (parity, -_digit_sum(n), n)


def _correct_order(items: list[int]) -> list[int]:
    return sorted(items, key=_sort_key)


LEARNING_CORRECT = [_correct_order(items) for items in LEARNING_INPUTS]
TEST_CORRECTS = [_correct_order(items) for items in TEST_INPUTS]


def _first_inversion(submitted: list[int], correct: list[int]) -> Optional[tuple]:
    rank = {v: i for i, v in enumerate(correct)}
    for i in range(len(submitted) - 1):
        a, b = submitted[i], submitted[i + 1]
        if rank.get(a, 0) > rank.get(b, 0):
            return (i, i + 1)
    for i, v in enumerate(submitted):
        if v != correct[i]:
            for j in range(i + 1, len(submitted)):
                if submitted[j] == correct[i]:
                    return (i, j)
    return None


@dataclass
class _SortSubmission:
    reasoning: str
    sorted_list: list[int]


@kbench.task(
    name="adaptive_sort_rule_proc_learning",
    description=(
        "Learn a hidden parityâdigit-sumâvalue sorting rule across 5 practice rounds with swap hints, "
        "then apply it in four no-hint tests. Score = weighted_learning_efficiencyÃ0.5 + (tests_passed/4)Ã0.5."
    ),
)
def adaptive_sort_proc_learning(llm) -> float:
    """5 practice rounds (hints, up to 14 attempts each) then 4 no-hint tests."""
    phases = []

    with kbench.chats.new("adaptive_sort"):
        learning_scores = []
        initial_prompt = ""
        for idx, (items, correct) in enumerate(zip(LEARNING_INPUTS, LEARNING_CORRECT)):
            turns = []
            solved = False
            num_steps = 0

            intro = (
                (
                    "You will sort numbers using a HIDDEN rule. "
                    "Each wrong attempt gives one hint: the first pair of positions to swap. "
                    "After 5 practice rounds you will face four final lists with no hints.\n\n"
                    "Scoring note: Your score has four components â transfer (30%): passing the "
                    "final no-hint tests; asymptote (25%): skill in the later practice rounds; "
                    "trajectory (25%): whether your performance improves across practice rounds "
                    "(a rising curve beats a flat one even at the same average); consistency (20%): "
                    "overall quality with later rounds weighted more. Solving each practice round "
                    "in fewer attempts also boosts your within-round efficiency score.\n\n"
                )
                if idx == 0
                else ""
            )

            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 â Numbers: {items}\n"
                f"Attempt 1 of {BUDGET}. Submit your sorted list."
            )

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                if idx == 0 and turn == 1:
                    initial_prompt = next_prompt
                try:
                    submission: _SortSubmission = llm.prompt(
                        next_prompt, schema=_SortSubmission
                    )
                except Exception:
                    entry = {
                        "turn": turn,
                        "submitted": "PARSE_ERROR",
                        "feedback": "Failed to parse response â turn wasted.",
                    }
                    turns.append(entry)
                    next_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Submit your sorted list."
                    continue

                try:
                    submitted = [int(x) for x in submission.sorted_list]
                except (TypeError, ValueError):
                    submitted = []

                entry = {
                    "turn": turn,
                    "submitted": submitted,
                    "reasoning": submission.reasoning,
                }

                if submitted == correct:
                    solved = True
                    turns.append(entry)
                    break

                inv = _first_inversion(submitted, correct) if submitted else None
                if inv is not None:
                    i, j = inv
                    a, b = submitted[i], submitted[j]
                    feedback = f"WRONG. Positions {i} ({a}) and {j} ({b}) should be swapped â {b} before {a}."
                else:
                    feedback = "WRONG. Unusual arrangement â review which elements belong at the start."

                entry["feedback"] = feedback
                turns.append(entry)
                next_prompt = f"{feedback}\n\nAttempt {turn + 1} of {BUDGET}. Submit your revised sorted list."

            eff = efficiency_score(solved, num_steps, BUDGET, MIN_NECESSARY)
            learning_scores.append(eff)
            phases.append(
                {
                    "label": f"Practice {idx + 1}/5",
                    "correct": correct,
                    "turns": turns,
                    "solved": solved,
                    "steps": num_steps,
                    "score": eff,
                }
            )

        test_correct_count = 0
        for ti, (test_items, test_correct) in enumerate(
            zip(TEST_INPUTS, TEST_CORRECTS), start=1
        ):
            test_prompt = (
                f"Final test {ti}/4 â Numbers: {test_items}\n"
                "No hints will be given. Submit your sorted list."
            )
            try:
                test_submission: _SortSubmission = llm.prompt(
                    test_prompt, schema=_SortSubmission
                )
            except Exception:
                test_submission = None

            try:
                test_submitted = (
                    [int(x) for x in test_submission.sorted_list]
                    if test_submission is not None
                    else []
                )
            except (TypeError, ValueError):
                test_submitted = []

            passed = test_submitted == test_correct
            if passed:
                test_correct_count += 1
            phases.append(
                {
                    "label": f"Final test {ti}/4",
                    "correct": test_correct,
                    "turns": [
                        {
                            "turn": 1,
                            "submitted": test_submitted,
                            "reasoning": test_submission.reasoning
                            if test_submission is not None
                            else "",
                        }
                    ],
                    "solved": passed,
                    "steps": 1,
                    "score": 1.0 if passed else 0.0,
                }
            )

    learning_score = weighted_learning_mean(learning_scores)
    test_score = test_correct_count / 4.0
    final_score = procedural_composite_score(learning_scores, test_score)
    _log_trace("ADAPTIVE SORT", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    adaptive_sort_proc_learning.run(kbench.llm)