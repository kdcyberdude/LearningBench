#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

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
    """Normalised OLS slope of round scores → [0,1] (0.5 = flat, 1.0 = perfect rise)."""
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

    transfer    0.30  — test_score: transfer to novel instances without feedback
    asymptote   0.25  — mean of latter half of round_scores: peak skill reached
    trajectory  0.25  — learning_curve_slope: evidence of genuine improvement
    consistency 0.20  — weighted_learning_mean: overall quality, later rounds weighted more
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

# ─────────────────────────────────────────────────────────────────────────────


_TASK_DESCRIPTION = (
    "Tests whether the model can reverse-engineer a hidden SQL WHERE clause filtering an employees table "
    "(id, dept, salary) by querying candidate clauses and comparing matched row IDs against the hidden "
    "clause's output, across 5 practice instances with a 10-action budget. The hidden clauses range from "
    "simple single-condition filters to compound OR expressions. What makes it hard is that the model sees "
    "only row IDs — not column values — requiring iterative refinement of the clause hypothesis."
)

# Minimum probes before a zero-prior agent can pin a WHERE clause from ID sets alone (~hypothesis elimination).
MIN_NECESSARY = 4


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
            print(f"    Turn {t['turn']}  submitted: {t['submitted']}", end="")
            if "feedback" in t:
                print(f"  →  {t['feedback']}", end="")
            print()
        status = "PASS ✓" if phase["solved"] else "FAIL ✗"
        print(f"    {status}  steps={phase['steps']}  score={phase['score']:.4f}")
    print(f"\n  Final score : {final_score:.4f}")
    print(f"{sep}\n")


BUDGET = 10

TABLE = [
    {"id": 1, "dept": "eng", "salary": 75000},
    {"id": 2, "dept": "sales", "salary": 55000},
    {"id": 3, "dept": "hr", "salary": 60000},
    {"id": 4, "dept": "eng", "salary": 90000},
    {"id": 5, "dept": "ops", "salary": 48000},
    {"id": 6, "dept": "sales", "salary": 80000},
    {"id": 7, "dept": "hr", "salary": 65000},
    {"id": 8, "dept": "eng", "salary": 70000},
    {"id": 9, "dept": "ops", "salary": 85000},
    {"id": 10, "dept": "sales", "salary": 52000},
]

LEARNING_WHERE = [
    "salary > 70000",
    "dept == 'eng'",
    "salary < 60000",
    "id < 5",
    "dept == 'hr' or salary > 80000",
]
TEST_WHERE = [
    "salary > 65000 and dept == 'eng'",
    "dept == 'ops'",
    "id > 7 or salary < 55000",
    "salary >= 60000 and id <= 3",
]


def _eval_where(where_clause, table):
    matched = []
    for row in table:
        try:
            if eval(where_clause.replace("AND", "and").replace("OR", "or"), {}, row):
                matched.append(row["id"])
        except Exception:
            pass
    return matched


def _apply_where(row, clause):
    try:
        return bool(eval(clause.replace("AND", "and").replace("OR", "or"), {}, row))
    except Exception:
        return False


def _parse_simple_where(clause, row):
    try:
        py = (
            clause.replace("AND", "and")
            .replace("OR", "or")
            .replace("=", "==")
            .replace("====", "==")
            .replace("!==", "!=")
        )
        return bool(eval(py, {}, row))
    except Exception:
        return False


LEARNING_HIDDEN_ROWS = [_eval_where(w, TABLE) for w in LEARNING_WHERE]
TEST_HIDDEN_ROWS = [_eval_where(w, TABLE) for w in TEST_WHERE]

TABLE_PREVIEW = "\n".join(
    f"  id={r['id']}, dept={r['dept']}, salary={r['salary']}" for r in TABLE
)


@dataclass
class _SQLAction:
    action: str
    where_clause: str


@kbench.task(
    name="sql_reverse_engineering_proc_learning",
    description=(
        "Reverse-engineer a hidden SQL WHERE clause over an employees table by querying candidate clauses "
        "and comparing results against hidden row output. Score = weighted_learning×0.5 + (tests_passed/4)×0.5."
    ),
)
def sql_reverse_engineering_proc_learning(llm) -> float:
    """5 practice instances then 4 no-hint tests. Score = weighted_learning×0.5 + test_fraction×0.5."""
    phases = []

    with kbench.chats.new("sql_reverse_engineering"):
        learning_scores = []
        initial_prompt = ""
        for idx, (hidden_where, hidden_rows) in enumerate(
            zip(LEARNING_WHERE, LEARNING_HIDDEN_ROWS)
        ):
            turns = []
            solved = False
            num_steps = 0

            intro = (
                (
                    "You must reverse-engineer a hidden SQL WHERE clause filtering this employees table:\n"
                    f"{TABLE_PREVIEW}\n\n"
                    "Columns: id (int), dept (str), salary (int).\n"
                    "Actions:\n"
                    "  action='query', where_clause='...'  → returns matching row IDs\n"
                    "  action='submit', where_clause='...' → graded (must match same rows as hidden clause)\n"
                    "The hidden row IDs are always shown. Use queries to narrow down the clause.\n\n"
                    "Scoring note: Your score has four components — transfer (30%): your WHERE "
                    "clause matching exactly the same rows as the hidden clause in the final "
                    "instances; asymptote (25%): clause accuracy in the later practice instances; "
                    "trajectory (25%): whether your accuracy improves across practice instances "
                    "(a rising curve beats a flat one even at the same average); consistency (20%): "
                    "overall quality with later instances weighted more. Using fewer query probes "
                    "per practice instance also boosts your within-round efficiency score.\n\n"
                )
                if idx == 0
                else ""
            )

            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 — Hidden clause matches rows: {hidden_rows}\n"
                f"Attempt 1 of {BUDGET}. Submit action='query' to probe or action='submit' to guess the WHERE clause."
            )

            if idx == 0:
                initial_prompt = next_prompt

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                try:
                    sub = llm.prompt(next_prompt, schema=_SQLAction)
                except Exception:
                    entry = {
                        "turn": turn,
                        "submitted": "PARSE_ERROR",
                        "feedback": "Failed to parse response — turn wasted.",
                    }
                    turns.append(entry)
                    next_prompt = (
                        f"Your last response could not be parsed. Please follow the schema exactly.\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Submit query or revised WHERE clause."
                    )
                    continue
                action = (sub.action or "").strip().lower()
                clause = (sub.where_clause or "").strip()
                entry = {"turn": turn, "submitted": clause}

                if action == "query":
                    matched = [r["id"] for r in TABLE if _parse_simple_where(clause, r)]
                    feedback = f"QUERY '{clause}' → matches rows: {matched}"
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\nHidden clause still matches rows: {hidden_rows}\n"
                        f"Attempt {turn + 1} of {BUDGET}. Submit query or submit your WHERE clause guess."
                    )
                elif action == "submit":
                    submitted_rows = [
                        r["id"] for r in TABLE if _parse_simple_where(clause, r)
                    ]
                    if submitted_rows == hidden_rows:
                        solved = True
                        turns.append(entry)
                        break
                    fp = [i for i in submitted_rows if i not in hidden_rows]
                    fn = [i for i in hidden_rows if i not in submitted_rows]
                    feedback = f"WRONG. false_positives={fp}, false_negatives={fn}"
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\nHidden clause matches rows: {hidden_rows}\n"
                        f"Attempt {turn + 1} of {BUDGET}. Submit query or revised WHERE clause."
                    )
                else:
                    feedback = f"INVALID action '{action}'. Use 'query' or 'submit'."
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\nHidden clause matches rows: {hidden_rows}\n"
                        f"Attempt {turn + 1} of {BUDGET}. Use action='query' or action='submit'."
                    )

            eff = efficiency_score(solved, num_steps, BUDGET, MIN_NECESSARY)
            learning_scores.append(eff)
            phases.append(
                {
                    "label": f"Practice {idx + 1}/5",
                    "correct": hidden_where,
                    "turns": turns,
                    "solved": solved,
                    "steps": num_steps,
                    "score": eff,
                }
            )

        test_ok = 0
        for ti, (hidden_rows, gold_where) in enumerate(
            zip(TEST_HIDDEN_ROWS, TEST_WHERE), start=1
        ):
            try:
                test_sub = llm.prompt(
                    f"Final test {ti}/4 — Hidden clause matches rows: {hidden_rows}\n"
                    "No hints. Submit action='submit' with your WHERE clause.",
                    schema=_SQLAction,
                )
            except Exception:
                test_sub = None
            test_clause = (
                (test_sub.where_clause or "").strip() if test_sub is not None else ""
            )
            test_submitted_rows = [
                r["id"] for r in TABLE if _parse_simple_where(test_clause, r)
            ]
            passed = test_submitted_rows == hidden_rows
            if passed:
                test_ok += 1
            phases.append(
                {
                    "label": f"Final test {ti}/4",
                    "correct": gold_where,
                    "turns": [{"turn": 1, "submitted": test_clause}],
                    "solved": passed,
                    "steps": 1,
                    "score": 1.0 if passed else 0.0,
                }
            )

    learning_score = weighted_learning_mean(learning_scores)
    test_score = test_ok / 4.0
    final_score = procedural_composite_score(learning_scores, test_score)
    _log_trace("SQL REVERSE ENGINEERING", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    sql_reverse_engineering_proc_learning.run(kbench.llm)

