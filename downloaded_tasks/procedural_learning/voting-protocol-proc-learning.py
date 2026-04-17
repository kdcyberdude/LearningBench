#!/usr/bin/env python
# coding: utf-8

from collections import Counter, defaultdict
from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether the model can identify which of 4 voting rules (Plurality, Borda, IRV, Approval) "
    "determines election winners by submitting discriminating ballot configurations and observing outcomes, "
    "across 5 practice instances with a 10-action budget. The hidden rule is a fixed voting algorithm. "
    "What makes it hard is that some ballot configurations produce identical winners under multiple rules — "
    "the model must design adversarial ballots that separate the rules and then commit to the right one."
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


CANDIDATES = ["A", "B", "C", "D"]
RULES = ["PLURALITY", "BORDA", "INSTANT_RUN", "APPROVAL"]
BUDGET = 10

LEARNING_INSTANCES = [
    {"rule": "PLURALITY"},
    {"rule": "BORDA"},
    {"rule": "INSTANT_RUN"},
    {"rule": "APPROVAL"},
    {"rule": "PLURALITY"},
]
TEST_INSTANCE = {"rule": "BORDA"}


def _plurality_winner(ballots):
    counts = Counter(b[0] for b in ballots if b)
    return max(CANDIDATES, key=lambda c: (counts.get(c, 0), -CANDIDATES.index(c)))


def _borda_winner(ballots):
    scores = defaultdict(int)
    n = len(CANDIDATES)
    for ballot in ballots:
        for rank, c in enumerate(ballot):
            scores[c] += n - 1 - rank
    return max(CANDIDATES, key=lambda c: (scores[c], -CANDIDATES.index(c)))


def _irv_winner(ballots):
    remaining = list(CANDIDATES)
    active_ballots = [list(b) for b in ballots]
    while len(remaining) > 1:
        effective = []
        for b in active_ballots:
            for c in b:
                if c in remaining:
                    effective.append(c)
                    break
        counts = Counter(effective)
        total = sum(counts.values())
        for c in remaining:
            if counts.get(c, 0) > total / 2:
                return c
        min_count = min(counts.get(c, 0) for c in remaining)
        to_eliminate = next(c for c in sorted(remaining) if counts.get(c, 0) == min_count)
        remaining.remove(to_eliminate)
    return remaining[0]


def _approval_winner(ballots):
    counts = Counter()
    for b in ballots:
        for c in b[:2]:
            counts[c] += 1
    return max(CANDIDATES, key=lambda c: (counts[c], -CANDIDATES.index(c)))


def _apply_rule(ballots, rule):
    if rule == "PLURALITY":
        return _plurality_winner(ballots)
    if rule == "BORDA":
        return _borda_winner(ballots)
    if rule == "INSTANT_RUN":
        return _irv_winner(ballots)
    return _approval_winner(ballots)


@dataclass
class _VotingAction:
    action: str      # "vote" or "identify"
    ballots: list    # list of ranked ballots; [] when action="identify"
    rule: str        # rule name; "" when action="vote"


@kbench.task(
    name="voting_protocol_proc_learning",
    description=(
        "Identify which of 4 voting rules (Plurality/Borda/IRV/Approval) determines winners by submitting "
        "discriminating ballot configurations across 5 instances. Score = learning_efficiency×0.5 + test_pass×0.5."
    ),
)
def voting_protocol_proc_learning(llm) -> float:
    """5 practice voting-protocol instances (submit ballots to see winner, identify rule), then 1 no-hint test. Score=learning_avg×0.5+test×0.5."""
    phases = []
    test_passed = False

    with kbench.chats.new("voting_protocol"):
        learning_scores = []
        initial_prompt = ""
        for idx, inst in enumerate(LEARNING_INSTANCES):
            rule = inst["rule"]
            turns = []
            solved = False
            num_steps = 0

            intro = (
                "A hidden voting rule determines the winner of elections over 4 candidates: A, B, C, D.\n\n"
                "Possible rules:\n"
                "  PLURALITY   — most first-place votes wins\n"
                "  BORDA       — sum of positional scores (3/2/1/0 per rank)\n"
                "  INSTANT_RUN — iterative elimination of weakest until majority\n"
                "  APPROVAL    — each voter approves their top 2; most approvals wins\n\n"
                "Actions:\n"
                "  action='vote', ballots=[[ranking1], ...]  → 'winner: X'\n"
                "  action='identify', rule='PLURALITY'|...  → graded\n\n"
                "Each ballot must rank all 4 candidates (most preferred first).\n"
                "After 5 practice instances you will face a final instance.\n\n"
            ) if idx == 0 else ""

            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 — Hidden rule #{idx + 1}. Identify the voting rule.\n"
                f"Attempt 1 of {BUDGET}. Submit ballots to observe winners or identify the rule."
            )

            if idx == 0:
                initial_prompt = next_prompt

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                try:
                    sub = llm.prompt(next_prompt, schema=_VotingAction)
                except Exception:
                    entry = {"turn": turn, "submitted": "PARSE_ERROR", "feedback": "Failed to parse response — turn wasted."}
                    turns.append(entry)
                    next_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Submit ballots to observe winners or identify the rule."
                    continue
                action = sub.action.strip().lower()

                if action == "vote":
                    raw_ballots = sub.ballots
                    try:
                        ballots = [[str(c).upper() for c in b] for b in raw_ballots]
                    except (TypeError, ValueError):
                        ballots = []
                    valid = bool(ballots) and all(sorted(b) == sorted(CANDIDATES) for b in ballots)
                    entry = {"turn": turn, "submitted": f"vote {len(ballots)} ballots"}
                    if not valid:
                        feedback = "INVALID ballots — each ballot must rank all 4 candidates (A, B, C, D)."
                    else:
                        winner = _apply_rule(ballots, rule)
                        feedback = f"vote {len(ballots)} ballots → winner: {winner}"
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Submit more ballots or identify the rule."
                    )
                elif action == "identify":
                    ans = sub.rule.strip().upper()
                    entry = {"turn": turn, "submitted": ans}
                    if ans == rule:
                        solved = True
                        turns.append(entry)
                        break
                    feedback = "WRONG. That is not the hidden rule."
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Submit ballots to gather more evidence or try again."
                    )
                else:
                    entry = {"turn": turn, "submitted": action}
                    feedback = f"INVALID action '{action}'. Use 'vote' or 'identify'."
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Use action='vote' or action='identify'."
                    )

            eff = _efficiency_score(solved, num_steps, BUDGET)
            learning_scores.append(eff)
            phases.append({
                "label": f"Practice {idx + 1}/5",
                "correct": rule,
                "turns": turns,
                "solved": solved,
                "steps": num_steps,
                "score": eff,
            })

        test_rule = TEST_INSTANCE["rule"]
        try:
            test_sub = llm.prompt(
                f"Final test — Hidden rule #{len(LEARNING_INSTANCES) + 1}. Identify the voting rule.\n"
                "This is your only attempt. No hints. Submit action='identify' with your answer.",
                schema=_VotingAction,
            )
        except Exception:
            test_sub = None
        test_ans = test_sub.rule.strip().upper() if test_sub is not None else ""
        test_passed = test_ans == test_rule
        phases.append({
            "label": "Final test",
            "correct": test_rule,
            "turns": [{"turn": 1, "submitted": test_ans}],
            "solved": test_passed,
            "steps": 1,
            "score": 1.0 if test_passed else 0.0,
        })

    final_score = sum(learning_scores) / 5 * 0.5 + (1.0 if test_passed else 0.0) * 0.5
    _log_trace("VOTING PROTOCOL", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    voting_protocol_proc_learning.run(kbench.llm)

