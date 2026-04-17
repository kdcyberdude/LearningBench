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
    "Tests whether the model can identify and best-respond to a hidden behavioural rule "
    "across 5 practice 12-round Iterated Prisoner's Dilemma games, accumulating >=30 points each. "
    "The five rules are RULE_ALPHA through RULE_EPSILON -- novel automata not named in standard "
    "game-theory literature. Discovering each rule requires deliberate probing (e.g. alternating "
    "C/D, sustaining CC, observing move lag). Success requires >=30 in both practice and final games."
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
            print(f"    Turn {t['turn']}  submitted: {t['submitted']}", end="")
            if "feedback" in t:
                print(f"  ->  {t['feedback']}", end="")
            print()
        status = "PASS ok" if phase["solved"] else "FAIL x"
        print(f"    {status}  steps={phase['steps']}  score={phase['score']:.4f}")
    print(f"\n  Final score : {final_score:.4f}")
    print(f"{sep}\n")


BUDGET = 12
TARGET = 30
MIN_NECESSARY = 12  # All rounds needed to accumulate score against any unknown rule.

# Five novel behavioural rules with no canonical literature names.
# RULE_ALPHA   -- mirrors your PREVIOUS move (starts C)
# RULE_BETA    -- Win-Stay Lose-Shift: repeats own last move if payoff >= 3; flips otherwise (starts C)
# RULE_GAMMA   -- cooperates until you defect TWICE cumulatively, then permanently defects
# RULE_DELTA   -- "Parity-Echo": plays your move from 2 rounds ago (defaults C for rounds 1-2)
# RULE_EPSILON -- "Spike": cooperates every round EXCEPT rounds 3, 6, 9, 12

LEARNING_INSTANCES = [
    {"rule": "RULE_ALPHA"},
    {"rule": "RULE_BETA"},
    {"rule": "RULE_GAMMA"},
    {"rule": "RULE_DELTA"},
    {"rule": "RULE_EPSILON"},
]
TEST_INSTANCES = [
    {"rule": "RULE_ALPHA"},
    {"rule": "RULE_BETA"},
    {"rule": "RULE_GAMMA"},
    {"rule": "RULE_DELTA"},
]

_PAYOFF = {"CC": (3, 3), "CD": (0, 5), "DC": (5, 0), "DD": (1, 1)}


class _RuleAlpha:
    """Mirrors your previous move; starts C."""
    def __init__(self):
        self.last_model = "C"

    def respond(self, model_last: str, round_num: int) -> str:
        r = self.last_model
        self.last_model = model_last
        return r


class _RuleBeta:
    """Win-Stay Lose-Shift: repeat own last move if payoff >= 3, else flip. Starts C."""
    def __init__(self):
        self.last_self = "C"
        self.last_model = "C"

    def respond(self, model_last: str, round_num: int) -> str:
        payoff_map = {"CC": 3, "CD": 0, "DC": 5, "DD": 1}
        got = payoff_map.get(self.last_self + model_last, 3)
        prev = self.last_self
        self.last_self = prev if got >= 3 else ("C" if prev == "D" else "D")
        self.last_model = model_last
        return self.last_self


class _RuleGamma:
    """Cooperates until you defect twice cumulatively; then permanently defects."""
    def __init__(self):
        self.d_count = 0

    def respond(self, model_last: str, round_num: int) -> str:
        if model_last == "D":
            self.d_count += 1
        return "D" if self.d_count >= 2 else "C"


class _RuleDelta:
    """Parity-Echo: plays your move from 2 rounds ago; defaults C for first 2 rounds."""
    def __init__(self):
        self.history: list = []

    def respond(self, model_last: str, round_num: int) -> str:
        self.history.append(model_last)
        if len(self.history) < 3:
            return "C"
        return self.history[-3]


class _RuleEpsilon:
    """Spike: cooperates every round except rounds 3, 6, 9, 12 (always D on multiples of 3)."""
    def respond(self, model_last: str, round_num: int) -> str:
        return "D" if round_num % 3 == 0 else "C"


def _make_opponent(rule_name: str):
    if rule_name == "RULE_ALPHA":
        return _RuleAlpha()
    if rule_name == "RULE_BETA":
        return _RuleBeta()
    if rule_name == "RULE_GAMMA":
        return _RuleGamma()
    if rule_name == "RULE_DELTA":
        return _RuleDelta()
    return _RuleEpsilon()


@dataclass
class _PDAction:
    action: str


@kbench.task(
    name="opponent_strategy_proc_learning",
    description=(
        "Identify and best-respond to a hidden behavioural rule (RULE_ALPHA - RULE_EPSILON) in an "
        "Iterated Prisoner's Dilemma across 5 practice games (12 rounds, target >=30 each), then "
        "4 final games. Score = procedural_composite(learning_scores, finals_passed/4)."
    ),
)
def opponent_strategy_proc_learning(llm) -> float:
    """5 practice games then 4 test games; pass each if score >=30 after 12 rounds."""
    phases = []

    with kbench.chats.new("opponent_strategy"):
        learning_scores = []
        initial_prompt = ""

        for idx, instance in enumerate(LEARNING_INSTANCES):
            rule_name = instance["rule"]
            opp = _make_opponent(rule_name)
            turns = []
            model_score = 0
            model_last = "C"
            round_num = 0

            intro = (
                (
                    "You are playing a 12-round Iterated Prisoner's Dilemma.\n"
                    "Payoffs per round (your score, opponent score):\n"
                    "  Both C: 3,3 | You C, Opp D: 0,5 | You D, Opp C: 5,0 | Both D: 1,1\n"
                    "The opponent follows one FIXED private behavioural rule.\n"
                    "You must discover the rule through play and exploit it to score >=30 points.\n"
                    f"After 5 practice games you face four final games.\n\n"
                    "Scoring note: Your score has four components — transfer (30%): scoring ≥30 "
                    "in the final games by exploiting the opponent's rule; asymptote (25%): "
                    "cumulative score in the later practice games; trajectory (25%): whether "
                    "your score per game improves across practice games (a rising curve beats a "
                    "flat one even at the same average); consistency (20%): overall quality with "
                    "later games weighted more. Identifying and exploiting the rule sooner "
                    "also boosts your within-round efficiency score.\n\n"
                )
                if idx == 0
                else ""
            )

            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 -- Round 1 of {BUDGET}. Your score: 0\n"
                "Submit C (cooperate) or D (defect)."
            )

            if idx == 0:
                initial_prompt = next_prompt

            for turn in range(1, BUDGET + 1):
                round_num = turn
                try:
                    submission = llm.prompt(next_prompt, schema=_PDAction)
                except Exception:
                    entry = {
                        "turn": turn,
                        "submitted": "PARSE_ERROR",
                        "feedback": "Failed to parse response -- turn wasted.",
                    }
                    turns.append(entry)
                    next_prompt = (
                        f"Your last response could not be parsed. Follow the schema exactly.\n\n"
                        f"Round {turn + 1} of {BUDGET}. Your score: {model_score}\nSubmit C or D."
                    )
                    continue

                model_action = submission.action.strip().upper()
                if model_action not in ("C", "D"):
                    model_action = "C"

                opp_action = opp.respond(model_last, round_num)
                model_last = model_action

                key = model_action + opp_action
                mp, _ = _PAYOFF[key]
                model_score += mp

                entry = {"turn": turn, "submitted": model_action}

                if turn < BUDGET:
                    feedback = f"R{turn}: you={model_action}, opp={opp_action}, +{mp} -> total={model_score}"
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Round {turn + 1} of {BUDGET}. Your score: {model_score}\n"
                        "Submit C or D."
                    )
                else:
                    feedback = f"R{turn}: you={model_action}, opp={opp_action}, +{mp} -> total={model_score}"
                    entry["feedback"] = feedback
                    turns.append(entry)

            solved = model_score >= TARGET
            eff = efficiency_score(solved, BUDGET, BUDGET, MIN_NECESSARY)
            learning_scores.append(eff)
            phases.append({
                "label": f"Practice {idx + 1}/5",
                "correct": {"rule": rule_name, "target": TARGET},
                "turns": turns, "solved": solved,
                "steps": BUDGET, "score": eff,
            })

        finals_passed = 0
        for fidx, tins in enumerate(TEST_INSTANCES, start=1):
            rule_name = tins["rule"]
            opp = _make_opponent(rule_name)
            turns = []
            model_score = 0
            model_last = "C"
            round_num = 0

            next_prompt = (
                f"Final game {fidx}/4 -- Round 1 of {BUDGET}. Your score: 0\n"
                "Submit C or D."
            )

            for turn in range(1, BUDGET + 1):
                round_num = turn
                try:
                    submission = llm.prompt(next_prompt, schema=_PDAction)
                except Exception:
                    next_prompt = (
                        f"Your last response could not be parsed.\n\n"
                        f"Round {turn + 1} of {BUDGET}. Your score: {model_score}\nSubmit C or D."
                    )
                    turns.append({"turn": turn, "submitted": "PARSE_ERROR"})
                    continue

                model_action = submission.action.strip().upper()
                if model_action not in ("C", "D"):
                    model_action = "C"

                opp_action = opp.respond(model_last, round_num)
                model_last = model_action

                key = model_action + opp_action
                mp, _ = _PAYOFF[key]
                model_score += mp

                entry = {"turn": turn, "submitted": model_action}
                if turn < BUDGET:
                    feedback = f"R{turn}: you={model_action}, opp={opp_action}, +{mp} -> total={model_score}"
                    entry["feedback"] = feedback
                    next_prompt = (
                        f"{feedback}\n\nRound {turn + 1} of {BUDGET}. "
                        f"Your score: {model_score}\nSubmit C or D."
                    )
                else:
                    feedback = f"R{turn}: you={model_action}, opp={opp_action}, +{mp} -> total={model_score}"
                    entry["feedback"] = feedback
                turns.append(entry)

            passed = model_score >= TARGET
            if passed:
                finals_passed += 1
            phases.append({
                "label": f"Final game {fidx}/4",
                "correct": {"rule": rule_name, "target": TARGET},
                "turns": turns, "solved": passed,
                "steps": BUDGET, "score": 1.0 if passed else 0.0,
            })

    test_score = finals_passed / 4.0
    final_score = procedural_composite_score(learning_scores, test_score)
    _log_trace("OPPONENT STRATEGY", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    opponent_strategy_proc_learning.run(kbench.llm)

