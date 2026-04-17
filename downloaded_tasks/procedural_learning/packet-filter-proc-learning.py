#!/usr/bin/env python
# coding: utf-8

import json
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
    "Tests whether the model can reverse-engineer a hidden 2-attribute AND firewall blocking rule "
    "(combining src_port, dst_port, protocol, and/or direction) by sending test packets and observing "
    "BLOCKED/ALLOWED outcomes across 5 practice instances. The hidden rule requires ALL conditions to match "
    "(AND logic). What makes it hard is systematically isolating each attribute with minimal packets within "
    "a 12-action budget. Success requires exactly identifying both attributes of the rule."
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
                print(f"  →  {t['feedback']}", end="")
            print()
        status = "PASS ✓" if phase["solved"] else "FAIL ✗"
        print(f"    {status}  steps={phase['steps']}  score={phase['score']:.4f}")
    print(f"\n  Final score : {final_score:.4f}")
    print(f"{sep}\n")


BUDGET = 12
MIN_NECESSARY = 5
LOGIC = "AND"

SRC_PORTS = [22, 80, 443, 8080]
DST_PORTS = [22, 80, 443, 8080]
PROTOCOLS = ["TCP", "UDP", "ICMP"]
DIRECTIONS = ["inbound", "outbound"]

ATTRIBUTES = {
    "src_port": SRC_PORTS,
    "dst_port": DST_PORTS,
    "protocol": PROTOCOLS,
    "direction": DIRECTIONS,
}

LEARNING_RULES = [
    {"src_port": 80, "protocol": "TCP"},
    {"dst_port": 443, "direction": "inbound"},
    {"src_port": 22, "dst_port": 8080},
    {"protocol": "UDP", "direction": "outbound"},
    {"src_port": 8080, "protocol": "ICMP"},
]
TEST_RULES = [
    {"dst_port": 80, "direction": "outbound"},
    {"src_port": 443, "protocol": "TCP"},
    {"src_port": 22, "direction": "inbound"},
    {"dst_port": 8080, "protocol": "UDP"},
]


def _is_blocked(packet: dict, rule: dict, logic: str) -> bool:
    matches = [packet.get(attr) == val for attr, val in rule.items()]
    if logic == "AND":
        return all(matches)
    return any(matches)


@dataclass
class _PacketAction:
    action: str
    src_port: int
    dst_port: int
    protocol: str
    direction: str
    identified_rule: str


@kbench.task(
    name="packet_filter_proc_learning",
    description=(
        "Probe a hidden 2-attribute AND firewall rule (src_port/dst_port/protocol/direction) across 5 instances "
        "by sending test packets, then identify the exact rule. Score = weighted_learning×0.5 + (tests_passed/4)×0.5."
    ),
)
def packet_filter_proc_learning(llm) -> float:
    """5 practice instances then 4 independent firewall identification tests."""
    phases = []

    with kbench.chats.new("packet_filter"):
        learning_scores = []

        initial_prompt = ""
        for idx, rule in enumerate(LEARNING_RULES):
            turns = []
            solved = False
            num_steps = 0

            intro = (
                (
                    "A firewall has a hidden blocking rule.\n"
                    "A packet is BLOCKED if ALL of the rule's 2 conditions are satisfied (AND logic).\n\n"
                    "Packet attributes available:\n"
                    f"  src_port:  {SRC_PORTS}\n"
                    f"  dst_port:  {DST_PORTS}\n"
                    f"  protocol:  {PROTOCOLS}\n"
                    f"  direction: {DIRECTIONS}\n\n"
                    "Actions:\n"
                    "  action='send' + packet attributes → 'BLOCKED' or 'ALLOWED'\n"
                    "  action='identify' + identified_rule='{\"attr\": val, ...}' → graded\n\n"
                    "After 5 practice instances you face a final instance.\n\n"
                    "Scoring note: Your score has four components — transfer (30%): correctly "
                    "identifying the blocking rule in the final instances; asymptote (25%): "
                    "rule-identification accuracy in the later practice instances; trajectory (25%): "
                    "whether your accuracy improves across practice instances (a rising curve beats "
                    "a flat one even at the same average); consistency (20%): overall quality with "
                    "later instances weighted more. Using fewer packet probes per practice instance "
                    "also boosts your within-round efficiency score.\n\n"
                )
                if idx == 0
                else ""
            )

            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 — New firewall instance. Budget: {BUDGET} actions.\n"
                f"Attempt 1 of {BUDGET}. Send a packet or identify the rule."
            )

            if idx == 0:
                initial_prompt = next_prompt

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                try:
                    submission = llm.prompt(next_prompt, schema=_PacketAction)
                except Exception:
                    entry = {
                        "turn": turn,
                        "submitted": "PARSE_ERROR",
                        "feedback": "Failed to parse response — turn wasted.",
                    }
                    turns.append(entry)
                    next_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Send a packet or identify the rule."
                    continue

                action = submission.action.strip().lower()
                entry = {"turn": turn, "submitted": action}

                if action == "send":
                    packet = {
                        "src_port": submission.src_port,
                        "dst_port": submission.dst_port,
                        "protocol": submission.protocol,
                        "direction": submission.direction,
                    }
                    blocked = _is_blocked(packet, rule, LOGIC)
                    result_str = "BLOCKED" if blocked else "ALLOWED"
                    entry["submitted"] = f"send {packet}"
                    feedback = f"{result_str}"
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Send another packet or identify the rule."
                    )
                elif action == "identify":
                    rule_str = submission.identified_rule or "{}"
                    try:
                        submitted_rule = json.loads(rule_str)
                    except Exception:
                        feedback = "INVALID JSON in identified_rule."
                        entry["submitted"] = f"identify '{rule_str}'"
                        entry["feedback"] = feedback
                        turns.append(entry)
                        next_prompt = (
                            f"{feedback}\n\n"
                            f"Attempt {turn + 1} of {BUDGET}. Send a packet or identify the rule."
                        )
                        continue

                    entry["submitted"] = f"identify {submitted_rule}"

                    if submitted_rule == rule:
                        solved = True
                        turns.append(entry)
                        break
                    else:
                        feedback = "WRONG. Incorrect or missing conditions."
                        entry["feedback"] = feedback
                        turns.append(entry)
                        next_prompt = (
                            f"{feedback}\n\n"
                            f"Attempt {turn + 1} of {BUDGET}. Send a packet or try again to identify the rule."
                        )
                else:
                    entry["submitted"] = f"unknown action: {action}"
                    entry["feedback"] = "Unknown action. Use 'send' or 'identify'."
                    turns.append(entry)
                    next_prompt = (
                        f"Unknown action. Use action='send' or action='identify'.\n\n"
                        f"Attempt {turn + 1} of {BUDGET}."
                    )

            eff = efficiency_score(solved, num_steps, BUDGET, MIN_NECESSARY)
            learning_scores.append(eff)
            phases.append(
                {
                    "label": f"Practice {idx + 1}/5",
                    "correct": rule,
                    "turns": turns,
                    "solved": solved,
                    "steps": num_steps,
                    "score": eff,
                }
            )

        test_ok = 0
        for ti, test_rule in enumerate(TEST_RULES, start=1):
            test_turns = []
            test_solved = False
            test_num_steps = 0

            test_prompt = (
                f"Final test {ti}/4 — New firewall instance. Budget: {BUDGET} actions.\n"
                "No hints after wrong identification.\n"
                f"Attempt 1 of {BUDGET}. Send a packet or identify the rule."
            )

            for turn in range(1, BUDGET + 1):
                test_num_steps = turn
                try:
                    test_submission = llm.prompt(test_prompt, schema=_PacketAction)
                except Exception:
                    entry = {
                        "turn": turn,
                        "submitted": "PARSE_ERROR",
                        "feedback": "Failed to parse response — turn wasted.",
                    }
                    test_turns.append(entry)
                    test_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Send another packet or identify the rule."
                    continue

                action = test_submission.action.strip().lower()
                entry = {"turn": turn, "submitted": action}

                if action == "send":
                    packet = {
                        "src_port": test_submission.src_port,
                        "dst_port": test_submission.dst_port,
                        "protocol": test_submission.protocol,
                        "direction": test_submission.direction,
                    }
                    blocked = _is_blocked(packet, test_rule, LOGIC)
                    result_str = "BLOCKED" if blocked else "ALLOWED"
                    entry["submitted"] = f"send {packet}"
                    feedback = f"{result_str}"
                    entry["feedback"] = feedback
                    test_turns.append(entry)
                    test_prompt = (
                        f"{feedback}\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Send another packet or identify the rule."
                    )
                elif action == "identify":
                    rule_str = test_submission.identified_rule or "{}"
                    try:
                        submitted_rule = json.loads(rule_str)
                    except Exception:
                        submitted_rule = {}

                    entry["submitted"] = f"identify {submitted_rule}"

                    if submitted_rule == test_rule:
                        test_solved = True
                        test_turns.append(entry)
                        break
                    else:
                        feedback = "WRONG. Incorrect or missing conditions."
                        entry["feedback"] = feedback
                        test_turns.append(entry)
                        test_prompt = (
                            f"{feedback}\n\n"
                            f"Attempt {turn + 1} of {BUDGET}. Send a packet or try again."
                        )
                else:
                    entry["submitted"] = f"unknown action: {action}"
                    test_turns.append(entry)
                    test_prompt = f"Unknown action.\n\nAttempt {turn + 1} of {BUDGET}."

            if test_solved:
                test_ok += 1
            phases.append(
                {
                    "label": f"Final test {ti}/4",
                    "correct": test_rule,
                    "turns": test_turns,
                    "solved": test_solved,
                    "steps": test_num_steps,
                    "score": 1.0 if test_solved else 0.0,
                }
            )

    learning_score = weighted_learning_mean(learning_scores)
    test_score = test_ok / 4.0
    final_score = procedural_composite_score(learning_scores, test_score)
    _log_trace("PACKET FILTER", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    packet_filter_proc_learning.run(kbench.llm)

