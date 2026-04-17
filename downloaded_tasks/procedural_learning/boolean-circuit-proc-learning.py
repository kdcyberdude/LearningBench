#!/usr/bin/env python
# coding: utf-8

import kaggle_benchmarks as kbench
import random as _random
from dataclasses import dataclass, field


def _random_circuit(rng, n_gates, gate_set):
    wires = {0: lambda A, B, C: A, 1: lambda A, B, C: B, 2: lambda A, B, C: C}
    n_inputs = 3
    wire_count = n_inputs
    for _ in range(n_gates):
        gate = rng.choice(gate_set)
        if gate == "NOT":
            in_w = rng.randint(0, wire_count - 1)
            in_fn = wires[in_w]
            def make_not(f):
                return lambda A, B, C: 1 - f(A, B, C)
            wires[wire_count] = make_not(in_fn)
        else:
            in_w1 = rng.randint(0, wire_count - 1)
            in_w2 = rng.randint(0, wire_count - 1)
            f1 = wires[in_w1]
            f2 = wires[in_w2]
            if gate == "AND":
                def make_and(fa, fb):
                    return lambda A, B, C: fa(A, B, C) & fb(A, B, C)
                wires[wire_count] = make_and(f1, f2)
            else:
                def make_or(fa, fb):
                    return lambda A, B, C: fa(A, B, C) | fb(A, B, C)
                wires[wire_count] = make_or(f1, f2)
        wire_count += 1
    return wires[wire_count - 1]


_GATE_SET = ["AND", "OR", "NOT"]
_N_GATES = 4
_INPUTS_ORDER = [(A, B, C) for A in (0, 1) for B in (0, 1) for C in (0, 1)]

_circuits = []
for _seed in [11, 22, 33, 44, 55]:
    _rng = _random.Random(_seed)
    _fn = _random_circuit(_rng, _N_GATES, _GATE_SET)
    _tt = [_fn(A, B, C) for A, B, C in _INPUTS_ORDER]
    _circuits.append({"fn": _fn, "truth_table": _tt})

LEARNING_CIRCUITS = _circuits
_test_rng = _random.Random(99)
_test_fn = _random_circuit(_test_rng, _N_GATES, _GATE_SET)
TEST_CIRCUIT = {"fn": _test_fn, "truth_table": [_test_fn(A, B, C) for A, B, C in _INPUTS_ORDER]}

INPUTS_ORDER = _INPUTS_ORDER
BUDGET = 16


@dataclass
class _CircuitAction:
    action: str          # "probe" or "submit"
    A: int               # input bit; 0 when action="submit"
    B: int               # input bit; 0 when action="submit"
    C: int               # input bit; 0 when action="submit"
    truth_table: list    # 8-entry list; [] when action="probe"


_TASK_DESCRIPTION = (
    "Tests procedural learning of hidden 4-gate Boolean circuits (AND/OR/NOT, 3 inputs). "
    "The model probes input triples to observe outputs and must deduce the full 8-entry truth table. "
    "What makes it hard is the combinatorial space of possible gate compositions with a limited probe budget. "
    "Success = 50% learning efficiency across 5 circuits + 50% correct final test."
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


@kbench.task(
    name="boolean_circuit_proc_learning",
    description="Probe a hidden 4-gate Boolean circuit (AND/OR/NOT) with input triples to recover the full 8-entry truth table across 5 circuits. Score = learning_efficiency×0.5 + test_pass×0.5.",
)
def boolean_circuit_proc_learning(llm) -> float:
    """5 practice Boolean circuits (probe inputs, submit truth table, budget=9), then 1 no-hint test. Score=learning_avg×0.5+test×0.5."""
    phases = []
    test_passed = False

    with kbench.chats.new("boolean_circuit_proc_learning"):
        learning_scores = []
        initial_prompt = ""

        for idx, circuit in enumerate(LEARNING_CIRCUITS):
            fn = circuit["fn"]
            truth_table = circuit["truth_table"]
            turns = []
            solved = False
            num_steps = 0

            intro = (
                "You are probing a hidden 4-gate Boolean circuit with 3 inputs (A, B, C), "
                "each 0 or 1. Gates are AND, OR, NOT. Each turn you may either:\n"
                "  action='probe' with A=<0|1>, B=<0|1>, C=<0|1> to see the output, or\n"
                "  action='submit' with truth_table=[8 ints in order (A,B,C)=(0,0,0)..(1,1,1)] "
                "to guess the full truth table.\n"
                "You have a limited budget of turns per circuit.\n\n"
            ) if idx == 0 else ""

            next_prompt = (
                f"{intro}Practice {idx+1}/5 — new hidden circuit.\n"
                f"Budget: {BUDGET} turns total. Attempt 1 of {BUDGET}. Probe or submit."
            )

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                if idx == 0 and turn == 1:
                    initial_prompt = next_prompt
                try:
                    sub = llm.prompt(next_prompt, schema=_CircuitAction)
                except Exception:
                    entry = {"turn": turn, "submitted": "PARSE_ERROR", "feedback": "Failed to parse response — turn wasted."}
                    turns.append(entry)
                    next_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Probe or submit."
                    continue

                if sub.action == "submit":
                    submitted_tt = list(sub.truth_table) if sub.truth_table else []
                    submitted_str = f"submit: {submitted_tt}"
                    entry = {"turn": turn, "submitted": submitted_str}
                    if submitted_tt == truth_table:
                        solved = True
                        turns.append(entry)
                        break
                    n_wrong = sum(1 for a, b in zip(submitted_tt, truth_table) if a != b)
                    feedback = f"WRONG. {n_wrong} bits incorrect."
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = f"{feedback}\n\nAttempt {turn+1} of {BUDGET}. Probe more inputs or resubmit."
                else:
                    a_val = sub.A
                    b_val = sub.B
                    c_val = sub.C
                    output = fn(a_val, b_val, c_val)
                    feedback = f"probe ({a_val},{b_val},{c_val}) → {output}"
                    submitted_str = f"probe A={a_val}, B={b_val}, C={c_val}"
                    entry = {"turn": turn, "submitted": submitted_str, "feedback": feedback}
                    turns.append(entry)
                    next_prompt = f"{feedback}\n\nAttempt {turn+1} of {BUDGET}. Probe another input or submit truth table."

            eff = _efficiency_score(solved, num_steps, BUDGET)
            learning_scores.append(eff)
            phases.append({
                "label": f"Practice {idx+1}/5",
                "correct": truth_table,
                "turns": turns,
                "solved": solved,
                "steps": num_steps,
                "score": eff,
            })

        test_tt = TEST_CIRCUIT["truth_table"]
        try:
            test_sub = llm.prompt(
                "Final test — new hidden Boolean circuit.\n"
                "This is your only attempt. No hints. Submit the full 8-entry truth table "
                "for inputs (A,B,C) in order (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1).",
                schema=_CircuitAction,
            )
        except Exception:
            test_sub = None
        submitted_test_tt = list(test_sub.truth_table) if (test_sub is not None and test_sub.truth_table) else []
        test_passed = submitted_test_tt == test_tt
        phases.append({
            "label": "Final test",
            "correct": test_tt,
            "turns": [{"turn": 1, "submitted": f"submit: {submitted_test_tt}"}],
            "solved": test_passed,
            "steps": 1,
            "score": 1.0 if test_passed else 0.0,
        })

    final_score = sum(learning_scores) / 5 * 0.5 + (1.0 if test_passed else 0.0) * 0.5
    _log_trace("BOOLEAN CIRCUIT", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    boolean_circuit_proc_learning.run(kbench.llm)

