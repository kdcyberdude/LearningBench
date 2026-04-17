#!/usr/bin/env python
# coding: utf-8

import random as _random
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
    "Tests whether the model can find 8-symbol sequences accepted by 5 hidden DFAs over alphabet {A,B,C,D} "
    "by submitting probes and using rejection step feedback to extend valid prefixes. The hidden rule is a "
    "randomly generated DFA with 5 states — sequences are accepted only if they follow a specific path to "
    "the accept state. What makes it hard is that rejected paths must be diagnosed symbol by symbol using "
    "the rejection step, and the search space is 4^8 = 65,536 possible sequences. "
    "After practice, four new DFAs must each yield an accepting sequence. "
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
            print(f"    Turn {t['turn']}  submitted: {t['submitted']}", end="")
            if "feedback" in t:
                print(f"  →  {t['feedback']}", end="")
            print()
        status = "PASS ✓" if phase["solved"] else "FAIL ✗"
        print(f"    {status}  steps={phase['steps']}  score={phase['score']:.4f}")
    print(f"\n  Final score : {final_score:.4f}")
    print(f"{sep}\n")


BUDGET = 10
MIN_NECESSARY = 4
ALPHABET = list("ABCD")
SEQ_LEN = 8
NUM_STATES = 5

_TRAP = -1


def _build_dfa(
    rng: _random.Random, num_states: int, alphabet: list[str]
) -> tuple[dict, int, int]:
    states = list(range(num_states))
    start = 0
    accept = num_states - 1
    transitions: dict[tuple[int, str], int] = {}
    path_len = SEQ_LEN
    path_states = (
        [start] + [rng.choice(states[1:]) for _ in range(path_len - 1)] + [accept]
    )
    path_symbols = [rng.choice(alphabet) for _ in range(path_len)]
    for i, sym in enumerate(path_symbols):
        transitions[(path_states[i], sym)] = path_states[i + 1]
    for s in states:
        for sym in alphabet:
            if (s, sym) not in transitions:
                if rng.random() < 0.45:
                    transitions[(s, sym)] = _TRAP
                else:
                    transitions[(s, sym)] = rng.choice(states)
    return transitions, start, accept


def _run_dfa(
    transitions: dict, start: int, accept: int, sequence: str, alphabet: list[str]
) -> tuple[bool, int]:
    state = start
    for i, sym in enumerate(sequence):
        if sym not in alphabet:
            return False, i + 1
        next_state = transitions.get((state, sym), _TRAP)
        if next_state == _TRAP:
            return False, i + 1
        state = next_state
    return state == accept, len(sequence)


def _parse_sequence(seq: str, alphabet: list[str], seq_len: int):
    if not isinstance(seq, str):
        return None
    seq = seq.strip().upper().replace(" ", "").replace("-", "")
    if len(seq) != seq_len:
        return None
    if not all(c in alphabet for c in seq):
        return None
    return seq


_dfa_configs = []
for _seed in [101, 202, 303, 404, 505]:
    _rng = _random.Random(_seed)
    _trans, _st, _acc = _build_dfa(_rng, NUM_STATES, ALPHABET)
    _dfa_configs.append({"transitions": _trans, "start": _st, "accept": _acc})

LEARNING_DFAS = _dfa_configs

TEST_DFA_SEEDS = [606, 707, 808, 909]
TEST_DFAS = []
for _seed in TEST_DFA_SEEDS:
    _rng = _random.Random(_seed)
    _tr, _st, _acc = _build_dfa(_rng, NUM_STATES, ALPHABET)
    TEST_DFAS.append({"transitions": _tr, "start": _st, "accept": _acc})


@dataclass
class _SequenceSubmission:
    sequence: str


@kbench.task(
    name="state_machine_password_proc_learning",
    description=(
        "Probe 5 hidden DFAs to find accepting 8-symbol sequences over {A,B,C,D}, "
        "using rejection step info to extend valid prefixes, then four new DFAs. "
        "Score = weighted_learning_efficiency×0.5 + (tests_passed/4)×0.5."
    ),
)
def state_machine_password_proc_learning(llm) -> float:
    """5 practice DFAs (rejection step hints), then 4 final DFAs."""
    phases = []

    alpha_str = ", ".join(ALPHABET)

    with kbench.chats.new("state_machine_password"):
        learning_scores = []

        initial_prompt = ""
        for idx, dfa_cfg in enumerate(LEARNING_DFAS):
            transitions = dfa_cfg["transitions"]
            start = dfa_cfg["start"]
            accept = dfa_cfg["accept"]

            turns = []
            solved = False
            num_steps = 0
            history_lines: list[str] = []

            intro = (
                (
                    f"You are probing hidden state machines with alphabet {{{alpha_str}}}.\n\n"
                    f"Each probe is a sequence of exactly {SEQ_LEN} symbols.\n"
                    "The machine either ACCEPTS the sequence, or rejects it and tells you at which\n"
                    "step (1-based) the sequence was rejected.\n\n"
                    "If rejected at step K:\n"
                    "  • Symbols at positions 1 through K-1 were valid.\n"
                    "  • Symbol at position K caused a dead state.\n\n"
                    "Your goal: find a sequence that is ACCEPTED.\n"
                    "After 5 practice DFAs you face four final DFAs.\n\n"
                    "Scoring note: Your score has four components — transfer (30%): finding an "
                    "accepted sequence in the final DFAs; asymptote (25%): acceptance rate in "
                    "the later practice DFAs; trajectory (25%): whether your acceptance rate "
                    "improves across practice DFAs (a rising curve beats a flat one even at the "
                    "same average); consistency (20%): overall quality with later DFAs weighted "
                    "more. Using fewer probe sequences per practice DFA also boosts your "
                    "within-round efficiency score.\n\n"
                )
                if idx == 0
                else ""
            )

            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 — New DFA. Alphabet: {{{alpha_str}}}. Sequence length: {SEQ_LEN}.\n"
                f"Attempt 1 of {BUDGET}. Submit your {SEQ_LEN}-symbol sequence."
            )

            if idx == 0:
                initial_prompt = next_prompt

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                try:
                    submission = llm.prompt(next_prompt, schema=_SequenceSubmission)
                except Exception:
                    entry = {
                        "turn": turn,
                        "submitted": "PARSE_ERROR",
                        "feedback": "Failed to parse response — turn wasted.",
                    }
                    turns.append(entry)
                    next_prompt = (
                        f"Your last response could not be parsed. Please follow the schema exactly.\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Submit your {SEQ_LEN}-symbol sequence."
                    )
                    continue

                seq = _parse_sequence(submission.sequence, ALPHABET, SEQ_LEN)
                entry = {
                    "turn": turn,
                    "submitted": submission.sequence,
                }

                if seq is None:
                    feedback = f"INVALID: must be exactly {SEQ_LEN} symbols from {{{alpha_str}}}."
                    entry["feedback"] = feedback
                    history_lines.append(
                        f"  Attempt {turn}: {submission.sequence!r} → {feedback}"
                    )
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Submit a valid {SEQ_LEN}-symbol sequence."
                    )
                    continue

                entry["submitted"] = seq
                accepted, reject_step = _run_dfa(
                    transitions, start, accept, seq, ALPHABET
                )

                if accepted:
                    solved = True
                    history_lines.append(f"  Attempt {turn}: {seq!r} → ACCEPTED")
                    turns.append(entry)
                    break

                feedback = f"REJECTED at step {reject_step}"
                entry["feedback"] = feedback
                history_lines.append(f"  Attempt {turn}: {seq!r} → {feedback}")
                turns.append(entry)

                history_block = "\n".join(history_lines[-5:])
                next_prompt = (
                    f"{feedback}\n\n"
                    f"Previous attempts:\n{history_block}\n\n"
                    f"Attempt {turn + 1} of {BUDGET}. Submit your next {SEQ_LEN}-symbol sequence."
                )

            eff = efficiency_score(solved, num_steps, BUDGET, MIN_NECESSARY)
            learning_scores.append(eff)
            phases.append(
                {
                    "label": f"Practice {idx + 1}/5",
                    "correct": "ACCEPTED_SEQUENCE",
                    "turns": turns,
                    "solved": solved,
                    "steps": num_steps,
                    "score": eff,
                }
            )

        test_ok = 0
        for ti, test_cfg in enumerate(TEST_DFAS, start=1):
            test_turns = []
            test_solved = False
            test_num_steps = 0
            test_history: list[str] = []

            test_prompt = (
                f"Final test {ti}/4 — New DFA. Alphabet: {{{alpha_str}}}. Sequence length: {SEQ_LEN}.\n"
                f"Budget: {BUDGET} attempts.\n"
                f"Attempt 1 of {BUDGET}. Submit your {SEQ_LEN}-symbol sequence."
            )

            for turn in range(1, BUDGET + 1):
                test_num_steps = turn
                try:
                    test_submission = llm.prompt(test_prompt, schema=_SequenceSubmission)
                except Exception:
                    entry = {
                        "turn": turn,
                        "submitted": "PARSE_ERROR",
                        "feedback": "Failed to parse response — turn wasted.",
                    }
                    test_turns.append(entry)
                    test_prompt = (
                        f"Your last response could not be parsed. Please follow the schema exactly.\n\n"
                        f"Attempt {turn + 1} of {BUDGET}. Submit your {SEQ_LEN}-symbol sequence."
                    )
                    continue

                seq = _parse_sequence(test_submission.sequence, ALPHABET, SEQ_LEN)
                entry = {
                    "turn": turn,
                    "submitted": test_submission.sequence,
                }

                if seq is None:
                    feedback = (
                        f"INVALID: must be exactly {SEQ_LEN} symbols from {{{alpha_str}}}."
                    )
                    entry["feedback"] = feedback
                    test_history.append(
                        f"  Attempt {turn}: {test_submission.sequence!r} → {feedback}"
                    )
                    test_turns.append(entry)
                    test_prompt = f"{feedback}\n\nAttempt {turn + 1} of {BUDGET}. Submit a valid sequence."
                    continue

                entry["submitted"] = seq
                accepted, reject_step = _run_dfa(
                    test_cfg["transitions"],
                    test_cfg["start"],
                    test_cfg["accept"],
                    seq,
                    ALPHABET,
                )

                if accepted:
                    test_solved = True
                    test_history.append(f"  Attempt {turn}: {seq!r} → ACCEPTED")
                    test_turns.append(entry)
                    break

                feedback = f"REJECTED at step {reject_step}"
                entry["feedback"] = feedback
                test_history.append(f"  Attempt {turn}: {seq!r} → {feedback}")
                test_turns.append(entry)

                history_block = "\n".join(test_history[-5:])
                test_prompt = (
                    f"{feedback}\n\n"
                    f"Previous attempts:\n{history_block}\n\n"
                    f"Attempt {turn + 1} of {BUDGET}. Submit your next {SEQ_LEN}-symbol sequence."
                )

            if test_solved:
                test_ok += 1
            phases.append(
                {
                    "label": f"Final test {ti}/4",
                    "correct": "ACCEPTED_SEQUENCE",
                    "turns": test_turns,
                    "solved": test_solved,
                    "steps": test_num_steps,
                    "score": 1.0 if test_solved else 0.0,
                }
            )

    learning_score = weighted_learning_mean(learning_scores)
    test_score = test_ok / 4.0
    final_score = procedural_composite_score(learning_scores, test_score)
    _log_trace("STATE MACHINE PASSWORD", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    state_machine_password_proc_learning.run(kbench.llm)

