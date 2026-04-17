#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import random
import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests inference-time associative learning of a hidden CUMULATIVE STATE MACHINE "
    "rewriting rule — a fundamentally different mechanism from 'local token infection'. "
    "No rules are stated. The model must infer three facts from 500 input→output examples: "
    "(1) K and J are state-control tokens that do not appear in the output; "
    "(2) K increments a hidden integer register (mod 4), J decrements it (mod 4), "
    "starting at 0; every non-special token T in the sequence is output as P^state(T) "
    "where state is the register value AT THE MOMENT T is encountered — meaning K/J "
    "affect ALL subsequent non-special tokens, not just the next one; "
    "(3) P is the 4-cycle A→B→C→D→A (order 4), so state 0=identity, 1=+1 step, "
    "2=+2 steps, 3=−1 step (inverse). "
    "Key difficulty axes: "
    "(a) The mechanism is NON-LOCAL — every token's output depends on ALL preceding K/J tokens; "
    "(b) J from state 0 yields state 3 (negative-wrap trap); "
    "(c) JJ ≠ identity (J has order 4, not 2); "
    "(d) State accumulates PAST intervening non-special tokens; "
    "(e) Longer test sequences (6–7 tokens) require tracking 4–6 state transitions "
    "without error. "
    "Each test question is engineered to catch a different failure mode: "
    "local-infection assumption, JJ-cancels assumption, JJJ-period-3 assumption, "
    "state-reset-after-J assumption, and multi-step accumulation errors."
)

# ---------------------------------------------------------------------------
# Hidden rule (not shown to models):
#   P: A→B, B→C, C→D, D→A  (single 4-cycle, order 4)
#   Hidden register 'state', initialised to 0.
#   Reading left to right:
#     K  → state = (state + 1) mod 4 ; K does NOT appear in output
#     J  → state = (state - 1) mod 4 ; J does NOT appear in output
#     T  → output P^state(T)          ; state is unchanged by non-special tokens
#   Trailing K/J (no following non-special token) affect state but produce no output.
# ---------------------------------------------------------------------------

_CYCLE = {"A": "B", "B": "C", "C": "D", "D": "A"}
_DATA_TOKENS = ["A", "B", "C", "D"]
_CONTROL_TOKENS = ["K", "J"]


def _apply_p(token: str, steps: int) -> str:
    """Apply P^steps to a single data token (steps mod 4)."""
    steps = steps % 4
    t = token
    for _ in range(steps):
        t = _CYCLE[t]
    return t


def _apply_rule(sequence: list[str]) -> list[str]:
    """
    Given a list of tokens (data: A/B/C/D; control: K/J), apply the hidden
    cumulative-state-machine rule and return the output token list.
    K increments the state register (mod 4), J decrements it (mod 4);
    neither appears in output. Data tokens are emitted as P^state(token).
    """
    state = 0
    output: list[str] = []
    for tok in sequence:
        if tok == "K":
            state = (state + 1) % 4
        elif tok == "J":
            state = (state - 1) % 4
        else:
            output.append(_apply_p(tok, state))
    return output


def _seq_to_str(tokens: list[str]) -> str:
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Training-sample generator (seeded RNG → deterministic)
# ---------------------------------------------------------------------------

def _build_training_pairs(n: int = 100, seed: int = 12) -> list[tuple[str, str]]:
    """
    Generate `n` unique input→output pairs that collectively demonstrate the rule.
    Each sequence has 1–6 tokens drawn from {A,B,C,D,K,J}, with at least one
    data token so the output is non-empty.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []

    while len(pairs) < n:
        length = rng.randint(1, 6)
        # Ensure at least one data token by forcing the last token to be a data token
        seq: list[str] = []
        for i in range(length - 1):
            # Weight control tokens less heavily so sequences don't degenerate to all-K/J
            seq.append(rng.choice(_DATA_TOKENS * 3 + _CONTROL_TOKENS))
        seq.append(rng.choice(_DATA_TOKENS))  # guarantee ≥1 data token

        key = _seq_to_str(seq)
        if key in seen:
            continue
        seen.add(key)

        out = _apply_rule(seq)
        pairs.append((key, _seq_to_str(out)))

    return pairs


_TRAINING_PAIRS = _build_training_pairs()


def _build_training_prompt() -> str:
    lines = [
        "Study the following input → output transformations carefully.",
        "Infer the hidden rule and apply it to the questions below.",
        "",
        "Examples:",
    ]
    for inp, out in _TRAINING_PAIRS:
        lines.append(f"  {inp:<30} →  {out}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fixed test questions (engineered to catch specific failure modes)
# ---------------------------------------------------------------------------

_TEST_SEQUENCES: list[list[str]] = [
    ["K", "A", "J", "K", "B", "C"],   # Q1: local-infection trap
    ["J", "J", "A", "B"],              # Q2: JJ-cancels assumption
    ["B", "K", "K", "A", "C"],         # Q3: JJJ-period-3 assumption
    ["K", "K", "B", "J", "J", "J", "A"],  # Q4: state-reset-after-J
    ["K", "A", "B", "J", "J", "C", "D"],  # Q5: multi-step accumulation
    ["J", "K", "A", "B"],              # Q6: J-wrap then K recovery
    ["K", "K", "K", "A", "J", "K", "B"],  # Q7: long accumulation chain
]

_STATE_REWRITE_EXPECTED: dict[str, str] = {
    f"q_{i}": _seq_to_str(_apply_rule(seq))
    for i, seq in enumerate(_TEST_SEQUENCES, 1)
}


def _build_test_prompt() -> str:
    lines = ["Questions:"]
    for i, seq in enumerate(_TEST_SEQUENCES, 1):
        lines.append(f"  Question {i}: {_seq_to_str(seq)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log_trace(
    task: str, description: str, prompt: str, answers: dict, expected: dict, score: float
) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    print(f"\n  RESPONSES:")
    for key in expected:
        actual = answers.get(key, "?")
        exp = expected[key]
        match = "✓" if _str_match(str(exp), str(actual)) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _str_match(expected: str, actual: str) -> bool:
    """Return True if expected appears anywhere in actual (case-insensitive)."""
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


# ---------------------------------------------------------------------------
# Answer schema
# ---------------------------------------------------------------------------

@dataclass
class StateRewriteAnswer:
    q_1: str
    q_2: str
    q_3: str
    q_4: str
    q_5: str
    q_6: str
    q_7: str


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@kbench.task(
    name="cumulative_state_rewrite_assoc_learning",
    description=(
        "Infer a hidden cumulative-state-machine rewriting rule from 500 examples; "
        "no rules stated."
    ),
)
def cumulative_state_rewrite_assoc_learning(llm) -> float:
    """Hidden cumulative state machine inferred from examples; return fraction correct."""

    prompt = _build_training_prompt() + "\n" + _build_test_prompt()

    result = llm.prompt(prompt, schema=StateRewriteAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_STATE_REWRITE_EXPECTED)
    for key, exp_val in _STATE_REWRITE_EXPECTED.items():
        act = " ".join(str(getattr(result, key)).strip().upper().split())
        expn = " ".join(str(exp_val).strip().upper().split())
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _STATE_REWRITE_EXPECTED}
    _log_trace(
        "cumulative_state_rewrite_assoc_learning",
        _TASK_DESCRIPTION,
        prompt,
        answers,
        _STATE_REWRITE_EXPECTED,
        score,
    )
    return score


if __name__ == "__main__":
    cumulative_state_rewrite_assoc_learning.run(kbench.llm)

