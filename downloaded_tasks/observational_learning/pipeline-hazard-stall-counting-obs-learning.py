#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can jointly infer three latent constants from execution "
    "trace observations of a fictional 'Chrono-Forge' execution engine. "
    "The hidden rule is: total_ticks = boot_overhead + Σ tier_cost(instr) + "
    "contention_penalty × count_of_adjacent_same-tier_pairs, where "
    "tier costs (α=1, β=2, γ=3), boot_overhead=5, contention_penalty=3 are all "
    "unknown and must be inferred from the examples. Instruction types (tiers) are "
    "denoted by abstract symbols in the prompt, and register names are non-standard. "
    "The task requires simultaneous multi-variable inference, not simple pattern matching."
)

_FIXED_SEED = 42

# ---------------------------------------------------------------------------
# Hidden constants (never stated in the prompt)
# ---------------------------------------------------------------------------
_BOOT = 5          # boot overhead ticks added once per program
_TIER_COST = {     # ticks consumed by each instruction tier
    "alpha": 1,
    "beta":  2,
    "gamma": 3,
}
_CONTENTION = 3    # extra ticks when two adjacent instructions share the same tier


def _total_ticks(instructions: list) -> int:
    """
    total_ticks = _BOOT
                  + sum of tier costs for each instruction
                  + _CONTENTION for each adjacent pair (i, i+1) that share the same tier
    """
    base = _BOOT + sum(_TIER_COST[tier] for _, tier, _, _ in instructions)
    penalties = sum(
        _CONTENTION
        for i in range(len(instructions) - 1)
        if instructions[i][1] == instructions[i + 1][1]
    )
    return base + penalties


def _count_contention(instructions: list) -> int:
    return sum(
        1
        for i in range(len(instructions) - 1)
        if instructions[i][1] == instructions[i + 1][1]
    )


# ---------------------------------------------------------------------------
# Instruction format: (symbol, tier, dest_slot, src_slots)
# Tiers presented with abstract symbols in the prompt:
#   alpha → "◈"  (tier 1, cheap)
#   beta  → "⬟"  (tier 2, medium)
#   gamma → "⬡"  (tier 3, expensive)
# Register slots: non-standard names to prevent lookup (P0..P7)
# ---------------------------------------------------------------------------
_TIER_SYM = {"alpha": "◈", "beta": "⬟", "gamma": "⬡"}


def _fmt(prog):
    lines = []
    for i, (label, tier, dest, srcs) in enumerate(prog, 1):
        sym = _TIER_SYM[tier]
        src_str = " ".join(srcs)
        lines.append(f"  {i:2d}. [{sym}] {label}  {dest} <- {src_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demonstration programs
# Structured to allow unique inference:
#   - Several all-single-tier programs pin individual tier costs + boot
#   - Contention-varied programs of the same tier composition isolate penalty
#   - Mixed programs validate the combined formula
#
# Variable naming: (mnemonic_label, tier, dest_slot, [src_slots])
# Labels are abstract (VX, WY, etc.) to avoid domain priming
# ---------------------------------------------------------------------------

# --- Group A: Single-tier, no contention ---
# All instructions same tier → every adjacent pair has contention.
# A1: 3 × alpha, 2 contention → 5 + 3×1 + 2×3 = 5+3+6 = 14
_DEMO_A1 = [
    ("VX", "alpha", "P0", ["P1", "P2"]),
    ("WY", "alpha", "P3", ["P0", "P4"]),
    ("VX", "alpha", "P1", ["P3", "P5"]),
]

# A2: 3 × beta, 2 contention → 5 + 3×2 + 2×3 = 5+6+6 = 17
_DEMO_A2 = [
    ("WY", "beta", "P2", ["P0", "P3"]),
    ("VX", "beta", "P5", ["P1", "P4"]),
    ("ZQ", "beta", "P0", ["P2", "P5"]),
]

# A3: 3 × gamma, 2 contention → 5 + 3×3 + 2×3 = 5+9+6 = 20
_DEMO_A3 = [
    ("ZQ", "gamma", "P4", ["P0", "P1"]),
    ("VX", "gamma", "P2", ["P3", "P5"]),
    ("WY", "gamma", "P1", ["P4", "P2"]),
]

# --- Group B: Single-tier, no contention (alternating via spacing trick) ---
# Actually we use length=1 to get exact boot+tier_cost, isolating boot + single cost.

# B1: 1 × alpha, 0 contention → 5 + 1 = 6
_DEMO_B1 = [
    ("VX", "alpha", "P0", ["P1", "P2"]),
]

# B2: 1 × beta, 0 contention → 5 + 2 = 7
_DEMO_B2 = [
    ("ZQ", "beta", "P3", ["P1", "P4"]),
]

# B3: 1 × gamma, 0 contention → 5 + 3 = 8
_DEMO_B3 = [
    ("WY", "gamma", "P5", ["P0", "P2"]),
]

# --- Group C: Mixed tiers, no contention ---
# C1: alpha, beta, gamma (all different, 0 contention) → 5 + 1+2+3 = 11
_DEMO_C1 = [
    ("VX", "alpha", "P1", ["P0", "P2"]),
    ("ZQ", "beta",  "P3", ["P1", "P4"]),
    ("WY", "gamma", "P2", ["P3", "P5"]),
]

# C2: beta, gamma, alpha (all different, 0 contention) → 5 + 2+3+1 = 11
_DEMO_C2 = [
    ("ZQ", "beta",  "P0", ["P2", "P4"]),
    ("WY", "gamma", "P1", ["P0", "P3"]),
    ("VX", "alpha", "P4", ["P1", "P5"]),
]

# C3: alpha, gamma, beta, alpha (0 contention, no adjacent same-tier) → 5 + 1+3+2+1 = 12
_DEMO_C3 = [
    ("VX", "alpha", "P0", ["P2", "P5"]),
    ("WY", "gamma", "P3", ["P0", "P1"]),
    ("ZQ", "beta",  "P1", ["P3", "P4"]),
    ("VX", "alpha", "P5", ["P1", "P2"]),
]

# --- Group D: Mixed tiers with contention ---
# D1: alpha, alpha, beta, gamma → tiers: a,a,b,g
#   costs: 1+1+2+3=7, contention: (a,a)=1 pair → 5+7+3 = 15
_DEMO_D1 = [
    ("VX", "alpha", "P2", ["P0", "P4"]),
    ("WY", "alpha", "P4", ["P2", "P5"]),
    ("ZQ", "beta",  "P0", ["P1", "P3"]),
    ("VX", "gamma", "P3", ["P0", "P2"]),
]

# D2: gamma, gamma, alpha, beta, gamma → tiers: g,g,a,b,g
#   costs: 3+3+1+2+3=12, contention: (g,g)=1 pair → 5+12+3 = 20
_DEMO_D2 = [
    ("WY", "gamma", "P1", ["P0", "P3"]),
    ("ZQ", "gamma", "P5", ["P1", "P4"]),
    ("VX", "alpha", "P0", ["P5", "P2"]),
    ("WY", "beta",  "P3", ["P0", "P1"]),
    ("ZQ", "gamma", "P2", ["P3", "P5"]),
]

# D3: beta, beta, beta, alpha → tiers: b,b,b,a
#   costs: 2+2+2+1=7, contention: (b,b),(b,b)=2 pairs → 5+7+6 = 18
_DEMO_D3 = [
    ("ZQ", "beta",  "P3", ["P0", "P2"]),
    ("VX", "beta",  "P1", ["P3", "P4"]),
    ("WY", "beta",  "P0", ["P1", "P5"]),
    ("VX", "alpha", "P4", ["P0", "P3"]),
]

# D4: gamma, alpha, gamma, alpha, beta → tiers: g,a,g,a,b
#   costs: 3+1+3+1+2=10, contention: 0 pairs → 5+10+0 = 15
_DEMO_D4 = [
    ("WY", "gamma", "P0", ["P2", "P4"]),
    ("VX", "alpha", "P3", ["P0", "P5"]),
    ("ZQ", "gamma", "P1", ["P3", "P2"]),
    ("WY", "alpha", "P4", ["P1", "P0"]),
    ("VX", "beta",  "P2", ["P4", "P3"]),
]

# D5: alpha, beta, beta, gamma, gamma, alpha → tiers: a,b,b,g,g,a
#   costs: 1+2+2+3+3+1=12, contention: (b,b),(g,g)=2 pairs → 5+12+6 = 23
_DEMO_D5 = [
    ("VX", "alpha", "P5", ["P0", "P3"]),
    ("ZQ", "beta",  "P0", ["P5", "P1"]),
    ("WY", "beta",  "P2", ["P0", "P4"]),
    ("VX", "gamma", "P1", ["P2", "P5"]),
    ("ZQ", "gamma", "P3", ["P1", "P0"]),
    ("WY", "alpha", "P4", ["P3", "P2"]),
]

_ALL_DEMOS = [
    _DEMO_A1, _DEMO_A2, _DEMO_A3,
    _DEMO_B1, _DEMO_B2, _DEMO_B3,
    _DEMO_C1, _DEMO_C2, _DEMO_C3,
    _DEMO_D1, _DEMO_D2, _DEMO_D3, _DEMO_D4, _DEMO_D5,
]

# ---------------------------------------------------------------------------
# Test programs
# ---------------------------------------------------------------------------

# T1: gamma, beta, alpha, gamma, beta → tiers: g,b,a,g,b
#   costs: 3+2+1+3+2=11, contention: 0 pairs → 5+11+0 = 16
_TEST_PROG1 = [
    ("WY", "gamma", "P0", ["P3", "P5"]),
    ("ZQ", "beta",  "P3", ["P0", "P1"]),
    ("VX", "alpha", "P1", ["P3", "P4"]),
    ("WY", "gamma", "P4", ["P1", "P2"]),
    ("VX", "beta",  "P2", ["P4", "P0"]),
]

# T2: beta, beta, gamma, gamma, alpha, alpha → tiers: b,b,g,g,a,a
#   costs: 2+2+3+3+1+1=12, contention: (b,b),(g,g),(a,a)=3 pairs → 5+12+9 = 26
_TEST_PROG2 = [
    ("ZQ", "beta",  "P1", ["P0", "P4"]),
    ("WY", "beta",  "P4", ["P1", "P2"]),
    ("VX", "gamma", "P0", ["P4", "P3"]),
    ("ZQ", "gamma", "P2", ["P0", "P5"]),
    ("WY", "alpha", "P3", ["P2", "P1"]),
    ("VX", "alpha", "P5", ["P3", "P0"]),
]

# T3: alpha, gamma, alpha, gamma, alpha → tiers: a,g,a,g,a
#   costs: 1+3+1+3+1=9, contention: 0 pairs → 5+9+0 = 14
_TEST_PROG3 = [
    ("VX", "alpha", "P2", ["P0", "P5"]),
    ("ZQ", "gamma", "P5", ["P2", "P3"]),
    ("WY", "alpha", "P0", ["P5", "P4"]),
    ("VX", "gamma", "P4", ["P0", "P1"]),
    ("ZQ", "alpha", "P1", ["P4", "P2"]),
]

# T4: gamma, gamma, gamma, beta, alpha, gamma, gamma → tiers: g,g,g,b,a,g,g
#   costs: 3+3+3+2+1+3+3=18, contention: (g,g),(g,g),(g,g)=3 pairs → 5+18+9 = 32
_TEST_PROG4 = [
    ("WY", "gamma", "P3", ["P0", "P2"]),
    ("ZQ", "gamma", "P0", ["P3", "P5"]),
    ("VX", "gamma", "P5", ["P0", "P1"]),
    ("WY", "beta",  "P1", ["P5", "P4"]),
    ("ZQ", "alpha", "P4", ["P1", "P0"]),
    ("VX", "gamma", "P2", ["P4", "P3"]),
    ("WY", "gamma", "P0", ["P2", "P1"]),
]

_TEST_PROGRAMS = [_TEST_PROG1, _TEST_PROG2, _TEST_PROG3, _TEST_PROG4]
_TEST_CASES = [(prog, _total_ticks(prog)) for prog in _TEST_PROGRAMS]


def _prepare():
    lines = [
        "You are observing an execution engine process instruction sequences.",
        "Each instruction belongs to one of three execution tiers, denoted by the",
        "symbols ◈, ⬟, and ⬡. Registers are named P0 through P7.",
        "The engine records a total tick count for each program.",
        "",
        "Study the following observations carefully:",
        "",
        "Observations (program → total ticks):",
    ]
    for i, prog in enumerate(_ALL_DEMOS, 1):
        lines.append(f"\n  Program {i}:")
        lines.append(_fmt(prog))
        lines.append(f"  → {_total_ticks(prog)} ticks")

    lines += [
        "",
        "Predict the total ticks for each of the following programs:",
    ]
    for q, (prog, _) in enumerate(_TEST_CASES, 1):
        lines.append(f"\n  Q{q}:")
        lines.append(_fmt(prog))
        lines.append(f"  → ? ticks")

    lines += [
        "",
        "Submit your answers as ticks_1, ticks_2, ticks_3, ticks_4.",
    ]
    prompt = "\n".join(lines)

    def grade_fn(response):
        results = []
        correct = 0
        for q_idx, (prog, exp) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"ticks_{q_idx}", None)
            try:
                got = int(raw)
                is_correct = got == exp
            except (TypeError, ValueError):
                got = raw
                is_correct = False
            results.append(
                {
                    "q": q_idx,
                    "expected": exp,
                    "got": got,
                    "correct": is_correct,
                }
            )
            if is_correct:
                correct += 1
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    ticks_1: int
    ticks_2: int
    ticks_3: int
    ticks_4: int


@kbench.task(
    name="pipeline_hazard_stall_counting_obs_learning",
    description=(
        "Infer three hidden constants from 14 execution traces, then predict total ticks for 4 programs using the inferred rule."
    ),
)
def pipeline_hazard_stall_counting_obs_learning(llm) -> float:
    """
    total_ticks = BOOT + sum(TIER_COST[tier]) + CONTENTION * adjacent_same_tier
    (BOOT=5, TIER_COST={alpha:1, beta:2, gamma:3}, CONTENTION=3; all are hidden)
    """
    prompt, grade_fn = _prepare()

    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None

    if response is not None:
        score, test_results = grade_fn(response)
    else:
        score = 0.0
        test_results = [
            {"q": i, "expected": _TEST_CASES[i - 1][1], "got": None, "correct": False}
            for i in range(1, 5)
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )

    sep = "=" * 70
    print(f"\n{sep}\n  pipeline_hazard_stall_counting_obs_learning\n{sep}")
    print(f"\n  TASK: {_TASK_DESCRIPTION}")
    print(f"\n  PROMPT:\n{prompt}")
    if reasoning:
        print(f"\n  REASONING:\n{reasoning}")
    print(f"\n  TEST RESULTS:")
    for r in test_results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"    [{status}] Q{r['q']}: expected={r['expected']!r}  got={r['got']!r}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")

    return score


if __name__ == "__main__":
    pipeline_hazard_stall_counting_obs_learning.run(kbench.llm)

