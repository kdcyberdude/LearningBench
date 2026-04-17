#!/usr/bin/env python
# coding: utf-8

"""
Observational learning: infer opaque instruction semantics from execution traces.

Design goals:
  • Difficulty: Prompts expose only programs and snapshots (Q0..Q6); no textbook name
    for the formalism. Short demos pin each semantic clause; task programs include
    long runs (~150+ steps) that stress program-counter bookkeeping without large tally
    arithmetic (values stay small; difficulty is control-flow simulation).
  • Solution uniqueness (intended reference semantics):
      – Q0/Q1: single-line increments with the partner tally fixed (Obs 1–2).
      – Q2/Q3: saturating decrement (Obs 3 vs Obs 4–5 rule out wrap/negative DEC).
      – Q4/Q5: conditional jump when the *named* tally is exactly zero, fall-through
        otherwise; polarity pinned by Obs 6–7; register binding by Obs 8–9; no hidden
        tally change on a taken branch (Obs 10–11).
      – Operand after Q4/Q5 is an absolute line index (not pc-relative): Obs 13 takes
        JZ B 8 from pc=5; a pc-relative reading (pc+8) overshoots the program and
        changes the trace.
      – Q6: clean halt on decode (traces stop with next=Q6).
    Any swap of Q0↔Q1 or Q2↔Q3, inverted JZ polarity, nonsaturating DEC, or
    pc-relative jump targets breaks at least one observation.
  • Consistency: Single reference interpreter; all examples deterministic; tests
    precomputed under that semantics.
  • Novelty / anti-contamination: Surface syntax is ad hoc opcode labels plus traces,
    not a named “Minsky machine” or standard textbook problem statement.
  • Human vs model: Humans can trace pc/A/B on paper; long mixed jump paths are a
    known weakness for step-by-step LLM rollouts.

Internal instruction encoding (rendered as Q0..Q6 in the prompt):
  ("INC","A"), ("INC","B"), ("DEC","A"), ("DEC","B"),
  ("JZ","A",target), ("JZ","B",target), ("HALT",)
"""

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Infer semantics of opaque opcodes (Q0..Q6) from execution traces; run six fixed "
    "task programs and return final A,B. Demos pin saturated decrement, zero-tests, "
    "halt, and absolute jump targets."
)

Instr = tuple

# Maps internal tuple ops to opaque prompt tokens (stable permutation — not INC/DEC/JZ).
_Q = {
    ("INC", "A"): "Q0",
    ("INC", "B"): "Q1",
    ("DEC", "A"): "Q2",
    ("DEC", "B"): "Q3",
    ("HALT",): "Q6",
}


def _format_instr(ins: Instr) -> str:
    if ins[0] == "JZ":
        return f"Q4 {ins[2]}" if ins[1] == "A" else f"Q5 {ins[2]}"
    if ins[0] == "HALT":
        return "Q6"
    return _Q[(ins[0], ins[1])]


def _log_trace(task, description, prompt, test_results, score, reasoning=""):
    sep = "=" * 70
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    if reasoning:
        print(f"\n  REASONING:\n{reasoning}")
    print(f"\n  TEST RESULTS:")
    for r in test_results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"    [{status}] Q{r['q']}: expected={r['expected']!r}  got={r['got']!r}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _execute(prog: list, a0: int, b0: int, max_steps: int = 500) -> list:
    """Reference semantics: non-negative counters; DEC saturates at 0; JZ jumps when tested counter is 0."""
    a, b, pc = a0, b0, 0
    trace: list = [(pc, a, b)]
    for _ in range(max_steps):
        if pc >= len(prog):
            break
        instr = prog[pc]
        if instr[0] == "HALT":
            break
        if instr[0] == "INC":
            if instr[1] == "A":
                a += 1
            else:
                b += 1
            pc += 1
        elif instr[0] == "DEC":
            if instr[1] == "A":
                a = max(0, a - 1)
            else:
                b = max(0, b - 1)
            pc += 1
        elif instr[0] == "JZ":
            counter_val = a if instr[1] == "A" else b
            if counter_val == 0:
                pc = instr[2]
            else:
                pc += 1
        else:
            pc += 1
        trace.append((pc, a, b))
    return trace


def _render_trace(idx: int, prog: list, a0: int, b0: int, trace: list) -> str:
    lines = [f"=== Observation {idx} === (initial A={a0}, B={b0})", "Program:"]
    for i, ins in enumerate(prog):
        lines.append(f"  {i}: {_format_instr(ins)}")
    lines.append("Execution snapshots (pc points at the instruction about to be decoded next):")
    for pc, a, b in trace:
        if pc < len(prog):
            lines.append(f"  pc={pc} next={_format_instr(prog[pc])} | A={a}, B={b}")
        else:
            lines.append(f"  pc={pc} (past last line) | A={a}, B={b}")
    return "\n".join(lines)


def _parse_answer(s) -> tuple:
    if not isinstance(s, str):
        return None, None
    import re

    ma = re.search(r"A\s*=\s*(-?\d+)", s, re.IGNORECASE)
    mb = re.search(r"B\s*=\s*(-?\d+)", s, re.IGNORECASE)
    if ma and mb:
        return int(ma.group(1)), int(mb.group(1))
    return None, None


def _curated_demonstrations():
    """Return list of (prog, a0, b0). Order matters for how clauses are pinned."""
    ia, ib, da, db, halt = ("INC", "A"), ("INC", "B"), ("DEC", "A"), ("DEC", "B"), ("HALT",)

    demos = []

    # 1–2: increment which register
    demos.append(([ia, halt], 4, 7))
    demos.append(([ib, halt], 4, 7))

    # 3: decrement from positive (three steps on A)
    demos.append(([da, da, da, halt], 5, 2))

    # 4–5: saturated DEC — must stay 0
    demos.append(([da, halt], 0, 3))
    demos.append(([db, halt], 2, 0))

    # 6–7: JZ B (Q5): nonzero falls through; zero jumps
    # 0: Q5 3  1: Q1  2: Q0  3: Q6
    jz_b_demo = [
        ("JZ", "B", 3),
        ib,
        ia,
        halt,
    ]
    demos.append((jz_b_demo, 0, 2))
    demos.append((jz_b_demo, 0, 0))

    # 8–9: JZ A discrimination (two initial A values)
    # 0: Q4 2  1: Q6  2: Q0  — if A=0 skip Q0
    jz_a_short = [("JZ", "A", 2), halt, ia, halt]
    demos.append((jz_a_short, 0, 5))
    demos.append((jz_a_short, 2, 5))

    # 10: Q4 taken: A unchanged by branch itself (A=0 before/after line 0, B constant)
    # 0:Q4 3  1:Q0  2:Q4 1  3:Q6  — from A=0,B=0: jump to 3 halt; shows no hidden dec on branch
    branch_no_mut = [
        ("JZ", "A", 3),
        ia,
        ("JZ", "A", 1),
        halt,
    ]
    demos.append((branch_no_mut, 0, 0))

    # 11: Q5 taken from B=0, A unchanged across guard
    guard_b = [
        ("JZ", "B", 2),
        ib,
        halt,
    ]
    demos.append((guard_b, 9, 0))

    # 12: loop until A=0 using Q2 + Q4; Q5 with B=0 acts as unconditional back-edge
    # 0:Q4 3  1:Q2  2:Q5 0  3:Q6
    loop_dec_a = [
        ("JZ", "A", 3),
        da,
        ("JZ", "B", 0),
        halt,
    ]
    demos.append((loop_dec_a, 3, 0))

    # 13: absolute jump target vs pc-relative — from pc=5, Q5 8 must mean line 8, not pc+8
    #     (which would leave the program). Both branches taken; tallies stay zero.
    abs_jmp = [
        ("JZ", "A", 5),
        ia,
        ia,
        ia,
        ia,
        ("JZ", "B", 8),
        ib,
        ib,
        halt,
    ]
    demos.append((abs_jmp, 0, 0))

    return demos


def _curated_tests():
    """Six tests: mixed control flow plus two long traces (~150 steps) for pc/tally drift."""
    ia, ib, da, db, halt = ("INC", "A"), ("INC", "B"), ("DEC", "A"), ("DEC", "B"), ("HALT",)

    # T1: nested guards — B clears first, then A loop
    # 0:Q5 8  1:Q3  2:Q5 0  3:Q4 7  4:Q2  5:Q4 3  6:Q1  7:Q6  8:Q6
    t1 = [
        ("JZ", "B", 8),
        db,
        ("JZ", "B", 0),
        ("JZ", "A", 7),
        da,
        ("JZ", "A", 3),
        ib,
        halt,
        halt,
    ]

    # T2: bounce between two JZs until A exhausted; B acts as flag
    # 0:Q4 5  1:Q5 4  2:Q2  3:Q4 0  4:Q1  5:Q6
    t2 = [
        ("JZ", "A", 5),
        ("JZ", "B", 4),
        da,
        ("JZ", "A", 0),
        ib,
        halt,
    ]

    # T3: increment B inside A-loop; exit when A hits 0 after saturated decs
    # 0:Q4 5  1:Q1  2:Q2  3:Q4 0  4:Q0  5:Q6
    t3 = [
        ("JZ", "A", 5),
        ib,
        da,
        ("JZ", "A", 0),
        ia,
        halt,
    ]

    # T4: long chain mixing forward and backward jumps; stresses pc bookkeeping
    # 0:Q4 9  1:Q0  2:Q4 9  3:Q2  4:Q5 7  5:Q3  6:Q5 4  7:Q1  8:Q4 1  9:Q6
    t4 = [
        ("JZ", "A", 9),
        ia,
        ("JZ", "A", 9),
        da,
        ("JZ", "B", 7),
        db,
        ("JZ", "B", 4),
        ib,
        ("JZ", "A", 1),
        halt,
    ]

    # T5: same loop topology as Obs 12 but large A — long deterministic run
    loop_dec_a = [
        ("JZ", "A", 3),
        da,
        ("JZ", "B", 0),
        halt,
    ]

    # T6: unreachable padding lines 1–3; entry jumps into a shifted loop; on exit, B bumps once
    loop_offset = [
        ("JZ", "B", 4),
        halt,
        halt,
        halt,
        ("JZ", "A", 7),
        da,
        ("JZ", "B", 4),
        ib,
        halt,
    ]

    return [
        (t1, 4, 1),
        (t2, 3, 0),
        (t3, 2, 0),
        (t4, 2, 2),
        (loop_dec_a, 56, 0),
        (loop_offset, 48, 0),
    ]


def _build_prompt(demos: list, test_cases: list) -> str:
    demo_block = "\n\n".join(
        _render_trace(i + 1, prog, a0, b0, _execute(prog, a0, b0))
        for i, (prog, a0, b0) in enumerate(demos)
    )
    test_block_lines = []
    for j, (prog, ta0, tb0) in enumerate(test_cases, 1):
        prog_lines = "\n".join(f"  {i}: {_format_instr(ins)}" for i, ins in enumerate(prog))
        test_block_lines.append(
            f"=== Task program {j} ===\nInitial tallies: A={ta0}, B={tb0}\nProgram:\n{prog_lines}"
        )
    test_block = "\n\n".join(test_block_lines)
    return (
        "You are given complete execution records of an unknown discrete automaton.\n"
        "It maintains two non-negative tallies labeled A and B, and a program counter pc "
        "that selects a numbered instruction line.\n"
        "Each observation lists a program (opaque symbols Q0..Q6, some with an integer operand) "
        "and a sequence of snapshots. In every snapshot, pc is the index of the instruction "
        "that is about to be decoded and executed next; the tally values shown are the values "
        "before that instruction runs.\n"
        "Infer the precise effect of each symbol on (A, B, pc) from the observations alone.\n\n"
        f"{demo_block}\n\n"
        "Now run each task program until it halts (when the next instruction is Q6, or pc leaves "
        "the program after Q6).\n\n"
        f"{test_block}"
    )


def _prepare():
    demos = _curated_demonstrations()
    test_specs = _curated_tests()
    test_cases = [(p, a, b) for p, a, b in test_specs]
    test_gts = []
    for prog, a0, b0 in test_cases:
        tr = _execute(prog, a0, b0)
        fa, fb = tr[-1][1], tr[-1][2]
        test_gts.append((fa, fb))

    prompt = _build_prompt(demos, test_cases)

    def grade_fn(response):
        results = []
        for i, (gt_a, gt_b) in enumerate(test_gts, 1):
            raw = getattr(response, f"answer_{i}", None)
            got_a, got_b = _parse_answer(raw)
            correct = got_a == gt_a and got_b == gt_b
            expected_str = f"A={gt_a} B={gt_b}"
            results.append(
                {"q": i, "expected": expected_str, "got": raw, "correct": correct}
            )
        score = sum(r["correct"] for r in results) / len(test_gts)
        return score, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str
    answer_5: str
    answer_6: str


@kbench.task(
    name="two_counter_machine_obs_learning",
    description="Infer opaque opcode semantics (Q0..Q6) from execution traces; run six task programs; report final A,B.",
)
def two_counter_machine_obs_learning(llm) -> float:
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
            {"q": i + 1, "expected": "?", "got": None, "correct": False}
            for i in range(6)
        ]
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        "two_counter_machine_obs_learning",
        _TASK_DESCRIPTION,
        prompt,
        test_results,
        score,
        str(reasoning),
    )
    return score


if __name__ == "__main__":
    demos = _curated_demonstrations()
    assert len(demos) == 13, len(demos)
    tests = _curated_tests()
    expected = [(4, 0), (3, 1), (2, 1), (2, 2), (0, 0), (0, 1)]
    for i, ((prog, a0, b0), exp) in enumerate(zip(tests, expected), 1):
        tr = _execute(prog, a0, b0)
        got = (tr[-1][1], tr[-1][2])
        assert got == exp, (i, got, exp)
        print(f"TEST{i} final A={got[0]} B={got[1]} steps={len(tr)}")
    try:
        two_counter_machine_obs_learning.run(kbench.llm)
    except Exception as exc:
        print(f"benchmark run skipped: {exc}")

