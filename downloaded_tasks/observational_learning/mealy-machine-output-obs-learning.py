#!/usr/bin/env python
# coding: utf-8

"""Observational Mealy inference: six I/O demos, four held-out strings.

The concrete transition table is task-specific (not a named textbook automaton).
Exhaustive search over all 3-state tables consistent with the demos certifies a
single answer vector on the graded tests (see _verify_unique_test_predictions).
The prompt states three configurations, injective input→output per configuration,
and no self-loops — all consistent with the demonstrations.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import kaggle_benchmarks as kbench


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


_TASK_DESCRIPTION = (
    "Six I/O demos of a hidden minimal 3-state Mealy transducer on {0,1,2}. Each "
    "configuration maps the three inputs to three distinct outputs (a permutation "
    "row), which pins the latent state after one symbol once configurations are "
    "distinguished. Transitions are irregular; all nine (state,input) edges appear in "
    "the demos. A sixth short demo pins the observable map: exhaustive search shows "
    "every 3-state Mealy table consistent with all six demos agrees on the four test "
    "sequences. The prompt also states three configurations, injective outputs per "
    "configuration, and no self-loops (consistent with demos). Evaluates observational "
    "structure learning, not arithmetic."
)

_N_STATES = 3
_ALPHA = (0, 1, 2)

# Explicit transition table: (state, input) -> (output, next_state).
#
# Output rows (each is a permutation of {0,1,2}):
#   State 0:  input 0→out 0,  input 1→out 1,  input 2→out 2   (identity)
#   State 1:  input 0→out 1,  input 1→out 0,  input 2→out 2   (swap 0↔1)
#   State 2:  input 0→out 2,  input 1→out 0,  input 2→out 1   (rotate: 0→2,1→0,2→1)
#
# Transition structure (where you go next) is irregular and must be inferred from demos.
_FST_TABLE: Dict[Tuple[int, int], Tuple[int, int]] = {
    (0, 0): (0, 2),
    (0, 1): (1, 1),
    (0, 2): (2, 1),
    (1, 0): (1, 2),
    (1, 1): (0, 2),
    (1, 2): (2, 0),
    (2, 0): (2, 1),
    (2, 1): (0, 0),
    (2, 2): (1, 0),
}


def _build_fst() -> Dict[Tuple[int, int], Tuple[int, int]]:
    return dict(_FST_TABLE)


def _transduce(fst: dict, inp: list, start: int = 0) -> list:
    state = start
    outputs = []
    for sym in inp:
        out, nxt = fst[(state, sym)]
        outputs.append(out)
        state = nxt
    return outputs


def _state_trace(fst: dict, inp: list, start: int = 0) -> List[int]:
    states = [start]
    st = start
    for sym in inp:
        _, st = fst[(st, sym)]
        states.append(st)
    return states


def _coverage(fst: dict, demos: list) -> set:
    covered: set = set()
    for inp, _ in demos:
        state = 0
        for sym in inp:
            covered.add((state, sym))
            _, nxt = fst[(state, sym)]
            state = nxt
    return covered


def _verify_no_self_loops(fst: dict) -> None:
    for s in range(_N_STATES):
        for a in _ALPHA:
            if fst[(s, a)][1] == s:
                raise RuntimeError(f"Self-loop forbidden: state {s} on input {a}")


def _verify_mealy_minimal(fst: dict) -> None:
    """Partition refinement; must yield _N_STATES blocks for a minimal machine."""

    def out_vec(s: int) -> Tuple[int, ...]:
        return tuple(fst[(s, a)][0] for a in _ALPHA)

    by_vec: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for s in range(_N_STATES):
        by_vec[out_vec(s)].append(s)
    blocks = list(by_vec.values())
    changed = True
    while changed:
        changed = False
        block_of = {s: i for i, B in enumerate(blocks) for s in B}
        new_blocks: List[List[int]] = []
        for B in blocks:
            sub: Dict[Tuple[Tuple[int, int], ...], List[int]] = defaultdict(list)
            for s in B:
                key = tuple((fst[(s, a)][0], block_of[fst[(s, a)][1]]) for a in _ALPHA)
                sub[key].append(s)
            if len(sub) > 1:
                changed = True
            new_blocks.extend(sub.values())
        blocks = new_blocks
    if len(blocks) != _N_STATES:
        raise RuntimeError(
            f"Mealy machine not minimal: {len(blocks)} blocks, expected {_N_STATES}"
        )


def _verify_output_rows_are_permutations(fst: dict) -> None:
    """Each state's output row must be a permutation of {0,1,2}."""
    for s in range(_N_STATES):
        row = tuple(fst[(s, a)][0] for a in _ALPHA)
        if sorted(row) != [0, 1, 2]:
            raise RuntimeError(
                f"State {s} output row {row} is not a permutation of {{0,1,2}}"
            )


def _verify_prefix_map_welldefined(demos: list) -> None:
    """Same input prefix (from empty start) must always carry the same output prefix."""
    prefix_out: Dict[tuple, tuple] = {}
    for inp, out in demos:
        for k in range(len(inp) + 1):
            pi = tuple(inp[:k])
            po = tuple(out[:k])
            if pi in prefix_out and prefix_out[pi] != po:
                raise RuntimeError(f"Contradictory demos on input prefix {list(pi)!r}")
            prefix_out[pi] = po


def _fst_injective_output_rows(fst: dict) -> bool:
    """True iff each state's outputs on 0,1,2 are pairwise distinct (a permutation)."""
    try:
        for s in range(_N_STATES):
            row = [fst[(s, a)][0] for a in _ALPHA]
            if len(set(row)) != len(_ALPHA):
                return False
        return True
    except KeyError:
        return False


def _enumerate_demo_consistent_fsts() -> List[Dict[Tuple[int, int], Tuple[int, int]]]:
    """
    All complete 3-state Mealy tables consistent with _DEMOS (start state 0).
    Used to certify uniqueness of scored test outputs under the same structural
    assumptions as the prompt (injective rows + no self-loops).
    """
    solutions: List[Dict[Tuple[int, int], Tuple[int, int]]] = []
    seen: set = set()

    def rec(demos_idx: int, st: int, t: int, fst: dict) -> None:
        if demos_idx >= len(_DEMOS):
            for inp, out in _DEMOS:
                if _transduce(fst, inp) != out:
                    return
            key = tuple(sorted(fst.items()))
            if key not in seen:
                seen.add(key)
                solutions.append(dict(fst))
            return
        inp, out = _DEMOS[demos_idx]
        if t == len(inp):
            rec(demos_idx + 1, 0, 0, fst)
            return
        sym = inp[t]
        need_o = out[t]
        edge = (st, sym)
        if edge in fst:
            o, nst = fst[edge]
            if o != need_o:
                return
            rec(demos_idx, nst, t + 1, fst)
            return
        for nst in range(_N_STATES):
            nf = dict(fst)
            nf[edge] = (need_o, nst)
            rec(demos_idx, nst, t + 1, nf)

    rec(0, 0, 0, {})
    return solutions


def _verify_unique_test_predictions_across_demo_fsts() -> None:
    """
    Every 3-state Mealy machine consistent with _DEMOS must agree on _TEST_CASES.
    (With five demos only, several tables fit the demos but two disagree on tests;
    the sixth demo removes wrong families while preserving full edge coverage.)
    """
    candidates = _enumerate_demo_consistent_fsts()
    if not candidates:
        raise RuntimeError("No Mealy table consistent with demonstrations")
    sigs = {
        tuple(tuple(_transduce(f, inp)) for inp, _ in _TEST_CASES) for f in candidates
    }
    if len(sigs) != 1:
        raise RuntimeError(
            f"Demonstrations do not pin a unique test behaviour: {len(sigs)} distinct predictions"
        )
    filtered = [
        f
        for f in candidates
        if _fst_injective_output_rows(f)
        and all(f[(s, a)][1] != s for s in range(_N_STATES) for a in _ALPHA)
    ]
    if not filtered:
        raise RuntimeError(
            "No demo-consistent machine satisfies injective rows + no self-loops"
        )


def _verify_tests_compatible_with_demo_prefixes(
    fst: dict, demos: list, tests: list
) -> None:
    prefix_out: Dict[tuple, tuple] = {}
    for inp, out in demos:
        for k in range(len(inp) + 1):
            prefix_out[tuple(inp[:k])] = tuple(out[:k])
    for inp, exp in tests:
        for k in range(len(inp)):
            pi = tuple(inp[:k])
            if pi in prefix_out and prefix_out[pi] != tuple(exp[:k]):
                raise RuntimeError(
                    f"Test input shares prefix {list(pi)!r} with demos but outputs diverge"
                )


_FST = _build_fst()

# Six demonstrations: all nine (state, input) transitions are covered. With five demos
# only, six 3-state tables fit the I/O pairs and two disagree on the graded tests; the
# sixth demo ([0,1,0],[0,0,0]) eliminates wrong families so every consistent table
# matches on all four tests (two remain, I/O-equivalent on these strings). Demo lengths
# 3–5; tests length 5.
_DEMOS = [
    ([0, 1, 2], [0, 0, 2]),
    ([2, 0, 1], [2, 1, 0]),
    ([1, 2, 0], [1, 2, 0]),
    ([0, 2, 1, 0, 2], [0, 1, 1, 1, 1]),
    ([2, 1, 0, 2, 1], [2, 0, 2, 2, 1]),
    ([0, 1, 0], [0, 0, 0]),
]

_TEST_CASES = [
    ([0, 0, 1, 2, 0], [0, 2, 0, 1, 0]),
    ([2, 1, 2, 0, 1], [2, 0, 1, 0, 0]),
    ([1, 0, 0, 2, 1], [1, 1, 2, 2, 1]),
    ([0, 2, 0, 1, 2], [0, 1, 0, 0, 2]),
]


def _verify_at_import() -> None:
    fst = _FST
    _verify_no_self_loops(fst)
    _verify_mealy_minimal(fst)
    _verify_output_rows_are_permutations(fst)

    for i, (inp, expected) in enumerate(_DEMOS, 1):
        actual = _transduce(fst, inp)
        if actual != expected:
            raise RuntimeError(f"Demo {i} mismatch: expected {expected}, got {actual}")
    for i, (inp, expected) in enumerate(_TEST_CASES, 1):
        actual = _transduce(fst, inp)
        if actual != expected:
            raise RuntimeError(f"Test {i} mismatch: expected {expected}, got {actual}")

    _verify_prefix_map_welldefined(_DEMOS)
    _verify_tests_compatible_with_demo_prefixes(fst, _DEMOS, _TEST_CASES)

    cov = _coverage(fst, _DEMOS)
    if len(cov) != _N_STATES * len(_ALPHA):
        raise RuntimeError(
            f"Coverage {len(cov)}/{_N_STATES * len(_ALPHA)} — not all transitions covered"
        )

    _verify_unique_test_predictions_across_demo_fsts()


_verify_at_import()


def _build_prompt(demos: list, test_cases: list) -> str:
    demo_lines = []
    for i, (inp, outs) in enumerate(demos, 1):
        inp_str = " ".join(str(x) for x in inp)
        out_str = " ".join(str(x) for x in outs)
        demo_lines.append(
            f"Demo {i}:  input  [ {inp_str} ]\n"
            f"          output [ {out_str} ]"
        )
    demo_block = "\n\n".join(demo_lines)

    q_lines = "\n".join(
        f"Q{i}: [ {' '.join(str(x) for x in inp)} ]"
        for i, (inp, _) in enumerate(test_cases, 1)
    )

    return (
        "A hidden machine reads a sequence of symbols from {0, 1, 2} and produces "
        "a same-length output sequence of symbols from {0, 1, 2}.\n\n"
        "The machine has internal configurations that are completely hidden: you are "
        "never told which configuration it is in, only the input/output streams. "
        "Every sequence starts from the same initial configuration.\n\n"
        "Structural facts you may rely on (they are consistent with every demonstration "
        "below):\n"
        "• The machine has exactly three distinct internal configurations.\n"
        "• In any fixed configuration, the three inputs 0, 1, and 2 always produce three "
        "different outputs — no two inputs share the same output while the configuration "
        "stays the same.\n"
        "• Reading a symbol never leaves the machine in the same configuration it was in "
        "immediately before that read (every step changes configuration).\n\n"
        "Six demonstrations (input row, then output row):\n\n"
        f"{demo_block}\n\n"
        "=== YOUR TASK ===\n"
        "For each input sequence below, predict the full output sequence the machine "
        "would emit (same length as the input).\n\n"
        f"{q_lines}"
    )


def _prepare():
    prompt = _build_prompt(_DEMOS, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (inp, expected) in enumerate(_TEST_CASES, 1):
            got = getattr(response, f"output_{i}", None)
            ok = False
            if isinstance(got, list) and len(got) == len(expected):
                try:
                    ok = [int(x) for x in got] == expected
                except (TypeError, ValueError):
                    ok = False
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    output_1: List[int]
    output_2: List[int]
    output_3: List[int]
    output_4: List[int]


@kbench.task(
    name="finite_state_transducer_obs_learning",
    description="Six demos of a 3-state Mealy machine (no self-loops, injective per config); scoring is by held-out prediction.",
)
def finite_state_transducer_obs_learning(llm) -> float:
    """
    Infers a unique 3-state Mealy machine from six demos. Invariants: 3 configs, every input changes state, each config maps inputs to unique outputs. Tests latent state tracking with irregular transitions. No arithmetic.
    """
    prompt, grade_fn = _prepare()
    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None
    score, test_results = (
        grade_fn(response)
        if response is not None
        else (
            0.0,
            [
                {
                    "q": i,
                    "expected": _TEST_CASES[i - 1][1],
                    "got": None,
                    "correct": False,
                }
                for i in range(1, 5)
            ],
        )
    )
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="finite_state_transducer_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    finite_state_transducer_obs_learning.run(kbench.llm)

