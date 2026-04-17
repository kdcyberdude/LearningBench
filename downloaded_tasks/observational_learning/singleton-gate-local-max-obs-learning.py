#!/usr/bin/env python
# coding: utf-8

import kaggle_benchmarks as kbench
from collections import Counter
from dataclasses import dataclass


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
    "Hidden singleton-gate partitioning with intra-block local-maxima selection. "
    "One value appears exactly once in the sequence — that value is the gate. "
    "The gate splits the sequence into contiguous blocks (gate tokens excluded). "
    "Within each block, a token at position p is selected iff it is strictly "
    "greater than both its block-neighbors (boundary positions compare against "
    "their single interior neighbor only). Gate tokens are never selected."
)


# ── Hidden rule ───────────────────────────────────────────────────────────────
#
# Step 1 — Identify the GATE: the unique value that appears exactly once.
#   Every other value appears 2+ times in the sequence.
#
# Step 2 — Partition: split the sequence into contiguous blocks by removing
#   all occurrences of the gate value (there is exactly one occurrence).
#   This yields at most two blocks (one before and one after the gate).
#
# Step 3 — Select local maxima within each block:
#   For a token at block-position p (0-indexed within its block):
#     left_ok  = (p == 0)  OR  (val > block[p-1])
#     right_ok = (p == len(block)-1)  OR  (val < block[p+1])   <- NOTE: strict
#     selected = left_ok AND right_ok
#   Boundary tokens only compete against their single interior neighbor.
#   Ties are NOT selected (strict inequality required on both sides).
#
# Returns: sorted list of 0-based indices into the original sequence.
#
def _select(seq: list) -> list:
    counts = Counter(seq)
    (gate,) = [v for v, c in counts.items() if c == 1]
    blocks: list = []
    cur: list = []
    for idx, val in enumerate(seq):
        if val == gate:
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append((idx, val))
    if cur:
        blocks.append(cur)
    selected = []
    for block in blocks:
        n = len(block)
        for pos, (idx, val) in enumerate(block):
            left_ok = (pos == 0) or (val > block[pos - 1][1])
            right_ok = (pos == n - 1) or (val > block[pos + 1][1])
            if left_ok and right_ok:
                selected.append(idx)
    return sorted(selected)


def _fmt(seq: list) -> str:
    return "[" + ", ".join(str(v) for v in seq) + "]"


# ── Demo sequences ────────────────────────────────────────────────────────────
#
# Design invariant: every non-gate value appears EXACTLY TWICE in the sequence,
# ensuring the gate (appearing once) is unambiguously identifiable by the
# singleton criterion — and no other value qualifies.
#
# Phase A (D1–D3): gate is the MAXIMUM value (99, 88, 77). Establishes:
#   • what a gate looks like and where it can sit (middle, start, end).
#   • the partition and local-max rule in simple configurations.
#
# Phase B (D4–D8): gate is a MIDDLE or SMALL value — NOT the maximum. Forces:
#   • the model to identify the gate by singularity, not by magnitude.
#   • the distinction between "split at the singleton" vs "split at the max".
#   D4 gate=5  (max=8)   D5 gate=3 (max=9)   D6 gate=2 (max=9)
#   D7 gate=4  (max=9)   D8 gate=6 (max=9)
#
_DEMO_SEQS = [
    # D1: gate=99 at idx 3.  Blocks: [3,1,3] and [1,4,2,4,2].
    #   blk1 → local maxima: 3(boundary,3>1)→idx0; 3(3>1,boundary)→idx2
    #   blk2 → local maxima: 4(4>1,4>2)→idx5; 4(4>2,boundary)→idx7
    #   selected: [0, 2, 5, 7]
    [3, 1, 3, 99, 1, 4, 2, 4, 2],

    # D2: gate=88 at idx 0 (start).  One block: [5,2,5,3,2,1,4,1,4,3].
    #   local maxima: 5(boundary,5>2)→idx1; 5(5>2,5>3)→idx3; 4(4>1,4>1)→idx7; 4(4>1,boundary)→idx9
    #   selected: [1, 3, 7, 9]
    [88, 5, 2, 5, 3, 2, 1, 4, 1, 4, 3],

    # D3: gate=77 at idx 10 (end).  One block: [2,5,2,1,5,3,1,4,3,4].
    #   local maxima: 5(5>2,5>2)→idx1; 5(5>1,5>3)→idx4; 4(4>1,4>3)→idx7; 4(4>3,boundary)→idx9
    #   selected: [1, 4, 7, 9]
    [2, 5, 2, 1, 5, 3, 1, 4, 3, 4, 77],

    # D4: gate=5 (NOT max; max=8) at idx 5.  Blocks: [7,3,6,7,3] and [6,2,8,4,2,8,4].
    #   blk1 → 7(boundary,7>3)→idx0; 6(6>3,6>7 no)→skip; 7(7>3,boundary)→idx3; 3→skip
    #     wait: pos0=7: left-boundary, right? 7>3 yes → idx0
    #     pos1=3: 3>7 no
    #     pos2=6: 6>3 yes, 6>7? no
    #     pos3=7: 7>6 yes, 7>3 yes → idx3
    #     pos4=3: 3>7 no
    #   blk2 → pos0=6(boundary,6>2)→idx6; pos1=2(2>6 no); pos2=8(8>2,8>4)→idx8;
    #     pos3=4(4>8 no); pos4=2(2>4 no); pos5=8(8>2,boundary)→idx11; pos6=4(4>8 no)
    #   selected: [0, 3, 6, 8, 11]
    [7, 3, 6, 7, 3, 5, 6, 2, 8, 4, 2, 8, 4],

    # D5: gate=3 (NOT max; max=9) at idx 5.  Blocks: [4,6,4,8,6] and [8,5,7,5,9,7,9].
    #   blk1 → 4(boundary,4>6 no); 6(6>4,6>4)→idx1; 4(4>6 no); 8(8>4,boundary)→idx3(wait...)
    #     pos0=4: right? 4>6 no
    #     pos1=6: 6>4 yes, 6>4 yes → idx1
    #     pos2=4: 4>6 no
    #     pos3=8: 8>4 yes, 8>6 yes → idx3
    #     pos4=6: 6>8 no
    #   blk2 (starts at idx6) → pos0=8(boundary,8>5)→idx6; pos1=5(5>8 no);
    #     pos2=7(7>5,7>5)→idx8; pos3=5(5>7 no); pos4=9(9>5,9>7)→idx10;
    #     pos5=7(7>9 no); pos6=9(9>7,boundary)→idx12
    #   selected: [1, 3, 6, 8, 10, 12]
    [4, 6, 4, 8, 6, 3, 8, 5, 7, 5, 9, 7, 9],

    # D6: gate=2 (NOT max; even minimum; max=9) at idx 6.
    #   Blocks: [6,8,5,9,6,8] and [5,7,3,9,7,3].
    #   blk1 → pos0=6(boundary,6>8 no); pos1=8(8>6,8>5)→idx1; pos2=5(5>8 no);
    #     pos3=9(9>5,9>6)→idx3; pos4=6(6>9 no); pos5=8(8>6,boundary)→idx5
    #   blk2 → pos0=5(boundary,5>7 no); pos1=7(7>5,7>3)→idx8; pos2=3(3>7 no);
    #     pos3=9(9>3,9>7)→idx10; pos4=7(7>9 no); pos5=3(3>7 no)
    #     wait: idx mapping: gate at 6, blk2 starts at idx 7
    #     blk2: (7,5),(8,7),(9,3),(10,9),(11,7),(12,3)
    #     pos0=5: right? 5>7 no; pos1=7: 7>5,7>3→idx8; pos2=3: 3>7 no;
    #     pos3=9: 9>3,9>7→idx10; pos4=7: 7>9 no; pos5=3: 3>7 no
    #   selected: [1, 3, 5, 8, 10]
    [6, 8, 5, 9, 6, 8, 2, 5, 7, 3, 9, 7, 3],

    # D7: gate=4 (NOT max; max=9) at idx 0 (start).  One block: [9,5,7,9,5,3,7,3,6,1,6,1].
    #   pos0=9(boundary,9>5)→idx1; pos1=5(5>9 no); pos2=7(7>5,7>9 no);
    #   pos3=9(9>7,9>5)→idx4; pos4=5(5>9 no); pos5=3(3>5 no);
    #   pos6=7(7>3,7>3)→idx7; pos7=3(3>7 no); pos8=6(6>3,6>1)→idx9;
    #   pos9=1(1>6 no); pos10=6(6>1,boundary)→idx11; pos11=1(1>6 no)
    #   selected: [1, 4, 7, 9, 11]
    [4, 9, 5, 7, 9, 5, 3, 7, 3, 6, 1, 6, 1],

    # D8: gate=6 (NOT max; max=9) at idx 5.  Blocks: [8,3,5,8,3] and [5,2,9,4,2,9,4].
    #   blk1 → pos0=8(boundary,8>3)→idx0; pos1=3(3>8 no); pos2=5(5>3,5>8 no);
    #     pos3=8(8>5,8>3)→idx3; pos4=3(3>8 no)
    #   blk2 (starts at idx6) → pos0=5(boundary,5>2)→idx6; pos1=2(2>5 no);
    #     pos2=9(9>2,9>4)→idx8; pos3=4(4>9 no); pos4=2(2>4 no);
    #     pos5=9(9>2,boundary)→idx11; pos6=4(4>9 no)
    #     wait: gate at 5, blk2: (6,5),(7,2),(8,9),(9,4),(10,2),(11,9),(12,4)
    #     pos0=5→idx6; pos1=2: no; pos2=9: 9>2,9>4→idx8; pos3=4: no; pos4=2: no; pos5=9: 9>2,boundary→idx11; pos6=4: no
    #   selected: [0, 3, 6, 8, 11]
    [8, 3, 5, 8, 3, 6, 5, 2, 9, 4, 2, 9, 4],
]

_DEMOS = [(seq, _select(seq)) for seq in _DEMO_SEQS]


# ── Test cases ────────────────────────────────────────────────────────────────
#
# T1: gate=44 (max, at idx 3). Two blocks of unequal length.
#   Blocks: [6,2,4] and [6,3,2,5,1,3,5,1,4]
#   selected: [0, 2, 4, 7, 10, 12]
_T1 = [6, 2, 4, 44, 6, 3, 2, 5, 1, 3, 5, 1, 4]

# T2: gate=33 (max, at idx 9). Large first block, small second block.
#   Blocks: [4,1,3,4,3,5,1,2,5] and [6,2,6]
#   selected: [0, 3, 5, 8, 10, 12]
_T2 = [4, 1, 3, 4, 3, 5, 1, 2, 5, 33, 6, 2, 6]

# T3: gate=3 (NOT max; max=9, at idx 5). Gate is the MINIMUM.
#   Blocks: [8,5,7,8,5] and [7,4,6,9,4,6,9]
#   selected: [0, 3, 6, 9, 12]
_T3 = [8, 5, 7, 8, 5, 3, 7, 4, 6, 9, 4, 6, 9]

# T4: gate=11 (NOT max; max=17, at idx 6). Gate is a medium value among larger
#   block values — traps models that use max as gate.
#   Blocks: [14,12,16,14,16,12] and [15,13,17,15,13,17]
#   selected: [0, 2, 4, 7, 9, 12]
_T4 = [14, 12, 16, 14, 16, 12, 11, 15, 13, 17, 15, 13, 17]

_TEST_ITEM_SETS = [_T1, _T2, _T3, _T4]
_TEST_CASES = [(seq, _select(seq)) for seq in _TEST_ITEM_SETS]


def _build_prompt(demos: list, test_cases: list) -> str:
    lines = [
        "You are observing a selection process applied to integer sequences.",
        "Each input is a list of integers. The process outputs a sorted list",
        "of 0-based indices that were selected. The selection rule is hidden.",
        "",
        "Observations (sequence → selected indices):",
    ]
    for i, (seq, sel) in enumerate(demos, 1):
        lines.append(f"  {i:2d}. {_fmt(seq)} → {sel}")
    lines.append("")
    lines.append("Apply the same hidden rule to these 4 sequences:")
    for i, (seq, _) in enumerate(test_cases, 1):
        lines.append(f"  Q{i}: {_fmt(seq)} → ?")
    lines.append("")
    lines.append(
        "Submit answer_1 through answer_4 as sorted lists of 0-based indices, e.g. '[0, 2, 4]'."
    )
    return "\n".join(lines)


def _prepare():
    prompt = _build_prompt(_DEMOS, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (seq, expected) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"answer_{i}", None)
            got = None
            ok = False
            if isinstance(raw, list):
                try:
                    got = sorted(int(x) for x in raw)
                    ok = got == expected
                except (TypeError, ValueError):
                    pass
            elif isinstance(raw, str):
                try:
                    cleaned = raw.strip().strip("[]")
                    if cleaned:
                        got = sorted(
                            int(x.strip()) for x in cleaned.split(",") if x.strip()
                        )
                    else:
                        got = []
                    ok = got == expected
                except (TypeError, ValueError):
                    pass
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: list
    answer_2: list
    answer_3: list
    answer_4: list


@kbench.task(
    name="singleton_gate_local_max_obs_learning",
    description=(
        "Infer: the unique singleton value partitions the sequence; "
        "within each resulting block, local maxima (boundary-open, strict) are selected. "
        "Predict selected indices for 4 test sequences."
    ),
)
def singleton_gate_local_max_obs_learning(llm) -> float:
    """Infer singleton-gate partition + intra-block local-max rule from 8 phased examples; predict 4 test cases."""
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
    _log_trace(
        task="singleton_gate_local_max_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    singleton_gate_local_max_obs_learning.run(kbench.llm)

