#!/usr/bin/env python
# coding: utf-8

"""Observational learning: hidden two-pass rewrite on bracketed binary trees.

HIDDEN MECHANISM (never stated to the evaluated model; inferred only from I/O):

  Notation: each internal node is [L X Y] where L is a single uppercase letter
  and X, Y are subtrees (each either a single lowercase leaf letter or another
  bracketed node). This is a full binary tree language with arbitrary labels, not
  linguistic constituency.

  Pass 1 — bottom-up child swap (post-order):
    If a node's label is one of {E, F, G, H}, swap its left and right children
    (after their subtrees have been processed). Otherwise leave children as-is.

  Pass 2 — top-down label inheritance (pre-order):
    Visit the tree in pre-order. At each internal node, replace that node's label
    with the root label of its *current* left child (the left subtree's root
    letter as it exists at the start of this visit — i.e. after Pass 1, before
    Pass 2 has rewritten that child). Leaves are unchanged.

  The full observable transformation is Pass 1 then Pass 2.

DIFFICULTY:
  Two interacting passes must be discovered; Pass 2's read of the left child
  uses pre-order timing (parent reads child's label before the child is
  rewritten). Nested markers create chained swaps whose effect on Pass 2 is
  non-local. This is substantially harder than a single movement rule (e.g.
  adverb fronting) and stresses hierarchical bookkeeping.

SOLUTION UNIQUENESS (checked at import):
  Competing single-pass rules (swap-only, relabel-only) disagree on curated demos.
  Reversing pass order disagrees. Using markers {A,B,C,D} instead of {E,F,G,H}
  disagrees on any demo whose root or subtree root is in {E..H} and would swap
  under the true rule.

LOGICAL CONSISTENCY:
  All demo and test outputs are produced by the same deterministic simulator.
  No search or numeric computation is required once the rule is known.

NOVELTY / ANTI-CONTAMINATION:
  The label set and demo sequence are custom; the interaction of a fixed
  marker subset with pre-order relabeling is not a standard textbook puzzle.

HUMAN–MODEL ASYMMETRY:
  Humans can sketch trees, run two mental passes, and diff I/O pairs. Models
  often latch onto one pass (e.g. relabel-only) or confuse pre-order relabeling
  with post-order or with "symmetric" tree mirroring.
"""

import re
from dataclasses import dataclass

import kaggle_benchmarks as kbench

_MARKERS: frozenset[str] = frozenset("EFGH")

_TASK_DESCRIPTION = (
    "Observational learning on a custom bracketed binary-tree language. "
    "The evaluated system must infer a deterministic input→output mapping "
    "from demonstrations alone, then apply it to held-out trees."
)


def _root_label(sub: str | tuple) -> str:
    if isinstance(sub, str):
        return sub
    return sub[0]


def _pass1(t: str | tuple, markers: frozenset[str]) -> str | tuple:
    if isinstance(t, str):
        return t
    lab, left, right = t
    l2 = _pass1(left, markers)
    r2 = _pass1(right, markers)
    if lab in markers:
        return (lab, r2, l2)
    return (lab, l2, r2)


def _pass2(t: str | tuple) -> str | tuple:
    if isinstance(t, str):
        return t
    _lab, left, right = t
    new_lab = _root_label(left)
    l2 = _pass2(left)
    r2 = _pass2(right)
    return (new_lab, l2, r2)


def _transform(t: str | tuple, markers: frozenset[str] = _MARKERS) -> str | tuple:
    return _pass2(_pass1(t, markers))


def _format_tree(t: str | tuple) -> str:
    if isinstance(t, str):
        return t
    lab, left, right = t
    return f"[{lab} {_format_tree(left)} {_format_tree(right)}]"


def _parse_tree(s: str) -> str | tuple:
    s = s.strip()
    if not s.startswith("["):
        return s
    inner = s[1:-1]
    i = 0
    while i < len(inner) and inner[i] != " ":
        i += 1
    lab = inner[:i]
    rest = inner[i + 1 :].lstrip()
    left, rest = _parse_first(rest)
    right, rest = _parse_first(rest)
    if rest.strip():
        raise ValueError(f"trailing junk: {rest!r} in {s!r}")
    return (lab, left, right)


def _parse_first(s: str) -> tuple[str | tuple, str]:
    s = s.lstrip()
    if not s:
        raise ValueError("empty subtree")
    if s[0] != "[":
        return s[0], s[1:].lstrip()
    depth = 0
    for i, c in enumerate(s):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return _parse_tree(s[: i + 1]), s[i + 1 :].lstrip()
    raise ValueError(f"unbalanced: {s!r}")


def _wrong_order_transform(t: str | tuple, markers: frozenset[str]) -> str | tuple:
    return _pass1(_pass2(t), markers)


_DEMO_INPUT_STRS: list[str] = [
    # Phase A: no marker at root — Pass 1 is identity; reveals Pass 2 only.
    "[A [B a b] [C c d]]",
    "[D [B x y] [C u v]]",
    # Phase B: marker at root — swap + relabel; discriminates swap+relabel vs relabel-only.
    "[G [B a b] [C c d]]",
    "[F a b]",
    # Phase C: nested markers — chained swaps; order of passes matters.
    "[E [F a b] c]",
    "[G [D a b] [E c d]]",
    # Phase D: deeper non-marker path — relabel propagates without swap confusion.
    "[A [D [B a b] c] e]",
    # Phase E: multiple internals under marker; rules out trivial "mirror once".
    "[E [A a b] [B c d]]",
    "[G [F a b] [A c d]]",
    # Phase F: three-level marker nesting — stress test.
    "[H [G [E a b] c] [F d e]]",
    # Phase G: contrast B (non-marker) vs F (marker) on same leaf pattern.
    "[B a b]",
    # Phase H: marker under non-marker root — Pass 2 root takes label from swapped child.
    "[A [G b c] [D d e]]",
]


def _build_demos() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in _DEMO_INPUT_STRS:
        tin = _parse_tree(s)
        tout = _transform(tin)
        out.append((_format_tree(tin), _format_tree(tout)))
    return out


_DEMO_PAIRS = _build_demos()

_TEST_INPUT_STRS: list[str] = [
    "[G [E a b] [C c d]]",
    "[A [G b c] [D d e]]",
    "[F [H [B a b] c] [E d e]]",
    "[E [G [A a b] c] [F d e]]",
]

_TEST_PAIRS: list[tuple[str, str]] = [
    (_format_tree(_parse_tree(s)), _format_tree(_transform(_parse_tree(s))))
    for s in _TEST_INPUT_STRS
]


def _verify() -> None:
    for inp, exp in _DEMO_PAIRS:
        got = _format_tree(_transform(_parse_tree(inp)))
        assert got == exp, (inp, got, exp)
    for (inp, exp) in _TEST_PAIRS:
        got = _format_tree(_transform(_parse_tree(inp)))
        assert got == exp

    wrong_markers = frozenset("ABCD")
    assert any(
        _format_tree(_transform(_parse_tree(inp), _MARKERS))
        != _format_tree(_transform(_parse_tree(inp), wrong_markers))
        for inp, _ in _DEMO_PAIRS
    )

    assert any(
        _format_tree(_transform(_parse_tree(inp)))
        != _format_tree(_wrong_order_transform(_parse_tree(inp), _MARKERS))
        for inp in _DEMO_INPUT_STRS
    )

    assert any(
        _format_tree(_transform(_parse_tree(inp))) != _format_tree(_pass1(_parse_tree(inp), _MARKERS))
        for inp in _DEMO_INPUT_STRS
    )

    assert any(
        _format_tree(_transform(_parse_tree(inp))) != _format_tree(_pass2(_parse_tree(inp)))
        for inp in _DEMO_INPUT_STRS
    )


_verify()


def _log_trace(task: str, description: str, prompt: str, test_results: list, score: float, reasoning: str = "") -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    if reasoning:
        print(f"\n  REASONING:\n{reasoning}")
    print("\n  TEST RESULTS:")
    for r in test_results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"    [{status}] Q{r['q']}: expected={r['expected']!r}  got={r['got']!r}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _str_match(expected: str, actual: str) -> bool:
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


def _build_prompt(demos: list[tuple[str, str]], test_inputs: list[str]) -> str:
    lines = [
        "You are watching a fixed program rewrite bracketed binary trees.",
        "Each tree uses this surface form:",
        "  • Internal node: [L X Y] — L is one uppercase letter; X and Y are subtrees.",
        "  • Leaf: one lowercase letter.",
        "",
        "Observations (input tree → output tree):",
    ]
    for i, (inp, outp) in enumerate(demos, 1):
        lines.append(f"  {i}. {inp}")
        lines.append(f"      → {outp}")
    lines += [
        "",
        "Apply the same transformation to each test tree.",
        "Submit one output tree string per answer field (same bracket notation).",
        "",
    ]
    for j, tin in enumerate(test_inputs, 1):
        lines.append(f"Test {j}: {tin}")
    return "\n".join(lines)


def _prepare() -> tuple[str, callable]:
    demos = list(_DEMO_PAIRS)
    test_inputs = [inp for inp, _ in _TEST_PAIRS]
    ground_truths = [out for _, out in _TEST_PAIRS]
    prompt = _build_prompt(demos, test_inputs)

    def grade_fn(response: _Answer):
        results = []
        for i, gt in enumerate(ground_truths, 1):
            ans = getattr(response, f"answer_{i}", None)
            correct = isinstance(ans, str) and _str_match(gt, ans)
            results.append({"q": i, "expected": gt, "got": ans, "correct": correct})
        score = sum(r["correct"] for r in results) / 4
        return score, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str


@kbench.task(
    name="syntax_tree_rewrite_obs_learning",
    description="Infer a deterministic rewrite on bracketed binary trees from input→output demonstrations, then apply it to four new trees. Two interacting hidden passes; nested structure.",
)
def syntax_tree_rewrite_obs_learning(llm) -> float:
    """
    Hidden: marker-gated child swap (E–H) bottom-up, then pre-order relabel from
    left-child root. 12 demos, 4 tests; score = fraction correct.
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
            {"q": i + 1, "expected": "?", "got": None, "correct": False}
            for i in range(4)
        ]
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        "syntax_tree_rewrite_obs_learning",
        _TASK_DESCRIPTION,
        prompt,
        test_results,
        score,
        str(reasoning),
    )
    return score


# Backward-compatible alias (previous task id)
syntax_tree_adverb_rewrite_obs_learning = syntax_tree_rewrite_obs_learning


if __name__ == "__main__":
    syntax_tree_rewrite_obs_learning.run(kbench.llm)

