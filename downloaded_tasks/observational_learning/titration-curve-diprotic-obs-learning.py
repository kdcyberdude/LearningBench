#!/usr/bin/env python
# coding: utf-8

"""
Observational learning: infer a hidden piecewise rule mapping integer x to a label y.

The observable mapping is *not* chemistry; the filename is legacy. The generator uses
four contiguous regimes on the integer line. In each regime y is a single symbol chosen
by cycling through a regime-specific tuple with a hidden phase offset. The symbol
alphabets are pairwise disjoint, so every label reveals which regime applies and the
demos pin the three cut integers exactly. No floating-point read-offs, logarithms, or
multi-parameter nonlinear fits are involved—only segmentation plus modular indexing,
which humans handle by grouping symbols and counting beats; models often mis-align
phase or confuse regime boundaries when regimes use different periods.
"""

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Integer inputs x map to single-token outputs y. The law is a four-way split of "
    "the x-axis into contiguous regimes; inside each regime, y cycles through a "
    "regime-specific ordered tuple of symbols with a fixed phase offset. The four "
    "tuples use disjoint Unicode symbol sets, so the active regime is readable from y "
    "alone. Demo x-values never equal a regime cut, but they bracket each cut so the "
    "boundaries are uniquely determined. Demos in each regime expose several distinct "
    "positions in that regime's cycle, fixing the phase modulo the period."
)


# ── Hidden piecewise cyclic rule (opaque to the evaluated model) ───────────
# Cuts: x < B1 → regime 0 ; B1 ≤ x < B2 → regime 1 ; B2 ≤ x < B3 → regime 2 ; else 3
_B1 = 61
_B2 = 119
_B3 = 173

# Disjoint symbol sets (no token appears in more than one regime).
_SYM0 = ("◈", "◇", "◆", "◊", "◉")  # period 5
_SYM1 = ("▢", "▣", "▤", "▥")  # period 4
_SYM2 = ("⬠", "⬡", "⬢")  # period 3
_SYM3 = ("⨀", "⨁", "⨂", "⨃", "⨄", "⨅", "⨆")  # period 7

_OFF0 = 3
_OFF1 = 0
_OFF2 = 2
_OFF3 = 5


def _label(x: int) -> str:
    if x < _B1:
        return _SYM0[(x + _OFF0) % 5]
    if x < _B2:
        return _SYM1[(x + _OFF1) % 4]
    if x < _B3:
        return _SYM2[(x + _OFF2) % 3]
    return _SYM3[(x + _OFF3) % 7]


# Demos: sorted by x; each regime represented; x mod period varies within regimes;
# consecutive demos across cuts show alphabet change (pins B1, B2, B3).
_DEMO_XS = [
    11,
    19,
    28,
    37,
    46,
    55,  # regime 0 — varied x mod 5
    63,
    64,
    65,
    66,
    67,
    91,
    110,  # regime 1 — hits residues 0–3 mod 4, plus longer-span checks
    119,
    120,
    121,
    125,
    129,
    137,
    166,  # regime 2 — 119–121 exposes full period-3 cycle at the left edge
    176,
    185,
    194,
    206,
    215,  # regime 3
]

_TEST_XS = [25, 100, 125, 168, 205]


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


def _build_prompt(demos: list[tuple[int, str]], test_xs: list[int]) -> str:
    lines = [
        "You are observing a mystery system where an input quantity x is mapped to",
        "an output quantity y. Each observation below shows one (x → y) pair.",
        "",
        "Observations:",
    ]
    for i, (x, y) in enumerate(demos, 1):
        lines.append(f"  {i:2d}.  x = {x}  →  y = {y}")
    lines += [
        "",
        "Using only the pattern visible in the observations above, predict the value",
        "of y for each of the following inputs.",
        "Submit your answers as answer_1 through answer_5.",
        "",
    ]
    for j, x in enumerate(test_xs, 1):
        lines.append(f"  Test {j}: x = {x}")
    return "\n".join(lines)


def _normalize_token(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().replace("\u00a0", "").strip("'\"")


def _prepare():
    demos = [(x, _label(x)) for x in _DEMO_XS]
    ground_truths = [_label(x) for x in _TEST_XS]
    prompt = _build_prompt(demos, _TEST_XS)

    def grade_fn(response):
        results = []
        for i, gt in enumerate(ground_truths, 1):
            raw = getattr(response, f"answer_{i}", None)
            got = _normalize_token(raw)
            correct = got == gt
            results.append({"q": i, "expected": gt, "got": raw, "correct": correct})
        score = sum(r["correct"] for r in results) / len(ground_truths)
        return score, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str
    answer_5: str


@kbench.task(
    name="titration_curve_diprotic_obs_learning",
    description=(
        "Observe 25 (x→y) pairs where y is a Unicode token from four disjoint cycles. "
        "Infer regimes and cycles, then predict y for 5 test x. Exact token match required."
    ),
)
def titration_curve_diprotic_obs_learning(llm) -> float:
    """Observe 25 (x→y) pairs from a 4-regime, piecewise cycling rule with disjoint symbols per regime. Identify regime boundaries and predict y for 5 test x. Solution needs no arithmetic."""
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
            {
                "q": i + 1,
                "expected": _label(_TEST_XS[i]),
                "got": None,
                "correct": False,
            }
            for i in range(len(_TEST_XS))
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        "titration_curve_diprotic_obs_learning",
        _TASK_DESCRIPTION,
        prompt,
        test_results,
        score,
        str(reasoning),
    )
    return score


if __name__ == "__main__":
    titration_curve_diprotic_obs_learning.run(kbench.llm)

