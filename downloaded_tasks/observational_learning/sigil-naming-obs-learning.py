#!/usr/bin/env python
# coding: utf-8

"""Sigil Sequence Naming — observational learning task.

Each "sigil" is a sequence of 5 to 9 tiles. Every tile has two attributes:
  - mark:  H (hollow), F (filled), P (peaked), L (flat)
  - size:  s (small), m (medium), l (large)

A tile is written as [size][mark], e.g. "sH", "mF", "lP".

THE HIDDEN NAMING RULE (three independent components):

  1. PREFIX  — determined by the FIRST peaked (P) tile in the sequence:
       first P is small  → prefix "alpha"
       first P is medium → prefix "beta"
       first P is large  → prefix "gamma"
       no P at all       → prefix "null"

  2. BODY    — determined by the CENTER tile (index len//2, 0-indexed):
       center tile is small  → body "min"
       center tile is medium → body "mid"
       center tile is large  → body "max"

  3. SUFFIX  — determined by the count of F (filled) tiles in the sequence:
       0 filled tiles → suffix "void"
       1 filled tile  → suffix "mono"
       2 filled tiles → suffix "di"
       3 filled tiles → suffix "tri"
       4+ filled tiles → suffix "poly"

  Full name format:  "[prefix]-[body]-[suffix]"
  Example: "alpha-mid-di"

WHY THIS IS HARD FOR LLMs:
  - All three naming components are completely independent, so a model must
    track three separate pattern dimensions simultaneously.
  - Deception 1: the first 8 demonstrations all have a small first-P tile,
    so "alpha" always appears. Models that anchor on "alpha" will fail on
    later demos that expose "beta", "gamma", and "null" prefixes.
  - Deception 2: center tiles in early demos are often "mid" or "max", making
    it plausible that body always encodes the sequence length. Several demos
    share the same length but differ in center-tile size to break this.
  - Deception 3: filled-tile counts in early demos are all 1 or 2 ("mono",
    "di"). Demo 13 introduces 0 filled tiles ("void"), and demos 15–16
    introduce 3+ ("tri", "poly"), completing the suffix vocabulary.
  - The three rules never interact: the prefix depends ONLY on the first P,
    body ONLY on the center tile, suffix ONLY on F-count. Yet their
    simultaneous presence forces the solver to maintain three independent
    hypotheses without cross-contamination.

SOLUTION UNIQUENESS:
  - Deception 1 is broken by demos 9–11 (beta, gamma, null prefixes).
  - Deception 2 is broken by demos 7–8 and 13–14 (same length, different body).
  - Deception 3 is broken by demos 13–16 (void, tri, poly).
  - By demo 16, every prefix value, every body value, and every suffix value
    has appeared in at least one example. No alternative three-component rule
    is consistent with all 16 demonstrations.

HUMAN vs LLM ASYMMETRY:
  A human can naturally tabulate three separate columns (prefix rule, body
  rule, suffix rule) and fill them in independently while scanning the
  demonstrations. LLMs tend to seek a single unified pattern that predicts
  the entire name from a holistic structural description, failing when the
  three components are deliberately decoupled.

TEST CASES:
  All four test sequences have been chosen so that:
  - Q1: prefix=gamma, body=min, suffix=mono  (large first-P, small center, 1 F)
  - Q2: prefix=null,  body=max, suffix=di    (no P at all,   large center, 2 Fs)
  - Q3: prefix=beta,  body=mid, suffix=void  (medium first-P, medium center, 0 Fs)
  - Q4: prefix=alpha, body=max, suffix=poly  (small first-P,  large center, 4+ Fs)
  Each exercises a component combination never seen together in the demos.
"""

from dataclasses import dataclass

import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests whether a model can simultaneously infer three independent hidden rules "
    "from a sequence-of-tiles naming system. Each tile has a mark (H/F/P/L) and "
    "size (s/m/l). The name encodes: (1) which size the first P-tile is, "
    "(2) the size of the center tile, (3) how many F-tiles the sequence contains. "
    "Early demos reveal only one prefix value and two suffix values; "
    "later demos progressively expose the full vocabulary of all three components."
)

# ── Tile alphabet and name mappings ─────────────────────────────────────────
_PREFIX_MAP = {"s": "alpha", "m": "beta", "l": "gamma", None: "null"}
_BODY_MAP = {"s": "min", "m": "mid", "l": "max"}
_SUFFIX_MAP = {0: "void", 1: "mono", 2: "di", 3: "tri"}


def _name(tiles: list[str]) -> str:
    """Compute the sigil name for a tile sequence under the hidden rule."""
    # PREFIX: first P tile's size
    first_p_size = None
    for t in tiles:
        if t[1] == "P":
            first_p_size = t[0]
            break
    prefix = _PREFIX_MAP[first_p_size]

    # BODY: center tile's size
    center = tiles[len(tiles) // 2]
    body = _BODY_MAP[center[0]]

    # SUFFIX: count of F tiles
    f_count = sum(1 for t in tiles if t[1] == "F")
    suffix = _SUFFIX_MAP.get(f_count, "poly")

    return f"{prefix}-{body}-{suffix}"


def _fmt(tiles: list[str]) -> str:
    """Format a tile list as a human-readable sequence string."""
    return "  ".join(tiles)


# ── Demo sequences (16 demonstrations) ──────────────────────────────────────
#
# Notation: each entry is a list of "[size][mark]" strings.
# Sizes: s=small, m=medium, l=large
# Marks: H=hollow, F=filled, P=peaked, L=flat
#
# Structural purpose of each group:
#
#  Demos  1–4  (prefix=alpha, body varies, suffix=mono or di):
#    Establish the basic name format; body varies to hint it depends on
#    something per-tile; suffix varies between mono and di.
#
#  Demos  5–8  (prefix=alpha, body varies, suffix=mono or di):
#    Different lengths to rule out "body = length encoding".
#    Two demos of same length (5) but different center tile → same length,
#    different body proves body depends on center tile size, not length.
#
#  Demos  9–11 (prefix=beta, gamma, null):
#    Introduces medium-P (beta), large-P (gamma), no-P-at-all (null).
#    Breaks the "prefix is always alpha" assumption.
#
#  Demos 12–14 (body=min; same length as prior demos but center=small):
#    Completes the body vocabulary with "min". Demo 13 also introduces
#    suffix=void (0 filled tiles), demo 14 revisits suffix=mono.
#
#  Demos 15–16 (suffix=tri, poly):
#    Completes the suffix vocabulary. Demo 15: 3 filled tiles (tri).
#    Demo 16: 4 filled tiles (poly). Both have different prefixes/bodies
#    to confirm prefix and body are independent of suffix.

_DEMOS: list[list[str]] = [
    # ── Group 1: prefix=alpha, varied body, mono/di suffix ────────────────
    # Demo 1: len=5, center=tiles[2]=mH, 1 F → alpha-mid-mono
    ["sP", "mH", "mH", "mF", "lL"],
    # Demo 2: len=5, center=tiles[2]=lL, 1 F → alpha-max-mono
    ["sP", "mH", "lL", "lF", "sH"],
    # Demo 3: len=7, center=tiles[3]=mH, 2 Fs → alpha-mid-di
    ["sL", "sP", "mF", "mH", "lF", "sH", "lL"],
    # Demo 4: len=7, center=tiles[3]=lH, 2 Fs → alpha-max-di
    ["sH", "sP", "mH", "lH", "mF", "sF", "lL"],

    # ── Group 2: same/different lengths to isolate body rule ──────────────
    # Demo 5: len=5, center=tiles[2]=mF, 2 Fs → alpha-mid-di
    #   (same length as demos 1&2 but different body and suffix)
    ["sH", "sP", "mF", "sH", "lF"],
    # Demo 6: len=5, center=tiles[2]=lH, 1 F → alpha-max-mono
    #   (same length as demo 5, different center → different body)
    ["sH", "sP", "lH", "mF", "sL"],
    # Demo 7: len=9, center=tiles[4]=mL, 1 F → alpha-mid-mono
    ["sL", "sP", "mH", "lH", "mL", "sH", "lL", "mF", "sH"],
    # Demo 8: len=9, center=tiles[4]=lH, 2 Fs → alpha-max-di
    ["sH", "mH", "sP", "mF", "lH", "sF", "mL", "lL", "sH"],

    # ── Group 3: non-alpha prefixes ───────────────────────────────────────
    # Demo 9: first P is medium → prefix=beta; len=5, center=mH → mid; 1 F → mono
    #   → beta-mid-mono
    ["sH", "mP", "mH", "mF", "lL"],
    # Demo 10: first P is large → prefix=gamma; len=7, center=lH → max; 2 Fs → di
    #   → gamma-max-di
    ["mH", "sF", "lP", "lH", "mF", "sH", "sL"],
    # Demo 11: no P at all → prefix=null; len=5, center=mL → mid; 1 F → mono
    #   → null-mid-mono
    ["sH", "mL", "mL", "mF", "lH"],

    # ── Group 4: body=min (center tile is small) ──────────────────────────
    # Demo 12: len=5, center=tiles[2]=sH → min; prefix=alpha (sP early); 2 Fs → di
    #   → alpha-min-di
    ["sP", "lH", "sH", "mF", "lF"],
    # Demo 13: len=7, center=tiles[3]=sL → min; prefix=null (no P); 0 Fs → void
    #   → null-min-void
    ["mH", "lH", "mL", "sL", "lH", "mH", "sH"],
    # Demo 14: len=5, center=tiles[2]=sF → min; prefix=beta (mP); 1 F → mono
    #   → beta-min-mono  (note: center tile is sF, which counts toward suffix too)
    ["lH", "mP", "sF", "lH", "mL"],

    # ── Group 5: suffix=tri and poly ─────────────────────────────────────
    # Demo 15: len=7, center=tiles[3]=mH → mid; prefix=gamma (lP); 3 Fs → tri
    #   → gamma-mid-tri
    ["sF", "lP", "mF", "mH", "mF", "sH", "lH"],
    # Demo 16: len=9, center=tiles[4]=lL → max; prefix=alpha (sP); 4 Fs → poly
    #   → alpha-max-poly
    ["sP", "mF", "lH", "mF", "lL", "sF", "mH", "lF", "sH"],
]

# ── Test cases ───────────────────────────────────────────────────────────────
# Q1: prefix=gamma, body=min, suffix=mono  → "gamma-min-mono"
#   len=5, center=tiles[2]=sH; first P is lP (large); 1 F (mF at index 3)
_Q1 = ["mH", "lP", "sH", "mF", "lL"]

# Q2: prefix=null, body=max, suffix=di    → "null-max-di"
#   len=7, center=tiles[3]=lH; no P tiles; 2 Fs (at indices 0 and 5)
_Q2 = ["mF", "sH", "mL", "lH", "sH", "lF", "sL"]

# Q3: prefix=beta, body=mid, suffix=void  → "beta-mid-void"
#   len=9, center=tiles[4]=mL; first P is mP (medium, index 1); 0 Fs
_Q3 = ["lH", "mP", "sL", "lH", "mL", "sH", "lL", "sH", "lH"]

# Q4: prefix=alpha, body=max, suffix=poly → "alpha-max-poly"
#   len=7, center=tiles[3]=lH; first P is sP (small, index 2); 4 Fs
_Q4 = ["mF", "lF", "sP", "lH", "mF", "sF", "lL"]

_TEST_CASES: list[tuple[list[str], str]] = [
    (_Q1, _name(_Q1)),
    (_Q2, _name(_Q2)),
    (_Q3, _name(_Q3)),
    (_Q4, _name(_Q4)),
]


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


def _prepare():
    lines = [
        "You are learning a naming system for sigil sequences.",
        "Each sigil is a sequence of tiles. Every tile is written as [size][mark]:",
        "  size:  s = small   m = medium   l = large",
        "  mark:  H = hollow  F = filled   P = peaked  L = flat",
        "",
        "Each example shows a tile sequence and the name assigned to it under",
        "this system. Study the examples carefully to infer the naming rule.",
        "",
        "Examples:",
    ]
    for i, tiles in enumerate(_DEMOS, 1):
        seq_str = _fmt(tiles)
        name = _name(tiles)
        lines.append(f"  {i:2d}.  {seq_str}  →  {name}")

    lines += [
        "",
        "Apply the same naming rule to these four sequences:",
    ]
    for q, (tiles, _) in enumerate(_TEST_CASES, 1):
        lines.append(f"  Q{q}: {_fmt(tiles)}")

    lines += [
        "",
        "Provide the name for each sequence.",
        "Submit as name_1, name_2, name_3, name_4.",
    ]
    prompt = "\n".join(lines)

    def grade_fn(response):
        import re
        results = []
        correct = 0
        for q_idx, (tiles, expected) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"name_{q_idx}", None)
            got = str(raw).strip().lower() if raw is not None else ""
            is_correct = bool(
                re.search(re.escape(expected.strip().lower()), got, re.IGNORECASE)
            )
            results.append(
                {
                    "q": q_idx,
                    "expected": expected,
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
    name_1: str
    name_2: str
    name_3: str
    name_4: str


@kbench.task(
    name="sigil_naming_obs_learning",
    description=(
        "See 16 tile-sequence→name examples. The naming rule has 3 hidden parts: prefix (from first peaked-tile size), body (center tile size), and suffix (count of filled tiles). Early demos show only some options; later ones show all. Name 4 test sequences."
    ),
)
def molecule_naming_custom_rule_obs_learning(llm) -> float:
    """
    Hidden rule: name = prefix (first P tile size), body (center tile), suffix (count of F tiles).
    Demos intentionally mislead; tests use unseen combinations.
    Returns fraction correct (out of 4).
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
    _log_trace(
        task="sigil_naming_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )

    return score


if __name__ == "__main__":
    molecule_naming_custom_rule_obs_learning.run(kbench.llm)

