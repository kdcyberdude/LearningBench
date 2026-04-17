#!/usr/bin/env python
# coding: utf-8

"""Hidden Token Filter — observational learning task.

The model observes examples where a list of words is filtered to a sublist by
a hidden 2-condition rule:
  KEEP word iff (length is ODD) AND (no letter appears more than once).

Deception scaffold
------------------
Phase 1 (5 demos):  inputs mix odd-length-all-distinct words (KEEP) with
  even-length-all-distinct words (DROP).  A model observing only Phase 1
  might reasonably conclude "the rule is: keep odd-length words" — the
  unique-letters condition is never violated in Phase 1.

Phase 2 (5 demos):  inputs introduce odd-length words WITH repeated letters —
  those are dropped, exposing the second condition.  Now both conditions must
  hold simultaneously.

Test inputs:  4 lists where odd-length-but-repeat words and even-length-distinct
  words both appear, so the "odd length only" rule and the "distinct only" rule
  each give wrong answers.
"""

from dataclasses import dataclass

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
    "The model observes word-lists filtered by a hidden 2-condition rule: a word is "
    "KEPT only when it satisfies both (a) odd length AND (b) no letter is repeated. "
    "Phase 1 examples are deceptive — consistent with the simpler 'odd length only' rule. "
    "Phase 2 introduces odd-length words with repeated letters (which are dropped), "
    "disambiguating the true two-part rule. "
    "The model must identify the exact rule and apply it to four new lists."
)


# ── Ground-truth filter ────────────────────────────────────────────────────
def _keep(word: str) -> bool:
    """Keep a word iff (length is ODD) AND (all letters are distinct)."""
    if len(word) % 2 == 0:  # even length → drop
        return False
    return len(set(word)) == len(word)  # repeated letter → drop


def _filter(words: list) -> list:
    return [w for w in words if _keep(w)]


# ── Vocabulary ─────────────────────────────────────────────────────────────
# ODD length + ALL DISTINCT → kept
_PASS: list = [
    "strip",
    "world",
    "blend",
    "crimp",
    "flown",
    "plumb",
    "brand",
    "swift",
    "glyph",
    "fjord",
    "crypt",
    "blitz",
    "spork",
    "squib",
    "clunk",
    "blunt",
    "craft",
    "chimp",
    "depth",
    "frost",
    "brisk",
    "dropt",
    "clung",
    "swung",
    "spilt",
    "flung",
    "stomp",
    "stump",
    "cleft",
    "crept",
    "snort",
    "birch",
    "grind",
    "twist",
    "bloke",
    "shone",
    "plonk",
    "prism",
    "swamp",
    "stork",
]

# EVEN length + ALL DISTINCT → dropped (fails length only)
_EVEN_DISTINCT: list = [
    "fish",
    "bark",
    "glow",
    "hulk",
    "myth",
    "lynx",
    "dusk",
    "rift",
    "womb",
    "silk",
    "musk",
    "tusk",
    "loft",
    "volt",
    "silt",
    "gilt",
    "jilt",
    "hilt",
    "wilt",
    "melt",
]

# ODD length + REPEATED LETTER → dropped (fails uniqueness only)
_ODD_REPEAT: list = [
    "spill",
    "skill",
    "walls",
    "bells",
    "tells",
    "cells",
    "yells",
    "spell",
    "blood",
    "woods",
    "stood",
    "goods",
    "abbot",
    "speed",
    "greet",
    "fleet",
    "bleed",
    "treed",
    "flood",
    "freed",
]


# ── Demo construction ──────────────────────────────────────────────────────
# Phase 1: mix PASS words with EVEN_DISTINCT words only.
# The "odd length → keep" hypothesis perfectly explains all Phase 1 examples.
_PHASE1_DEMOS = [
    # Each list: some PASS (odd+distinct→keep), some EVEN_DISTINCT (even+distinct→drop)
    (
        ["strip", "fish", "blend", "bark", "flown"],
        _filter(["strip", "fish", "blend", "bark", "flown"]),
    ),
    (
        ["world", "glow", "crimp", "hulk", "swift"],
        _filter(["world", "glow", "crimp", "hulk", "swift"]),
    ),
    (
        ["plumb", "myth", "brand", "lynx", "glyph"],
        _filter(["plumb", "myth", "brand", "lynx", "glyph"]),
    ),
    (
        ["crypt", "dusk", "blitz", "rift", "fjord"],
        _filter(["crypt", "dusk", "blitz", "rift", "fjord"]),
    ),
    (
        ["spork", "womb", "squib", "silk", "clunk"],
        _filter(["spork", "womb", "squib", "silk", "clunk"]),
    ),
]

# Phase 2: add ODD_REPEAT words — they are odd-length but dropped.
# This breaks the "odd length → keep" hypothesis and reveals the second condition.
_PHASE2_DEMOS = [
    (
        ["blunt", "spill", "craft", "skill", "depth"],
        _filter(["blunt", "spill", "craft", "skill", "depth"]),
    ),
    (
        ["frost", "walls", "brisk", "bells", "dropt"],
        _filter(["frost", "walls", "brisk", "bells", "dropt"]),
    ),
    (
        ["clung", "tells", "swung", "cells", "spilt"],
        _filter(["clung", "tells", "swung", "cells", "spilt"]),
    ),
    (
        ["flung", "yells", "stomp", "abbot", "stump"],
        _filter(["flung", "yells", "stomp", "abbot", "stump"]),
    ),
    (
        ["cleft", "blood", "crept", "stood", "snort"],
        _filter(["cleft", "blood", "crept", "stood", "snort"]),
    ),
]

# ── Test cases ─────────────────────────────────────────────────────────────
# Each list contains: PASS words + EVEN_DISTINCT words + ODD_REPEAT words.
# "Odd length only" → wrong (keeps odd-repeat words).
# "Distinct letters only" → wrong (keeps even-distinct words).
# Only the true 2-part rule gives the right answer.
_RAW_TEST_INPUTS = [
    ["birch", "woods", "grind", "musk", "twist"],  # pass: birch,grind,twist
    ["bloke", "abbot", "shone", "loft", "chimp"],  # pass: bloke,shone,chimp
    ["plonk", "speed", "prism", "tusk", "greet"],  # pass: plonk,prism
    ["swamp", "fleet", "stork", "gilt", "spell"],  # pass: swamp,stork
]

_TEST_CASES = [(inp, _filter(inp)) for inp in _RAW_TEST_INPUTS]


# ── Prompt builder ─────────────────────────────────────────────────────────
def _fmt_list(words: list) -> str:
    return "[" + ", ".join(words) + "]"


def _build_prompt(demos, test_inputs):
    lines = [
        "You are studying a hidden filtering rule.",
        "In each example below an input word-list is transformed to an output word-list",
        "by keeping only the words that satisfy a hidden rule (original order is preserved).",
        "",
        "Observations:",
    ]
    for i, (inp, out) in enumerate(demos, 1):
        lines.append(f"  {i:2d}. {_fmt_list(inp)} → {_fmt_list(out)}")
    lines.append("")
    lines.append("Now apply the same hidden rule to these 4 new lists:")
    for i, inp in enumerate(test_inputs, 1):
        lines.append(f"  Q{i}: {_fmt_list(inp)} → ?")
    lines.append("")
    lines.append(
        "Submit answer_1 through answer_4 as comma-separated words in the original order "
        "(e.g. 'word1, word2'). If no word survives, submit an empty string."
    )
    return "\n".join(lines)


def _prepare():
    all_demos = _PHASE1_DEMOS + _PHASE2_DEMOS
    test_inputs = [inp for inp, _ in _TEST_CASES]
    prompt = _build_prompt(all_demos, test_inputs)

    def _parse_answer(raw) -> list:
        if raw is None:
            return []
        s = str(raw).strip()
        if not s:
            return []
        return [w.strip() for w in s.split(",") if w.strip()]

    def grade_fn(response):
        results = []
        correct = 0
        for i, (inp, expected) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"answer_{i}", None)
            got = _parse_answer(raw)
            ok = got == expected
            if ok:
                correct += 1
            results.append(
                {
                    "q": i,
                    "expected": ", ".join(expected),
                    "got": ", ".join(got),
                    "correct": ok,
                }
            )
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str


@kbench.task(
    name="hidden_token_filter_obs_learning",
    description="Observe word-lists filtered by a hidden 2-condition rule (odd length AND all letters distinct). Phase 1 is deceptive — consistent with 'odd length only'. Phase 2 reveals the second condition. Apply the rule to four new lists.",
)
def hidden_token_filter_obs_learning(llm) -> float:
    """Infer a 2-condition word filter (odd length AND distinct letters) from 10 phased examples."""
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
                "q": i,
                "expected": ", ".join(_TEST_CASES[i - 1][1]),
                "got": None,
                "correct": False,
            }
            for i in range(1, 5)
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="hidden_token_filter_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    hidden_token_filter_obs_learning.run(kbench.llm)

