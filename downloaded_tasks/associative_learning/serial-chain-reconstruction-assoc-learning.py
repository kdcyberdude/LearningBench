#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Dual-phase inference-time associative learning. Phase A: infer a hidden binary "
    "operation ⊛ on integers {1,…,9} from six numeric examples only (the rule is "
    "a ⊛ b = ((a×b + a) mod 9) + 1, i.e. a·(b+1) in Z₉ shifted to 1…9). Phase B: recover "
    "the unique bijection from nine invented glyph names to {1,…,9} using twelve "
    "symbolic equations of the form X ⊛ Y = Z. The operation is neither the affine "
    "mod-11 map used elsewhere nor commutative in the sense that bracketing matters: "
    "⊛ is used in compound queries to test non-associative reasoning. "
    "Twelve equations over-determine the bijection; computationally, each line is "
    "individually redundant (dropping any single equation still leaves a unique "
    "assignment), so solvers must maintain global consistency across a large clause "
    "set rather than rely on one decisive witness. Difficulty levers: (1) models must "
    "not collapse ⊛ to "
    "addition, multiplication, or XOR; (2) twelve coupled equations overload working "
    "memory; (3) compound queries require correct operator inference before bracket "
    "evaluation; (4) glyph names have no semantic cues."
)


def _bind_op(a: int, b: int) -> int:
    """Hidden operation: ((a*b + a) mod 9) + 1."""
    return ((a * b + a) % 9) + 1


# True assignment (unique under the twelve constraints below)
_VAL: dict[str, int] = {
    "Klym": 1,
    "Pers": 2,
    "Vost": 3,
    "Renn": 4,
    "Jyx": 5,
    "Orim": 6,
    "Tahl": 7,
    "Mev": 8,
    "Wold": 9,
}

# Twelve constraints; jointly pin the unique bijection (verified by enumeration).
_BIND_CONSTRAINTS: list[tuple[str, str, str]] = [
    ("Klym", "Klym", "Vost"),
    ("Klym", "Wold", "Pers"),
    ("Jyx", "Pers", "Tahl"),
    ("Pers", "Klym", "Jyx"),
    ("Pers", "Jyx", "Renn"),
    ("Renn", "Tahl", "Orim"),
    ("Renn", "Vost", "Mev"),
    ("Renn", "Mev", "Klym"),
    ("Tahl", "Mev", "Klym"),
    ("Vost", "Jyx", "Klym"),
    ("Wold", "Mev", "Klym"),
    ("Wold", "Pers", "Klym"),
]


def _log_trace(task: str, description: str, prompt: str, answers: dict, expected: dict, score: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    print(f"\n  RESPONSES:")
    for key in expected:
        actual = answers.get(key, "?")
        exp = expected[key]
        match = "✓" if _label_match(str(exp), str(actual)) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _label_match(expected: str, actual: str) -> bool:
    """Whole-word, negation-aware match for glyph names; also accepts exact int strings."""
    exp_s = expected.strip()
    act_s = actual.strip()
    if exp_s.isdigit() and act_s.isdigit():
        return int(act_s) == int(exp_s)
    token = re.escape(exp_s)
    m = re.search(rf"\b{token}\b", act_s, re.IGNORECASE)
    if not m:
        return False
    prefix_words = re.findall(r"\w+", act_s[: m.start()])[-3:]
    negations = {"not", "isn't", "isnt", "never", "no", "cannot", "can't", "cant", "neither", "without"}
    return not any(w.lower() in negations for w in prefix_words)


def _glyph_for_value(d: dict[str, int], v: int) -> str:
    for name, val in d.items():
        if val == v:
            return name.upper()
    raise KeyError(v)


_v = _VAL
_o = _bind_op
_EXPECTED_INT = {
    "q_1": _v["Klym"],
    "q_2": _v["Wold"],
    "q_4": _o(_o(_v["Klym"], _v["Pers"]), _v["Jyx"]),
    "q_5": _o(_v["Klym"], _o(_v["Pers"], _v["Jyx"])),
    "q_7": _o(_v["Mev"], _v["Renn"]),
    "q_8": _o(_v["Orim"], _v["Tahl"]),
}
_EXPECTED_NAME = {
    "q_3": _glyph_for_value(_v, _o(_v["Pers"], _v["Renn"])),
    "q_6": _glyph_for_value(_v, 3),
    "q_9": _glyph_for_value(_v, _o(_v["Wold"], _v["Klym"])),
    "q_10": _glyph_for_value(
        _v,
        sorted([_v["Pers"], _v["Renn"], _v["Tahl"]])[1],
    ),
}

_EXPECTED: dict[str, str] = {
    "q_1": str(_EXPECTED_INT["q_1"]),
    "q_2": str(_EXPECTED_INT["q_2"]),
    "q_3": _EXPECTED_NAME["q_3"],
    "q_4": str(_EXPECTED_INT["q_4"]),
    "q_5": str(_EXPECTED_INT["q_5"]),
    "q_6": _EXPECTED_NAME["q_6"],
    "q_7": str(_EXPECTED_INT["q_7"]),
    "q_8": str(_EXPECTED_INT["q_8"]),
    "q_9": _EXPECTED_NAME["q_9"],
    "q_10": _EXPECTED_NAME["q_10"],
}


@dataclass
class GlyphBindAnswer:
    q_1: str
    q_2: str
    q_3: str
    q_4: str
    q_5: str
    q_6: str
    q_7: str
    q_8: str
    q_9: str
    q_10: str


def _numeric_examples_block() -> list[str]:
    """Six numeric ⊛ examples; outputs are diverse and disambiguate the mod-9 rule."""
    pairs = [(1, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7)]
    lines = ["Operator ⊛ (six independent examples on integers 1–9):"]
    for a, b in pairs:
        lines.append(f"  {a} ⊛ {b} = {_bind_op(a, b)}")
    return lines


def _constraints_block() -> list[str]:
    lines = [
        "Glyph equations (each line is an equation; the right-hand side is a single glyph name):",
    ]
    for x, y, z in sorted(_BIND_CONSTRAINTS, key=lambda t: (t[0], t[1], t[2])):
        lines.append(f"  {x} ⊛ {y} = {z}")
    return lines


@kbench.task(
    name="glyph_bind_assoc_learning",
    description=(
        "Infer a novel modular nonlinear binary operator from numeric examples, then recover "
        "a unique 9-glyph-to-integer bijection from twelve coupled equations; answer questions "
        "including bracket-sensitive compounds."
    ),
)
def glyph_bind_assoc_learning(llm) -> float:
    """
    Tests inference-time learning of a hidden operation plus a symbolic bijection.
    Returns fraction correct across 10 questions.
    """

    prompt = "\n".join(
        [
            "You are given a hidden binary operator ⊛ that maps pairs of integers from",
            "{1, 2, …, 9} to integers in {1, 2, …, 9}.",
            "",
            "═══════════════════════════════════════════════════════",
            "  PART A — infer ⊛ from integers alone",
            "═══════════════════════════════════════════════════════",
            "",
            *_numeric_examples_block(),
            "",
            "═══════════════════════════════════════════════════════",
            "  PART B — nine glyphs and twelve equations",
            "═══════════════════════════════════════════════════════",
            "",
            "Nine distinct glyphs — Jyx, Klym, Mev, Orim, Pers, Renn, Tahl, Vost, Wold —",
            "each denote a UNIQUE integer secret in {1, …, 9}. No two glyphs share a secret.",
            "The operator ⊛ is the SAME function as in Part A, now applied to those secrets.",
            "",
            *_constraints_block(),
            "",
            "There is exactly one assignment of secrets to glyphs that satisfies all twelve lines.",
            "Use that assignment to answer the questions.",
            "",
            "For Q1, Q2, Q4, Q5, Q7, Q8: respond with a single integer in {1,…,9}.",
            "For Q3, Q6, Q9, Q10: respond with EXACTLY one glyph name from the list:",
            "Jyx, Klym, Mev, Orim, Pers, Renn, Tahl, Vost, Wold.",
            "",
            "  Q1:  What integer secret does Klym denote?",
            "       [1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9]",
            "",
            "  Q2:  What integer secret does Wold denote?",
            "       [1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9]",
            "",
            "  Q3:  Pers ⊛ Renn equals which glyph? (i.e. which glyph's secret is the result?)",
            "       [JYX / KLYM / MEV / ORIM / PERS / RENN / TAHL / VOST / WOLD]",
            "",
            "  Q4:  Evaluate (Klym ⊛ Pers) ⊛ Jyx  — compute Klym ⊛ Pers first, then ⊛ with Jyx.",
            "       [1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9]",
            "",
            "  Q5:  Evaluate Klym ⊛ (Pers ⊛ Jyx)  — compute Pers ⊛ Jyx first, then Klym ⊛ with that.",
            "       [1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9]",
            "",
            "  Q6:  Which glyph denotes the integer 3?",
            "       [JYX / KLYM / MEV / ORIM / PERS / RENN / TAHL / VOST / WOLD]",
            "",
            "  Q7:  What integer is Mev ⊛ Renn?",
            "       [1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9]",
            "",
            "  Q8:  What integer is Orim ⊛ Tahl?",
            "       [1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9]",
            "",
            "  Q9:  Wold ⊛ Klym equals which glyph?",
            "       [JYX / KLYM / MEV / ORIM / PERS / RENN / TAHL / VOST / WOLD]",
            "",
            "  Q10: Sort the three glyphs Pers, Renn, Tahl by their integer secrets (low to high).",
            "       Which glyph is in the middle (the median)?",
            "       [JYX / KLYM / MEV / ORIM / PERS / RENN / TAHL / VOST / WOLD]",
            "",
        ]
    )

    result = llm.prompt(prompt, schema=GlyphBindAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_EXPECTED)

    for key, exp_val in _EXPECTED.items():
        act = str(getattr(result, key)).strip()
        expn = str(exp_val).strip()
        if _label_match(expn, act):
            correct += 1
        assertions.assert_equal(expn.upper(), act.upper(), expectation=f"`{key}` must match {expn}.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _EXPECTED}
    _log_trace("glyph_bind_assoc_learning", _TASK_DESCRIPTION, prompt, answers, _EXPECTED, score)
    return score


if __name__ == "__main__":
    glyph_bind_assoc_learning.run(kbench.llm)

