#!/usr/bin/env python
# coding: utf-8

"""
Inference-time associative learning: four novel infix operators on digits 0–9.
No formulas are stated — only example lines. Models must induce each operator,
then answer direct applications, composition, a modular inverse, and a
commutativity probe.
"""

from dataclasses import dataclass

import random

import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Pure inference-time operator induction on digits 0–9. Four unrelated infix "
    "operators (QREL, VARN, TYLX, WIRM) are never defined in prose; ~128 example "
    "lines mix all four. Eight held-out queries test direct use, parenthesized "
    "composition, a single-digit inverse for QREL, non-commutativity of VARN, and "
    "a two-operator compound. Operator names avoid KREL (used as a stimulus token "
    "elsewhere in the suite). Exact semantics are defined only in code, not in "
    "the model-facing prompt."
)


def _qrel(a: int, b: int) -> int:
    return (3 * a + b) % 10


def _varn(a: int, b: int) -> int:
    return (a * (b + 1)) % 10


def _tylx(a: int, b: int) -> int:
    return (a + 2 * b) % 10


def _wirm(a: int, b: int) -> int:
    return (a + 3 * b + a * b) % 10


def _smallest_x_qrel(result: int, b: int) -> int:
    for x in range(10):
        if _qrel(x, b) == result:
            return x
    raise ValueError(f"no x in 0..9 with QREL(x,{b}) == {result}")


_OPS: dict[str, tuple[str, object]] = {
    "QREL": ("QREL", _qrel),
    "VARN": ("VARN", _varn),
    "TYLX": ("TYLX", _tylx),
    "WIRM": ("WIRM", _wirm),
}

# Do not leak exact tuples used in the eight test questions.
_EXCLUDE: set[tuple[int, int, str]] = {
    (7, 4, "QREL"),
    (6, 1, "VARN"),
    (8, 6, "TYLX"),
    (5, 4, "WIRM"),
    (3, 2, "QREL"),
    (1, 4, "VARN"),
    (2, 3, "QREL"),
    (7, 9, "TYLX"),
}


def _training_lines(seed: int = 42, per_op: int = 32) -> list[str]:
    rng = random.Random(seed)
    rows: list[tuple[str, int, int, int]] = []
    counts = {k: 0 for k in _OPS}
    while any(counts[k] < per_op for k in _OPS):
        name = rng.choice(list(_OPS.keys()))
        if counts[name] >= per_op:
            continue
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if (a, b, name) in _EXCLUDE:
            continue
        fn = _OPS[name][1]
        rows.append((name, a, b, int(fn(a, b))))
        counts[name] += 1
    rng.shuffle(rows)
    return [f"  {a} {nm} {b} = {r}" for nm, a, b, r in rows]


_TRAINING_BLOCK = _training_lines()


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
        match = "✓" if _match_q(key, exp, actual) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _yes_no_norm(s: object) -> str:
    t = str(s).strip().upper().rstrip(".")
    return t if t in ("YES", "NO") else ""


def _match_q(key: str, exp: object, actual: object) -> bool:
    if key == "q_7":
        return _yes_no_norm(actual) == str(exp).strip().upper()
    try:
        return int(str(actual).strip()) == int(exp)
    except (TypeError, ValueError):
        return False


@dataclass
class InferenceDyadOperatorsAnswer:
    q_1: int
    q_2: int
    q_3: int
    q_4: int
    q_5: int
    q_6: int
    q_7: str
    q_8: int


_EXPECTED = {
    "q_1": _qrel(7, 4),
    "q_2": _varn(6, 1),
    "q_3": _tylx(8, 6),
    "q_4": _varn(_qrel(3, 2), 4),
    "q_5": _wirm(5, 4),
    "q_6": _smallest_x_qrel(8, 5),
    "q_7": "NO",
    "q_8": _tylx(_wirm(5, 4), _qrel(2, 3)),
}


@kbench.task(
    name="inference_dyad_operators_assoc_learning",
    description=(
        "H-IO: Four undisclosed digit operators learned only from ~128 mixed examples; "
        "8 queries: direct, composition, inverse (mod 10), VARN commutativity, WIRM+TYLX+QREL chain."
    ),
)
def inference_dyad_operators_assoc_learning(llm) -> float:
    """
    Four-operator inference from examples only. Score = fraction correct / 8.
    """

    prompt = "\n".join(
        [
            "Below is a log of equations over single decimal digits 0–9.",
            "Each line uses ONE of four infix operators: QREL, VARN, TYLX, or WIRM.",
            "The operators are unrelated to each other and are not named anywhere else.",
            "There is no further documentation — infer each operator from the examples.",
            "",
            "Training log (unordered):",
            "",
            *_TRAINING_BLOCK,
            "",
            "Answer using ONLY the patterns consistent with the log above.",
            "For Q1–Q6 and Q8: give a single digit 0–9.",
            "For Q7: answer YES or NO.",
            "",
            "  Q1:  7 QREL 4 = ?",
            "",
            "  Q2:  6 VARN 1 = ?",
            "",
            "  Q3:  8 TYLX 6 = ?",
            "",
            "  Q4:  (3 QREL 2) VARN 4 = ?",
            "       (Evaluate the parenthesized part first.)",
            "",
            "  Q5:  5 WIRM 4 = ?",
            "",
            "  Q6:  Find a digit x in {0,…,9} such that x QREL 5 = 8.",
            "       If more than one digit works, choose the smallest.",
            "",
            "  Q7:  Is VARN commutative? That is, is a VARN b always equal to b VARN a",
            "       for all digits a, b? Answer YES or NO.",
            "",
            "  Q8:  (5 WIRM 4) TYLX (2 QREL 3) = ?",
            "       (Evaluate each parenthesized part first, then TYLX.)",
            "",
        ]
    )

    result = llm.prompt(prompt, schema=InferenceDyadOperatorsAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_EXPECTED)

    for key, exp_val in _EXPECTED.items():
        act = getattr(result, key)
        if key == "q_7":
            act_s = _yes_no_norm(act)
            if act_s == str(exp_val).strip().upper():
                correct += 1
            assertions.assert_equal(str(exp_val), act_s, expectation=f"`{key}` must be {exp_val}.")
        else:
            try:
                act_i = int(act)
            except (TypeError, ValueError):
                act_i = None
            if act_i == int(exp_val):
                correct += 1
            assertions.assert_equal(str(exp_val), str(act_i), expectation=f"`{key}` must be {exp_val}.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _EXPECTED}
    _log_trace(
        "inference_dyad_operators_assoc_learning",
        _TASK_DESCRIPTION,
        prompt,
        answers,
        {k: str(v) for k, v in _EXPECTED.items()},
        score,
    )
    return score


if __name__ == "__main__":
    inference_dyad_operators_assoc_learning.run(kbench.llm)

