#!/usr/bin/env python
# coding: utf-8

import re
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
    "An alien organism uses a 3-nucleotide codon system where the first and third "
    "nucleotide positions jointly determine the amino acid via a hidden structural rule. "
    "The middle nucleotide is always G (a fixed separator). The model observes 10 mRNA "
    "sequences and their translated peptides, exposing 11 of the 16 possible (nuc1, nuc3) "
    "combinations. Test questions require translating sequences whose codons use the 5 "
    "unseen combinations, which can be inferred by recognising the Latin-square structure "
    "in the observed data. The rule has a unique solution: there is exactly one 4x4 "
    "assignment consistent with the Latin-square constraint and all 11 observed mappings."
)

# Nucleotide ordering: A=0, U=1, G=2, C=3
_NUC_VAL = {"A": 0, "U": 1, "G": 2, "C": 3}
# 4 invented amino acids (not standard single-letter codes for common AAs)
_AA = ["P", "Q", "R", "S"]
# Middle nucleotide is always G — a fixed structural constraint, observable from demos
_MID = "G"

# The hidden rule: AA = _AA[(col - row) mod 4]
# where row = _NUC_VAL[nuc1], col = _NUC_VAL[nuc3]
# This produces a cyclic Latin square — every row and column contains each AA exactly once.
#
# Full 4x4 grid (nuc1 × nuc3):
#        A    U    G    C
#  A:    P    Q    R    S
#  U:    S    P    Q    R
#  G:    R    S    P    Q
#  C:    Q    R    S    P


def _encode(nuc1: str, nuc3: str) -> str:
    row = _NUC_VAL[nuc1]
    col = _NUC_VAL[nuc3]
    return _AA[(col - row) % 4]


def _translate(codon_list: list) -> str:
    return "".join(_encode(c[0], c[2]) for c in codon_list)


def _str_match(expected: str, actual: str) -> bool:
    if actual is None:
        return False
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


# ---------------------------------------------------------------------------
# Demonstration sequences.
# These 12 observations collectively expose 11 of the 16 possible (nuc1, nuc3)
# combinations.  The 5 reserved for test questions are:
#   (A,G) -> R   (U,A) -> S   (G,A) -> R   (C,G) -> S   (C,C) -> P
# With 11 of 16 cells of a 4×4 Latin square known, the remaining 5 are each
# uniquely forced by the row-and-column constraint — no arithmetic needed.
# ---------------------------------------------------------------------------
_RAW_DEMOS = [
    (["AGA", "AGU"],            "PQ"),
    (["UGU", "UGG"],            "PQ"),
    (["GGG", "GGC"],            "PQ"),
    (["AGC", "UGC"],            "SR"),
    (["GGU", "CGU"],            "SR"),
    (["AGA", "UGG", "GGC"],     "PQQ"),
    (["AGU", "GGG"],            "QP"),
    (["UGU", "AGC"],            "PS"),
    (["CGU", "GGU", "AGU"],     "RSQ"),
    (["UGG", "CGU"],            "QR"),
    (["AGC", "GGC", "UGC"],     "SQR"),
    (["CGA", "AGU"],            "QQ"),
]

_VERIFIED_DEMOS = []
for _codons, _expected_pep in _RAW_DEMOS:
    _actual = _translate(_codons)
    assert _actual == _expected_pep, f"Demo error: {_codons} -> {_actual!r} != {_expected_pep!r}"
    _VERIFIED_DEMOS.append((_codons, _actual))

# ---------------------------------------------------------------------------
# Test cases — every codon in each test uses one of the 5 unseen (nuc1,nuc3)
# combinations.  The correct peptide is uniquely determined by the Latin square
# constraint derivable from the 12 demonstration examples.
# ---------------------------------------------------------------------------
_TEST_CASES = [
    (["AGG", "UGA"],              "RS"),
    (["GGA", "CGG", "AGG"],       "RSR"),
    (["CGC", "UGA", "GGA"],       "PSR"),
    (["AGG", "CGC", "GGA", "UGA"], "RPRS"),
]

for _tc_codons, _tc_expected in _TEST_CASES:
    _tc_actual = _translate(_tc_codons)
    assert _tc_actual == _tc_expected, (
        f"Test case error: {_tc_codons} -> {_tc_actual!r} != {_tc_expected!r}"
    )


def _build_prompt(demos: list, test_cases: list) -> str:
    lines = [
        "You are studying the genetic code of an alien organism.",
        "Each codon is 3 nucleotides from {A, U, G, C}.",
        "Each codon translates to exactly one amino acid: P, Q, R, or S.",
        "",
        "Below are observed mRNA sequences and their translated peptide sequences.",
        "Each codon is shown separated by hyphens:",
        "",
        "Observations:",
    ]
    for i, (codons, pep) in enumerate(demos, 1):
        lines.append(f"  {i:2d}. {'-'.join(codons)}  ->  {pep}")

    lines.append("")
    lines.append("Translate these 4 new mRNA sequences:")
    for i, (codons, _) in enumerate(test_cases, 1):
        lines.append(f"  Q{i}: {'-'.join(codons)}  ->  ?")

    lines.append("")
    lines.append(
        "Provide answer_1 through answer_4 as the translated peptide string "
        "(e.g., PQR for a 3-codon sequence)."
    )
    return "\n".join(lines)


def _prepare():
    prompt = _build_prompt(_VERIFIED_DEMOS, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (_, expected) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"answer_{i}", None)
            got = raw.strip() if isinstance(raw, str) else raw
            ok = _str_match(expected, str(got)) if got is not None else False
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / len(_TEST_CASES), results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str


@kbench.task(
    name="codon_table_translation_obs_learning",
    description=(
        "Observe 12 mRNA→peptide translations under a custom alien genetic code. "
        "Each codon uses only the first and third nucleotides to encode one of four "
        "amino acids (P, Q, R, S) via a hidden structural rule."
    ),
)
def codon_table_translation_obs_learning(llm) -> float:
    """Infer a Latin-square codon rule from 12 examples; translate 4 new mRNA sequences."""
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
            for i in range(1, len(_TEST_CASES) + 1)
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="codon_table_translation_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    codon_table_translation_obs_learning.run(kbench.llm)

