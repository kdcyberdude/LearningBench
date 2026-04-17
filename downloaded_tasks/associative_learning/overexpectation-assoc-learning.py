#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests associative overexpectation from text logs. Phase 1 gives solo baselines; "
    "Phase 2 shows a subadditive compound (joint below the sum of solos). Phase 3 "
    "records new solo runs after the compound block. The Phase-3 solo rates match "
    "each cue's effective contribution during Phase 2 (they sum exactly to the "
    "observed joint rate) but they are NOT a proportional extrapolation from Phase-1 "
    "solos — e.g. Lab Alder: naive proportional split of 12 would be 9+3 from 15:5, "
    "whereas Phase 3 reveals 8+4. Q5 and Q8 ask whether Phase-2 attribution preserves "
    "the Phase-1 solo ratio; the correct answer is NO, while proportional splitting (9/3 "
    "or 6/2) yields YES — a consistency trap across questions. "
    "Lab Brume mirrors the same structure with different numerals (mirk/selt). "
    "Eight answers: six integers, two YES/NO. No formulas, laws, or cue roles are named."
)


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
        match = "✓" if _answer_matches(key, exp, actual) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _str_match(expected: str, actual: str) -> bool:
    e = expected.strip()
    a = actual.strip()
    if e.upper() in ("YES", "NO"):
        return bool(re.search(rf"\b{re.escape(e)}\b", a, re.IGNORECASE))
    return bool(re.search(re.escape(e), a, re.IGNORECASE))


def _answer_matches(key: str, expected, actual) -> bool:
    if key.endswith("_yesno"):
        return _str_match(str(expected), str(actual))
    try:
        return int(actual) == int(expected)
    except (TypeError, ValueError):
        return False


@dataclass
class OverexpectationAnswer:
    # Alder: P1 kyre 15 / walt 5; P2 joint 12; P3 kyre 8 / walt 4 (≠ 9/3 from proportion)
    kyre_effective_share_phase2: int
    walt_effective_share_phase2: int
    naive_sum_of_solos: int
    compound_versus_naive_shortfall: int
    # NO: Phase-2 attribution (8:4) does not preserve Phase-1 solo ratio (15:5); YES=proportion trap
    phase2_kyre_walt_ratio_matches_phase1_solo_ratio_yesno: str

    # Brume: P1 mirk 12 / selt 4; P2 joint 8; P3 mirk 5 / selt 3 (≠ 6/2 from proportion)
    mirk_effective_share_phase2: int
    selt_effective_share_phase2: int
    # NO: 5:3 ≠ 12:4; proportional 6:2 would wrongly say YES
    phase2_mirk_selt_ratio_matches_phase1_solo_ratio_yesno: str


_OVEREXP_EXPECTED = {
    "kyre_effective_share_phase2": 8,
    "walt_effective_share_phase2": 4,
    "naive_sum_of_solos": 20,
    "compound_versus_naive_shortfall": 8,
    "phase2_kyre_walt_ratio_matches_phase1_solo_ratio_yesno": "NO",
    "mirk_effective_share_phase2": 5,
    "selt_effective_share_phase2": 3,
    "phase2_mirk_selt_ratio_matches_phase1_solo_ratio_yesno": "NO",
}


# ── Ground truth (not in prompt) ────────────────────────────────────────────
#
# Alder
#   Phase 1 solo: kyre 15, walt 5  → naive additive expectation for joint = 20
#   Phase 2 joint: 12  → overexpectation deficit = 8
#   Wrong shortcut: split 12 by Phase-1 ratio 15:5 → 9 and 3 (incorrect here)
#   Phase 3 solo (after compound): kyre 8, walt 4 → sum 12 = Phase-2 joint; these
#   are the effective Phase-2 contributions (revaluation, not proportionality).
#
# Brume
#   Phase 1: mirk 12, selt 4 → naive sum 16; Phase 2 joint 8; deficit 8
#   Proportional split of 8 would be 6+2; Phase 3 shows 5+3.
# ─────────────────────────────────────────────────────────────────────────────


@kbench.task(
    name="overexpectation_assoc_learning",
    description=(
        "H-06: Overexpectation from summary logs. Assess solo, joint (subadditive), and post-compound solo. Match to effective shares, not to proportional split. Lab transfer (mirk/selt). Eight answers; prompt gives no rules."
    ),
)
def overexpectation_assoc_learning(llm) -> float:
    """
    Integrate three phases; avoid proportional shortcut from Phase 1 alone.
    Returns fraction of eight fields correct.
    """

    prompt = "\n".join([
        "Two labs publish chronological trial logs. Each line is one observation.",
        "Infer whatever you can from the sequence alone; nothing here names laws,",
        "formulas, or roles of the cues beyond what the numbers show.",
        "",
        "── Lab Alder (kyre, walt) ──────────────────────────────────────────────",
        "",
        "  Phase 1 — solo",
        "    Trial  1: cue active: kyre        → 15 pulses/cycle",
        "    Trial  2: cue active: kyre        → 15 pulses/cycle",
        "    Trial  3: cue active: kyre        → 15 pulses/cycle",
        "    Trial  4: cue active: kyre        → 15 pulses/cycle",
        "    Trial  5: cue active: walt        →  5 pulses/cycle",
        "    Trial  6: cue active: walt        →  5 pulses/cycle",
        "    Trial  7: cue active: walt        →  5 pulses/cycle",
        "    Trial  8: cue active: walt        →  5 pulses/cycle",
        "",
        "  Phase 2 — both cues on",
        "    Trial  9: cues active: kyre, walt → 12 pulses/cycle",
        "    Trial 10: cues active: kyre, walt → 12 pulses/cycle",
        "    Trial 11: cues active: kyre, walt → 12 pulses/cycle",
        "    Trial 12: cues active: kyre, walt → 12 pulses/cycle",
        "",
        "  Phase 3 — solo again",
        "    Trial 13: cue active: walt        →  4 pulses/cycle",
        "    Trial 14: cue active: kyre        →  8 pulses/cycle",
        "    Trial 15: cue active: walt        →  4 pulses/cycle",
        "    Trial 16: cue active: kyre        →  8 pulses/cycle",
        "",
        "── Lab Brume (mirk, selt) ───────────────────────────────────────────────",
        "",
        "  Phase 1 — solo",
        "    Trial  1: cue active: mirk        → 12 pulses/cycle",
        "    Trial  2: cue active: mirk        → 12 pulses/cycle",
        "    Trial  3: cue active: selt        →  4 pulses/cycle",
        "    Trial  4: cue active: selt        →  4 pulses/cycle",
        "",
        "  Phase 2 — both cues on",
        "    Trial  5: cues active: mirk, selt →  8 pulses/cycle",
        "    Trial  6: cues active: mirk, selt →  8 pulses/cycle",
        "",
        "  Phase 3 — solo again",
        "    Trial  7: cue active: selt        →  3 pulses/cycle",
        "    Trial  8: cue active: mirk        →  5 pulses/cycle",
        "    Trial  9: cue active: selt        →  3 pulses/cycle",
        "    Trial 10: cue active: mirk        →  5 pulses/cycle",
        "",
        "── Questions (Alder unless noted) ─────────────────────────────────────",
        "",
        "  Q1 (integer): During Phase 2, how many pulses/cycle of the joint output",
        "      should be attributed to kyre?",
        "",
        "  Q2 (integer): During Phase 2, how many pulses/cycle of the joint output",
        "      should be attributed to walt?",
        "",
        "  Q3 (integer): If Phase 2 had matched the sum of the Phase-1 solo rates,",
        "      how many pulses/cycle would you have expected?",
        "",
        "  Q4 (integer): By how much does that expectation exceed the observed Phase-2",
        "      joint rate (same units)?",
        "",
        "  Q5 (YES or NO): Is the ratio of your Q1 answer to your Q2 answer exactly",
        "      equal to the ratio of kyre's Phase-1 solo rate to walt's Phase-1 solo rate?",
        "",
        "  Q6 (integer, Brume): During Phase 2 in Lab Brume, how many pulses/cycle of",
        "      the joint output should be attributed to mirk?",
        "",
        "  Q7 (integer, Brume): During Phase 2 in Lab Brume, how many pulses/cycle of",
        "      the joint output should be attributed to selt?",
        "",
        "  Q8 (YES or NO): Is the ratio of your Q6 answer to your Q7 answer exactly",
        "      equal to the ratio of mirk's Phase-1 solo rate to selt's Phase-1 solo rate?",
        "",
    ])

    result = llm.prompt(prompt, schema=OverexpectationAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_OVEREXP_EXPECTED)

    int_keys = (
        "kyre_effective_share_phase2",
        "walt_effective_share_phase2",
        "naive_sum_of_solos",
        "compound_versus_naive_shortfall",
        "mirk_effective_share_phase2",
        "selt_effective_share_phase2",
    )
    for key in int_keys:
        exp_val = _OVEREXP_EXPECTED[key]
        try:
            act_int = int(getattr(result, key))
        except (TypeError, ValueError):
            act_int = None
        if act_int == exp_val:
            correct += 1
        assertions.assert_equal(
            str(exp_val), str(act_int), expectation=f"`{key}` must equal {exp_val}."
        )

    for key, expn in (
        ("phase2_kyre_walt_ratio_matches_phase1_solo_ratio_yesno", "NO"),
        ("phase2_mirk_selt_ratio_matches_phase1_solo_ratio_yesno", "NO"),
    ):
        act = str(getattr(result, key)).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _OVEREXP_EXPECTED}
    _log_trace(
        "overexpectation", _TASK_DESCRIPTION, prompt, answers, _OVEREXP_EXPECTED, score
    )
    return score


if __name__ == "__main__":
    overexpectation_assoc_learning.run(kbench.llm)

