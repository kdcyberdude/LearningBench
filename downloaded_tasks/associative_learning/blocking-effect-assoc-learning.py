#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests five distinct aspects of associative learning inference across a 20-trial log "
    "with six compounds (Vorn, Grix, Kael, Nulm, Plex, Reth, Wex, Tael). "
    "8 questions covering: "
    ""
    "(A) Causal insufficiency / UNKNOWN — 4 questions, each for a distinct structural reason: "
    "    Q1 Plex: appeared exclusively in ALERT-context compound trials (always with Vorn or Grix). "
    "       Co-occurrence is 3/3 ALERT — strongest co-occurrence trap. UNKNOWN because an "
    "       established ALERT predictor fully accounts for each trial; Plex may be inert. "
    "    Q2 Reth: appeared with Kael (QUIET context) T14-T15 AND with Vorn (ALERT context) T16. "
    "       Outcome flips with the co-present compound — irresolvably confounded. UNKNOWN. "
    "    Q3 Wex: appeared exclusively with Nulm (QUIET) T17-T18. QUIET co-occurrence trap. "
    "       UNKNOWN because Nulm may be doing all the work; Wex could be ALERT, QUIET, or inert. "
    "    Q4 Tael: appeared with Vorn→ALERT in T19 and with Nulm→QUIET in T20. Two trials, "
    "       opposite predictors, opposite outcomes. Irresolvably confounded. UNKNOWN. "
    ""
    "(B) Confirmed causal attribution — Q5 and Q6: "
    "    Q5 Grix alone → ALERT (solo-established T06-T08). Tests positive evidence retrieval. "
    "    Q6 Kael alone → QUIET (solo-established T04-T05). Tests negative evidence retrieval. "
    ""
    "(C) Compositional prediction — Q7: "
    "    Grix + Nulm → ALERT. Grix is a confirmed ALERT cause; Nulm is confirmed QUIET "
    "    (non-suppressing). A confirmed ALERT cause dominates; Nulm adds nothing. "
    "    Model trap: UNKNOWN (never seen this pair) or QUIET (Nulm was present). "
    ""
    "(D) Blocking recognition — Q8 (integer): "
    "    How many of {Plex, Reth, Wex, Tael} have a conclusively established solo effect? → 0. "
    "    Tests meta-epistemic awareness: model must recognise that none of the four test "
    "    compounds have been isolated from confounds, so none can be conclusively classified. "
    "    Model trap: counting co-occurrence appearances as 'evidence' and answering > 0. "
)


def _log_trace(task, description, prompt, answers, expected, score):
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    print(f"\n  RESPONSES:")
    for key in expected:
        actual = answers.get(key, "?")
        exp = expected[key]
        match = "✓" if _str_match(str(exp), str(actual)) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _str_match(expected: str, actual: str) -> bool:
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


# ---------------------------------------------------------------------------
# 20-trial log — verified ground truth
#
# ESTABLISHED compounds (solo-tested, all confirmed):
#   Vorn → ALERT  (T01, T02, T03)
#   Kael → QUIET  (T04, T05)
#   Grix → ALERT  (T06, T07, T08)
#   Nulm → QUIET  (T09, T10)
#
# TEST compounds (never solo-tested, all UNKNOWN for distinct reasons):
#   Plex — T11 (Vorn+Plex→ALERT), T12 (Vorn+Plex→ALERT), T13 (Grix+Plex→ALERT)
#          100% ALERT co-occurrence but always with established ALERT predictor → UNKNOWN
#
#   Reth — T14 (Kael+Reth→QUIET), T15 (Kael+Reth→QUIET), T16 (Vorn+Reth→ALERT)
#          QUIET in QUIET context, ALERT in ALERT context → irresolvably confounded → UNKNOWN
#
#   Wex  — T17 (Nulm+Wex→QUIET), T18 (Nulm+Wex→QUIET)
#          100% QUIET co-occurrence but always with established QUIET predictor → UNKNOWN
#
#   Tael — T19 (Vorn+Tael→ALERT), T20 (Nulm+Tael→QUIET)
#          One trial each with opposite-outcome predictors → confounded → UNKNOWN
#
# Q7 ground truth — Grix + Nulm → ALERT:
#   Grix is a confirmed ALERT cause (T06-T08).
#   Nulm is confirmed QUIET, meaning it does not trigger ALERT on its own.
#   As stated in the prompt rules, a QUIET compound does not suppress ALERT from others.
#   Therefore Grix + Nulm = Grix's ALERT contribution + Nulm's zero contribution = ALERT.
#   This pair is never directly observed; the model must apply compositional reasoning.
#
# Q8 ground truth — 0 of {Plex, Reth, Wex, Tael} have a conclusively established effect:
#   All four were only observed in compound trials with known predictors acting as confounds.
#   None were solo-tested; none can be isolated. Count = 0.
# ---------------------------------------------------------------------------


@dataclass
class BlockingAnswer:
    q_plex: str         # Q1: Plex alone?                      UNKNOWN
    q_reth: str         # Q2: Reth alone?                      UNKNOWN
    q_wex: str          # Q3: Wex alone?                       UNKNOWN
    q_tael: str         # Q4: Tael alone?                      UNKNOWN
    q_grix: str         # Q5: Grix alone?                      ALERT
    q_kael: str         # Q6: Kael alone?                      QUIET
    q_grix_nulm: str    # Q7: Grix + Nulm together?            ALERT
    q_established: int  # Q8: How many of {Plex,Reth,Wex,Tael} are conclusively established?  0


@kbench.task(
    name="blocking_effect_assoc_learning",
    description=(
        "H-01: 8-question associative learning inference spanning 5 aspects — "
        "causal insufficiency ×4 "
        "confirmed attribution ×2 "
        "compositional prediction ×1 "
        "meta-epistemic counting ×1."
    ),
)
def blocking_effect_assoc_learning(llm) -> float:
    """
    Multi-aspect associative learning inference across a 20-trial phased log.
    Returns fraction correct across 8 questions.
    """

    prompt = "\n".join([
        "A detection system learns from experience which substances trigger an ALERT response.",
        "You will see a complete history of 20 trials. Each trial lists which substances were",
        "present and the system's observed signal (ALERT or QUIET).",
        "",
        "Read the entire history carefully before answering.",
        "",
        "━━━ Trial History ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        " T01  │  Vorn                      │  ALERT",
        " T02  │  Vorn                      │  ALERT",
        " T03  │  Vorn                      │  ALERT",
        " T04  │  Kael                      │  QUIET",
        " T05  │  Kael                      │  QUIET",
        " T06  │  Grix                      │  ALERT",
        " T07  │  Grix                      │  ALERT",
        " T08  │  Grix                      │  ALERT",
        " T09  │  Nulm                      │  QUIET",
        " T10  │  Nulm                      │  QUIET",
        " T11  │  Vorn + Plex               │  ALERT",
        " T12  │  Vorn + Plex               │  ALERT",
        " T13  │  Grix + Plex               │  ALERT",
        " T14  │  Kael + Reth               │  QUIET",
        " T15  │  Kael + Reth               │  QUIET",
        " T16  │  Vorn + Reth               │  ALERT",
        " T17  │  Nulm + Wex                │  QUIET",
        " T18  │  Nulm + Wex                │  QUIET",
        " T19  │  Vorn + Tael               │  ALERT",
        " T20  │  Nulm + Tael               │  QUIET",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "Use these definitions for reaction questions (Q1–Q7):",
        "  ALERT   — the evidence conclusively establishes this substance triggers ALERT.",
        "  QUIET   — the evidence conclusively establishes this substance does not trigger ALERT.",
        "  UNKNOWN — the evidence is insufficient to determine either way.",
        "",
        "Key reasoning rules:",
        "  Rule 1 — A substance present in an ALERT trial does NOT prove it caused the ALERT.",
        "           Another substance in the same trial may be fully responsible.",
        "  Rule 2 — A substance present in a QUIET trial is not necessarily QUIET itself.",
        "           Another substance may be suppressing the outcome.",
        "  Rule 3 — Conclusive evidence requires isolating a substance so that no other",
        "           substance in the same trial can account for the observed signal.",
        "  Rule 4 — A QUIET substance does not suppress ALERT produced by another substance.",
        "           If a confirmed ALERT-cause is present, the outcome is ALERT regardless",
        "           of how many confirmed QUIET substances share the trial.",
        "",
        " Q1  What does Plex alone produce?",
        "      (Choose ALERT, QUIET, or UNKNOWN.)",
        "",
        " Q2  What does Reth alone produce?",
        "      (Choose ALERT, QUIET, or UNKNOWN.)",
        "",
        " Q3  What does Wex alone produce?",
        "      (Choose ALERT, QUIET, or UNKNOWN.)",
        "",
        " Q4  What does Tael alone produce?",
        "      (Choose ALERT, QUIET, or UNKNOWN.)",
        "",
        " Q5  What does Grix alone produce?",
        "      (Choose ALERT, QUIET, or UNKNOWN.)",
        "",
        " Q6  What does Kael alone produce?",
        "      (Choose ALERT, QUIET, or UNKNOWN.)",
        "",
        " Q7  If Grix and Nulm are both present (and nothing else), what does the system produce?",
        "      (Choose ALERT, QUIET, or UNKNOWN.)",
        "",
        " Q8  Consider only these four substances: Plex, Reth, Wex, Tael.",
        "      How many of them have a conclusively established solo effect based on the",
        "      trial history above? Give a non-negative integer.",
        "",
    ])

    result = llm.prompt(prompt, schema=BlockingAnswer)
    assertions = kbench.assertions
    correct = 0
    total = 8

    str_checks = {
        "q_plex":      "UNKNOWN",  # Q1: ALERT co-occurrence trap
        "q_reth":      "UNKNOWN",  # Q2: conflicting-context trap
        "q_wex":       "UNKNOWN",  # Q3: QUIET co-occurrence trap
        "q_tael":      "UNKNOWN",  # Q4: confounded opposite-predictor trap
        "q_grix":      "ALERT",    # Q5: confirmed positive attribution
        "q_kael":      "QUIET",    # Q6: confirmed negative attribution
        "q_grix_nulm": "ALERT",    # Q7: compositional (ALERT-cause + QUIET-cause)
    }
    for key, expn in str_checks.items():
        act = str(getattr(result, key)).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must be {expn}.")

    try:
        act_int = int(getattr(result, "q_established"))
    except (TypeError, ValueError):
        act_int = None
    if act_int == 0:
        correct += 1
    assertions.assert_equal("0", str(act_int), expectation="`q_established` must equal 0.")

    score = correct / total
    expected = {**str_checks, "q_established": "0"}
    answers = {k: getattr(result, k) for k in [*str_checks, "q_established"]}
    _log_trace("blocking", _TASK_DESCRIPTION, prompt, answers, expected, score)
    return score


if __name__ == "__main__":
    blocking_effect_assoc_learning.run(kbench.llm)

