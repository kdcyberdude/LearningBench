#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Four-phase cross-interference extinction with asymmetric partial reinforcement, "
    "blocking disambiguation, and second-order associative-strength tracking. "
    "Phase 1 establishes a three-link chain KREL→MOVEN→THASP→alarm (100%). "
    "Phase 2 directly re-pairs KREL→alarm (100%) while extinguishing MOVEN in isolation (0%); "
    "DROVEN introduced only in compound with KREL at 60% reinforcement (blocking scenario). "
    "Phase 3 partially extinguishes KREL→alarm (50%), while THASP is re-introduced at 40%. "
    "Phase 4 fully extinguishes THASP (0%) and introduces BRYNN only in compound with DROVEN (100%). "
    "Final-state signal summary: "
    "KREL=partially excitatory (positive residual), "
    "THASP=extinguished, MOVEN=extinguished, "
    "DROVEN=UNKNOWN (blocking ambiguity; never isolated), "
    "BRYNN=UNKNOWN (depends on DROVEN's uncertain status). "
    "Probe answer derivations: "
    "Q1 MOVEN alone: MOVEN extinguished (Phase 2 direct extinction), THASP second-order path also dead → NO. "
    "Q2 THASP alone: re-introduced Phase 3, then fully extinguished Phase 4 → NO. "
    "Q3 KREL alone: partial extinction (50%) leaves positive residual → YES. "
    "Q4 MOVEN+THASP: both zero → NO. "
    "Q5 DROVEN alone: blocking by KREL in Phase 2; never isolated → UNKNOWN. "
    "Q6 KREL+THASP: KREL positive, THASP zero; net positive → YES. "
    "Q7 BRYNN alone: trained only with DROVEN whose status is UNKNOWN → UNKNOWN. "
    "Q8 KREL+MOVEN: KREL positive, MOVEN zero; net positive → YES. "
    "Q9 DROVEN+THASP: DROVEN unknown, THASP zero; net unknown → UNKNOWN. "
    "Q10 KREL+DROVEN: KREL positive; DROVEN unknown but ≥ 0; net ≥ KREL > 0 → YES (lower-bound dominance). "
    "Exact answer key: q1=NO, q2=NO, q3=YES, q4=NO, q5=UNKNOWN, q6=YES, q7=UNKNOWN, q8=YES, q9=UNKNOWN, q10=YES."
)


def _log_trace(task: str, description: str, prompt: str, answers: dict, expected: dict, score: float) -> None:
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


@dataclass
class DeepExtinctionAnswer:
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


_EXPECTED = {
    "q_1":  "NO",
    "q_2":  "NO",
    "q_3":  "YES",
    "q_4":  "NO",
    "q_5":  "UNKNOWN",
    "q_6":  "YES",
    "q_7":  "UNKNOWN",
    "q_8":  "YES",
    "q_9":  "UNKNOWN",
    "q_10": "YES",
}


@kbench.task(
    name="deep_second_order_extinction_assoc_learning",
    description=(
        "Four-phase extinction with asymmetric partial reinforcement, blocking disambiguation, "
        "second-order chain re-evaluation, and compound probes requiring lower-bound dominance "
        "reasoning under partial uncertainty."
    ),
)
def deep_second_order_extinction_assoc_learning(llm) -> float:
    """Deep second-order extinction with blocking and partial reinforcement; return fraction correct (10 questions)."""

    prompt = "\n".join([
        "You are given a complete experimental log for a synthetic predictive system.",
        "Read every phase carefully before answering.",
        "",
        "Definitions:",
        "  • EXCITATORY: net positive associative strength with the alarm at the end of all phases.",
        "  • EXTINGUISHED: association fully reduced to zero through non-reinforced trials.",
        "  • PARTIAL EXTINCTION: non-reinforced trials reduce strength but do not eliminate it.",
        "    A signal that undergoes partial extinction retains a positive (reduced) strength.",
        "  • BLOCKING: when a new signal appears exclusively in compound with an already-excitatory",
        "    signal, the established predictor may prevent the new signal from acquiring its own",
        "    independent associative strength. Whether blocking occurred must be inferred from the",
        "    training design; it cannot be assumed either way without isolation evidence.",
        "  • COMPOUND PROBE RULE:",
        "    — If a signal's strength is unknown, its contribution to a compound is unknown.",
        "    — If a signal's strength is zero (extinguished), its contribution is zero.",
        "    — If at least one signal in the compound has confirmed positive strength, and all",
        "      others have zero or positive (but possibly unknown) strength, the compound net is",
        "      confirmed positive.",
        "    — If all signals in the compound are zero, the net is zero.",
        "    — If the net cannot be determined, the compound is UNKNOWN.",
        "",
        "─────────────────────────────────────────",
        "PHASE 1  (trials 1–40)",
        "─────────────────────────────────────────",
        "  • THASP is paired directly with the alarm on every trial: 100% reinforcement (40 / 40).",
        "  • MOVEN precedes THASP on every trial; the chain MOVEN→THASP→alarm is followed",
        "    100% of the time (40 / 40).",
        "  • KREL precedes MOVEN on every trial; the chain KREL→MOVEN→THASP→alarm is followed",
        "    100% of the time (40 / 40).",
        "  • BRYNN and DROVEN do not appear in phase 1.",
        "",
        "─────────────────────────────────────────",
        "PHASE 2  (trials 41–100)",
        "─────────────────────────────────────────",
        "  • THASP is removed entirely from phase 2.",
        "  • MOVEN appears alone (no THASP, no KREL, no alarm): 0% reinforcement (0 / 30).",
        "  • KREL appears alone, directly followed by the alarm: 100% reinforcement (40 / 40).",
        "  • DROVEN appears for the first time, but ONLY in compound with KREL.",
        "    On these compound trials: DROVEN+KREL→alarm, 60% reinforcement (24 / 40).",
        "    DROVEN never appears without KREL in phase 2.",
        "",
        "─────────────────────────────────────────",
        "PHASE 3  (trials 101–160)",
        "─────────────────────────────────────────",
        "  • KREL appears alone; alarm occurs on 50% of trials: 50% reinforcement (30 / 60).",
        "  • THASP is re-introduced; alarm occurs on 40% of THASP trials: 40% reinforcement (16 / 40).",
        "  • MOVEN, DROVEN, and BRYNN do not appear in phase 3.",
        "",
        "─────────────────────────────────────────",
        "PHASE 4  (trials 161–200)",
        "─────────────────────────────────────────",
        "  • THASP appears alone with no alarm: 0% reinforcement (0 / 40).",
        "  • BRYNN appears for the first time, always in compound with DROVEN.",
        "    On every compound trial: BRYNN+DROVEN→alarm, 100% reinforcement (20 / 20).",
        "    BRYNN never appears without DROVEN in phase 4.",
        "  • KREL and MOVEN do not appear in phase 4.",
        "",
        "─────────────────────────────────────────",
        "PROBES  (test trials — no alarm is ever delivered during probes)",
        "─────────────────────────────────────────",
        "",
        "For each probe answer YES (alarm expected), NO (no alarm expected),",
        "or UNKNOWN (the evidence is insufficient to determine the outcome).",
        "",
        "Q1:  MOVEN appears alone.",
        "",
        "Q2:  THASP appears alone.",
        "",
        "Q3:  KREL appears alone.",
        "",
        "Q4:  MOVEN and THASP appear together (no other signals present).",
        "",
        "Q5:  DROVEN appears alone.",
        "",
        "Q6:  KREL and THASP appear together (no other signals present).",
        "",
        "Q7:  BRYNN appears alone.",
        "",
        "Q8:  KREL and MOVEN appear together (no other signals present).",
        "",
        "Q9:  DROVEN and THASP appear together (no other signals present).",
        "",
        "Q10: KREL and DROVEN appear together (no other signals present).",
        "",
    ])

    result = llm.prompt(prompt, schema=DeepExtinctionAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_EXPECTED)

    for key, expn in _EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must be {expn}.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _EXPECTED}
    _log_trace(
        "deep_second_order_extinction",
        _TASK_DESCRIPTION,
        prompt,
        answers,
        _EXPECTED,
        score,
    )
    return score


if __name__ == "__main__":
    deep_second_order_extinction_assoc_learning.run(kbench.llm)

