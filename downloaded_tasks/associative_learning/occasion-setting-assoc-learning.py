#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "H-03 (expanded): Dual-path hierarchical occasion setting with a third inert pulse family. "
    "Hidden rule: OPEN iff (i) Dysis arrives and the core relay is permissive, OR "
    "(ii) Vetch arrives AND Toral is ACTIVE AND the core relay is permissive — where the core relay "
    "is permissive iff Keth is ACTIVE AND (Rym is IDLE OR Ves is ACTIVE). "
    "Nelt is categorically non-gateable (never opens), regardless of modulators. "
    "This is strictly richer than classic occasion setting: the model must learn that Toral is a "
    "pulse-specific occasion setter (enables Vetch only) without blocking or substituting for Dysis, "
    "while still inferring Rym as first-order blocker and Ves as second-order restorer on the shared core. "
    "Training exploits two independent co-occurrence traps mirrored across pulse types: "
    "(A) Every OPEN trial with Ves ACTIVE also has Rym ACTIVE, so shallow statistics suggest "
    "'Vetch/Dysis opens only when Ves co-occurs with Rym' — failing Q7 and Q9 which need Ves ACTIVE "
    "with Rym IDLE (Ves is vacuously true but required by logic). "
    "(B) Every Dysis-mediated OPEN has Toral IDLE, so statistics suggest Toral is incompatible with "
    "or irrelevant to Dysis in a directional way — failing Q8 (Dysis + Toral ACTIVE + minimal core). "
    "Fourteen probes mix recall, cross-pulse composition, and novel tuples absent from the 26 trials."
)


# ─── Hidden structure ─────────────────────────────────────────────────────────
#
#  Pulses: Dysis (primary trigger), Vetch (secondary trigger; needs Toral to occasion it),
#           Nelt (never opens), none.
#
#  Core(Keth,Rym,Ves)  :=  Keth=ACTIVE  ∧  (Rym=IDLE ∨ Ves=ACTIVE)
#
#  OPEN  iff  Nelt does not arrive, and:
#            ( Dysis arrives ∧ Core )  ∨  ( Vetch arrives ∧ Toral=ACTIVE ∧ Core )
#
#  Keth   — arms the shared relay for any gateable pulse (Dysis or occasioned Vetch).
#  Rym    — first-order brake on the core; Ves cancels that brake only.
#  Ves    — restorer: neutralizes Rym's brake; no standalone gating power.
#  Toral  — occasions Vetch only; does not participate in the Dysis disjunct except as
#            irrelevant concomitant (Dysis OPEN does not require Toral IDLE logically,
#            but every Dysis OPEN in training has Toral IDLE — co-occurrence trap Q8).
#
# ─── Co-occurrence traps ───────────────────────────────────────────────────────
#
#  Trap A (Ves): All training rows where Ves=ACTIVE and gate=OPEN have Rym=ACTIVE.
#       Ves is never ACTIVE with Rym=IDLE on an OPEN outcome — though logic allows it.
#
#  Trap B (Toral×Dysis): All Dysis-mediated OPEN trials have Toral=IDLE.
#       Toral ACTIVE with Dysis never appears in OPEN training — though logic allows OPEN.
#
# ─── Trial table (26) — verified against _true_open ───────────────────────────


def _true_open(keth: bool, rym: bool, ves: bool, toral: bool, pulse: str) -> bool:
    pulse_l = pulse.strip().lower()
    if pulse_l == "nelt":
        return False
    if pulse_l in ("", "none", "no pulse"):
        return False
    core = keth and (not rym or ves)
    if pulse_l == "dysis":
        return core
    if pulse_l == "vetch":
        return toral and core
    return False


def _label_from_bool(opening: bool) -> str:
    return "OPEN" if opening else "CLOSED"


_RELAY_EXPECTED = {
    "q_1":  "CLOSED",  # recall: no Keth, Dysis
    "q_2":  "OPEN",    # recall: vanilla Dysis
    "q_3":  "CLOSED",  # recall: Rym blocks, no Ves
    "q_4":  "OPEN",    # recall: Ves restores under Rym
    "q_5":  "CLOSED",  # recall: Nelt never
    "q_6":  "CLOSED",  # recall: Vetch path without Keth
    "q_7":  "OPEN",    # NOVEL: Dysis + Keth + Rym IDLE + Ves ACTIVE + Toral IDLE (Trap A)
    "q_8":  "OPEN",    # NOVEL: Dysis + Toral ACTIVE — Trap B; core minimal
    "q_9":  "OPEN",    # NOVEL: Vetch + Trap A on occasioned path (Ves redundant, Rym idle)
    "q_10": "CLOSED",  # NOVEL: Vetch without Toral (even with easy core)
    "q_11": "CLOSED",  # NOVEL: Dysis + Rym block + Toral ACTIVE (spurious Toral)
    "q_12": "CLOSED",  # NOVEL: Nelt + full modulator stack
    "q_13": "CLOSED",  # NOVEL: Vetch + Toral but Keth off
    "q_14": "CLOSED",  # NOVEL: Vetch + Rym block, no Ves, Toral on
}

# Sanity-check: expected answers match the true rule.
_RULE_CASES = {
    "q_1":  (False, False, False, False, "dysis"),
    "q_2":  (True,  False, False, False, "dysis"),
    "q_3":  (True,  True,  False, False, "dysis"),
    "q_4":  (True,  True,  True,  False, "dysis"),
    "q_5":  (True,  False, False, False, "nelt"),
    "q_6":  (False, True,  True,  True,  "vetch"),
    "q_7":  (True,  False, True,  False, "dysis"),
    "q_8":  (True,  False, False, True,  "dysis"),
    "q_9":  (True,  False, True,  True,  "vetch"),
    "q_10": (True,  False, False, False, "vetch"),
    "q_11": (True,  True,  False, True,  "dysis"),
    "q_12": (True,  True,  True,  True,  "nelt"),
    "q_13": (False, False, False, True,  "vetch"),
    "q_14": (True,  True,  False, True,  "vetch"),
}
for _k, (_ke, _ry, _ve, _to, _pu) in _RULE_CASES.items():
    _got = _label_from_bool(_true_open(_ke, _ry, _ve, _to, _pu))
    assert _got == _RELAY_EXPECTED[_k], f"Sanity-check failed for {_k}: rule gives {_got}, expected {_RELAY_EXPECTED[_k]}"


def _log_trace(task: str, description: str, prompt: str, answers: dict, expected: dict, score: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    print(f"\n  RESPONSES:")
    for key in expected:
        actual = answers.get(key, "?")
        exp = expected[key]
        match = "✓" if _label_match(exp, str(actual)) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _label_match(expected: str, actual: str) -> bool:
    """Return True if `expected` appears as a whole word in `actual` (case-insensitive)
    and is not immediately preceded by a negation word.
    """
    token = re.escape(expected.strip())
    m = re.search(rf"\b{token}\b", actual.strip(), re.IGNORECASE)
    if not m:
        return False
    prefix = actual[: m.start()].strip()
    last_words = re.findall(r"\w+", prefix)[-3:]
    negations = {"not", "isn't", "isnt", "never", "no", "cannot", "can't", "cant", "neither", "without"}
    return not any(w.lower() in negations for w in last_words)


@dataclass
class RelayHierarchyAnswer:
    q_1:  str
    q_2:  str
    q_3:  str
    q_4:  str
    q_5:  str
    q_6:  str
    q_7:  str
    q_8:  str
    q_9:  str
    q_10: str
    q_11: str
    q_12: str
    q_13: str
    q_14: str


@kbench.task(
    name="occasion_setting_assoc_learning",
    description=(
        "Dual-path occasion setting: Keth arms, Rym brakes unless Ves restores. "
        "Dysis triggers on core; Vetch needs Toral, Nelt never opens. 26 trials, 14 Qs, hard novel probes."
    ),
)
def occasion_setting_assoc_learning(llm) -> float:
    """Dual-path hierarchical modulation with cross-pulse occasion setting; fraction correct over 14 questions."""

    prompt = "\n".join([
        "A relay station log records whether a downstream gate OPENS or stays CLOSED after each trial.",
        "Each trial lists four status indicators (Keth, Rym, Ves, Toral — each ACTIVE or IDLE) and which",
        "pulse arrived: a Dysis pulse, a Vetch pulse, a Nelt pulse, or no pulse. The gate either OPENS",
        "or stays CLOSED.",
        "",
        "Here are ALL 26 trials recorded in this log:",
        "",
        "  Trial  1: Keth IDLE,    Rym IDLE, Ves IDLE, Toral IDLE, Dysis pulse -> Gate CLOSED.",
        "  Trial  2: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE, Dysis pulse -> Gate OPENS.",
        "  Trial  3: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE, Dysis pulse -> Gate OPENS.",
        "  Trial  4: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE, no pulse      -> Gate CLOSED.",
        "  Trial  5: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE, Vetch pulse  -> Gate CLOSED.",
        "  Trial  6: Keth ACTIVE,  Rym ACTIVE, Ves IDLE, Toral IDLE, Dysis pulse -> Gate CLOSED.",
        "  Trial  7: Keth ACTIVE,  Rym ACTIVE, Ves IDLE, Toral IDLE, Dysis pulse -> Gate CLOSED.",
        "  Trial  8: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral IDLE, Dysis pulse -> Gate OPENS.",
        "  Trial  9: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral IDLE, Dysis pulse -> Gate OPENS.",
        "  Trial 10: Keth IDLE,    Rym ACTIVE, Ves ACTIVE, Toral IDLE, Dysis pulse -> Gate CLOSED.",
        "  Trial 11: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral ACTIVE, Vetch pulse -> Gate OPENS.",
        "  Trial 12: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral ACTIVE, Vetch pulse -> Gate OPENS.",
        "  Trial 13: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral ACTIVE, Vetch pulse -> Gate OPENS.",
        "  Trial 14: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral ACTIVE, Vetch pulse -> Gate OPENS.",
        "  Trial 15: Keth ACTIVE,  Rym ACTIVE, Ves IDLE, Toral ACTIVE, Vetch pulse -> Gate CLOSED.",
        "  Trial 16: Keth IDLE,    Rym IDLE, Ves IDLE, Toral ACTIVE, Vetch pulse -> Gate CLOSED.",
        "  Trial 17: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral IDLE, Vetch pulse -> Gate CLOSED.",
        "  Trial 18: Keth ACTIVE,  Rym ACTIVE, Ves IDLE, Toral IDLE, no pulse      -> Gate CLOSED.",
        "  Trial 19: Keth ACTIVE,  Rym ACTIVE, Ves IDLE, Toral ACTIVE, Dysis pulse -> Gate CLOSED.",
        "  Trial 20: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE, Nelt pulse    -> Gate CLOSED.",
        "  Trial 21: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral ACTIVE, Nelt pulse -> Gate CLOSED.",
        "  Trial 22: Keth IDLE,    Rym IDLE, Ves IDLE, Toral IDLE, Vetch pulse  -> Gate CLOSED.",
        "  Trial 23: Keth IDLE,    Rym ACTIVE, Ves ACTIVE, Toral ACTIVE, Dysis pulse -> Gate CLOSED.",
        "  Trial 24: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral IDLE, no pulse      -> Gate CLOSED.",
        "  Trial 25: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE, Dysis pulse -> Gate OPENS.",
        "  Trial 26: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral ACTIVE, Vetch pulse -> Gate OPENS.",
        "",
        "Based ONLY on these 26 trials, answer each question with exactly one token: OPEN or CLOSED.",
        "",
        "  Q1:  Keth IDLE,    Rym IDLE, Ves IDLE, Toral IDLE,    Dysis pulse  — does the gate open?",
        "  Q2:  Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE,    Dysis pulse  — does the gate open?",
        "  Q3:  Keth ACTIVE,  Rym ACTIVE, Ves IDLE, Toral IDLE,    Dysis pulse  — does the gate open?",
        "  Q4:  Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral IDLE,    Dysis pulse  — does the gate open?",
        "  Q5:  Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE,    Nelt pulse   — does the gate open?",
        "  Q6:  Keth IDLE,    Rym ACTIVE, Ves ACTIVE, Toral ACTIVE, Vetch pulse  — does the gate open?",
        "  Q7:  Keth ACTIVE,  Rym IDLE, Ves ACTIVE, Toral IDLE,    Dysis pulse  — does the gate open?",
        "  Q8:  Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral ACTIVE,    Dysis pulse  — does the gate open?",
        "  Q9:  Keth ACTIVE,  Rym IDLE, Ves ACTIVE, Toral ACTIVE,    Vetch pulse  — does the gate open?",
        "  Q10: Keth ACTIVE,  Rym IDLE, Ves IDLE, Toral IDLE,    Vetch pulse  — does the gate open?",
        "  Q11: Keth ACTIVE,  Rym ACTIVE, Ves IDLE, Toral ACTIVE,    Dysis pulse  — does the gate open?",
        "  Q12: Keth ACTIVE,  Rym ACTIVE, Ves ACTIVE, Toral ACTIVE,    Nelt pulse   — does the gate open?",
        "  Q13: Keth IDLE,    Rym IDLE, Ves IDLE, Toral ACTIVE,    Vetch pulse  — does the gate open?",
        "  Q14: Keth ACTIVE,  Rym ACTIVE, Ves IDLE, Toral ACTIVE,    Vetch pulse  — does the gate open?",
        "",
    ])

    result = llm.prompt(prompt, schema=RelayHierarchyAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_RELAY_EXPECTED)

    for key, expn in _RELAY_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        if _label_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must be {expn}.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _RELAY_EXPECTED}
    _log_trace("occasion_setting", _TASK_DESCRIPTION, prompt, answers, _RELAY_EXPECTED, score)
    return score


if __name__ == "__main__":
    occasion_setting_assoc_learning.run(kbench.llm)

