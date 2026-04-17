#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests multi-cue inhibitory summation with hierarchical modulation and compositional transfer. "
    "Six novel signals (gorv, telk, pryn, shael, dozn, felk) interact in a reactor system that may "
    "produce one of three outcomes (FLARE, DAMP, INERT). Hidden roles: gorv is a strong excitor "
    "(FLARE), felk is a weak excitor (DAMP), telk is a partial inhibitor (-1 step), pryn is a full "
    "inhibitor (-2 steps), shael is a modulator that doubles the strength of each co-present "
    "inhibitor, and dozn is neutral. The model must infer all six roles and the inhibitory-strength "
    "algebra from 18 observations, then predict 8 novel combinations that span all three outcome "
    "levels. Key challenges: (1) distinguishing two excitors of different strength, (2) partial vs "
    "full inhibition, (3) recognizing shael's multiplier role, (4) compositional predictions for "
    "unseen multi-signal combinations. No rules or roles are stated."
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
    """Return True if expected appears anywhere in actual (case-insensitive)."""
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))

@dataclass
class InhibitorySummationAnswer:
    q_1: str
    q_2: str
    q_3: str
    q_4: str
    q_5: str
    q_6: str
    q_7: str
    q_8: str


# Signal roles & excitation/inhibition values:
#   gorv  = strong excitor   -> contributes +2 to outcome level
#   felk  = weak excitor     -> contributes +1 to outcome level
#   telk  = partial inhibitor -> contributes -1 to outcome level
#   pryn  = full inhibitor    -> contributes -2 to outcome level
#   shael = inhibitory multiplier -> doubles each inhibitor's contribution
#                                    (telk becomes -2, pryn becomes -4)
#                                    has NO effect on excitors or alone
#   dozn  = neutral           -> contributes 0
#
# Outcome mapping (clamped to [0, 2]):
#   level >= 2 -> FLARE
#   level == 1 -> DAMP
#   level <= 0 -> INERT
#
# Derivation from 18 observations:
#
# Obs  1: gorv                 -> FLARE   [gorv = +2]
# Obs  2: gorv                 -> FLARE   [confirms]
# Obs  3: felk                 -> DAMP    [felk = +1]
# Obs  4: felk                 -> DAMP    [confirms]
# Obs  5: gorv, dozn           -> FLARE   [dozn = 0, gorv still +2]
# Obs  6: gorv, dozn           -> FLARE   [confirms]
# Obs  7: gorv, telk           -> DAMP    [+2 -1 = +1 -> DAMP]
# Obs  8: gorv, telk           -> DAMP    [confirms]
# Obs  9: gorv, pryn           -> INERT   [+2 -2 = 0 -> INERT]
# Obs 10: gorv, pryn           -> INERT   [confirms]
# Obs 11: felk, telk           -> INERT   [+1 -1 = 0 -> INERT]
# Obs 12: felk, telk           -> INERT   [confirms]
# Obs 13: gorv, telk, shael    -> INERT   [+2, telk*2 = -2, net 0 -> INERT]
# Obs 14: gorv, telk, shael    -> INERT   [confirms]
# Obs 15: gorv, shael          -> FLARE   [shael alone (no inhibitor) = no effect, +2 -> FLARE]
# Obs 16: felk, shael          -> DAMP    [shael alone (no inhibitor) = no effect, +1 -> DAMP]
# Obs 17: gorv, dozn, telk     -> DAMP    [+2, dozn=0, telk=-1, net +1 -> DAMP]
# Obs 18: telk, dozn           -> INERT   [no excitor, net <= 0 -> INERT]
#
# Test questions and derivations:
#
# Q1: gorv, felk               -> FLARE   [+2 +1 = +3, clamped to FLARE]
#     (novel: two excitors combined; never seen. Tests additive excitation.)
#
# Q2: gorv, pryn, shael        -> INERT   [+2, pryn*2=-4, net -2, floor 0 -> INERT]
#     (shael amplifies pryn, already past INERT; tests shael+full-inhibitor.)
#
# Q3: felk, pryn               -> INERT   [+1 -2 = -1, floor 0 -> INERT]
#     (novel: weak excitor + full inhibitor. Tests that pryn overpowers felk.)
#
# Q4: gorv, telk, dozn, shael  -> INERT   [+2, telk*2=-2, dozn=0, net 0 -> INERT]
#     (same as obs 13-14 + dozn neutral; tests compositional combination.)
#
# Q5: felk, telk, shael        -> INERT   [+1, telk*2=-2, net -1, floor 0 -> INERT]
#     (novel: shael doubles telk against weak excitor; would be DAMP without shael.)
#
# Q6: gorv, felk, telk         -> FLARE   [+2 +1 -1 = +2 -> FLARE]
#     (novel: two excitors + partial inhibitor. Requires adding all three contributions.)
#
# Q7: gorv, felk, pryn         -> DAMP    [+2 +1 -2 = +1 -> DAMP]
#     (CRITICAL: two excitors + full inhibitor = DAMP. This is the hardest question.)
#     (Model must recognize both excitors contribute and pryn only partially cancels the sum.)
#
# Q8: dozn, shael              -> INERT   [no excitor, 0+0 = 0 -> INERT]
#     (novel: two non-excitor, non-inhibitor signals. Straightforward.)

_INHIBITORY_EXPECTED = {
    "q_1": "FLARE",
    "q_2": "INERT",
    "q_3": "INERT",
    "q_4": "INERT",
    "q_5": "INERT",
    "q_6": "FLARE",
    "q_7": "DAMP",
    "q_8": "INERT",
}

@kbench.task(
    name="inhibitory_summation_assoc_learning",
    description="H-07: Infer six hidden signal roles (strong/weak excitor, partial/full inhibitor, inhibitory multiplier, neutral) from reactor logs; predict novel combinations.",
)
def inhibitory_summation_assoc_learning(llm) -> float:
    """Multi-cue inhibitory summation with hierarchical modulation; return fraction correct."""

    prompt = "\n".join([
        "A reactor's output depends on which signals are active. Study these logs to infer how signals interact.",
        "",
        "Reactor logs (each line = one trial):",
        "  Trial  1: signals active: gorv                    -> outcome: FLARE",
        "  Trial  2: signals active: gorv                    -> outcome: FLARE",
        "  Trial  3: signals active: felk                    -> outcome: DAMP",
        "  Trial  4: signals active: felk                    -> outcome: DAMP",
        "  Trial  5: signals active: gorv, dozn              -> outcome: FLARE",
        "  Trial  6: signals active: gorv, dozn              -> outcome: FLARE",
        "  Trial  7: signals active: gorv, telk              -> outcome: DAMP",
        "  Trial  8: signals active: gorv, telk              -> outcome: DAMP",
        "  Trial  9: signals active: gorv, pryn              -> outcome: INERT",
        "  Trial 10: signals active: gorv, pryn              -> outcome: INERT",
        "  Trial 11: signals active: felk, telk              -> outcome: INERT",
        "  Trial 12: signals active: felk, telk              -> outcome: INERT",
        "  Trial 13: signals active: gorv, telk, shael       -> outcome: INERT",
        "  Trial 14: signals active: gorv, telk, shael       -> outcome: INERT",
        "  Trial 15: signals active: gorv, shael             -> outcome: FLARE",
        "  Trial 16: signals active: felk, shael             -> outcome: DAMP",
        "  Trial 17: signals active: gorv, dozn, telk        -> outcome: DAMP",
        "  Trial 18: signals active: telk, dozn              -> outcome: INERT",
        "",
        "Predict the reactor outcome for each combination below.",
        "Answer FLARE, DAMP, or INERT for each question.",
        "",
        "  Q1: signals active: gorv, felk",
        "  Q2: signals active: gorv, pryn, shael",
        "  Q3: signals active: felk, pryn",
        "  Q4: signals active: gorv, telk, dozn, shael",
        "  Q5: signals active: felk, telk, shael",
        "  Q6: signals active: gorv, felk, telk",
        "  Q7: signals active: gorv, felk, pryn",
        "  Q8: signals active: dozn, shael",
        "",
    ])

    result = llm.prompt(prompt, schema=InhibitorySummationAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_INHIBITORY_EXPECTED)

    for key, expn in _INHIBITORY_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _INHIBITORY_EXPECTED}
    _log_trace("inhibitory_summation", _TASK_DESCRIPTION, prompt, answers, _INHIBITORY_EXPECTED, score)
    return score

if __name__ == "__main__":
    inhibitory_summation_assoc_learning.run(kbench.llm)

