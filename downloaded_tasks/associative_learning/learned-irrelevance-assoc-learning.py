#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests inference of the learned-irrelevance (uncorrelated pre-exposure) effect "
    "from raw numerical conditioning-rate data across four experimental groups. "
    "The model is given session-by-session conditioned-response (CR) rates for "
    "Groups A–D that differ in Phase-1 pre-exposure regime. It must infer which "
    "group acquired the fastest, which the slowest, identify the latent ordering, "
    "diagnose which group is uninformative for isolating the learned-irrelevance "
    "effect (a counter-conditioning confound group), and project CR rates for a "
    "novel 'Group E' described only by its pre-exposure statistics. "
    "Answers are not recoverable from training data: they require inference from "
    "the provided numbers. "
    "Critical difficulty: Groups B and C differ only in pre-exposure correlation "
    "sign (+0.8 vs −0.8); both are faster than Group A (r=0) but slower than "
    "Group D (novel); the model must avoid conflating 'any pre-exposure' with "
    "'uncorrelated pre-exposure' by reasoning about directionality of correlation."
)


def _log_trace(
    task: str,
    description: str,
    prompt: str,
    answers: dict,
    expected: dict,
    score: float,
) -> None:
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


# ---------------------------------------------------------------------------
# Hidden experimental structure (never shown in prompt)
#
# DESIGN:  Tone = CS.   Food = US.   Session unit = 16-trial block.
#          CR rate = fraction of Tone presentations that elicit anticipatory CR.
#          Phase 1: 40 sessions of pre-exposure treatment (varies by group).
#          Phase 2: 10 sessions of Tone->Food pairings (same for all groups).
#
# Group A  Phase-1 regime: Tone and Food presented in alternating, uncorrelated
#          schedule (r = 0.00 across 40 sessions, equal marginal frequencies).
#          Effect: learned irrelevance — Tone marked as an unreliable signal.
#          Phase-2 CR trajectory: flat for sessions 1–4, gradual rise sessions 5–10.
#          Final (session 10) CR rate: 0.41
#
# Group B  Phase-1 regime: Tone and Food positively correlated (r = +0.80) but
#          pairing density is only 30 % of Phase-2 density; weak partial conditioning.
#          Effect: partial prior association facilitates — moderate head start.
#          Phase-2 CR trajectory: fast rise sessions 1–3, plateau sessions 6–10.
#          Final CR rate: 0.82
#
# Group C  Phase-1 regime: Tone and Food negatively correlated (r = −0.80),
#          i.e., Tone predicts ABSENCE of food in Phase 1.
#          Effect: inhibitory conditioning — Tone acquires inhibitory strength,
#          which must be extinguished before excitatory Phase-2 learning begins.
#          Phase-2 CR trajectory: suppressed sessions 1–3, delayed rise sessions 5–10.
#          Final CR rate: 0.37
#
# Group D  Phase-1 regime: Tone ABSENT entirely (no pre-exposure).
#          Effect: pure novelty; no prior history — fastest acquisition baseline.
#          Phase-2 CR trajectory: rapid rise sessions 1–5, high plateau 6–10.
#          Final CR rate: 0.88
#
# Group E (query, never shown data): "Phase-1 schedule had Tone and Food events
#          each occurring on 50 % of trials, arranged so Tone neither predicted
#          nor counter-predicted Food (r = 0.00), but the inter-trial-interval
#          separating Tone-alone and Food-alone trials was 10× longer than in
#          Group A's schedule."
#          Effect: same correlation direction as A (r=0) but longer ITI reduces
#          associability-change relative to A.  CR rate expected BETWEEN A and D
#          (closer to D than A because interference is weaker).
#          Correct answer: BETWEEN_A_AND_D
#
# ANSWER KEY DERIVATIONS:
#
#  Q1: Which group shows the clearest learned-irrelevance effect?
#      Learned irrelevance = slowest acquisition in Phase 2 despite receiving
#      equal Phase-2 pairings, caused specifically by uncorrelated pre-exposure.
#      Group A (r=0) or Group C (r=−0.80)?
#      Group C is slower (0.37 vs 0.41) BUT its mechanism is inhibitory
#      conditioning, not learned irrelevance.  The PUREST learned-irrelevance
#      effect is Group A: uncorrelated, same marginal frequencies, no inhibition.
#      Answer: A
#
#  Q2: Which group is CONFOUNDED for isolating learned irrelevance?
#      Group C: its slow acquisition could be attributed to conditioned inhibition
#      (a different mechanism) rather than to learned irrelevance. Its Phase-1
#      regime was r=−0.80, meaning the Tone WAS correlated — just negatively.
#      This confounds the isolation of 'zero-correlation' as the causal factor.
#      Answer: C
#
#  Q3: Correct acquisition speed order (fastest to slowest) for Phase 2?
#      D (no pre-exposure, no history) > B (partial positive conditioning) >
#      A (uncorrelated pre-exposure) > C (inhibitory pre-exposure).
#      Numerical: D=0.88, B=0.82, A=0.41, C=0.37.
#      Answer: D_B_A_C
#
#  Q4: Between Group A and Group C, which effect is harder to reverse in Phase 2?
#      Group C requires first extinguishing inhibitory strength before excitatory
#      associations form — a two-stage process that is slower to reverse than
#      the associability decrement in Group A.
#      Answer: C
#
#  Q5: If a Group F were run with Tone pre-exposed but at r = 0, using 4× MORE
#      pre-exposure sessions than Group A, would its Phase-2 final CR be
#      expected to be HIGHER, LOWER, or SAME as Group A?
#      More uncorrelated pre-exposure → deeper associability reduction →
#      slower subsequent conditioning → LOWER final Phase-2 CR.
#      Answer: LOWER
#
#  Q6: Does Group B's faster acquisition prove that any prior Tone experience
#      facilitates Phase-2 conditioning? YES or NO?
#      NO — Group A had prior Tone experience but showed impaired acquisition.
#      Therefore prior exposure alone does not guarantee facilitation; the
#      correlation sign during pre-exposure determines direction of effect.
#      Answer: NO
#
#  Q7: For the novel Group E (r=0, 10× longer ITI than Group A), where would
#      its final Phase-2 CR rate fall relative to the observed groups?
#      Answer: BETWEEN_A_AND_D
#
#  Q8: Which mechanistic property distinguishes Group A's impairment from
#      Group C's impairment — REDUCED_ASSOCIABILITY or INHIBITORY_STRENGTH?
#      Group A = REDUCED_ASSOCIABILITY (Pearce-Hall / Mackintosh model: cue
#      loses alpha because it predicts nothing).
#      Group C = INHIBITORY_STRENGTH (Rescorla-Wagner: cue acquires negative V).
#      The question asks what drives Group A specifically.
#      Answer: REDUCED_ASSOCIABILITY
#
# ---------------------------------------------------------------------------


@dataclass
class LearnedIrrelevanceAnswer:
    q_1: str
    q_2: str
    q_3: str
    q_4: str
    q_5: str
    q_6: str
    q_7: str
    q_8: str


_LEARNED_IRRELEVANCE_EXPECTED = {
    "q_1": "A",
    "q_2": "C",
    "q_3": "D_B_A_C",
    "q_4": "C",
    "q_5": "LOWER",
    "q_6": "NO",
    "q_7": "BETWEEN_A_AND_D",
    "q_8": "REDUCED_ASSOCIABILITY",
}


@kbench.task(
    name="learned_irrelevance_assoc_learning",
    description=(
        "H-11: Infer learned-irrelevance effect and its mechanistic distinction from "
        "conditioned inhibition by reading four groups' Phase-2 CR-rate trajectories; "
        "answer eight inference questions including a held-out Group E projection."
    ),
)
def learned_irrelevance_assoc_learning(llm) -> float:
    """Learned irrelevance vs conditioned inhibition; infer from CR data; return fraction correct."""

    prompt = "\n".join([
        "A conditioning laboratory ran four groups of subjects (A, B, C, D) through",
        "two phases. In Phase 1 each group received a different pre-exposure schedule",
        "involving a Tone and occasional Food deliveries. In Phase 2 every group",
        "received identical training: 10 sessions in which every Tone was immediately",
        "followed by Food (same number of pairings per session for all groups).",
        "",
        "Phase-1 pre-exposure regime for each group:",
        "  Group A:  Tone and Food each appeared on 50 % of trials, INDEPENDENTLY",
        "            arranged so that knowing the Tone appeared gave zero information",
        "            about whether Food would appear on that trial (r = 0.00).",
        "  Group B:  Tone and Food appeared together on 80 % of trials in which",
        "            either occurred (r = +0.80). Pairing density was 30 % of what",
        "            Phase 2 would use.",
        "  Group C:  On trials when the Tone appeared, Food was withheld on 80 %",
        "            of those trials relative to non-Tone trials (r = −0.80).",
        "  Group D:  Tone was NEVER presented during Phase 1 (no pre-exposure).",
        "",
        "Phase-2 conditioned-response (CR) rates per session (fraction of Tone",
        "presentations eliciting an anticipatory CR):",
        "",
        "         Sess 1  Sess 2  Sess 3  Sess 4  Sess 5  Sess 6  Sess 7  Sess 8  Sess 9  Sess 10",
        "  Grp A:  0.04    0.06    0.07    0.09    0.14    0.21    0.28    0.35    0.38    0.41",
        "  Grp B:  0.31    0.48    0.62    0.69    0.73    0.77    0.79    0.81    0.82    0.82",
        "  Grp C:  0.02    0.02    0.03    0.06    0.11    0.19    0.27    0.32    0.35    0.37",
        "  Grp D:  0.22    0.41    0.57    0.68    0.75    0.81    0.84    0.86    0.87    0.88",
        "",
        "Answer the following questions using ONLY the information above.",
        "Use the exact token(s) shown in brackets for your answer.",
        "",
        "  Q1: Which single group most cleanly demonstrates the learned-irrelevance",
        "      effect (slowest Phase-2 acquisition attributable specifically to",
        "      UNCORRELATED pre-exposure, not to any prior directional association)?",
        "      [A / B / C / D]",
        "",
        "  Q2: Which group introduces a CONFOUND that prevents it from isolating",
        "      the learned-irrelevance mechanism, because its slow Phase-2 acquisition",
        "      could instead be explained by a DIFFERENT established learning effect?",
        "      [A / B / C / D]",
        "",
        "  Q3: State the correct order of Phase-2 acquisition speed from FASTEST to",
        "      SLOWEST across the four groups, using underscore-separated letters.",
        "      [D_B_A_C  or some other ordering, e.g. D_A_B_C]",
        "",
        "  Q4: Between Group A and Group C, which group's Phase-2 impairment is",
        "      harder to reverse over the course of Phase 2, based on the data?",
        "      [A / C]",
        "",
        "  Q5: Suppose Group F is identical to Group A (r = 0.00, same marginal",
        "      frequencies) but undergoes 4× as many Phase-1 sessions. Relative to",
        "      Group A, would Group F's final Phase-2 (session-10) CR be",
        "      HIGHER, LOWER, or the SAME?",
        "      [HIGHER / LOWER / SAME]",
        "",
        "  Q6: Does Group B's faster Phase-2 acquisition demonstrate that ANY prior",
        "      Tone experience (regardless of correlation) facilitates Phase-2",
        "      conditioning?  YES or NO.",
        "      [YES / NO]",
        "",
        "  Q7: A new Group E is described: Phase-1 schedule had Tone and Food each",
        "      on 50 % of trials with r = 0.00 (same as Group A), but the",
        "      inter-trial interval separating Tone-alone from Food-alone trials",
        "      was 10 times longer than in Group A's schedule. Where would Group E's",
        "      final Phase-2 CR rate most likely fall?",
        "      [BELOW_C / BETWEEN_C_AND_A / BETWEEN_A_AND_D / ABOVE_D]",
        "",
        "  Q8: The impairment seen in Group A is best explained by which mechanistic",
        "      property — the cue losing its ability to form new associations",
        "      (REDUCED_ASSOCIABILITY) or the cue acquiring an active suppressive",
        "      learned value (INHIBITORY_STRENGTH)?",
        "      [REDUCED_ASSOCIABILITY / INHIBITORY_STRENGTH]",
        "",
    ])

    result = llm.prompt(prompt, schema=LearnedIrrelevanceAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_LEARNED_IRRELEVANCE_EXPECTED)
    for key, expn in _LEARNED_IRRELEVANCE_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _LEARNED_IRRELEVANCE_EXPECTED}
    _log_trace(
        "learned_irrelevance",
        _TASK_DESCRIPTION,
        prompt,
        answers,
        _LEARNED_IRRELEVANCE_EXPECTED,
        score,
    )
    return score


if __name__ == "__main__":
    learned_irrelevance_assoc_learning.run(kbench.llm)

