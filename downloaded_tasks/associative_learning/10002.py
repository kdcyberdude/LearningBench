#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests latent inhibition (pre-exposure / CS pre-exposure effect). "
    "Tone FAINT was presented 30 times without reinforcement before Phase B/C conditioning; "
    "Tone BRIGHT was novel and immediately reinforced in Phase B. "
    "The model must recognise that pre-exposure retards subsequent conditioning, making BRIGHT the stronger predictor after Phase B, "
    "while confirming that Phase C pairing eventually overcomes latent inhibition for FAINT."
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
        match = "✓" if str(actual).upper() == str(exp).upper() else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


@dataclass
class LatentInhibitionAnswer:
    q_1: str
    q_2: str


_LATENT_INHIBITION_EXPECTED = {
    "q_1": "BRIGHT",
    "q_2": "YES",
}


@kbench.task(
    name="latent_inhibition__assoc_learning",
    description="H-09: Non-reinforced pre-exposure to FAINT slows subsequent conditioning vs novel BRIGHT; pairing later overcomes it.",
)
def latent_inhibition__assoc_learning(llm) -> float:
    """Latent inhibition / pre-exposure effect; return fraction correct."""

    prompt = "\n".join([
        "Rodent lab: two tones used as cues.",
        "  Phase A (30 days): Tone FAINT plays at random times; food never follows FAINT.",
        "  Phase B (same 30 days): Tone BRIGHT is new; each BRIGHT is followed by food pellet.",
        "",
        "  Phase C (next 20 days): Every FAINT is followed by food. Every BRIGHT is still followed by food.",
        "",
        "Based ONLY on this:",
        "",
        "  Q1: After Phase B only, which tone is the stronger / faster-learned food predictor — FAINT or BRIGHT?",
        "  Q2: After Phase C, does FAINT reliably predict food? YES or NO",
        "",
        'Reply with ONLY valid JSON: {"q_1": "FAINT/BRIGHT", "q_2": "YES/NO"}',
    ])

    result = llm.prompt(prompt, schema=LatentInhibitionAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_LATENT_INHIBITION_EXPECTED)
    for key, expn in _LATENT_INHIBITION_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        if act == expn:
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _LATENT_INHIBITION_EXPECTED}
    _log_trace("latent_inhibition", _TASK_DESCRIPTION, prompt, answers, _LATENT_INHIBITION_EXPECTED, score)
    return score


if __name__ == "__main__":
    latent_inhibition__assoc_learning.run(kbench.llm)

