#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests composition of symbol-based meaning flips using paired * and @ modifiers. "
    "Each matching pair of identical surrounding symbols flips a word to its exact opposite; unmatched symbols are ignored; multiple pairs compose (two flips restore the original). "
    "The model must determine the final meaning of five symbol-wrapped words. "
    "Success requires correctly counting matched pairs per symbol type and composing an even/odd number of flips."
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
        match = "✓" if str(actual).lower() == str(exp).lower() else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


@dataclass
class ContextualFlipAnswer:
    word_1: str
    word_2: str
    word_3: str
    word_4: str
    word_5: str


_CONTEXTUAL_FLIP_EXPECTED = {
    "word_1": "hot",
    "word_2": "slow",
    "word_3": "good",
    "word_4": "dark",
    "word_5": "short",
}


@kbench.task(
    name="contextual_flip_assoc__learning",
    description="Compose symbol-based meaning flips: paired * and @ each invert a word's meaning; multiple pairs compose.",
)
def contextual_flip_assoc_learning(llm) -> float:
    """Apply composed contextual modifiers (* and @); return fraction correct."""

    prompt = "\n".join([
        "Words have their normal meaning, but can be modified by symbols: asterisks (*) and at-signs (@).",
        "Each MATCHING PAIR of identical symbols surrounding the word flips its meaning to its exact opposite.",
        "Unmatched symbols are ignored.",
        "Multiple matching pairs compose. For example, one matching pair flips the meaning. A second matching pair flips it back to the original.",
        "",
        "Provide the meaning (a single word) for each of these:",
        "  Word 1: *@hot@*",
        "  Word 2: *fast*",
        "  Word 3: @@good@@",
        "  Word 4: *bright*",
        "  Word 5: @tall@",
        "",
        'Reply with ONLY valid JSON: {"word_1": "meaning", "word_2": "meaning", "word_3": "meaning", "word_4": "meaning", "word_5": "meaning"}',
    ])

    result = llm.prompt(prompt, schema=ContextualFlipAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_CONTEXTUAL_FLIP_EXPECTED)
    for key, exp_val in _CONTEXTUAL_FLIP_EXPECTED.items():
        act = str(getattr(result, key)).strip().lower()
        expn = str(exp_val).strip().lower()
        if act == expn:
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _CONTEXTUAL_FLIP_EXPECTED}
    _log_trace("contextual_flip", _TASK_DESCRIPTION, prompt, answers, _CONTEXTUAL_FLIP_EXPECTED, score)
    return score


if __name__ == "__main__":
    contextual_flip_assoc_learning.run(kbench.llm)

