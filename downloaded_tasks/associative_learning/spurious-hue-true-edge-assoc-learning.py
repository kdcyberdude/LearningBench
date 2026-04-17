#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests resistance to spurious correlation combined with configural (XOR) rule induction. "
    "Twelve training examples are given over three features: tint (veld/oske), facet (riven/zolth/morde/quell), "
    "and grain (pellate/frice/skaff/drent). "
    "Tint=veld is a 75%-reliable but broken correlate of GROUP_X; tint=oske is 75%-reliable for GROUP_Y. "
    "The true rule is an exclusive-OR over two binary partitions: "
    "  GROUP_X iff exactly one of (facet ∈ {riven,zolth}) or (grain ∈ {pellate,frice}) holds. "
    "Three examples deliberately show tint disagreeing with the label, breaking tint's apparent reliability. "
    "All four XOR quadrants are represented in training. "
    "Four test items are provided; every test item has tint pointing to the wrong group, "
    "so success requires fully discarding the spurious correlate and applying the XOR rule. "
    "No scaffolding is provided. All feature tokens are semantically neutral invented labels "
    "to eliminate leakage from semantic priors."
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
class SpuriousHueAnswer:
    item_1: str
    item_2: str
    item_3: str
    item_4: str

_SPURIOUS_HUE_EXPECTED = {
    "item_1": "GROUP_Y",
    "item_2": "GROUP_X",
    "item_3": "GROUP_Y",
    "item_4": "GROUP_X",
}

@kbench.task(
    name="spurious_hue_true_edge_assoc_learning",
    description=(
        "Spurious tint (75% reliable) vs true XOR rule over facet × grain partitions. "
        "All test items have tint pointing to the wrong group. Tests spurious-correlation "
        "resistance and configural rule induction simultaneously."
    ),
)
def spurious_hue_true_edge_assoc_learning(llm) -> float:
    """Spurious-tint vs XOR structural rule classification; return fraction correct."""

    # True rule: GROUP_X iff (facet in {riven,zolth}) XOR (grain in {pellate,frice}).
    # Spurious correlate: tint=veld predicts GROUP_X with 75% accuracy (9/12 examples agree),
    # tint=oske predicts GROUP_Y with 75% accuracy. Three examples break this pattern.
    # All four test items have tint disagreeing with the true label.
    prompt = "\n".join([
        "Items are classified into GROUP_X or GROUP_Y.",
        "",
        "Labeled examples:",
        "   1. tint=veld, facet=riven,  grain=skaff   -> GROUP_X",
        "   2. tint=veld, facet=zolth,  grain=drent   -> GROUP_X",
        "   3. tint=veld, facet=morde,  grain=frice   -> GROUP_X",
        "   4. tint=veld, facet=quell,  grain=pellate -> GROUP_X",
        "   5. tint=veld, facet=riven,  grain=pellate -> GROUP_Y",
        "   6. tint=veld, facet=morde,  grain=skaff   -> GROUP_Y",
        "   7. tint=oske, facet=morde,  grain=skaff   -> GROUP_Y",
        "   8. tint=oske, facet=quell,  grain=drent   -> GROUP_Y",
        "   9. tint=oske, facet=zolth,  grain=pellate -> GROUP_Y",
        "  10. tint=oske, facet=riven,  grain=frice   -> GROUP_Y",
        "  11. tint=oske, facet=morde,  grain=drent   -> GROUP_Y",
        "  12. tint=oske, facet=quell,  grain=frice   -> GROUP_X",
        "",
        "Classify these new items:",
        "  Item 1: tint=veld, facet=riven,  grain=frice",
        "  Item 2: tint=oske, facet=zolth,  grain=skaff",
        "  Item 3: tint=veld, facet=quell,  grain=drent",
        "  Item 4: tint=oske, facet=morde,  grain=pellate",
        "",
    ])

    result = llm.prompt(prompt, schema=SpuriousHueAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_SPURIOUS_HUE_EXPECTED)
    for key, exp_val in _SPURIOUS_HUE_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        expn = str(exp_val).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _SPURIOUS_HUE_EXPECTED}
    _log_trace("spurious_hue_true_edge", _TASK_DESCRIPTION, prompt, answers, _SPURIOUS_HUE_EXPECTED, score)
    return score

if __name__ == "__main__":
    spurious_hue_true_edge_assoc_learning.run(kbench.llm)

