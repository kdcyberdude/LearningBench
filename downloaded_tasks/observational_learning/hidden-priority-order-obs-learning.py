#!/usr/bin/env python
# coding: utf-8

import random
import re
from dataclasses import dataclass
from typing import List

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
    "The model observes word lists sorted by a hidden 3-key comparator: primary key is "
    "vowel count ascending, secondary key is last character descending (z-a), tertiary "
    "key is word length descending. Tiebreaker cases are essential: standard alphabetical "
    "or length sorting superficially agrees with many demos; only careful analysis of "
    "tie cases fully reveals all three keys."
)

_FIXED_SEED = 0

VOWELS = set("aeiouAEIOU")

_REAL_WORDS = [
    "apple",
    "mango",
    "brave",
    "crane",
    "drift",
    "shelf",
    "spoke",
    "glint",
    "frost",
    "blaze",
    "crisp",
    "plumb",
    "groan",
    "tryst",
    "kneel",
    "swamp",
    "flute",
    "prism",
    "bloom",
    "epoch",
    "ivory",
    "ulcer",
    "oaken",
    "abbey",
    "anvil",
    "berry",
    "cedar",
    "dogma",
    "ember",
    "fungi",
    "aster",
    "broil",
    "cleft",
    "depth",
    "exult",
    "flame",
    "gripe",
    "hoist",
    "inter",
    "joust",
    "knack",
    "lemon",
    "marsh",
    "notch",
    "optic",
    "prowl",
    "quest",
    "raven",
    "smelt",
    "thrum",
    "until",
    "vigor",
    "whirl",
    "xylem",
    "yearn",
    "zonal",
    "abbot",
    "acorn",
    "aglow",
    "aisle",
    "algae",
    "angel",
    "annex",
    "arbor",
]

_NONSENSE = [
    "zprxt",
    "vlfmb",
    "qrnds",
    "bztpk",
    "kwlvf",
    "xmdrc",
    "plvzt",
    "snkwb",
    "fqrmt",
    "djvlk",
    "trbzn",
    "gpxlc",
    "mrfvt",
    "wbzkp",
    "hxrnd",
    "czpvl",
    "brtng",
    "fxpwm",
    "klzrd",
    "vrntb",
    "splxc",
    "dwrtz",
    "fnjpb",
    "hrtxz",
    "mgrvl",
    "pbnzt",
    "qxlrm",
    "sznbd",
    "tvkpw",
    "xlnfz",
]


def _sort_key(word: str) -> tuple:
    vowels = sum(1 for c in word if c in VOWELS)
    last = word[-1].lower() if word else "z"
    # secondary: last char descending => higher 'last' letter = lower rank
    return (vowels, chr(ord("z") - ord(last)), -len(word))


def _expert_sort(items: list) -> list:
    return sorted(items, key=_sort_key)


def _gen_list(rng: random.Random, length: int, pool: list) -> list:
    items = rng.sample(pool, min(length, len(pool)))
    rng.shuffle(items)
    return items


def _str_match(expected: str, actual: str) -> bool:
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


def _make_test_cases():
    rng = random.Random(_FIXED_SEED + 77)
    cases = []
    for _ in range(4):
        lst = _gen_list(rng, 8, _REAL_WORDS)
        expected = _expert_sort(lst)
        cases.append((lst, expected))
    return cases


_TEST_CASES = _make_test_cases()


def _build_prompt(demos: list, test_lists: list) -> str:
    demo_lines = "\n".join(
        f"  Demo {i + 1}:\n    Unsorted: {inp}\n    Sorted:   {out}"
        for i, (inp, out) in enumerate(demos)
    )
    q_lines = "\n".join(f"  Q{i + 1}: {lst}" for i, (lst, _) in enumerate(test_lists))
    return (
        "You are observing an expert sort lists of words using a SPECIFIC ordering rule.\n\n"
        "Demonstrations:\n"
        f"{demo_lines}\n\n"
        "Now sort each of these 4 test lists using exactly the same rule:\n"
        f"{q_lines}\n\n"
        "Submit your answers as sorted_list_1, sorted_list_2, sorted_list_3, sorted_list_4."
    )


def _prepare():
    rng = random.Random(_FIXED_SEED)
    demos = []
    for _ in range(12):
        lst = _gen_list(rng, 8, _REAL_WORDS)
        demos.append((lst, _expert_sort(lst)))

    prompt = _build_prompt(demos, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (lst, expected) in enumerate(_TEST_CASES, 1):
            got = getattr(response, f"sorted_list_{i}", None)
            ok = False
            if isinstance(got, list) and len(got) == len(expected):
                ok = all(_str_match(str(e), str(a)) for e, a in zip(expected, got))
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    sorted_list_1: List[str]
    sorted_list_2: List[str]
    sorted_list_3: List[str]
    sorted_list_4: List[str]


@kbench.task(
    name="custom_comparator_sort_obs_learning",
    description=(
        "Observe lists sorted by a hidden 3-key comparator (vowel count asc, last char "
        "desc, length desc). Infer the full ordering rule from 12 demos and submit the "
        "correctly sorted version of 4 new lists."
    ),
)
def custom_comparator_sort_obs_learning(llm) -> float:
    """Infer a hidden 3-key sort comparator from 12 demos; sort 4 new lists correctly."""
    prompt, grade_fn = _prepare()
    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None
    score, test_results = (
        grade_fn(response)
        if response is not None
        else (
            0.0,
            [
                {
                    "q": i,
                    "expected": _TEST_CASES[i - 1][1],
                    "got": None,
                    "correct": False,
                }
                for i in range(1, 5)
            ],
        )
    )
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="custom_comparator_sort_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    custom_comparator_sort_obs_learning.run(kbench.llm)

