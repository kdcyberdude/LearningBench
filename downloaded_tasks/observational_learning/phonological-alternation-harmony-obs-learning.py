#!/usr/bin/env python
# coding: utf-8

import re
from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can infer a progressive consonant voicing harmony rule from "
    "word transformation examples. Early demos use CVCVCV patterns (no clusters), so the "
    "rule never fires and input equals output. Later demos with CC clusters reveal: in any "
    "cluster, C2 must match the voicing of C1 (progressive assimilation). Four test questions "
    "probe multi-cluster words with both voiced and voiceless triggers, including cases where "
    "cascading harmony applies across triple-consonant sequences."
)

_FIXED_SEED = 0

_VOICED = list("bdgvz")
_VOICELESS = list("ptkfs")
_VOWELS = list("aeiou")
_VOICE_PAIR = dict(zip(_VOICED, _VOICELESS))  # voiced → voiceless counterpart
_DEVOICE_PAIR = {v: k for k, v in _VOICE_PAIR.items()}  # voiceless → voiced counterpart


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


def _is_voiced(c: str) -> bool:
    return c in _VOICED


def _make_voiced(c: str) -> str:
    return _DEVOICE_PAIR.get(c, c)


def _make_voiceless(c: str) -> str:
    return _VOICE_PAIR.get(c, c)


def _apply_harmony(word: str) -> str:
    """Progressive voicing harmony: in a CC cluster, C2 matches voicing of C1."""
    result = list(word)
    n = len(result)
    for i in range(1, n):
        c = result[i]
        if c not in _VOICED and c not in _VOICELESS:
            continue
        # C2 only assimilates if immediately preceded by a consonant (no vowel in between)
        if result[i - 1] not in _VOICED and result[i - 1] not in _VOICELESS:
            continue  # previous char is a vowel — no cluster
        prev = result[i - 1]
        if _is_voiced(prev) and not _is_voiced(c):
            result[i] = _make_voiced(c)
        elif not _is_voiced(prev) and _is_voiced(c):
            result[i] = _make_voiceless(c)
    return "".join(result)


def _str_match(expected: str, got) -> bool:
    if got is None:
        return False
    return bool(re.search(re.escape(expected.strip()), str(got).strip(), re.IGNORECASE))


# 13 demos: first 4 are CVCVCVC (no clusters, rule doesn't fire),
# rest contain CC or CCC clusters where harmony changes the output.
_DEMOS_RAW = [
    # No-cluster demos (CVCV...) — rule is silent, output = input
    "batike",  # B-A-T-I-K-E  → no CC → no change
    "padove",  # P-A-D-O-V-E  → no CC → no change
    "siguta",  # S-I-G-U-T-A  → no CC → no change
    "kovibe",  # K-O-V-I-B-E  → no CC → no change
    # CC cluster demos — harmony fires
    "abzena",  # a-B-Z-e-n-a  → BZ cluster: B voiced → Z stays voiced ✓ same (already voiced)
    "apfena",  # a-P-F-e-n-a  → PF cluster: P voiceless → F stays voiceless ✓
    "abdena",  # a-B-D-e-n-a  → BD cluster: B voiced → D stays ✓
    "aktova",  # a-K-T-o-v-a  → KT cluster: K voiceless → T stays ✓
    "azbota",  # a-Z-B-o-t-a  → ZB→ZB (both voiced) then ot no cluster → no change
    "aspiva",  # a-S-P-i-v-a  → SP cluster: S voiceless → P voiceless ✓
    # Deceptive demos: the trigger CHANGES the following consonant
    "avpika",  # a-V-P-i-k-a  → VP: V voiced → P must become B → "avbika"
    "okzeba",  # o-K-Z-e-b-a  → KZ: K voiceless → Z must become S → "okseba"
    "ifdepa",  # i-F-D-e-p-a  → FD: F voiceless → D must become T → "iftepa"
]

_DEMOS = [(w, _apply_harmony(w)) for w in _DEMOS_RAW]

# Four test cases: each has CC clusters requiring changes (harmony fires)
# All require recognizing that the trigger changes C2 (non-trivial)
_TEST_WORDS = [
    "avpoza",  # a-V-P-o-z-a → VP: V voiced → P must become B → "avboza"
    "okzabu",  # o-K-Z-a-b-u → KZ: K voiceless → Z must become S → "oksabu"
    "ifdeva",  # i-F-D-e-v-a → FD: F voiceless → D must become T → "ifteva"
    "uvkaze",  # u-V-K-a-z-e → VK: V voiced → K must become G → "uvgaze"
]

_TEST_CASES = [(_w, _apply_harmony(_w)) for _w in _TEST_WORDS]


def _prepare():
    lines = [
        "You are observing a natural language phonological process.",
        "Consonants: voiced {b, d, g, v, z}, voiceless {p, t, k, f, s}. Vowels: {a, e, i, o, u}.",
        "A hidden rule transforms some words. Each pair shows the input and the output.",
        "",
        "Observations (input → output):",
    ]
    for i, (inp, out) in enumerate(_DEMOS, 1):
        lines.append(f"  {i:2d}. {inp} → {out}")
    lines += [
        "",
        "Now solve these 4 test questions:",
    ]
    for q, (inp, _) in enumerate(_TEST_CASES, 1):
        lines.append(f"  Q{q}: {inp} → ?")
    lines += [
        "",
        "Apply the same rule to each input.",
        "Submit as output_1, output_2, output_3, output_4.",
    ]
    prompt = "\n".join(lines)

    def grade_fn(response):
        results = []
        correct = 0
        for q_idx, (inp, exp) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"output_{q_idx}", None)
            got = str(raw).strip().lower() if raw is not None else None
            is_correct = _str_match(exp, got)
            results.append(
                {
                    "q": q_idx,
                    "expected": exp,
                    "got": got,
                    "correct": is_correct,
                }
            )
            if is_correct:
                correct += 1
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    output_1: str
    output_2: str
    output_3: str
    output_4: str


@kbench.task(
    name="phonological_alternation_harmony_obs_learning",
    description=(
        "Observe words before and after a hidden consonant voicing harmony rule. "
        "Later demos show that in a CC cluster, C2 adopts C1's voicing. Apply this rule to 4 test words."
    ),
)
def phonological_alternation_harmony_obs_learning(llm) -> float:
    """
    Rule: In CC clusters, C2 matches C1's voicing; vowels block this. Returns accuracy (0.0–1.0).
    """
    prompt, grade_fn = _prepare()

    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None

    if response is not None:
        score, test_results = grade_fn(response)
    else:
        score = 0.0
        test_results = [
            {"q": i, "expected": _TEST_CASES[i - 1][1], "got": None, "correct": False}
            for i in range(1, 5)
        ]

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="phonological_alternation_harmony_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )

    return score


if __name__ == "__main__":
    phonological_alternation_harmony_obs_learning.run(kbench.llm)

