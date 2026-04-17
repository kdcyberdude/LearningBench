#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "SKLONVETH tests Semitic-style root-and-pattern morphology. "
    "Trilateral consonant roots (C1C2C3) are woven into CV-skeleton PATTERNS to derive words. "
    "8 patterns exist encoding different TAMV (tense/aspect/mood/voice) values. "
    "Additionally, NEGATION triggers PROSODIC INVERSION: the prefix 'dren-' is added AND "
    "the first vowel of the pattern is doubled (lengthened). "
    "Model must discover the 8 patterns, the root-weaving mechanism, and the negation rule. "
    "Exam includes wug-test items with novel roots. Score = accuracy * efficiency."
)


def _log_trace(task, turns, exam_results, final_score, examples_used,
               exam_prompt="", exam_raw=None):
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    for t in turns:
        print(f"\n[USER — Turn {t['turn']}]\n{t.get('prompt','')}")
        print(f"\n[ASSISTANT — Turn {t['turn']}]\naction: {t['action']}\nanswer: {t.get('response','(none)')}")
    if exam_prompt:
        print(f"\n[USER — Exam]\n{exam_prompt}")
    if exam_raw:
        print("\n[ASSISTANT — Exam]\n" + "\n".join(exam_raw))
    for r in exam_results:
        s = "CORRECT" if r["correct"] else "WRONG  "
        print(f"  Test {r['item']}: {s}  expected={r['expected']!r}  got={r['answer']!r}")
    ok = sum(1 for r in exam_results if r["correct"])
    print(f"\n  Examples: {examples_used}/{MAX_EXAMPLES}  Correct: {ok}/{len(exam_results)}  Score: {final_score:.4f}\n{sep}\n")


NUM_TEST_ITEMS = 4
FREE_THRESHOLD = 6

_PATTERNS = {
    "PRES.ACT":  ["C", "a", "C", "a", "C"],
    "PAST.ACT":  ["C", "a", "C", "i", "C"],
    "FUT.ACT":   ["C", "u", "C", "a", "C"],
    "PRES.PASS": ["C", "i", "C", "a", "C"],
    "PAST.PASS": ["C", "a", "C", "u", "C"],
    "HABIT.ACT": ["C", "a", "C", "C", "a", "C"],
    "INFIN":     ["C", "a", "C", "a", "C", "a", "n"],
    "PTCP":      ["m", "a", "C", "C", "u", "C"],
}


def _apply_pattern(root_consonants, pattern_key):
    template = _PATTERNS[pattern_key]
    c_iter = iter(root_consonants)
    last_consonant = root_consonants[-1]
    result = []
    for slot in template:
        if slot == "C":
            try:
                last_consonant = next(c_iter)
            except StopIteration:
                pass  # gemination: repeat the last consonant
            result.append(last_consonant)
        else:
            result.append(slot)
    return "".join(result)


def _negate(word, pattern_key):
    vowels = "aeiou"
    chars = list(word)
    for i, c in enumerate(chars):
        if c in vowels:
            chars.insert(i+1, c)
            break
    return "dren-" + "".join(chars)


_ROOTS = {
    "skr": ["s","k","r"],
    "drl": ["d","r","l"],
    "vnt": ["v","n","t"],
    "brf": ["b","r","f"],
    "plk": ["p","l","k"],
}

import itertools as _itools
import random as _random
_rng2 = _random.Random(77)
_all_combos2 = list(_itools.product(list(_ROOTS.keys()), list(_PATTERNS.keys()), [False, True]))
_rng2.shuffle(_all_combos2)
_selected2 = _all_combos2[:24]

_ALL_INPUTS2 = _selected2
_ALL_OUTPUTS2 = []
for root_key, pat_key, negated in _selected2:
    word = _apply_pattern(_ROOTS[root_key], pat_key)
    if negated:
        word = _negate(word, pat_key)
    _ALL_OUTPUTS2.append(word)


def _derive(root_key, pat_key, negated):
    word = _apply_pattern(_ROOTS[root_key], pat_key)
    return _negate(word, pat_key) if negated else word


MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 6

_TEST_COMBOS2 = [
    ("vnt",  "PAST.ACT",  False),
    ("skr",  "PRES.PASS", True),
    ("brf",  "HABIT.ACT", False),
    ("plk",  "PTCP",      True),
]
_TEST_EXPECTED2 = [_derive(r, p, n) for r, p, n in _TEST_COMBOS2]


def _fmt_spec2(root_key, pat_key, negated):
    return f"root={root_key}({','.join(_ROOTS[root_key])}), pattern={pat_key}, negated={'yes' if negated else 'no'}"


def _concept_score(correct_count, examples_used, max_examples, initial_examples):
    accuracy = correct_count / NUM_TEST_ITEMS
    if accuracy == 0:
        return 0.0
    eff_free = max(initial_examples, FREE_THRESHOLD)
    if max_examples <= eff_free or examples_used <= eff_free:
        efficiency = 1.0
    else:
        efficiency = max(0.0, 1.0 - (examples_used - eff_free) / (max_examples - eff_free))
    return accuracy * (0.40 + 0.60 * efficiency)


def _surface_equal(expected: str, actual: str) -> bool:
    def _norm(s: str) -> str:
        t = (s or "").strip()
        t = " ".join(t.split())
        return unicodedata.normalize("NFC", t)

    return _norm(expected) == _norm(actual)


@dataclass
class _ConceptAction:
    action: str
    answer: str


@dataclass
class _ExamAnswers:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str


@kbench.task(
    name="sklonveth_root_pattern_lang_learning",
    description="Learn SKLONVETH root-and-pattern morphology (8 CV templates + negation prosodic inversion) from examples; produce 4 surface forms. Score=accuracy*efficiency.",
)
def sklonveth_root_pattern_lang_learning(llm) -> float:
    """Infer SKLONVETH trilateral root-weaving into 8 CV patterns plus negation lengthening from examples; produce 4 novel forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying SKLONVETH morphology.",
        "SKLONVETH words are formed by weaving a 3-consonant ROOT into a vowel-consonant PATTERN.",
        "The pattern encodes tense/aspect/mood/voice. C slots are filled left-to-right from the root.",
        "",
        "Example (root skr=[s,k,r], pattern PRES.ACT=C-a-C-a-C):",
        "  s + a + k + a + r = 'sakar'",
        "",
        "NEGATION: add prefix 'dren-' AND double the first vowel of the resulting word.",
        "  neg('sakar') = 'dren-saakr' → actually: 'dren-s' + doubled-vowel + rest",
        "",
        "Labeled examples (root + pattern + negated → surface word):",
    ]
    for i in range(INITIAL_EXAMPLES):
        r, p, n = _ALL_INPUTS2[i]
        neg_str = "NEG" if n else "AFF"
        lines.append(f"  Example {i+1}: root={r}({','.join(_ROOTS[r])}), {p}, {neg_str} → {_ALL_OUTPUTS2[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover all 8 patterns and the negation rule.",
        "Scoring note: Getting each examination pattern realization right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in root-and-pattern morphology (including novel roots), then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("sklonveth_rp"):
        for turn in range(1, MAX_EXAMPLES + 2):
            cur = next_prompt
            try:
                sub = llm.prompt(cur, schema=_ConceptAction)
            except Exception:
                turns.append({"turn": turn, "action": "PARSE_ERROR", "prompt": cur})
                next_prompt = "Parse error. Use action='request' or action='submit'."
                continue
            action = (sub.action or "").strip().lower()
            entry = {"turn": turn, "action": action, "prompt": cur,
                     "response": (sub.answer or "").strip()}
            if action == "request":
                if examples_shown >= MAX_EXAMPLES:
                    turns.append(entry)
                    next_prompt = "No more examples. Use action='submit'."
                else:
                    idx = examples_shown
                    r, p, n = _ALL_INPUTS2[idx]
                    neg_str = "NEG" if n else "AFF"
                    ex = f"Example {idx+1}: root={r}({','.join(_ROOTS[r])}), {p}, {neg_str} → {_ALL_OUTPUTS2[idx]}"
                    examples_shown += 1
                    turns.append(entry)
                    next_prompt = (f"{ex}\n\nSeen {examples_shown}/{MAX_EXAMPLES}.\n"
                                   "action='request' or action='submit'.")
            elif action == "submit":
                turns.append(entry)
                break
            else:
                turns.append(entry)
                next_prompt = "Use action='request' or action='submit'."

        exam_lines = [
            "EXAMINATION — Produce the correct SKLONVETH surface form for each specification.",
            "Weave the root consonants into the pattern, then apply negation if required.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, combo in enumerate(_TEST_COMBOS2):
            exam_lines.append(f"  Item {i+1}: {_fmt_spec2(*combo)}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": _fmt_spec2(*_TEST_COMBOS2[i]),
                "expected": _TEST_EXPECTED2[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED2[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("SKLONVETH ROOT-AND-PATTERN", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    sklonveth_root_pattern_lang_learning.run(kbench.llm)

