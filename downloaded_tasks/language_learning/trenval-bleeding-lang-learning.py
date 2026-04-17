#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "TRENVAL tests all 4 rule ordering relationships: feeding, bleeding, counterfeeding, counterbleeding. "
    "Four phonological rules interact in pairs: "
    "  R_FEED: A→B / _C (creates context for R2); R_BLEED: D→∅ / VC_ (destroys context for R3). "
    "  R_CTRFEED: E→F applied BEFORE G→H would apply (G→H doesn't see F's output = counterfeeding). "
    "  R_CTRBLEED: I→J applies AFTER K→∅ would apply (K deletion doesn't happen because J is opaque). "
    "Model must identify which rule pair applies in each word and what the ordering relationship is. "
    "Score = accuracy * efficiency."
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
FREE_THRESHOLD = 5

_VOWELS_TR = set("aeiou")


def _is_v(c):
    return c in _VOWELS_TR


def _is_c(c):
    return c.isalpha() and not _is_v(c)


def _apply_feeding(word):
    chars = list(word)
    for i in range(len(chars)-1):
        if chars[i] == "n" and _is_v(chars[i+1]):
            chars[i] = "m"
    for i in range(len(chars)-1):
        if chars[i] == "m" and chars[i+1] == "k":
            chars.insert(i+1, "p")
            break
    return "".join(chars)


def _apply_bleeding(word):
    chars = list(word)
    i = 0
    while i < len(chars)-2:
        if _is_v(chars[i]) and _is_c(chars[i+1]) and chars[i+1] == chars[i+2]:
            del chars[i+2]
        else:
            i += 1
    return "".join(chars)


def _apply_counterfeeding(word):
    chars = list(word)
    for i in range(len(chars)):
        if chars[i] == "e" and (i == 0 or not _is_v(chars[i-1])):
            chars[i] = "i"
    return "".join(chars)


def _apply_counterbleeding(word):
    chars = list(word)
    for i in range(len(chars)-1):
        if chars[i] == "s" and chars[i+1] == "t":
            chars[i] = "z"
    return "".join(chars)


_TR_RULE_MAP = {
    "feed": _apply_feeding,
    "bleed": _apply_bleeding,
    "ctrfeed": _apply_counterfeeding,
    "ctrbleed": _apply_counterbleeding,
}

_TR_WORDS = {
    "feed":     ["ankov","enkalt","onvak","inkor","ankaven","onventu"],
    "bleed":    ["asskol","ellprak","ottreln","akkoven","usskel","onntrak"],
    "ctrfeed":  ["eskrol","eblank","enkov","etran","elvak","efrant"],
    "ctrbleed": ["strelko","stovan","skront","stevan","strelk","stonvar"],
}

import itertools as _it_tr
_ALL_TR_SPECS = []
for rtype, words in _TR_WORDS.items():
    for w in words:
        _ALL_TR_SPECS.append((w, rtype))

import random as _rng_tr
_rng_tr_obj = _rng_tr.Random(57)
_rng_tr_obj.shuffle(_ALL_TR_SPECS)

_ALL_TR_OUTPUTS = [(_TR_RULE_MAP[rtype](w), rtype) for w, rtype in _ALL_TR_SPECS]
_ALL_TR_FORMS = [o[0] for o in _ALL_TR_OUTPUTS]
_ALL_TR_RTYPES = [o[1] for o in _ALL_TR_OUTPUTS]

MAX_EXAMPLES = 18
INITIAL_EXAMPLES = 5

_TEST_TR_SPECS = [
    ("ankov",   "feed"),
    ("asskol",  "bleed"),
    ("eskrol",  "ctrfeed"),
    ("strelko", "ctrbleed"),
]
_TEST_TR_EXPECTED = [_TR_RULE_MAP[rtype](w) for w, rtype in _TEST_TR_SPECS]


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
    name="trenval_bleeding_lang_learning",
    description="Learn TRENVAL 4 rule-ordering types (feeding/bleeding/counterfeeding/counterbleeding) from form examples; produce correct surface forms for 4 test words. Score=accuracy*efficiency.",
)
def trenval_bleeding_lang_learning(llm) -> float:
    """Infer TRENVAL rule-ordering relationships (all 4 types) from before/after pairs; produce 4 surface forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying TRENVAL phonological rule interactions.",
        "TRENVAL has 4 rule sets that interact in different ways:",
        "  FEEDING (feed): Rule A creates the environment for Rule B to apply.",
        "    (n→m before vowel FEEDS mp-insertion after 'm_k')",
        "  BLEEDING (bleed): Rule C destroys the environment where Rule D would apply.",
        "    (geminate deletion after vowel BLEEDS environment for other rules)",
        "  COUNTERFEEDING (ctrfeed): Rule E applies first, but E's output would have triggered Rule F.",
        "    However F does NOT apply to E's output (F is ordered after E but is opaque).",
        "    (e→i word-initially, but following consonant rule doesn't see 'i')",
        "  COUNTERBLEEDING (ctrbleed): Rule G applies, but G's output restores an environment.",
        "    Rule H applies to the output even though the original feeding would have been deleted.",
        "    (s→z before t, even though st-cluster was the trigger for H)",
        "",
        "Each underlying word undergoes exactly ONE rule set. Identify which rule set from the surface form.",
        "",
        "Labeled examples (underlying → surface, rule-ordering-type):",
    ]
    for i in range(INITIAL_EXAMPLES):
        w, rtype = _ALL_TR_SPECS[i]
        lines.append(f"  Example {i+1}: '{w}' → '{_ALL_TR_FORMS[i]}' ({rtype})")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover which rule applies to each word type.",
        "Scoring note: Getting each examination surface form right (rule-order effects) matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in how rule order affects surface forms, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("trenval_bleeding"):
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
                    w, rtype = _ALL_TR_SPECS[idx]
                    ex = f"Example {idx+1}: '{w}' → '{_ALL_TR_FORMS[idx]}' ({rtype})"
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
            "EXAMINATION — Produce the SURFACE FORM for each underlying form.",
            "You are told which rule-ordering type to apply. Apply the correct rule.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (w, rtype) in enumerate(_TEST_TR_SPECS):
            exam_lines.append(f"  Item {i+1}: underlying='{w}', rule-type={rtype}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_TR_SPECS[i][0]}({_TEST_TR_SPECS[i][1]})",
                "expected": _TEST_TR_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_TR_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("TRENVAL RULE ORDERING (ALL 4 TYPES)", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    trenval_bleeding_lang_learning.run(kbench.llm)

