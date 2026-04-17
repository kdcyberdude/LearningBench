#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "SKELTH tests multi-condition allomorphy triggered by BOTH phonological AND morphological environments. "
    "The same suffix has 5 allomorphs: "
    "  -ak after voiceless stops; -ek after voiced stops/fricatives; "
    "  -ok after nasals and liquids; -ik in the diminutive construction; -uk in the nominalization. "
    "Additionally, a SECOND suffix stack changes when the first suffix begins with a vowel (V-initial mutation). "
    "Model must identify BOTH conditioning environments and ALL 5 allomorphs. Score = accuracy * efficiency."
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

_VOICELESS_STOPS = set("ptkc")
_VOICED_STOPS_FRICS = set("bdgvzj")
_NASALS_LIQUIDS = set("mnlr")


def _agr_allomorph(root, construction):
    last_c = None
    for c in reversed(root):
        if c.isalpha() and c not in "aeiou":
            last_c = c
            break
    if construction == "DIM":
        return "-ik"
    if construction == "NOM":
        return "-uk"
    if last_c in _VOICELESS_STOPS:
        return "-ak"
    if last_c in _VOICED_STOPS_FRICS:
        return "-ek"
    if last_c in _NASALS_LIQUIDS:
        return "-ok"
    return "-ak"


def _second_suffix(first_suf, construction):
    base = {"DIM": "vren", "NOM": "stol", "POSS": "drak", "PL": "stan"}[construction]
    if first_suf[1:2] in "aeiou":
        return "n-" + base
    return base


def _skelth_form(root, construction):
    s1 = _agr_allomorph(root, construction)
    s2 = _second_suffix(s1, construction)
    return root + s1 + s2


_ROOTS_SK = [
    "skovel","dranta","brelko","volmak","trenten","kolfan",
    "preskam","drovnal","skelvan","fortlek",
]
_CONSTRUCTIONS_SK = ["DIM","NOM","POSS","PL"]

import itertools as _it6
_ALL_SPECS_SK = list(_it6.product(_ROOTS_SK, _CONSTRUCTIONS_SK))
import random as _rng7m
_rng7 = _rng7m.Random(19)
_rng7.shuffle(_ALL_SPECS_SK)
_ALL_OUTPUTS_SK = [_skelth_form(r,c) for r,c in _ALL_SPECS_SK]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 6

_TEST_SPECS_SK = [
    ("brelko",  "POSS"),
    ("trenten", "DIM"),
    ("skovel",  "NOM"),
    ("drovnal", "PL"),
]
_TEST_EXPECTED_SK = [_skelth_form(r,c) for r,c in _TEST_SPECS_SK]


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
    name="skelth_allomorph_lang_learning",
    description="Learn SKELTH 5-allomorph suffix system (phonological+morphological conditioning: -ak/-ek/-ok/-ik/-uk) plus V-initial second-suffix mutation; produce 4 forms. Score=accuracy*efficiency.",
)
def skelth_allomorph_lang_learning(llm) -> float:
    """Infer SKELTH 5-allomorph system with dual conditioning (phonological+morphological) and V-initial mutation from examples; produce 4 forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying SKELTH nominal morphology.",
        "SKELTH nouns take two stacked suffixes for different constructions.",
        "FIRST SUFFIX — 5 allomorphs, determined by BOTH phonological and morphological environment:",
        "  -ak  : after voiceless stops (p/t/k/c)",
        "  -ek  : after voiced stops or fricatives (b/d/g/v/z/j)",
        "  -ok  : after nasals and liquids (m/n/l/r)",
        "  -ik  : in DIMINUTIVE construction (regardless of phonology)",
        "  -uk  : in NOMINALIZATION construction (regardless of phonology)",
        "SECOND SUFFIX — base form depends on construction; BUT if the first suffix begins with a vowel,",
        "  the second suffix is prefixed by 'n-' (V-initial mutation: vowel triggers extra 'n' onset)",
        "  Constructions: DIM=vren, NOM=stol, POSS=drak, PL=stan",
        "",
        "Labeled examples (root + construction → form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        r,c = _ALL_SPECS_SK[i]
        lines.append(f"  Example {i+1}: root='{r}', construction={c} → {_ALL_OUTPUTS_SK[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover both conditioning environments.",
        "Scoring note: Getting each examination surface form right (allomorphy) matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in allomorph selection, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("skelth_allomorph"):
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
                    r,c = _ALL_SPECS_SK[idx]
                    ex = f"Example {idx+1}: root='{r}', construction={c} → {_ALL_OUTPUTS_SK[idx]}"
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
            "EXAMINATION — Produce the correct SKELTH nominal form for each specification.",
            "Apply the correct allomorph, then the second suffix (with V-initial mutation if needed).",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (r,c) in enumerate(_TEST_SPECS_SK):
            exam_lines.append(f"  Item {i+1}: root='{r}', construction={c}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_SPECS_SK[i][0]}.{_TEST_SPECS_SK[i][1]}",
                "expected": _TEST_EXPECTED_SK[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_SK[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("SKELTH MULTI-CONDITION ALLOMORPHY", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    skelth_allomorph_lang_learning.run(kbench.llm)

