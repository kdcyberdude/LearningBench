#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "GRELKAN tests context-sensitive verbal suppletion. "
    "Three verb classes show completely different stems across paradigm cells; class is not phonologically predictable. "
    "Class A: IMPF vs PERF stem alternation. "
    "Class B: 1st-person IMPF stem vs 2nd/3rd-person IMPF stem vs unified PERF stem. "
    "Class C: 5-way stem split (person × aspect), with dual using rare 4th stems. "
    "Model must infer each verb's class from partial paradigm evidence, then fill unseen cells. "
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
FREE_THRESHOLD = 6

_AGR = {
    ("1","sg"):"am", ("2","sg"):"ot", ("3","sg"):"es",
    ("1","du"):"avan",("2","du"):"otel",("3","du"):"esan",
    ("1","pl"):"ami", ("2","pl"):"oti", ("3","pl"):"esi",
}
_ASP = {"IMPF":"rek","PERF":"val"}

_VERBS = {
    "skrel": ("A", {"IMPF":"skrel","PERF":"fovan"}),
    "drath": ("A", {"IMPF":"drath","PERF":"klenva"}),
    "pelvo": ("A", {"IMPF":"pelvo","PERF":"thraski"}),
    "brentor": ("B", {
        ("1","IMPF"):"brentor", ("2","IMPF"):"skolven", ("3","IMPF"):"skolven", "PERF":"felvrak",
    }),
    "vrantkel": ("B", {
        ("1","IMPF"):"vrantkel",("2","IMPF"):"dorspev",("3","IMPF"):"dorspev","PERF":"troklav",
    }),
    "thovral": ("C", {
        ("1","sg","IMPF"):"thovral",("2","sg","IMPF"):"skleven",("3","sg","IMPF"):"porthan",
        ("1","du","IMPF"):"blavren",("2","du","IMPF"):"skleven",("3","du","IMPF"):"porthan",
        ("1","pl","IMPF"):"thovral",("2","pl","IMPF"):"skleven",("3","pl","IMPF"):"porthan",
        ("1","sg","PERF"):"dronfal",("2","sg","PERF"):"skelpov",("3","sg","PERF"):"gronvak",
        ("1","du","PERF"):"trelskon",("2","du","PERF"):"skelpov",("3","du","PERF"):"gronvak",
        ("1","pl","PERF"):"dronfal",("2","pl","PERF"):"skelpov",("3","pl","PERF"):"gronvak",
    }),
}


def _get_stem(verb, vclass, paradigm, person, number, aspect):
    if vclass == "A":
        return paradigm[aspect]
    if vclass == "B":
        if aspect == "PERF":
            return paradigm["PERF"]
        return paradigm.get((person, "IMPF"), paradigm.get(("3","IMPF"), "?"))
    if vclass == "C":
        key = (person, number, aspect)
        return paradigm.get(key, paradigm.get((person, "sg", aspect), "?"))
    return "?"


def _build_form(verb_root, person, number, aspect):
    vclass, paradigm = _VERBS[verb_root]
    stem = _get_stem(verb_root, vclass, paradigm, person, number, aspect)
    return stem + "-" + _ASP[aspect] + "-" + _AGR[(person, number)]


import itertools as _it4
_PERSONS = ["1","2","3"]
_NUMBERS = ["sg","du","pl"]
_ASPECTS = ["IMPF","PERF"]

_ALL_SPECS_G = [(v,p,n,a) for v in _VERBS for p in _PERSONS for n in _NUMBERS for a in _ASPECTS]
import random as _rng4m
_rng4 = _rng4m.Random(41)
_rng4.shuffle(_ALL_SPECS_G)
_ALL_OUTPUTS_G = [_build_form(v,p,n,a) for v,p,n,a in _ALL_SPECS_G]

MAX_EXAMPLES = 22
INITIAL_EXAMPLES = 7

_TEST_SPECS_G = [
    ("brentor",  "1","du","IMPF"),
    ("thovral",  "2","pl","PERF"),
    ("vrantkel", "3","sg","IMPF"),
    ("drath",    "1","du","PERF"),
]
_TEST_EXPECTED_G = [_build_form(v,p,n,a) for v,p,n,a in _TEST_SPECS_G]


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
    name="grelkan_suppletion_lang_learning",
    description="Learn GRELKAN 3-class verbal suppletion (A=aspect-split, B=person-split, C=5-way) from examples; infer class and fill 4 paradigm cells. Score=accuracy*efficiency.",
)
def grelkan_suppletion_lang_learning(llm) -> float:
    """Infer GRELKAN unpredictable verbal suppletion class (A/B/C) from partial paradigm; fill 4 unseen cells. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying GRELKAN verbal morphology.",
        "GRELKAN verbs show LEXICAL SUPPLETION: entirely different stems in different paradigm cells.",
        "Three suppletion classes (NOT phonologically predictable):",
        "  Class A: one stem for IMPERFECTIVE aspect, another for PERFECTIVE",
        "  Class B: 1st-person IMPF stem vs 2nd/3rd-person IMPF stem vs unified PERF stem",
        "  Class C: 5-way stem split by person×aspect (dual uses rare 4th stems)",
        "",
        "Verb form = SUPPLETIVE-STEM + -ASPECT-SUFFIX + -AGREEMENT-SUFFIX",
        "Aspect suffixes: -rek (IMPF), -val (PERF)",
        "Agreement: 1sg=-am, 2sg=-ot, 3sg=-es, 1du=-avan, 2du=-otel, 3du=-esan, 1pl=-ami, 2pl=-oti, 3pl=-esi",
        "",
        "Labeled examples (verb + person.number.aspect → form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        v,p,n,a = _ALL_SPECS_G[i]
        lines.append(f"  Example {i+1}: {v} {p}.{n}.{a} → {_ALL_OUTPUTS_G[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Infer each verb's class, then fill unseen cells.",
        "Scoring note: Getting each examination inflected surface form right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in suppletion and stem alternations, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("grelkan_suppletion"):
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
                    v,p,n,a = _ALL_SPECS_G[idx]
                    ex = f"Example {idx+1}: {v} {p}.{n}.{a} → {_ALL_OUTPUTS_G[idx]}"
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
            "EXAMINATION — Produce the correct GRELKAN verb form for each specification.",
            "Identify the verb's class from previously seen forms, then build the correct form.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (v,p,n,a) in enumerate(_TEST_SPECS_G):
            exam_lines.append(f"  Item {i+1}: verb='{v}', {p}.{n}.{a}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_SPECS_G[i][0]}_{_TEST_SPECS_G[i][1]}.{_TEST_SPECS_G[i][2]}.{_TEST_SPECS_G[i][3]}",
                "expected": _TEST_EXPECTED_G[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_G[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("GRELKAN CONTEXT-SENSITIVE SUPPLETION", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    grelkan_suppletion_lang_learning.run(kbench.llm)

