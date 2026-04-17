#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "PELVAN tests multi-probe verbal agreement: person×number×gender×animacy×case. "
    "Verbs agree with up to TWO arguments simultaneously (split agreement). "
    "Subject agreement is encoded by PREFIX; object agreement by SUFFIX. "
    "Animacy overrides gender in marking: ANIMATE uses one paradigm, INANIMATE another. "
    "Case also affects agreement: when the S is LOC-case, subject agreement is suppressed. "
    "Model must discover the 5-dimensional agreement grid from labeled examples, "
    "then assemble 4 correct verb complexes. Score = accuracy * efficiency."
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
FREE_THRESHOLD = 7

_S_AGR = {
    ("1","sg","m","anim"):   "ka",    ("1","sg","f","anim"):   "ke",
    ("1","pl","m","anim"):   "kasto", ("1","pl","f","anim"):   "keste",
    ("2","sg","m","anim"):   "vi",    ("2","sg","f","anim"):   "ve",
    ("2","pl","m","anim"):   "visto", ("2","pl","f","anim"):   "veste",
    ("3","sg","m","anim"):   "no",    ("3","sg","f","anim"):   "ne",
    ("3","pl","m","anim"):   "nosto", ("3","pl","f","anim"):   "neste",
    ("1","sg","m","inanim"): "ta",    ("1","sg","f","inanim"): "te",
    ("1","pl","m","inanim"): "tasto", ("1","pl","f","inanim"): "teste",
    ("2","sg","m","inanim"): "pa",    ("2","sg","f","inanim"): "pe",
    ("2","pl","m","inanim"): "pasto", ("2","pl","f","inanim"): "peste",
    ("3","sg","m","inanim"): "ra",    ("3","sg","f","inanim"): "re",
    ("3","pl","m","inanim"): "rasto", ("3","pl","f","inanim"): "reste",
}
_O_AGR = {
    ("3","sg","m","anim"):   "lov",  ("3","sg","f","anim"):   "lev",
    ("3","pl","m","anim"):   "lovsto",("3","pl","f","anim"):  "levste",
    ("3","sg","m","inanim"): "tov",  ("3","sg","f","inanim"): "tev",
    ("3","pl","m","inanim"): "tovsto",("3","pl","f","inanim"):"tevste",
}
_VERB_ROOTS_PV = {"see":"vanr","carry":"trel","find":"preth","give":"drun","make":"skof"}


def _pelvan_form(s_spec, o_spec, verb_en, s_case):
    s_p,s_n,s_g,s_a = s_spec
    s_prefix = "" if s_case == "LOC" else _S_AGR.get((s_p,s_n,s_g,s_a),"??")
    o_p,o_n,o_g,o_a = o_spec
    o_suffix = _O_AGR.get((o_p,o_n,o_g,o_a),"??")
    v = _VERB_ROOTS_PV[verb_en]
    return s_prefix + v + o_suffix


_PV_SPECS = [
    (("1","sg","m","anim"),   ("3","sg","m","anim"),   "see",   "NOM"),
    (("2","sg","f","anim"),   ("3","pl","f","inanim"), "carry", "NOM"),
    (("3","sg","m","inanim"), ("3","sg","f","anim"),   "find",  "NOM"),
    (("1","pl","f","anim"),   ("3","sg","m","inanim"), "give",  "NOM"),
    (("3","sg","m","anim"),   ("3","pl","m","anim"),   "make",  "LOC"),
    (("2","pl","m","anim"),   ("3","sg","f","anim"),   "see",   "NOM"),
    (("1","sg","f","inanim"), ("3","sg","m","inanim"), "carry", "NOM"),
    (("3","pl","f","anim"),   ("3","sg","f","inanim"), "find",  "NOM"),
    (("2","sg","m","anim"),   ("3","pl","f","anim"),   "give",  "LOC"),
    (("1","pl","m","anim"),   ("3","sg","m","anim"),   "make",  "NOM"),
    (("3","sg","f","inanim"), ("3","pl","m","inanim"), "see",   "NOM"),
    (("2","pl","f","anim"),   ("3","sg","m","anim"),   "carry", "NOM"),
    (("1","sg","m","anim"),   ("3","sg","f","inanim"), "find",  "LOC"),
    (("3","sg","m","anim"),   ("3","sg","m","inanim"), "give",  "NOM"),
    (("2","sg","f","inanim"), ("3","pl","f","anim"),   "make",  "NOM"),
    (("1","pl","f","anim"),   ("3","sg","f","anim"),   "see",   "NOM"),
    (("3","pl","m","anim"),   ("3","sg","m","anim"),   "carry", "LOC"),
    (("2","sg","m","inanim"), ("3","sg","f","inanim"), "find",  "NOM"),
    (("1","sg","f","anim"),   ("3","pl","m","anim"),   "give",  "NOM"),
    (("3","sg","f","anim"),   ("3","sg","m","inanim"), "make",  "NOM"),
]

_ALL_OUTPUTS_PV = [_pelvan_form(s,o,v,c) for s,o,v,c in _PV_SPECS]

MAX_EXAMPLES = 18
INITIAL_EXAMPLES = 6

_TEST_PV_SPECS = [
    (("1","sg","m","anim"),   ("3","sg","f","anim"),   "see",   "LOC"),
    (("3","pl","f","anim"),   ("3","sg","m","inanim"), "carry", "NOM"),
    (("2","sg","m","anim"),   ("3","pl","f","anim"),   "find",  "NOM"),
    (("1","pl","f","inanim"), ("3","sg","m","anim"),   "give",  "NOM"),
]
_TEST_EXPECTED_PV = [_pelvan_form(s,o,v,c) for s,o,v,c in _TEST_PV_SPECS]


def _fmt_pv(s,o,verb,c):
    sp,sn,sg,sa = s
    op,on,og,oa = o
    return f"S={sp}.{sn}.{sg}.{sa}({c}), O={op}.{on}.{og}.{oa}, V={verb}"


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
    name="pelvan_agreement_lang_learning",
    description="Learn PELVAN 5-dimensional split agreement (person×number×gender×animacy×case; LOC suppresses S-prefix) from examples; produce 4 verb complexes. Score=accuracy*efficiency.",
)
def pelvan_agreement_lang_learning(llm) -> float:
    """Infer PELVAN split verbal agreement (S-prefix from 5-dim grid; O-suffix; LOC suppresses S) from examples; produce 4 forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying PELVAN verbal agreement morphology.",
        "PELVAN verbs carry split agreement: S-AGREEMENT as PREFIX, O-AGREEMENT as SUFFIX.",
        "Subject agreement indexes: person (1/2/3) × number (sg/pl) × gender (m/f) × animacy (anim/inanim)",
        "Object agreement indexes: person (always 3rd) × number × gender × animacy",
        "",
        "SPECIAL RULE: when the SUBJECT is in LOCATIVE case, subject agreement prefix is SUPPRESSED (omitted).",
        "",
        "Labeled examples (subject-spec + object-spec + verb + s-case → verb complex):",
    ]
    for i in range(INITIAL_EXAMPLES):
        s,o,v,c = _PV_SPECS[i]
        lines.append(f"  Example {i+1}: [{_fmt_pv(s,o,v,c)}] → {_ALL_OUTPUTS_PV[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover the 5-dimensional agreement grid.",
        "Scoring note: Getting each examination agreeing form right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in agreement patterns, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("pelvan_agr"):
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
                    s,o,v,c = _PV_SPECS[idx]
                    ex = f"Example {idx+1}: [{_fmt_pv(s,o,v,c)}] → {_ALL_OUTPUTS_PV[idx]}"
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
            "EXAMINATION — Produce the correct PELVAN verb complex for each specification.",
            "Remember: S prefix + verb root + O suffix; LOC subject → no S prefix.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (s,o,v,c) in enumerate(_TEST_PV_SPECS):
            exam_lines.append(f"  Item {i+1}: {_fmt_pv(s,o,v,c)}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": _fmt_pv(*_TEST_PV_SPECS[i]),
                "expected": _TEST_EXPECTED_PV[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_PV[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("PELVAN 5-DIMENSIONAL SPLIT AGREEMENT", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    pelvan_agreement_lang_learning.run(kbench.llm)

