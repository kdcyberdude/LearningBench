#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "SKOLVREN tests polysynthetic verb morphology with a 6-slot template: "
    "[S1:subj-person] [S2:subj-number] [S3:subj-animacy] [S4:tense] [S5:verb-root+noun-incorporation] [S6:evidentiality+mood]. "
    "When the direct object is an INANIMATE noun, it is incorporated into S5 and triggers a classifier on the verb root. "
    "The model must discover all 6 slots, their ordering, all paradigm values, and the noun-incorporation rule from examples, "
    "then assemble 4 novel verb complexes in examination. Score = accuracy * efficiency."
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

_S1 = {"1": "ka", "2": "vi", "3": "no"}
_S2 = {"sg": "l", "du": "rv", "pl": "sto"}
_S3 = {"anim": "en", "inanim": "ab"}
_S4 = {"past": "dreth", "pres": "olan", "fut": "skev"}
_VERB_ROOTS = {"eat": "bok", "carry": "trel", "see": "van", "make": "skof", "give": "drun"}
_CLASSIFIERS = {"round": "ga", "elongated": "pri", "flat": "mes", "container": "vok", "other": "lu"}
_NOUN_CLASS = {
    "stone": "round", "ball": "round",
    "stick": "elongated", "spear": "elongated",
    "leaf": "flat", "cloth": "flat",
    "pot": "container", "basket": "container",
    "tool": "other",
}
_NOUN_ROOTS = {k: k[:3] for k in _NOUN_CLASS}
_S6 = {
    ("direct", "decl"):  "ven",
    ("infer", "decl"):   "mek",
    ("hearsay", "decl"): "sto",
    ("direct", "inter"): "venka",
    ("infer", "inter"):  "mekra",
    ("direct", "imp"):   "voli",
}


def _build_verb(subj_p, subj_n, subj_a, tense, verb, evid, mood, obj_noun=None):
    s1 = _S1[subj_p]
    s2 = _S2[subj_n]
    s3 = _S3[subj_a]
    s4 = _S4[tense]
    if obj_noun and obj_noun in _NOUN_CLASS:
        clf = _CLASSIFIERS[_NOUN_CLASS[obj_noun]]
        noun_root = _NOUN_ROOTS[obj_noun]
        s5 = _VERB_ROOTS[verb] + clf + noun_root
    else:
        s5 = _VERB_ROOTS[verb]
    s6 = _S6.get((evid, mood), "ven")
    return s1 + s2 + s3 + s4 + s5 + s6


_ALL_SPECS = [
    ("1","sg","anim","past","eat",None,"direct","decl"),     # direct+decl (baseline)
    ("2","pl","anim","pres","see",None,"infer","decl"),      # infer+decl
    ("3","sg","inanim","fut","carry","stone","hearsay","decl"), # hearsay+decl + incorporation
    ("3","sg","anim","pres","carry",None,"infer","inter"),   # infer+inter → mekra (critical for test 2)
    ("2","pl","inanim","fut","make","pot","direct","imp"),   # direct+imp → voli (critical for test 4)
    ("1","du","anim","past","make",None,"direct","inter"),   # direct+inter
    ("2","sg","anim","fut","give",None,"direct","decl"),
    ("1","sg","inanim","past","see","leaf","direct","decl"),
    ("3","du","anim","pres","carry",None,"hearsay","decl"),
    ("1","pl","anim","past","eat",None,"infer","decl"),
    ("3","sg","anim","pres","give",None,"direct","inter"),
    ("2","du","inanim","past","see","ball","hearsay","decl"),
    ("1","sg","anim","fut","carry",None,"direct","decl"),
    ("3","pl","inanim","pres","eat","basket","infer","decl"),
    ("2","sg","anim","past","make",None,"direct","decl"),
    ("1","du","inanim","fut","give","cloth","direct","decl"),
    ("3","sg","anim","pres","see",None,"direct","imp"),
    ("2","pl","inanim","past","carry","spear","infer","decl"),
    ("1","sg","anim","fut","eat",None,"hearsay","decl"),
    ("3","pl","inanim","pres","eat","stick","infer","decl"),
]


def _spec_to_surface(spec):
    p,n,a,t,v,obj,evid,mood = spec
    return _build_verb(p,n,a,t,v,evid,mood,obj)


_ALL_OUTPUTS = [_spec_to_surface(s) for s in _ALL_SPECS]


def _fmt_spec(spec):
    p,n,a,t,v,obj,evid,mood = spec
    obj_part = f", incorporating OBJ='{obj}'" if obj else ""
    return f"SUBJ={p}.{n}.{a}, TENSE={t}, VERB={v}{obj_part}, EVID={evid}, MOOD={mood}"


MAX_EXAMPLES = 18
INITIAL_EXAMPLES = 6

_TEST_SPECS = [
    ("3","sg","inanim","past","eat","stone","direct","decl"),
    ("1","pl","anim","fut","see",None,"infer","inter"),
    ("2","du","inanim","pres","carry","leaf","hearsay","decl"),
    ("3","pl","anim","past","give",None,"direct","imp"),
]
_TEST_EXPECTED = [_spec_to_surface(s) for s in _TEST_SPECS]


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
    name="skolvren_polysynthetic_lang_learning",
    description="Learn SKOLVREN 6-slot polysynthetic verb template (subj-person/number/animacy + tense + verb+noun-incorporation + evid+mood) from examples. Score=accuracy*efficiency.",
)
def skolvren_polysynthetic_lang_learning(llm) -> float:
    """Infer SKOLVREN 6-slot polysynthetic morpheme template including noun incorporation from examples; assemble 4 novel verbs. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying SKOLVREN, a polysynthetic language.",
        "Entire sentences are expressed as a single VERB COMPLEX built from 6 ordered morpheme slots.",
        "Each example shows the grammatical specification and its corresponding surface verb complex.",
        "",
        "Labeled examples:",
    ]
    for i in range(INITIAL_EXAMPLES):
        lines.append(f"  Example {i+1}: [{_fmt_spec(_ALL_SPECS[i])}] → '{_ALL_OUTPUTS[i]}'")
    lines += [
        "",
        "Note: when the direct object is an INANIMATE noun, it is INCORPORATED into the verb complex.",
        "This changes the verb root and adds a CLASSIFIER morpheme that depends on the noun's shape/material class.",
        "",
        f"Actions: action='request' (more examples, up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover all 6 slots and the noun incorporation rule.",
        "Scoring note: Getting each examination polysynthetic verb complex right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in the polysynthetic verb template, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("skolvren_poly"):
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
                    ex = f"Example {idx+1}: [{_fmt_spec(_ALL_SPECS[idx])}] → '{_ALL_OUTPUTS[idx]}'"
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
            "EXAMINATION — Build the correct SKOLVREN verb complex for each specification.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, spec in enumerate(_TEST_SPECS):
            exam_lines.append(f"  Item {i+1}: {_fmt_spec(spec)}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": _fmt_spec(_TEST_SPECS[i]),
                "expected": _TEST_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("SKOLVREN POLYSYNTHETIC TEMPLATE", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    skolvren_polysynthetic_lang_learning.run(kbench.llm)

