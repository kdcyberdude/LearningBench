#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "VRENDEL tests circumfix + prosodic templatic morphology + allomorphy. "
    "CIRCUMFIXES: the prefix and suffix are fused into one morphological unit (neither can appear alone). "
    "The circumfix changes based on: verb class (I/II/III) + TAM category. "
    "TEMPLATIC CONSTRAINT: the resulting word must fit a minimal prosodic template (minimum 2 syllables). "
    "ALLOMORPHY: Class I verbs have two prefix-allomorphs depending on the following vowel (ka- vs ke-). "
    "Model must infer the circumfix grid AND the prosodic template AND allomorphy. Score = accuracy * efficiency."
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

_CIRCUMFIXES = {
    ("I",  "PRES"):  ("ka-/ke-", "-ven"),
    ("I",  "PAST"):  ("ka-/ke-", "-dreth"),
    ("I",  "FUT"):   ("ka-/ke-", "-skev"),
    ("II", "PRES"):  ("no-",     "-lan"),
    ("II", "PAST"):  ("no-",     "-vrak"),
    ("II", "FUT"):   ("no-",     "-skev"),
    ("III","PRES"):  ("dru-",    "-sto"),
    ("III","PAST"):  ("dru-",    "-fen"),
    ("III","FUT"):   ("dru-",    "-skev"),
}
_VOWELS_VR = set("aeiou")
_FRONT_VOWELS = set("ei")


def _syllable_count(w):
    return sum(1 for c in w if c in _VOWELS_VR)


def _get_prefix(vclass, tam, root):
    key = (vclass, tam)
    pref_raw, _ = _CIRCUMFIXES[key]
    if "/" in pref_raw:
        parts = pref_raw.split("/")
        p1, p2 = parts[0], parts[1]
        first_v = next((c for c in root if c in _VOWELS_VR), "a")
        return p2.rstrip("-") if first_v in _FRONT_VOWELS else p1.rstrip("-")
    return pref_raw.rstrip("-")


def _vrendel_form(root, vclass, tam):
    pref = _get_prefix(vclass, tam, root)
    key = (vclass, tam)
    _, suf = _CIRCUMFIXES[key]
    word = pref + root + suf
    while _syllable_count(word) < 2:
        word = pref + "a" + word
    return word


_ROOTS_VR = {
    "see":  ("vanr", "I"),   "carry": ("trel", "I"),
    "find": ("preth","II"),  "give":  ("drun","II"),
    "make": ("skof", "III"), "run":   ("skopal","III"),
}
_TAMS = ["PRES","PAST","FUT"]

import itertools as _it_vr
_ALL_VR_SPECS = [(r, rc, t) for r,(root,rc) in _ROOTS_VR.items() for t in _TAMS]

import random as _rng_vr
_rng_vr_obj = _rng_vr.Random(83)
_rng_vr_obj.shuffle(_ALL_VR_SPECS)

_ALL_VR_OUTPUTS = [_vrendel_form(_ROOTS_VR[r][0], _ROOTS_VR[r][1], t) for r,_,t in _ALL_VR_SPECS]

MAX_EXAMPLES = 16
INITIAL_EXAMPLES = 5

_TEST_VR_SPECS = [
    ("see",  "PAST"),
    ("find", "FUT"),
    ("make", "PRES"),
    ("carry","FUT"),
]
_TEST_VR_EXPECTED = [_vrendel_form(_ROOTS_VR[r][0], _ROOTS_VR[r][1], t) for r,t in _TEST_VR_SPECS]


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
    name="vrendel_templatic_lang_learning",
    description="Learn VRENDEL circumfix morphology (3-class × 3-TAM grid, ka-/ke- allomorphy, 2-syl minimum template) from examples; produce 4 verb forms. Score=accuracy*efficiency.",
)
def vrendel_templatic_lang_learning(llm) -> float:
    """Infer VRENDEL 9-cell circumfix grid with vowel-conditioned allomorphy and prosodic template from examples; produce 4 verb forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying VRENDEL verbal morphology.",
        "VRENDEL uses CIRCUMFIXES: a prefix AND suffix are added simultaneously (they form one morphological unit).",
        "The circumfix is chosen based on VERB CLASS (I/II/III) × TAM (PRES/PAST/FUT):",
        "  Class I:   PRES=ka/ke-...-ven, PAST=ka/ke-...-dreth, FUT=ka/ke-...-skev",
        "  Class II:  PRES=no-...-lan,    PAST=no-...-vrak,     FUT=no-...-skev",
        "  Class III: PRES=dru-...-sto,   PAST=dru-...-fen,     FUT=dru-...-skev",
        "",
        "Class I ALLOMORPHY: if the root's first vowel is front (e/i), use ke- prefix; otherwise ka-.",
        "PROSODIC TEMPLATE: the resulting word must have at least 2 syllables;",
        "  if too short, an extra -a- is inserted after the prefix.",
        "",
        "Labeled examples (verb + class + TAM → form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        r, _, t = _ALL_VR_SPECS[i]
        vclass = _ROOTS_VR[r][1]
        lines.append(f"  Example {i+1}: verb='{r}'(class {vclass}), TAM={t} → {_ALL_VR_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover the circumfix grid, allomorphy, and template.",
        "Scoring note: Getting each examination verb form right (circumfix + template) matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in the circumfix grid, allomorphy, and template, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("vrendel_template"):
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
                    r, _, t = _ALL_VR_SPECS[idx]
                    vclass = _ROOTS_VR[r][1]
                    ex = f"Example {idx+1}: verb='{r}'(class {vclass}), TAM={t} → {_ALL_VR_OUTPUTS[idx]}"
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
            "EXAMINATION — Produce the correct VRENDEL verb form for each specification.",
            "Identify the circumfix for the class×TAM combination, apply allomorphy, check the template.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (r, t) in enumerate(_TEST_VR_SPECS):
            vclass = _ROOTS_VR[r][1]
            exam_lines.append(f"  Item {i+1}: verb='{r}'(class {vclass}), TAM={t}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            r, t = _TEST_VR_SPECS[i]
            exam_results.append({
                "item": i+1, "input": f"{r}.{t}",
                "expected": _TEST_VR_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_VR_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("VRENDEL CIRCUMFIX+TEMPLATE+ALLOMORPHY", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    vrendel_templatic_lang_learning.run(kbench.llm)

