#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "KELSTRAN tests lexical tones that encode grammatical distinctions. "
    "The same segmental form surfaces with different tones to mark: "
    "DECLARATIVE (H-L), INTERROGATIVE (L-H), IMPERATIVE (H-H), SUBJUNCTIVE (L-L). "
    "Monosyllabic verbs use different contour tones (DECL=falling, INTER=rising, IMP=level-H, SUBJ=level-L). "
    "Additionally: aspect (IMPF vs PERF) triggers a 'tone displacement' that shifts all tones one position left. "
    "Model must infer the tone patterns AND displacement from examples. Score = accuracy * efficiency."
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
FREE_THRESHOLD = 3

_ACUTE = {"a":"á","e":"é","i":"í","o":"ó","u":"ú"}
_GRAVE = {"a":"à","e":"è","i":"ì","o":"ò","u":"ù"}
_CIRC  = {"a":"â","e":"ê","i":"î","o":"ô","u":"û"}
_CARON = {"a":"ǎ","e":"ě","i":"ǐ","o":"ǒ","u":"ǔ"}


def _mark(v, tone):
    if tone == "H":  return _ACUTE.get(v, v)
    if tone == "L":  return _GRAVE.get(v, v)
    if tone == "HL": return _CIRC.get(v, v)
    if tone == "LH": return _CARON.get(v, v)
    return v


def _base(c):
    return unicodedata.normalize("NFD", c)[0]


def _is_vowel(c):
    return _base(c) in "aeiou"


def _syllable_peaks(word):
    return [i for i, c in enumerate(word) if _is_vowel(c)]


_MOOD_MELODY = {
    "DECL":  ["HL","L"],
    "INTER": ["L","LH"],
    "IMP":   ["H","H"],
    "SUBJ":  ["L","L"],
}
_MOOD_MONO = {
    "DECL": "HL", "INTER": "LH", "IMP": "H", "SUBJ": "L",
}


def _flip_tone_token(t):
    return {"H": "L", "L": "H", "HL": "LH", "LH": "HL"}.get(t, t)


def _apply_melody(word, melody, displace=False):
    peaks = _syllable_peaks(word)
    chars = list(word)
    if displace:
        shifted = melody[1:] + [melody[0]]
        melody = shifted
    for i, p in enumerate(peaks):
        if i < len(melody):
            chars[p] = _mark(_base(chars[p]), melody[i])
    return "".join(chars)


def _kelstran_form(root, mood, aspect, episodic):
    peaks = _syllable_peaks(root)
    displace = (aspect == "PERF")
    if len(peaks) == 1:
        mel = _MOOD_MONO[mood]
        if episodic == "HABIT":
            mel = _flip_tone_token(mel)
        melody = [mel]
    else:
        base_mel = _MOOD_MELODY[mood]
        if episodic == "HABIT":
            melody = [_flip_tone_token(x) for x in base_mel]
        else:
            melody = list(base_mel)
    return _apply_melody(root, melody, displace)


_ROOTS_KT = ["skolen","dravath","belko","frentak","solva","trevel","prondak","skrevol"]
_MOODS = ["DECL","INTER","IMP","SUBJ"]
_ASPECTS_KT = ["IMPF","PERF"]
_EPISODIC_KT = ["NARR","HABIT"]

import itertools as _it5
_ALL_SPECS_KT = list(_it5.product(_ROOTS_KT, _MOODS, _ASPECTS_KT, _EPISODIC_KT))
import random as _rng5m
_rng5 = _rng5m.Random(97)
_rng5.shuffle(_ALL_SPECS_KT)

_ALL_OUTPUTS_KT = [_kelstran_form(r,m,a,e) for r,m,a,e in _ALL_SPECS_KT]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 5

_TEST_SPECS_KT = [
    ("dravath", "INTER", "PERF", "HABIT"),
    ("solva", "IMP", "IMPF", "NARR"),
    ("skolen", "SUBJ", "PERF", "NARR"),
    ("prondak", "DECL", "IMPF", "HABIT"),
]
_TEST_EXPECTED_KT = [_kelstran_form(r, m, a, e) for r, m, a, e in _TEST_SPECS_KT]


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
    name="kelstran_tone_lang_learning",
    description="Learn KELSTRAN mood tones + HABIT polarity flip + PERF displacement from examples; 4 tonal forms. Score=accuracy*efficiency.",
)
def kelstran_tone_lang_learning(llm) -> float:
    """Infer KELSTRAN tone melodies, episodic HABIT flip, and PERF displacement from examples; produce 4 forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying KELSTRAN tonal morphology.",
        "KELSTRAN uses TONES to encode grammatical mood (tone marks are on vowels):",
        "  DECLARATIVE:   H-L melody  (high on 1st syllable, low on 2nd)",
        "  INTERROGATIVE: L-LH melody (low on 1st, rising LH on 2nd)",
        "  IMPERATIVE:    H-H melody  (high throughout)",
        "  SUBJUNCTIVE:   L-L melody  (low throughout)",
        "Monosyllabic verbs: DECL=falling(HL), INTER=rising(LH), IMP=H, SUBJ=L",
        "",
        "ASPECT TONE DISPLACEMENT: PERFECTIVE aspect shifts the entire tone melody ONE POSITION LEFT",
        "(cyclic shift: melody[HL,L] → [L,HL])",
        "EPISODIC: NARR leaves the mood melody as above; HABIT flips each tonal target (H↔L, HL↔LH) BEFORE displacement applies.",
        "",
        "Labeled examples (root + mood + aspect + episodic → tonal form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        r, m, a, e = _ALL_SPECS_KT[i]
        lines.append(f"  Example {i+1}: root='{r}', {m}, {a}, {e} → {_ALL_OUTPUTS_KT[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}.",
        "Scoring note: Getting each examination tonal assignment right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in tone or accent assignment, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("kelstran_tone"):
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
                    r, m, a, e = _ALL_SPECS_KT[idx]
                    ex = f"Example {idx+1}: root='{r}', {m}, {a}, {e} → {_ALL_OUTPUTS_KT[idx]}"
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
            "EXAMINATION — Produce the correct tonal verb form for each specification.",
            "Apply mood melody, apply HABIT flip if episodic=HABIT, then PERF displacement if aspect=PERF.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (r, m, a, e) in enumerate(_TEST_SPECS_KT):
            exam_lines.append(f"  Item {i+1}: root='{r}', mood={m}, aspect={a}, episodic={e}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_SPECS_KT[i][0]}.{_TEST_SPECS_KT[i][1]}.{_TEST_SPECS_KT[i][2]}.{_TEST_SPECS_KT[i][3]}",
                "expected": _TEST_EXPECTED_KT[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_KT[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("KELSTRAN GRAMMATICAL TONE", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    kelstran_tone_lang_learning.run(kbench.llm)

