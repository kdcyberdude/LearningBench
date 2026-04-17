#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "WUKAL tests tone spreading + Obligatory Contour Principle (OCP) + blocking. "
    "High tones spread rightward to adjacent toneless vowels (spreading stops at a L tone). "
    "OCP: two adjacent identical tones (HH or LL) merge into one contour tone (HL or LH). "
    "BLOCKING: tone spreading is blocked when a nasal consonant intervenes. "
    "Model must discover all 3 tonal processes AND their precedence from examples. "
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

_ACUTE_W = {"a":"á","e":"é","i":"í","o":"ó","u":"ú"}
_GRAVE_W = {"a":"à","e":"è","i":"ì","o":"ò","u":"ù"}
_CIRC_W  = {"a":"â","e":"ê","i":"î","o":"ô","u":"û"}
_CARON_W = {"a":"ǎ","e":"ě","i":"ǐ","o":"ǒ","u":"ǔ"}
_NASALS_W = set("mnŋ")
_BASE_V = {"á":"a","é":"e","í":"i","ó":"o","ú":"u",
           "à":"a","è":"e","ì":"i","ò":"o","ù":"u",
           "â":"a","ê":"e","î":"i","ô":"o","û":"u",
           "ǎ":"a","ě":"e","ǐ":"i","ǒ":"o","ǔ":"u"}


def _base_v(c):
    return _BASE_V.get(c, c)


def _is_v(c):
    return _base_v(c) in "aeiou"


def _is_n(c):
    return c in _NASALS_W


def _tone_of(c):
    if c in "áéíóú": return "H"
    if c in "àèìòù": return "L"
    if c in "âêîôû": return "HL"
    if c in "ǎěǐǒǔ": return "LH"
    if _is_v(c):      return "0"
    return None


def _mark_v(base, tone):
    if tone == "H":  return _ACUTE_W.get(base, base)
    if tone == "L":  return _GRAVE_W.get(base, base)
    if tone == "HL": return _CIRC_W.get(base, base)
    if tone == "LH": return _CARON_W.get(base, base)
    return base


def _spread_h(chars):
    out = list(chars)
    i = 0
    while i < len(out):
        t = _tone_of(out[i])
        if t == "H":
            j = i + 1
            while j < len(out):
                if _is_n(out[j]):
                    break
                if _is_v(out[j]):
                    next_t = _tone_of(out[j])
                    if next_t == "L" or next_t == "HL" or next_t == "LH":
                        break
                    if next_t == "0":
                        out[j] = _mark_v(_base_v(out[j]), "H")
                    elif next_t == "H":
                        break
                j += 1
        i += 1
    return out


def _ocp_merge(chars):
    out = list(chars)
    i = 0
    while i < len(out) - 1:
        if _is_v(out[i]) and _is_v(out[i+1]):
            t1 = _tone_of(out[i])
            t2 = _tone_of(out[i+1])
            if t1 == "H" and t2 == "H":
                out[i] = _mark_v(_base_v(out[i]), "HL")
                out[i+1] = _base_v(out[i+1])
            elif t1 == "L" and t2 == "L":
                out[i] = _mark_v(_base_v(out[i]), "LH")
                out[i+1] = _base_v(out[i+1])
        i += 1
    return out


def _wukal_derive(underlying):
    chars = list(underlying)
    chars = _spread_h(chars)
    chars = _ocp_merge(chars)
    return "".join(chars)


_WUKAL_INPUTS = [
    "ákolan",  "sóvren",  "tràvel",  "drénvak",  "ànkova",
    "skoláven", "prétank", "bólmreth", "frànskev", "drólvank",
    "étankov",  "skrévan",  "àmpreth", "nótrevel", "bólnakov",
    "ántrevan",  "skóvnam",  "drèlkov",  "réntavel", "bólnakev",
]
_ALL_WUKAL_OUTPUTS = [_wukal_derive(w) for w in _WUKAL_INPUTS]

MAX_EXAMPLES = 16
INITIAL_EXAMPLES = 4

_TEST_WUKAL_INPUTS = ["ákónvan","drèntevel","sóvránam","préntov"]
_TEST_WUKAL_EXPECTED = [_wukal_derive(w) for w in _TEST_WUKAL_INPUTS]


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
    name="wukal_tones_lang_learning",
    description="Learn WUKAL 3-process tonal system (H-spreading blocked by nasal, OCP HH→HL, OCP LL→LH) from examples; produce 4 surface tonal forms. Score=accuracy*efficiency.",
)
def wukal_tones_lang_learning(llm) -> float:
    """Infer WUKAL H-spreading (nasal-blocked), OCP merge (HH→HL, LL→LH) from examples; produce 4 tonal surface forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying WUKAL tonal phonology.",
        "Tones are marked on vowels: á/é=H, à/è=L, â/ê=HL(falling), ǎ/ě=LH(rising).",
        "Toneless vowels are unmarked (a/e/i/o/u).",
        "",
        "THREE tonal processes interact:",
        "  1. H-SPREADING: a H-toned vowel spreads H rightward to adjacent toneless vowels.",
        "     BLOCKING: spreading STOPS at a nasal consonant (m/n/ŋ) — nasal is a spreading barrier.",
        "  2. OCP (Obligatory Contour Principle): two adjacent identical tones MERGE into a contour:",
        "     H + H adjacent vowels → first becomes HL (falling), second becomes toneless",
        "     L + L adjacent vowels → first becomes LH (rising), second becomes toneless",
        "  3. The processes apply in order: H-spreading first, then OCP.",
        "",
        "Labeled examples (underlying form → surface form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        lines.append(f"  Example {i+1}: {_WUKAL_INPUTS[i]} → {_ALL_WUKAL_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover spreading, blocking, and OCP.",
        "Scoring note: Getting each examination tonal output right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in the tonal rules, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("wukal_tones"):
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
                    ex = f"Example {idx+1}: {_WUKAL_INPUTS[idx]} → {_ALL_WUKAL_OUTPUTS[idx]}"
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
            "EXAMINATION — Produce the correct WUKAL surface form for each underlying form.",
            "Apply H-spreading (blocked by nasals) then OCP. Provide all 4 answers at once.",
            "",
        ]
        for i, w in enumerate(_TEST_WUKAL_INPUTS):
            exam_lines.append(f"  Item {i+1}: underlying='{w}'")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": _TEST_WUKAL_INPUTS[i],
                "expected": _TEST_WUKAL_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_WUKAL_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("WUKAL TONE SPREADING + OCP + BLOCKING", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    wukal_tones_lang_learning.run(kbench.llm)

