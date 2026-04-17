#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "GWELTHAR tests a 3-way tonal interaction: lexical tone, evidential suffix tone, and mirative floating H. "
    "Each verb root has a lexical tone melody (H or L per syllable). "
    "Evidential suffixes add their own tones. "
    "A MIRATIVE (unexpected information) marker floats an H tone that docks on the penultimate syllable; "
    "if that syllable already has H, it spreads rightward (tone doubling). "
    "The model must infer all three tonal systems and their precedence order from examples, "
    "then produce 4 correct surface forms. Score = accuracy * efficiency."
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

_ACUTE = {"a":"á","e":"é","i":"í","o":"ó","u":"ú"}
_GRAVE = {"a":"à","e":"è","i":"ì","o":"ò","u":"ù"}


def _mark_h(v):
    return _ACUTE.get(v, v)


def _mark_l(v):
    return _GRAVE.get(v, v)


def _base_v(c):
    return unicodedata.normalize("NFD", c)[0]


def _is_vowel(c):
    return _base_v(c) in "aeiou"


def _syllable_peaks(word):
    return [i for i, c in enumerate(word) if _is_vowel(c)]


def _apply_lex_tone(syllables, lex_melody):
    chars = list(syllables)
    peaks = _syllable_peaks(syllables)
    for i, p in enumerate(peaks):
        if i < len(lex_melody):
            tone = lex_melody[i]
            base = _base_v(chars[p])
            chars[p] = _mark_h(base) if tone == "H" else _mark_l(base)
    return "".join(chars)


def _derive_word(root_str, lex_melody, evid_suffix, evid_melody, mirative):
    with_lex = _apply_lex_tone(root_str, lex_melody)
    full_word = with_lex + evid_suffix
    chars = list(full_word)
    peaks = _syllable_peaks(full_word)
    for j, p in enumerate(peaks[len(_syllable_peaks(root_str)):], start=len(_syllable_peaks(root_str))):
        idx_in_melody = j - len(_syllable_peaks(root_str))
        if idx_in_melody < len(evid_melody):
            tone = evid_melody[idx_in_melody]
            base = _base_v(chars[p])
            chars[p] = _mark_h(base) if tone == "H" else _mark_l(base)
    full_word = "".join(chars)
    if mirative:
        chars = list(full_word)
        all_peaks = _syllable_peaks(full_word)
        if len(all_peaks) >= 2:
            penult_idx = all_peaks[-2]
            base = _base_v(chars[penult_idx])
            existing = chars[penult_idx]
            existing_tone = "H" if existing in "áéíóú" else "L"
            if existing_tone == "H":
                last_idx = all_peaks[-1]
                base_last = _base_v(chars[last_idx])
                chars[last_idx] = _mark_h(base_last)
            else:
                chars[penult_idx] = _mark_h(base)
        elif len(all_peaks) == 1:
            p = all_peaks[0]
            chars[p] = _mark_h(_base_v(chars[p]))
        full_word = "".join(chars)
    return full_word


_ROOTS = {
    "drelko":  ("drelko", ["H","L"]),
    "skovar":  ("skovar", ["L","H"]),
    "antrel":  ("antrel", ["H","H"]),
    "bolven":  ("bolven", ["L","L"]),
    "treskav": ("treskav",["H","L","H"]),
    "omfelak": ("omfelak",["L","H","L"]),
}
_EVIDENTIALS = {
    "direct":  ("ven",  ["H"]),
    "infer":   ("mela", ["L","H"]),
    "hearsay": ("sto",  ["L"]),
    "assume":  ("prak", ["H","L"]),
}

import itertools as _itools
import random as _random
_rng = _random.Random(42)
_all_combos = list(_itools.product(list(_ROOTS.keys()), list(_EVIDENTIALS.keys()), [False, True]))
_rng.shuffle(_all_combos)
_selected = _all_combos[:22]

_ALL_INPUTS  = _selected


def _to_surface(root_key, evid_key, mir):
    rstr, rmelo = _ROOTS[root_key]
    esuf, emelo = _EVIDENTIALS[evid_key]
    return _derive_word(rstr, rmelo, esuf, emelo, mir)


_ALL_OUTPUTS = [_to_surface(r, e, m) for r, e, m in _selected]

MAX_EXAMPLES = 18
INITIAL_EXAMPLES = 5

_TEST_COMBOS = [
    ("drelko", "infer",   True),
    ("skovar", "direct",  False),
    ("antrel", "hearsay", True),
    ("bolven", "assume",  False),
]
_TEST_EXPECTED = [_to_surface(r, e, m) for r, e, m in _TEST_COMBOS]


def _fmt_spec(r, e, m):
    return f"root='{r}', evidential={e}, mirative={'yes' if m else 'no'}"


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
    name="gwelthar_mirative_evidential_tone_lang_learning",
    description="Learn GWELTHAR 3-way tonal interaction (lexical tone + evidential suffix tone + mirative floating H) from examples; produce 4 surface forms. Score=accuracy*efficiency.",
)
def gwelthar_mirative_evidential_tone_lang_learning(llm) -> float:
    """Infer GWELTHAR 3-system tone interaction (lex+evid+mirative floating-H spreading) from examples; produce 4 surface forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying GWELTHAR verbal morphology.",
        "GWELTHAR words carry tones marked on vowels: á=H(high), à=L(low).",
        "Three tonal systems interact:",
        "  1. LEXICAL TONE: each root has an inherent H/L melody.",
        "  2. EVIDENTIAL SUFFIX TONE: suffixes add their own tones after the root.",
        "  3. MIRATIVE: when information is surprising, a floating H docks on the PENULTIMATE syllable.",
        "     If that syllable is already H, the floating H SPREADS to the final syllable instead.",
        "",
        "Labeled examples (root + evidential + mirative → surface word):",
    ]
    for i in range(INITIAL_EXAMPLES):
        r, e, m = _ALL_INPUTS[i]
        mir_str = "MIR" if m else "non-MIR"
        lines.append(f"  Example {i+1}: root='{r}', {e}, {mir_str} → {_ALL_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover all 3 tonal systems and their interaction.",
        "Scoring note: Getting each examination marked form right (mirative, evidential, tone) matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in mirative, evidential, and tone interaction, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("gwelthar_tone"):
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
                    r, e, m = _ALL_INPUTS[idx]
                    mir_str = "MIR" if m else "non-MIR"
                    ex = f"Example {idx+1}: root='{r}', {e}, {mir_str} → {_ALL_OUTPUTS[idx]}"
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
            "EXAMINATION — Produce the correct GWELTHAR surface form for each specification.",
            "Apply all three tonal systems in the correct order. Provide all 4 answers at once.",
            "",
        ]
        for i, (r, e, m) in enumerate(_TEST_COMBOS):
            exam_lines.append(f"  Item {i+1}: {_fmt_spec(r,e,m)}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": _fmt_spec(*_TEST_COMBOS[i]),
                "expected": _TEST_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("GWELTHAR MIRATIVE×EVIDENTIAL×TONE", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    gwelthar_mirative_evidential_tone_lang_learning.run(kbench.llm)

