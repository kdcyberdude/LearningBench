#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "DIMVAL tests morphophonology: R0 suffix vowel harmony (ə/ɨ realize from stem's LAST vowel: "
    "a/o/u→[ə→a, ɨ→u] else [ə→e, ɨ→i]), then R1 syncope, R2 metathesis on listed CC pairs, R3 nasal hardening. "
    "Harmony uses the template BEFORE syncope. Model infers R0–R3 order from examples; one surface string per exam item. "
    "Score = accuracy × efficiency."
)


def _log_trace(task, turns, exam_results, final_score, examples_used,
               exam_prompt="", exam_raw=None):
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {_TASK_DESCRIPTION}")
    print(f"\n{sep}\n  CONVERSATION\n{sep}")
    for t in turns:
        print(f"\n[USER — Turn {t['turn']}]")
        print(t.get("prompt", ""))
        print(f"\n[ASSISTANT — Turn {t['turn']}]")
        print(f"action: {t['action']}")
        response = t.get("response", "")
        print(f"answer: {response if response else '(none)'}")
    print(f"\n{sep}\n  EXAMINATION\n{sep}")
    if exam_prompt:
        print("\n[USER — Exam]")
        print(exam_prompt)
    if exam_raw:
        print("\n[ASSISTANT — Exam]")
        print("\n".join(exam_raw))
    print(f"\n{sep}\n  RESULTS\n{sep}")
    for r in exam_results:
        status = "CORRECT" if r["correct"] else "WRONG  "
        print(f"  Test {r['item']}: {status}   input={r['input']!r}   expected={r['expected']!r}   got={r['answer']!r}")
    correct = sum(1 for r in exam_results if r["correct"])
    print(f"\n  Examples used : {examples_used}/{MAX_EXAMPLES}")
    print(f"  Exam accuracy : {correct}/{len(exam_results)}")
    print(f"  Final score   : {final_score:.4f}")
    print(f"{sep}\n")


NUM_TEST_ITEMS = 4
FREE_THRESHOLD = 3

_META_SWAP = {
    "st": "ts", "sk": "ks", "sp": "ps",
    "ft": "tf", "lk": "kl", "rn": "nr",
    "ln": "nl", "rm": "mr",
}
_OBSTRUENTS = set("ptkbdgfs")


def _is_vowel(c):
    return c in "aeiou"


def _stem_back_domain(stem):
    vowels = [c for c in stem.lower() if c in "aeiou"]
    if not vowels:
        return True
    return vowels[-1] in "aou"


def _harmonize_suffix(stem, template):
    back = _stem_back_domain(stem)
    out = []
    for c in template:
        if c == "ə":
            out.append("a" if back else "e")
        elif c == "ɨ":
            out.append("u" if back else "i")
        else:
            out.append(c)
    return "".join(out)


def _apply_syncope(stem, suffix):
    if not suffix or not _is_vowel(suffix[0]):
        return stem
    chars = list(stem)
    vowel_count = 0
    for i, c in enumerate(chars):
        if _is_vowel(c):
            vowel_count += 1
            if vowel_count == 2:
                del chars[i]
                return "".join(chars)
    return stem


def _apply_metathesis(word):
    if len(word) < 2:
        return word
    tail = word[-2:]
    return word[:-2] + _META_SWAP[tail] if tail in _META_SWAP else word


def _apply_nasal_hardening(word):
    if len(word) >= 2 and word[-1] == "n" and word[-2] in _OBSTRUENTS:
        return word[:-1] + "d"
    return word


def _derive(stem, suffix_template):
    suffix = _harmonize_suffix(stem, suffix_template)
    after_r1 = _apply_syncope(stem, suffix)
    combined = after_r1 + suffix
    after_r2 = _apply_metathesis(combined)
    return _apply_nasal_hardening(after_r2)


_ALL_PAIRS = [
    ("talvet", "ura"),
    ("prandol", "ɨlo"),
    ("talvet", "nak"),
    ("talvet", "ɨnak"),
    ("morkast", "elə"),
    ("bornaf", "nak"),
    ("trelvan", "ən"),
    ("preskol", "rek"),
    ("vrenkat", "ana"),
    ("prandol", "rek"),
    ("preskol", "elə"),
    ("talvet", "onə"),
    ("kostem", "ən"),
    ("kostem", "ona"),
    ("morkast", "iven"),
    ("skeltam", "ɨnak"),
]

_ALL_OUTPUTS = [_derive(s, f) for s, f in _ALL_PAIRS]

MAX_EXAMPLES = 16
INITIAL_EXAMPLES = 3

_TEST_PAIRS = [
    ("brenkol", "ɨven"),
    ("skarveth", "urə"),
    ("plomket", "ən"),
    ("treskaf", "nak"),
]
_TEST_EXPECTED = [_derive(s, f) for s, f in _TEST_PAIRS]


def _concept_score(correct_count, examples_used, max_examples, initial_examples):
    accuracy = correct_count / NUM_TEST_ITEMS
    if accuracy == 0:
        return 0.0
    effective_free = max(initial_examples, FREE_THRESHOLD)
    if max_examples <= effective_free or examples_used <= effective_free:
        efficiency = 1.0
    else:
        paid_used = examples_used - effective_free
        paid_budget = max_examples - effective_free
        efficiency = max(0.0, 1.0 - paid_used / paid_budget)
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
    name="dimval_metathesis_lang_learning",
    description="Learn DIMVAL R0 harmony then R1–R3 phonology from examples; 4 held-out stems. Score=accuracy*efficiency.",
)
def dimval_metathesis_lang_learning(llm) -> float:
    """Infer DIMVAL harmony+syncope+metathesis+hardening from examples; 4 novel stems. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    initial_lines = [
        "You are studying DIMVAL morphophonology. Each item is stem + SUFFIX-TEMPLATE (ə and ɨ are harmonic archiphonemes).",
        "R0 — Vowel harmony: look at the LAST vowel of the stem (left-to-right). If it is a, o, or u, the suffix is [+back] (ə→a, ɨ→u).",
        "        Otherwise the suffix is [-back] (ə→e, ɨ→i). Other suffix letters are unchanged.",
        "R1 — Syncope: if the REALIZED suffix (after R0) begins with a vowel, delete the SECOND vowel of the stem.",
        "R2 — Metathesis: if the whole word ends in a CC cluster listed in the data, swap those two consonants (e.g. sk→ks).",
        "R3 — Nasal hardening: word-final n becomes d after an obstruent (p,t,k,b,d,g,f,s).",
        "Rules apply in order R0 → R1 → R2 → R3 on the string stem+realized_suffix (R1 shortens only the stem).",
        "",
        "Labeled examples (stem + suffix-template → surface form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        stem, suf = _ALL_PAIRS[i]
        initial_lines.append(f"  Example {i+1}: {stem} + -{suf} → {_ALL_OUTPUTS[i]}")
    initial_lines += [
        "",
        "Actions:",
        "  action='request' — LEARN: get one more labeled example (up to "
        f"{MAX_EXAMPLES} total)",
        "  action='submit'  — EXAMINE: enter examination (4 test items, one attempt each, no feedback)",
        "",
        f"You have seen {INITIAL_EXAMPLES} of {MAX_EXAMPLES} available examples.",
        "Discover R0–R3 and their ordering, then enter examination.",
        "",
        "Scoring note: Getting the examination surface forms right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in the ordered phonological rules, then use action='submit'.",
    ]
    next_prompt = "\n".join(initial_lines)

    exam_results = []

    with kbench.chats.new("dimval_metathesis"):
        for turn in range(1, MAX_EXAMPLES + 2):
            current_prompt = next_prompt
            try:
                sub = llm.prompt(current_prompt, schema=_ConceptAction)
            except Exception:
                turns.append({"turn": turn, "action": "PARSE_ERROR",
                              "prompt": current_prompt})
                next_prompt = "Parse error. Use action='request' or action='submit'."
                continue

            action = (sub.action or "").strip().lower()
            entry = {"turn": turn, "action": action, "prompt": current_prompt,
                     "response": (sub.answer or "").strip()}

            if action == "request":
                if examples_shown >= MAX_EXAMPLES:
                    entry["feedback"] = "No more examples. You must submit."
                    turns.append(entry)
                    next_prompt = ("No more examples available. Use action='submit' to begin "
                                   "examination (answer field ignored).")
                else:
                    idx = examples_shown
                    stem, suf = _ALL_PAIRS[idx]
                    ex_line = f"Example {idx+1}: {stem} + -{suf}(template) → {_ALL_OUTPUTS[idx]}"
                    examples_shown += 1
                    remaining = MAX_EXAMPLES - examples_shown
                    entry["feedback"] = f"Showed example {examples_shown}."
                    turns.append(entry)
                    next_prompt = (f"{ex_line}\n\nYou have seen {examples_shown} examples. "
                                   f"{remaining} more available.\n\n"
                                   "action='request' for another or action='submit' to examine.")
            elif action == "submit":
                entry["feedback"] = "Entering examination."
                turns.append(entry)
                break
            else:
                entry["feedback"] = "Unknown action."
                turns.append(entry)
                next_prompt = "Use action='request' or action='submit'."

        exam_lines = [
            "EXAMINATION — Produce the correct DIMVAL surface form for each item (apply R0–R3).",
            "Suffix strings are TEMPLATES (ə, ɨ harmonize); output the full word after all rules.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (stem, suf) in enumerate(_TEST_PAIRS):
            exam_lines.append(f"  Item {i+1}: stem='{stem}', suffix-template='-{suf}'")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]

        for i in range(NUM_TEST_ITEMS):
            answer = (raw_answers[i] or "").strip()
            correct = _surface_equal(_TEST_EXPECTED[i], answer)
            exam_results.append({
                "item": i + 1,
                "input": f"{_TEST_PAIRS[i][0]}+-{_TEST_PAIRS[i][1]}",
                "expected": _TEST_EXPECTED[i],
                "answer": answer,
                "correct": correct,
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("DIMVAL METATHESIS × SYNCOPE", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    dimval_metathesis_lang_learning.run(kbench.llm)

