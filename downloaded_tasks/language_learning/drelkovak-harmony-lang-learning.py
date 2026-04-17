#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "DRELKOVAK tests multi-dimensional harmony: two independent systems apply simultaneously. "
    "SYSTEM A (Vowel Harmony): roots are [+back] (a/o/u) or [-back] (e/i/ü/ö); "
    "suffix neutral vowels ə→a/e and ɨ→u/i depending on root class. "
    "SYSTEM B (Pharyngeal Consonant Harmony): roots with ħ or ʕ trigger suffix consonant pharyngealization "
    "(s→sˤ, t→tˤ, n→nˤ, l→lˤ, r→rˤ). "
    "Both systems apply simultaneously; model must discover both from examples. "
    "Hardest items: pharyngeal roots that are [-back] (cross-system). Score = accuracy * efficiency."
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

_BACK_VOWELS = set("aouAOU")
_PHARYNGEALS = {"ħ", "ʕ"}
_PHARY_MAP = {"s":"sˤ","t":"tˤ","n":"nˤ","l":"lˤ","r":"rˤ","k":"kˤ","p":"pˤ","d":"dˤ"}


def _root_is_back(root):
    vowels = [c for c in root if c.lower() in "aoueiüö"]
    if not vowels:
        return True
    back = sum(1 for v in vowels if v.lower() in "aou")
    return back > len(vowels) / 2


def _root_has_pharyngeal(root):
    return any(c in _PHARYNGEALS for c in root)


def _apply_vowel_harmony(suffix, back):
    result = []
    for c in suffix:
        if c == "ə":
            result.append("a" if back else "e")
        elif c == "ɨ":
            result.append("u" if back else "i")
        else:
            result.append(c)
    return "".join(result)


def _apply_pharyngeal(suffix):
    result = []
    i = 0
    while i < len(suffix):
        if i + 1 < len(suffix) and suffix[i+1] == "ˤ":
            result.append(suffix[i])
            result.append("ˤ")
            i += 2
        elif suffix[i] in _PHARY_MAP:
            result.append(_PHARY_MAP[suffix[i]])
            i += 1
        else:
            result.append(suffix[i])
            i += 1
    return "".join(result)


def _derive(root, suffix_template):
    back = _root_is_back(root)
    phary = _root_has_pharyngeal(root)
    suffix = _apply_vowel_harmony(suffix_template, back)
    if phary:
        suffix = _apply_pharyngeal(suffix)
    return root + suffix


_ROOTS_BACK    = ["skolva","drovak","turkol","bolgar","pronkal"]
_ROOTS_FRONT   = ["skrivel","belkin","trevil","fesnöm","glepki"]
_ROOTS_B_PHARY = ["skolħa","droħak","turʕol"]
_ROOTS_F_PHARY = ["belħin","treʕil","skrɨʕel"]

_SUFFIXES = [
    ("nəl",  "AGENT.NOM"),
    ("ɨtə",  "PAST.TR"),
    ("sənə", "PRES.PROG"),
    ("rɨk",  "PL"),
    ("tərn", "ACC"),
]

import itertools as _it3
import random as _rng3m
_rng3 = _rng3m.Random(31)
all_roots3 = _ROOTS_BACK + _ROOTS_FRONT + _ROOTS_B_PHARY + _ROOTS_F_PHARY
_combos3 = list(_it3.product(all_roots3, _SUFFIXES))
_rng3.shuffle(_combos3)
_selected3 = _combos3[:24]

_ALL_INPUTS3  = [(r, suf, gloss) for r, (suf, gloss) in _selected3]
_ALL_OUTPUTS3 = [_derive(r, suf) for r, (suf, gloss) in _selected3]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 5

_TEST_ITEMS3 = [
    ("turʕol",  "nəl",  "AGENT.NOM"),
    ("fesnöm",  "ɨtə",  "PAST.TR"),
    ("bolgar",  "sənə", "PRES.PROG"),
    ("belħin",  "rɨk",  "PL"),
]
_TEST_EXPECTED3 = [_derive(r, suf) for r, suf, _ in _TEST_ITEMS3]


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
    name="drelkovak_harmony_lang_learning",
    description="Learn DRELKOVAK two-system harmony (vowel harmony ə/ɨ + pharyngeal consonant harmony ħ/ʕ→suffix pharyngealization) from examples; produce 4 forms. Score=accuracy*efficiency.",
)
def drelkovak_harmony_lang_learning(llm) -> float:
    """Infer DRELKOVAK vowel+pharyngeal harmony simultaneously from examples; suffix neutral vowels and pharyngealization. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying DRELKOVAK morphophonology.",
        "Suffix templates use NEUTRAL vowels ə and ɨ that realize differently by root class.",
        "TWO harmony systems apply simultaneously:",
        "  SYSTEM A — Vowel Harmony: roots with a/o/u are [+back]; roots with e/i/ü/ö are [-back].",
        "    [+back] root: ə→a, ɨ→u   |   [-back] root: ə→e, ɨ→i",
        "  SYSTEM B — Pharyngeal Harmony: roots with ħ or ʕ cause suffix consonants to pharyngealize.",
        "    s→sˤ, t→tˤ, n→nˤ, l→lˤ, r→rˤ  (ˤ = pharyngealization diacritic)",
        "",
        "Labeled examples (root + suffix-template → surface form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        r, suf, gloss = _ALL_INPUTS3[i]
        lines.append(f"  Example {i+1}: {r} + -{suf}({gloss}) → {_ALL_OUTPUTS3[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover both harmony systems.",
        "Scoring note: Getting each examination harmonized surface form right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in vowel harmony domains and triggers, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("drelkovak_harmony"):
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
                    r, suf, gloss = _ALL_INPUTS3[idx]
                    ex = f"Example {idx+1}: {r} + -{suf}({gloss}) → {_ALL_OUTPUTS3[idx]}"
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
            "EXAMINATION — Produce the correct surface form for each root + suffix-template combination.",
            "Apply both harmony systems simultaneously. Provide all 4 answers at once.",
            "",
        ]
        for i, (r, suf, gloss) in enumerate(_TEST_ITEMS3):
            exam_lines.append(f"  Item {i+1}: root='{r}', suffix-template='-{suf}'({gloss})")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_ITEMS3[i][0]}+-{_TEST_ITEMS3[i][1]}",
                "expected": _TEST_EXPECTED3[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED3[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("DRELKOVAK MULTI-DIMENSIONAL HARMONY", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    drelkovak_harmony_lang_learning.run(kbench.llm)

