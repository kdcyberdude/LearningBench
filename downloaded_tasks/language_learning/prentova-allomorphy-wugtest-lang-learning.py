#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "PRENTOVA tests opaque allomorphy + wug-test paradigm cell filling. "
    "Nouns belong to one of three declension classes (I/II/III) that are NOT phonologically predictable. "
    "Class I: regular suffixes; Class II: vowel mutation (ablaut) in marked cases + same suffixes; "
    "Class III: suppletive root in marked cases + same suffixes. "
    "Determiners also agree by class. "
    "Model must infer class membership from a partial paradigm, then fill unseen cells. "
    "The hardest items: novel roots with minimal class evidence. Score = accuracy * efficiency."
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

_SUFFIXES = {
    "NOM": "",  "ACC": "on", "GEN": "al", "DAT": "ek",
    "INS": "ur", "LOC": "ith",
}
_MARKED_CASES = {"GEN", "DAT", "INS", "LOC"}

_ABLAUT = {"a": "e", "o": "u", "e": "i", "u": "y", "i": "ei"}


def _apply_ablaut(root):
    chars = list(root)
    for i in range(len(chars)-1, -1, -1):
        if chars[i] in _ABLAUT:
            chars[i] = _ABLAUT[chars[i]]
            return "".join(chars)
    return root


def _inflect(root, noun_class, case, suppletive_root=None):
    suf = _SUFFIXES[case]
    if noun_class == "I":
        return root + suf
    elif noun_class == "II":
        stem = _apply_ablaut(root) if case in _MARKED_CASES else root
        return stem + suf
    elif noun_class == "III":
        stem = (suppletive_root if suppletive_root else root) if case in _MARKED_CASES else root
        return stem + suf
    return root + suf


_NOUNS = {
    "skovel":  ("I",   None),
    "dranta":  ("I",   None),
    "brelko":  ("II",  None),
    "volmak":  ("II",  None),
    "thovral": ("III", "skren"),
    "felpok":  ("III", "dravon"),
}

_CASES = list(_SUFFIXES.keys())


def _all_forms(noun_key):
    root, (cls, sup) = noun_key, _NOUNS[noun_key]
    return {c: _inflect(root, cls, c, sup) for c in _CASES}


_ALL_SPECS = []
for noun in _NOUNS:
    for case in _CASES:
        _ALL_SPECS.append((noun, case))

import random as _random
_rng = _random.Random(55)
_rng.shuffle(_ALL_SPECS)

_ALL_OUTPUTS = [_inflect(n, _NOUNS[n][0], c, _NOUNS[n][1]) for n, c in _ALL_SPECS]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 6

_TEST_SPECS = [
    ("brelko",  "DAT"),
    ("thovral", "GEN"),
    ("skovel",  "INS"),
    ("felpok",  "LOC"),
]
_TEST_EXPECTED = [_inflect(n, _NOUNS[n][0], c, _NOUNS[n][1]) for n, c in _TEST_SPECS]


def _fmt_spec(noun, case):
    return f"noun='{noun}', case={case}"


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
    name="prentova_allomorphy_wugtest_lang_learning",
    description="Learn PRENTOVA 3-class noun declension (regular/ablaut/suppletive) from examples; infer class and fill 4 unseen paradigm cells. Score=accuracy*efficiency.",
)
def prentova_allomorphy_wugtest_lang_learning(llm) -> float:
    """Infer PRENTOVA 3-class unpredictable allomorphy (I=regular, II=ablaut, III=suppletive) from examples; fill 4 paradigm cells. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying PRENTOVA noun declension.",
        "PRENTOVA nouns decline for 6 cases: NOM, ACC, GEN, DAT, INS, LOC.",
        "There are THREE declension classes — class membership is NOT phonologically predictable:",
        "  Class I:   regular suffix added to unchanged root",
        "  Class II:  root's last vowel MUTATES (ablaut) in GEN/DAT/INS/LOC cases",
        "  Class III: root is REPLACED by a completely different (suppletive) stem in GEN/DAT/INS/LOC",
        "All classes use the same case suffixes: NOM=∅, ACC=-on, GEN=-al, DAT=-ek, INS=-ur, LOC=-ith",
        "",
        "Labeled examples (noun + case → inflected form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        noun, case = _ALL_SPECS[i]
        lines.append(f"  Example {i+1}: {noun}+{case} → {_ALL_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Infer each noun's class, then predict unseen cells.",
        "Scoring note: Getting each examination surface form right (including novel roots) matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in allomorphy and wug-test generalization, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("prentova_allomorphy"):
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
                    noun, case = _ALL_SPECS[idx]
                    ex = f"Example {idx+1}: {noun}+{case} → {_ALL_OUTPUTS[idx]}"
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
            "EXAMINATION — Produce the correct inflected form for each noun+case combination.",
            "First infer the noun's class from previously seen forms, then apply the correct rule.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (noun, case) in enumerate(_TEST_SPECS):
            exam_lines.append(f"  Item {i+1}: {_fmt_spec(noun, case)}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": _fmt_spec(*_TEST_SPECS[i]),
                "expected": _TEST_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("PRENTOVA ALLOMORPHY WUG-TEST", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    prentova_allomorphy_wugtest_lang_learning.run(kbench.llm)

