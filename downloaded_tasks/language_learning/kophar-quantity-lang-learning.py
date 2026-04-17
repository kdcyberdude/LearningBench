#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "KOPHAR tests mass/count noun syntax and measure constructions. "
    "KOPHAR nouns are either MASS (no inherent number, require measure words) or COUNT (directly number). "
    "Mass nouns need a MEASURE WORD between the numeral and noun. "
    "Measure words have different shapes for liquids, granular, elongated, flat, and abstract. "
    "Additionally: KOPHAR numerals above 5 are expressed by a base-6 additive system. "
    "Model must discover noun classes AND measure words AND the numeral system. Score = accuracy * efficiency."
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

_NUMS = {1:"ven",2:"dro",3:"sketh",4:"palv",5:"ront",6:"trelvak",7:"trelvak-ven",8:"trelvak-dro",9:"trelvak-sketh",10:"trelvak-palv",11:"trelvak-ront",12:"dro-trelvak"}
_MASS_NOUNS = {
    "water":     ("kolvar", "liquid",    "meth"),
    "rice":      ("prantha", "granular", "skov"),
    "rope":      ("drelvak", "elongated","threp"),
    "cloth":     ("felthar", "flat",     "bren"),
    "knowledge": ("skolven", "abstract", "vel"),
}
_COUNT_NOUNS = {
    "stone":    "felmak",
    "warrior":  "drokveth",
    "tree":     "prethil",
    "basket":   "kolvrak",
}
_MEAS_CLASSIFIERS = {"liquid":"meth","granular":"skov","elongated":"threp","flat":"bren","abstract":"vel"}


def _num_word(n):
    return _NUMS.get(n, str(n))


def _np(noun_en, number):
    if noun_en in _MASS_NOUNS:
        root, mclass, mword = _MASS_NOUNS[noun_en]
        return f"{_num_word(number)} {mword} {root}"
    else:
        return f"{_num_word(number)} {_COUNT_NOUNS[noun_en]}"


_ALL_NP_SPECS = []
for n in list(_MASS_NOUNS.keys()) + list(_COUNT_NOUNS.keys()):
    for num in [1,2,3,4,5,6,7,8]:
        _ALL_NP_SPECS.append((n, num))

import random as _rng6m
_rng6 = _rng6m.Random(13)
_rng6.shuffle(_ALL_NP_SPECS)

# Ensure the initial examples cover all 5 test-relevant nouns so models
# see 'cloth' (flat mass) before the exam asks for it.  We promote one
# 'cloth' example into the first INITIAL_EXAMPLES slots without changing
# the total list order.
_cloth_idx = next(i for i, (n, _) in enumerate(_ALL_NP_SPECS) if n == "cloth")
if _cloth_idx >= 6:   # 6 = INITIAL_EXAMPLES defined below
    _ALL_NP_SPECS.insert(5, _ALL_NP_SPECS.pop(_cloth_idx))
_ALL_OUTPUTS_KP = [_np(n, num) for n, num in _ALL_NP_SPECS]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 6

_TEST_NP_SPECS = [
    ("water",   7),
    ("stone",   9),
    ("cloth",   3),
    ("warrior", 6),
]
_TEST_EXPECTED_KP = [_np(n, num) for n, num in _TEST_NP_SPECS]


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
    name="kophar_quantity_lang_learning",
    description="Learn KOPHAR mass/count noun distinction, measure words (5 classes), and base-6 numerals from examples; produce 4 correct NPs. Score=accuracy*efficiency.",
)
def kophar_quantity_lang_learning(llm) -> float:
    """Infer KOPHAR mass/count syntax, 5-class measure words, and additive base-6 numerals from examples; produce 4 NPs. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying KOPHAR noun phrases.",
        "KOPHAR nouns are either MASS (require a MEASURE WORD) or COUNT (directly combine with numerals).",
        "Structure of a mass noun NP:  NUMERAL + MEASURE-WORD + NOUN",
        "Structure of a count noun NP: NUMERAL + NOUN",
        "",
        "Measure words by material class:",
        "  liquid=meth, granular=skov, elongated=threp, flat=bren, abstract=vel",
        "",
        "KOPHAR numerals use BASE-6: 6='trelvak', 7='trelvak-ven' (6+1), 8='trelvak-dro' (6+2), ...",
        "1=ven, 2=dro, 3=sketh, 4=palv, 5=ront, 6=trelvak, 12=dro-trelvak (2×6)",
        "",
        "Labeled examples (noun + number → KOPHAR NP):",
    ]
    for i in range(INITIAL_EXAMPLES):
        n, num = _ALL_NP_SPECS[i]
        lines.append(f"  Example {i+1}: '{n}'×{num} → {_ALL_OUTPUTS_KP[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover noun classes, measure words, and numeral system.",
        "Scoring note: Getting each examination KOPHAR noun phrase right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in mass/count syntax, measure words, and numerals, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("kophar_quantity"):
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
                    n, num = _ALL_NP_SPECS[idx]
                    ex = f"Example {idx+1}: '{n}'×{num} → {_ALL_OUTPUTS_KP[idx]}"
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
            "EXAMINATION — Produce the correct KOPHAR noun phrase for each specification.",
            "Decide if the noun is mass or count, choose the correct measure word if needed,",
            "and use the correct KOPHAR numeral. Provide all 4 answers at once.",
            "",
        ]
        for i, (n, num) in enumerate(_TEST_NP_SPECS):
            exam_lines.append(f"  Item {i+1}: noun='{n}', number={num}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_NP_SPECS[i][0]}×{_TEST_NP_SPECS[i][1]}",
                "expected": _TEST_EXPECTED_KP[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_KP[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("KOPHAR MASS/COUNT + MEASURE WORDS + BASE-6", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    kophar_quantity_lang_learning.run(kbench.llm)

