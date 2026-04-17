#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "TREVKOVAN tests consonant gradation + morphological trigger grid. "
    "Syllable-initial consonants weaken in 'weak-grade' environments (closed syllables in the inflected form). "
    "Gradation pairs: p/b, t/d, k/g, pp/p, tt/t, kk/k, sk/sg, st/sd, sp/sb. "
    "TRIGGER GRID: different morphological categories trigger different grade environments: "
    "nominative=strong, genitive=weak, partitive=strong, inessive=weak, elative=strong, illative=weak. "
    "The model must discover all gradation pairs AND the trigger grid. Score = accuracy * efficiency."
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

_STRONG_TO_WEAK = {"pp":"p","tt":"t","kk":"k","p":"b","t":"d","k":"g","sk":"sg","st":"sd","sp":"sb"}
_CASES_TV = {
    "nom":  ("", False),  "gen":  ("n", True),
    "par":  ("a", False), "ines": ("ssa", True),
    "ela":  ("sta", False),"ill": ("an", True),
}


def _weaken(root):
    for strong, weak in sorted(_STRONG_TO_WEAK.items(), key=lambda x: -len(x[0])):
        if strong in root:
            return root.replace(strong, weak, 1)
    return root


def _trevkovan_form(root, case):
    suffix, weak = _CASES_TV[case]
    stem = _weaken(root) if weak else root
    return stem + suffix


_ROOTS_TK = [
    "skovel","dratta","brelkko","volppak","prensta","kolvran",
    "felskov","treppel","drontka","skelban","porttrev",
]
_CASES_LIST = list(_CASES_TV.keys())

import itertools as _it_tk
_ALL_TK_SPECS = list(_it_tk.product(_ROOTS_TK, _CASES_LIST))
import random as _rng_tk
_rng_tk_obj = _rng_tk.Random(63)
_rng_tk_obj.shuffle(_ALL_TK_SPECS)
_ALL_TK_OUTPUTS = [_trevkovan_form(r,c) for r,c in _ALL_TK_SPECS]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 6

_TEST_TK_SPECS = [
    ("dratta",  "gen"),
    ("brelkko", "ines"),
    ("prensta", "ill"),
    ("skovel",  "gen"),
]
_TEST_TK_EXPECTED = [_trevkovan_form(r,c) for r,c in _TEST_TK_SPECS]


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
    name="trevkovan_gradation_lang_learning",
    description="Learn TREVKOVAN consonant gradation (9 strong/weak pairs) + morphological trigger grid (nom/par/ela=strong; gen/ines/ill=weak) from examples; produce 4 inflected forms. Score=accuracy*efficiency.",
)
def trevkovan_gradation_lang_learning(llm) -> float:
    """Infer TREVKOVAN consonant gradation pairs and which cases trigger weak vs strong grade; produce 4 inflected forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying TREVKOVAN nominal morphology.",
        "TREVKOVAN consonants WEAKEN in certain morphological contexts (consonant gradation).",
        "Gradation pairs (strong→weak):",
        "  pp→p, tt→t, kk→k (geminate reduction)",
        "  p→b, t→d, k→g (voicing)",
        "  sk→sg, st→sd, sp→sb (cluster voicing)",
        "",
        "TRIGGER GRID — which case triggers WEAK vs STRONG grade:",
        "  STRONG grade: nominative (∅), partitive (-a), elative (-sta)",
        "  WEAK grade:   genitive (-n), inessive (-ssa), illative (-an)",
        "",
        "Form: GRADED-STEM + CASE-SUFFIX",
        "",
        "Labeled examples (root + case → inflected form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        r,c = _ALL_TK_SPECS[i]
        lines.append(f"  Example {i+1}: root='{r}', case={c} → {_ALL_TK_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover gradation pairs and trigger grid.",
        "Scoring note: Getting each examination inflected form right (gradation + case) matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in gradation pairs and which morphological case triggers strong vs weak grade, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("trevkovan_grad"):
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
                    r,c = _ALL_TK_SPECS[idx]
                    ex = f"Example {idx+1}: root='{r}', case={c} → {_ALL_TK_OUTPUTS[idx]}"
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
            "EXAMINATION — Produce the correct inflected TREVKOVAN noun form.",
            "Determine if the case triggers weak or strong grade, then apply the correct gradation.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (r,c) in enumerate(_TEST_TK_SPECS):
            exam_lines.append(f"  Item {i+1}: root='{r}', case={c}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_TK_SPECS[i][0]}.{_TEST_TK_SPECS[i][1]}",
                "expected": _TEST_TK_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_TK_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("TREVKOVAN CONSONANT GRADATION", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    trevkovan_gradation_lang_learning.run(kbench.llm)

