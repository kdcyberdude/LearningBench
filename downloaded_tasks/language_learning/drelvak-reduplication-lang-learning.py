#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "DRELVAK tests semantic-contrastive reduplication. "
    "Three reduplication types produce DIFFERENT surface forms AND DIFFERENT meanings: "
    "TYPE 1 (full copy STEM-STEM) = distributive 'all kinds of X'; "
    "TYPE 2 (heavy-syllable prefix CVC-STEM) = attenuative 'somewhat X'; "
    "TYPE 3 (light-syllable suffix STEM-CV) = intensifying 'extremely X'. "
    "When stem ends in a nasal: Type 2 geminates the nasal in prefix; Type 3 nasalizes the copied vowel. "
    "Model must discover all 3 types, their forms, AND their meanings. "
    "Exam: given a stem+meaning, produce the surface form. Score = accuracy * efficiency."
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
FREE_THRESHOLD = 4

_NASALS = set("mnŋ")
_NASAL_TILDE = {"a":"ã","e":"ẽ","i":"ĩ","o":"õ","u":"ũ"}
_VOWELS = set("aeiou")


def _first_cvc(stem):
    for i, c in enumerate(stem):
        if c.lower() in _VOWELS:
            onset_start = max(0, i-2)
            coda_end = i+1
            while coda_end < len(stem) and stem[coda_end].lower() not in _VOWELS:
                coda_end += 1
                if coda_end - (i+1) >= 1:
                    break
            return stem[onset_start:coda_end]
    return stem[:3]


def _last_cv(stem):
    for i in range(len(stem)-1, -1, -1):
        if stem[i].lower() in _VOWELS:
            onset = i-1
            while onset >= 0 and stem[onset].lower() not in _VOWELS:
                onset -= 1
            onset += 1
            cv = stem[onset:i+1]
            ends_nasal = bool(stem[i+1:]) and stem[-1] in _NASALS
            return cv, ends_nasal
    return stem[-2:], False


def reduplicate(stem, rtype):
    if rtype == 1:
        return stem + "-" + stem
    if rtype == 2:
        cvc = _first_cvc(stem)
        if len(cvc) >= 1 and cvc[-1] in _NASALS:
            prefix = cvc + cvc[-1]
        else:
            prefix = cvc
        return prefix + "-" + stem
    if rtype == 3:
        cv, ends_nasal = _last_cv(stem)
        if ends_nasal:
            new_cv = "".join(_NASAL_TILDE.get(c, c) if c.lower() in _VOWELS else c for c in cv)
        else:
            new_cv = cv
        return stem + "-" + new_cv
    return stem


_STEMS_DRELVAK = [
    "skovel","pranta","drenkov","felmak","skloven",
    "travan","kolvram","brespan","flektiv","dornak",
]
_RED_SEM = {1:"distributive/all-kinds-of", 2:"attenuative/somewhat", 3:"intensifying/extremely"}

_ALL_SPECS_D = [(st, rt) for st in _STEMS_DRELVAK for rt in [1,2,3]]
_ALL_OUTPUTS_D = [reduplicate(st, rt) for st, rt in _ALL_SPECS_D]

MAX_EXAMPLES = 20
INITIAL_EXAMPLES = 6

_TEST_SPECS_D = [
    ("skovel",  1),
    ("travan",  2),
    ("dornak",  3),
    ("skloven", 3),
]
_TEST_EXPECTED_D = [reduplicate(st, rt) for st, rt in _TEST_SPECS_D]


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
    name="drelvak_reduplication_lang_learning",
    description="Learn DRELVAK 3-type semantic-contrastive reduplication (full/heavy-prefix/light-suffix) with nasal complications; produce 4 forms given stem+meaning. Score=accuracy*efficiency.",
)
def drelvak_reduplication_lang_learning(llm) -> float:
    """Infer DRELVAK 3 reduplication types (form+meaning) and nasal complications from examples; produce forms for given stem+meaning. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying DRELVAK morphology.",
        "DRELVAK has 3 types of reduplication, each with a DISTINCT FORM and DISTINCT MEANING:",
        "  TYPE 1 — FULL COPY (STEM-STEM): distributive, meaning 'all kinds of X'",
        "  TYPE 2 — HEAVY PREFIX (CVC-STEM): attenuative, meaning 'somewhat X'",
        "  TYPE 3 — LIGHT SUFFIX (STEM-CV): intensifying, meaning 'extremely X'",
        "NASAL COMPLICATION: if the stem ends in a nasal consonant (m/n/ng):",
        "  Type 2: the nasal is GEMINATED in the prefix (extra nasal added)",
        "  Type 3: the copied vowel in the suffix is NASALIZED (ã/ẽ/ĩ/õ/ũ)",
        "",
        "Labeled examples (stem + type + meaning → surface form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        st, rt = _ALL_SPECS_D[i]
        lines.append(f"  Example {i+1}: '{st}' + TYPE{rt}({_RED_SEM[rt]}) → {_ALL_OUTPUTS_D[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover all 3 types and nasal behavior.",
        "Scoring note: Getting each examination reduplicated form right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in reduplication shape and meaning, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("drelvak_redupliction"):
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
                    st, rt = _ALL_SPECS_D[idx]
                    ex = f"Example {idx+1}: '{st}' + TYPE{rt}({_RED_SEM[rt]}) → {_ALL_OUTPUTS_D[idx]}"
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
            "EXAMINATION — Produce the correct surface form for each stem + reduplication type.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, (st, rt) in enumerate(_TEST_SPECS_D):
            exam_lines.append(f"  Item {i+1}: stem='{st}', type=TYPE{rt}({_RED_SEM[rt]})")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_SPECS_D[i][0]},TYPE{_TEST_SPECS_D[i][1]}",
                "expected": _TEST_EXPECTED_D[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_D[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("DRELVAK SEMANTIC-CONTRASTIVE REDUPLICATION", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    drelvak_reduplication_lang_learning.run(kbench.llm)

