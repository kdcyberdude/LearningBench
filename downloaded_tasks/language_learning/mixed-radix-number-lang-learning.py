#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "KROMATH (legacy task id: mixed_radix) tests compositional verb morphology on novel stems: "
    "PAST geminates the first stem letter, PASS prefixes va, INFER appends nk; order is fixed. "
    "No arithmetic — only morphological composition. Score = accuracy × efficiency."
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
FREE_THRESHOLD = 8

_STEM = {
    "rain": "skovr",
    "ash": "tlenp",
    "mist": "vroms",
    "dew": "klef",
    "hail": "brant",
    "frost": "spelg",
    "sleet": "drenk",
}


def _kromath_surface(lex, voice, tense, evid):
    s = _STEM[lex]
    if tense == "PAST":
        s = s[0] + s[0] + s[1:]
    if voice == "PASS":
        s = "va" + s
    if evid == "INFER":
        s = s + "nk"
    return s


def _fmt_spec(lex, voice, tense, evid):
    return f"LEX={lex} VOICE={voice} TENSE={tense} EVID={evid}"


_KROMATH_SPECS = [
    ("hail", "PASS", "NOW", "VIS"),
    ("ash", "ACT", "PAST", "VIS"),
    ("sleet", "PASS", "PAST", "INFER"),
    ("hail", "PASS", "NOW", "INFER"),
    ("rain", "ACT", "NOW", "VIS"),
    ("ash", "PASS", "NOW", "VIS"),
    ("frost", "ACT", "PAST", "INFER"),
    ("rain", "ACT", "PAST", "INFER"),
    ("sleet", "PASS", "NOW", "INFER"),
    ("sleet", "ACT", "NOW", "INFER"),
    ("ash", "ACT", "NOW", "VIS"),
    ("dew", "PASS", "NOW", "VIS"),
    ("dew", "PASS", "PAST", "VIS"),
    ("mist", "ACT", "PAST", "VIS"),
    ("dew", "ACT", "PAST", "INFER"),
    ("hail", "ACT", "NOW", "VIS"),
]

_ALL_OUTPUTS_KR = [_kromath_surface(*t) for t in _KROMATH_SPECS]

MAX_EXAMPLES = 16
INITIAL_EXAMPLES = 8

_TEST_KR = [
    ("rain", "PASS", "PAST", "INFER"),
    ("ash", "PASS", "PAST", "VIS"),
    ("hail", "ACT", "NOW", "INFER"),
    ("frost", "PASS", "PAST", "VIS"),
]
_TEST_EXPECTED_KR = [_kromath_surface(*t) for t in _TEST_KR]


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
    name="mixed_radix_number_lang_learning",
    description="Learn KROMATH morphology (PAST geminate, PASS va-, INFER -nk) from SPE→surface pairs; 4 held-out specs. Score=accuracy*efficiency.",
)
def mixed_radix_number_lang_learning(llm) -> float:
    """Active learning: induce KROMATH exponence order from labeled SPE lines; generalize to 4 novel feature bundles. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are learning KROMATH verb exponence (morphological composition only, not arithmetic).",
        "Each labeled line gives four features — LEX (English weather tag), VOICE (ACT or PASS), "
        "TENSE (NOW or PAST), EVID (VIS or INFER) — and the corresponding KROMATH surface string.",
        "The mapping is compositional and total: every feature bundle has exactly one surface.",
        "Infer how ACT/NOW/VIS, PASS, PAST, and INFER are exponed on the lexeme stem, including the ORDER in which operations apply.",
        "",
        "Labeled examples (SPE → surface):",
    ]
    for i in range(INITIAL_EXAMPLES):
        t = _KROMATH_SPECS[i]
        lines.append(f"  Example {i+1}: {_fmt_spec(*t)} → {_ALL_OUTPUTS_KR[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}.",
        "Scoring note: Getting each examination surface right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in the exponence system, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("kromath_morph"):
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
                    t = _KROMATH_SPECS[idx]
                    ex = f"Example {idx+1}: {_fmt_spec(*t)} → {_ALL_OUTPUTS_KR[idx]}"
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
            "EXAMINATION — Produce the KROMATH surface string for each SPE line.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, t in enumerate(_TEST_KR):
            exam_lines.append(f"  Item {i+1}: {_fmt_spec(*t)}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i + 1,
                "input": _fmt_spec(*_TEST_KR[i]),
                "expected": _TEST_EXPECTED_KR[i],
                "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_KR[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("KROMATH COMPOSITIONAL MORPHOLOGY", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    mixed_radix_number_lang_learning.run(kbench.llm)

