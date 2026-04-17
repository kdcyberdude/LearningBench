#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "THRELKAV tests quantifier scope × split ergativity interaction. "
    "Case-marking shifts by tense: PAST=ergative/absolutive, PRES=nominative/accusative, FUT=agentive/deictic. "
    "SCOPE RULE: morphologically ZERO-MARKED cases yield WIDE scope; morphologically MARKED cases yield NARROW scope. "
    "This creates a tense-conditioned scope reversal: the SAME NP can have wide scope in one tense but narrow in another. "
    "Model must discover the 3-way case split AND the zero-marking=wide/overt-marking=narrow scope rule. "
    "Exam items require identifying the scope reading for a given sentence. Score = accuracy * efficiency."
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

_CASE_MARKING = {
    "past":  {"S": ("erg", "-ak"),  "O": ("abs", "")},
    "pres":  {"S": ("nom", ""),     "O": ("acc", "-on")},
    "fut":   {"S": ("agt", "-ev"),  "O": ("del", "-ul")},
}

_ZERO_MARKED_CASES = {"abs", "nom"}


def _scope_of_case(tense, role):
    case_name, marker = _CASE_MARKING[tense][role]
    return "wide" if case_name in _ZERO_MARKED_CASES else "narrow"


_VERBS = {"see": "vanrel", "chase": "drolak", "defeat": "skonvel", "find": "prethak"}
_QUANT_NOUNS = {
    "every hunter":  ("every",   "skrelvo"),
    "some trader":   ("some",    "brantal"),
    "a warrior":     ("indef",   "drokveth"),
    "each scout":    ("each",    "felkran"),
}

_TENSES = ["past", "pres", "fut"]


def _build_sentence(tense, subj_en, obj_en, verb_en):
    s_name, s_marker = _CASE_MARKING[tense]["S"]
    o_name, o_marker = _CASE_MARKING[tense]["O"]
    s_drav = _QUANT_NOUNS[subj_en][1] + s_marker
    o_drav = _QUANT_NOUNS[obj_en][1] + o_marker
    verb = _VERBS[verb_en]
    tense_sfx = {"past": "-dreth", "pres": "-olan", "fut": "-skev"}[tense]
    sent = f"{s_drav} {o_drav} {verb}{tense_sfx}"
    s_scope = _scope_of_case(tense, "S")
    o_scope = _scope_of_case(tense, "O")
    return sent, s_scope, o_scope


_SPECS = [
    ("past", "every hunter",  "some trader",  "see"),
    ("pres", "every hunter",  "some trader",  "see"),
    ("fut",  "every hunter",  "some trader",  "see"),
    ("past", "each scout",    "a warrior",    "chase"),
    ("pres", "each scout",    "a warrior",    "chase"),
    ("fut",  "each scout",    "a warrior",    "chase"),
    ("past", "some trader",   "every hunter", "defeat"),
    ("pres", "some trader",   "every hunter", "defeat"),
    ("fut",  "some trader",   "every hunter", "defeat"),
    ("past", "a warrior",     "each scout",   "find"),
    ("pres", "a warrior",     "each scout",   "find"),
    ("fut",  "a warrior",     "each scout",   "find"),
    ("past", "every hunter",  "each scout",   "defeat"),
    ("pres", "some trader",   "a warrior",    "find"),
    ("fut",  "each scout",    "every hunter", "see"),
    ("past", "a warrior",     "some trader",  "chase"),
    ("pres", "every hunter",  "a warrior",    "chase"),
    ("fut",  "some trader",   "each scout",   "defeat"),
    ("past", "each scout",    "every hunter", "find"),
    ("pres", "a warrior",     "some trader",  "see"),
]


def _spec_to_data(spec):
    tense, subj, obj, verb = spec
    sent, s_scope, o_scope = _build_sentence(tense, subj, obj, verb)
    return sent, s_scope, o_scope


_ALL_OUTPUTS_FULL = [_spec_to_data(s) for s in _SPECS]
_ALL_INPUTS  = [(s[0], s[1], s[2], s[3]) for s in _SPECS]
_ALL_OUTPUTS = [f"subj-scope={d[1]}, obj-scope={d[2]}" for d in _ALL_OUTPUTS_FULL]
_ALL_SENTENCES = [d[0] for d in _ALL_OUTPUTS_FULL]

MAX_EXAMPLES = 16
INITIAL_EXAMPLES = 5

_TEST_SPECS = [
    ("past", "some trader",   "every hunter", "see"),
    ("pres", "each scout",    "a warrior",    "defeat"),
    ("fut",  "every hunter",  "some trader",  "chase"),
    ("pres", "a warrior",     "each scout",   "find"),
]
_TEST_DATA = [_spec_to_data(s) for s in _TEST_SPECS]
_TEST_EXPECTED = [f"subj-scope={d[1]}, obj-scope={d[2]}" for d in _TEST_DATA]
_TEST_SENTENCES = [d[0] for d in _TEST_DATA]


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
    name="threlkav_scope_ergativity_lang_learning",
    description="Learn THRELKAV quantifier scope × split ergativity (zero-marking=wide scope, overt=narrow) across 3 tenses; predict scope readings for 4 sentences. Score=accuracy*efficiency.",
)
def threlkav_scope_ergativity_lang_learning(llm) -> float:
    """Infer THRELKAV tense-conditioned ergativity + scope rule (zero-case=wide, marked=narrow); identify scope of 4 test sentences. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying THRELKAV grammar and quantifier scope.",
        "THRELKAV uses DIFFERENT case systems depending on TENSE:",
        "  PAST:    ergative(-ak) subject,  absolutive(∅) object",
        "  PRESENT: nominative(∅) subject,  accusative(-on) object",
        "  FUTURE:  agentive(-ev) subject,  deictic(-ul) object",
        "",
        "SCOPE RULE: a zero-marked (∅) case NP takes WIDE scope over the sentence;",
        "            an overtly-marked case NP takes NARROW scope.",
        "This creates a tense-conditioned scope reversal for the same quantifier.",
        "",
        "Labeled examples (sentence → scope readings):",
    ]
    for i in range(INITIAL_EXAMPLES):
        lines.append(f"  Example {i+1}: '{_ALL_SENTENCES[i]}' → {_ALL_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover the case system and scope rule.",
        "Scoring note: Getting each examination scope judgment right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in quantifier scope and ergativity marking, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("threlkav_scope"):
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
                    ex = f"Example {idx+1}: '{_ALL_SENTENCES[idx]}' → {_ALL_OUTPUTS[idx]}"
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
            "EXAMINATION — For each THRELKAV sentence, state the scope of the subject and object.",
            "Format: 'subj-scope=wide/narrow, obj-scope=wide/narrow'",
            "",
        ]
        for i, sent in enumerate(_TEST_SENTENCES):
            exam_lines.append(f"  Item {i+1}: '{sent}'")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": _TEST_SENTENCES[i],
                "expected": _TEST_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("THRELKAV SCOPE × ERGATIVITY", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    threlkav_scope_ergativity_lang_learning.run(kbench.llm)

