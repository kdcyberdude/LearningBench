#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "SKOVAR tests deletion + epenthesis interaction (cluster repair). "
    "SKOVAR disallows: (1) word-final clusters CC, (2) word-initial clusters CCC. "
    "REPAIR RULES: "
    "  Final CC → delete C2 UNLESS C2 is a sonorant, in which case insert vowel 'e' before C2. "
    "  Initial CCC → delete C2 (middle of three). "
    "  Additionally: if deletion produces a word-initial vowel, a glottal stop ʔ is prefixed. "
    "These rules interact: deletion can feed epenthesis and vice versa. "
    "Model must discover both repair rules and their interaction order. Score = accuracy * efficiency."
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

_VOWELS_SV = set("aeiou")
_SONORANTS_SV = set("mnlrwy")


def _is_vowel_sv(c):
    return c.lower() in _VOWELS_SV


def _is_consonant_sv(c):
    return c.isalpha() and not _is_vowel_sv(c)


def _is_sonorant_sv(c):
    return c.lower() in _SONORANTS_SV


def _apply_final_cluster_repair(word):
    i = len(word) - 1
    consonants_at_end = []
    while i >= 0 and _is_consonant_sv(word[i]):
        consonants_at_end.insert(0, (i, word[i]))
        i -= 1
    if len(consonants_at_end) >= 2:
        c1_idx, c1 = consonants_at_end[-2]
        c2_idx, c2 = consonants_at_end[-1]
        if _is_sonorant_sv(c2):
            return word[:c2_idx] + "e" + word[c2_idx:]
        else:
            return word[:c2_idx]
    return word


def _apply_initial_cluster_repair(word):
    if len(word) >= 3 and _is_consonant_sv(word[0]) and _is_consonant_sv(word[1]) and _is_consonant_sv(word[2]):
        return word[0] + word[2:]
    return word


def _apply_glottal_stop(word):
    if word and _is_vowel_sv(word[0]):
        return "ʔ" + word
    return word


def _skovar_derive(underlying):
    w = underlying
    w = _apply_initial_cluster_repair(w)
    w = _apply_final_cluster_repair(w)
    w = _apply_glottal_stop(w)
    return w


_SKOVAR_UNDERLYINGS = [
    "skrelvant", "drobnask", "trelvomk", "skelkrant", "brelspand",
    "krondelf", "splotrak", "drenvolsk", "brentask", "skranvelt",
    "plotkram", "drontelm", "skrevland", "brontvelk", "sprendolm",
    "kravteln", "dreskmont", "splotven", "brenktral", "skolvrand",
]
_ALL_SKOVAR_OUTPUTS = [_skovar_derive(w) for w in _SKOVAR_UNDERLYINGS]

MAX_EXAMPLES = 16
INITIAL_EXAMPLES = 4

_TEST_SKOVAR = ["skolvrank","drentspam","plotmren","bresktalv"]
_TEST_EXPECTED_SV = [_skovar_derive(w) for w in _TEST_SKOVAR]


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
    name="skovar_deletion_lang_learning",
    description="Learn SKOVAR cluster repair (final-CC deletion/epenthesis + initial-CCC deletion + glottal stop insertion) from examples; produce 4 surface forms. Score=accuracy*efficiency.",
)
def skovar_deletion_lang_learning(llm) -> float:
    """Infer SKOVAR cluster repair rules (deletion+epenthesis+glottal stop) and their interaction order from examples; produce 4 surface forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying SKOVAR phonological cluster repair.",
        "SKOVAR has two phonotactic constraints and repair rules:",
        "  CONSTRAINT 1 — No word-final consonant clusters (CC at end):",
        "    If C2 is a SONORANT (m/n/l/r/w/y): insert 'e' between C1 and C2",
        "    Otherwise: DELETE C2",
        "  CONSTRAINT 2 — No word-initial triple clusters (CCC at start):",
        "    DELETE the MIDDLE consonant (C2 of CCC)",
        "  CONSTRAINT 3 — No word-initial vowels:",
        "    If repair creates a word-initial vowel, prefix glottal stop ʔ",
        "",
        "Labeled examples (underlying form → surface form):",
    ]
    for i in range(INITIAL_EXAMPLES):
        lines.append(f"  Example {i+1}: {_SKOVAR_UNDERLYINGS[i]} → {_ALL_SKOVAR_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover all 3 repair rules and their order.",
        "Scoring note: Getting each examination repaired surface form right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in cluster repair and rule order, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("skovar_deletion"):
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
                    ex = f"Example {idx+1}: {_SKOVAR_UNDERLYINGS[idx]} → {_ALL_SKOVAR_OUTPUTS[idx]}"
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
            "EXAMINATION — Produce the correct SKOVAR surface form for each underlying form.",
            "Apply the repair rules in the correct order. Provide all 4 answers at once.",
            "",
        ]
        for i, w in enumerate(_TEST_SKOVAR):
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
                "item": i+1, "input": _TEST_SKOVAR[i],
                "expected": _TEST_EXPECTED_SV[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_SV[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("SKOVAR CLUSTER REPAIR", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    skovar_deletion_lang_learning.run(kbench.llm)

