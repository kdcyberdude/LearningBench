#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "VRELTHAN tests phonological rule-interaction OPACITY. "
    "Three rules apply in fixed order: R1 Nasal Place Assimilation (nasal copies place of following obstruent), "
    "R2 Obstruent Voicing (obstruent voices between sonorants), "
    "R3 Nasal Deletion Before Voiced Obstruent (nasal deletes before voiced obstruent). "
    "R1 feeds R3 (assimilation creates ND context), R2 feeds R3 (voicing creates ND context), "
    "but the surface form is opaque: you can't see the intermediate stages. "
    "Model must infer all three rules AND their strict ordering from examples, "
    "then produce 4 surface forms in examination. Score = accuracy * efficiency."
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

_NASALS = {"m": "bilabial", "n": "alveolar", "ng": "velar"}
_VOICED_OBS = set("bdgvz")
_VOICELESS_OBS = {"p": "b", "t": "d", "k": "g", "f": "v", "s": "z"}
_SONORANTS = set("mnlrwy") | set("ng")


def _tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        if s[i:i+2] in ("ng", "th", "sk", "st", "sp"):
            tokens.append(s[i:i+2])
            i += 2
        else:
            tokens.append(s[i])
            i += 1
    return tokens


def _is_sonorant(t):
    return t in _SONORANTS or t[0] in "aeioulrmn"


def _is_nasal(t):
    return t in ("m", "n", "ng")


def _is_obstruent(t):
    return t in _VOICELESS_OBS or t in _VOICED_OBS


def _nasal_place(nasal, following_obs):
    if following_obs in ("p", "b"):
        return "m"
    if following_obs in ("k", "g"):
        return "ng"
    return "n"


def _apply_r1(tokens):
    out = list(tokens)
    for i in range(len(out) - 1):
        if _is_nasal(out[i]) and _is_obstruent(out[i+1]):
            out[i] = _nasal_place(out[i], out[i+1])
    return out


def _apply_r2(tokens):
    out = list(tokens)
    for i in range(len(out)):
        if out[i] in _VOICELESS_OBS:
            left = i > 0 and _is_sonorant(out[i-1])
            right = i < len(out)-1 and _is_sonorant(out[i+1])
            if left and right:
                out[i] = _VOICELESS_OBS[out[i]]
    return out


def _apply_r3(tokens):
    out = []
    i = 0
    while i < len(tokens):
        if _is_nasal(tokens[i]) and i < len(tokens)-1 and tokens[i+1] in _VOICED_OBS:
            i += 1
        else:
            out.append(tokens[i])
            i += 1
    return out


def _derive(underlying):
    t = _tokenize(underlying)
    t = _apply_r1(t)
    t = _apply_r2(t)
    t = _apply_r3(t)
    return "".join(t)


_ALL_INPUTS = [
    "antopa",  "enskalu", "omtiver", "inpakelo", "alskorel",
    "ontravel", "emfokis", "ansoltek", "imburel",  "engravet",
    "unpiskol", "omtelvar", "ankopris", "insoktev", "elmbatru",
    "amteklov", "onfeskal", "entsopir", "angoverp", "umtrelka",
]
_ALL_OUTPUTS = [_derive(w) for w in _ALL_INPUTS]

MAX_EXAMPLES = 16
INITIAL_EXAMPLES = 4

_TEST_INPUTS = ["ankopris", "insoktev", "emfokis", "omtiver"]
_TEST_EXPECTED = [_derive(w) for w in _TEST_INPUTS]


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
    name="vrelthan_rule_interaction_lang_learning",
    description="Learn 3 opaque VRELTHAN phonological rules (NPA→OV→NDV) and their ordering from examples; produce 4 surface forms. Score=accuracy*efficiency.",
)
def vrelthan_rule_interaction_lang_learning(llm) -> float:
    """Infer 3 ordered opaque VRELTHAN phonology rules from examples; surface forms are non-transparent. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying VRELTHAN phonology.",
        "An underlying word form undergoes a sequence of phonological rules to produce the surface form.",
        "The rules may INTERACT: applying one rule can change the environment for another.",
        "",
        "Labeled examples (underlying → surface):",
    ]
    for i in range(INITIAL_EXAMPLES):
        lines.append(f"  Example {i+1}: {_ALL_INPUTS[i]} → {_ALL_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (get more examples, up to {MAX_EXAMPLES} total) | "
        "action='submit' (enter 4-item exam, 1 attempt each, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover the rules and their ordering.",
        "Hint: there are exactly 3 ordered rules. The surface form may not reveal intermediate stages.",
        "Scoring note: Getting each examination surface form right (opaque rule interaction) matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in the interacting phonological rules and their order, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("vrelthan_rules"):
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
                    ex = f"Example {idx+1}: {_ALL_INPUTS[idx]} → {_ALL_OUTPUTS[idx]}"
                    examples_shown += 1
                    turns.append(entry)
                    next_prompt = (f"{ex}\n\nSeen {examples_shown}/{MAX_EXAMPLES}.\n"
                                   "action='request' for more or action='submit' to examine.")
            elif action == "submit":
                turns.append(entry)
                break
            else:
                turns.append(entry)
                next_prompt = "Use action='request' or action='submit'."

        exam_lines = [
            "EXAMINATION — Produce the VRELTHAN surface form for each underlying form.",
            "Apply all rules in the correct order. Provide all 4 answers at once.",
            "",
        ]
        for i, w in enumerate(_TEST_INPUTS):
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
                "item": i+1, "input": _TEST_INPUTS[i],
                "expected": _TEST_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("VRELTHAN RULE INTERACTION", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    vrelthan_rule_interaction_lang_learning.run(kbench.llm)

