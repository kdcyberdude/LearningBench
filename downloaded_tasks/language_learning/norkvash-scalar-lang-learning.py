#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "NORKVASH tests scalar implicature, entailment, and exhaustivity. "
    "Sentences use scalar expressions (all/most/some/few/none) with gradable predicates. "
    "Each example shows a sentence + its PRAGMATIC READING: "
    "whether 'some' implicates 'not all' (SI), whether negation reverses the scale, "
    "and whether exhaustivity applies (only-interpretation). "
    "The model must learn the scale hierarchy and implicature rules, "
    "then determine the pragmatic reading of 4 novel sentences. "
    "Score = accuracy * efficiency."
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

_SCALE = ["none", "few", "some", "most", "all"]
_SCALE_RANK = {s: i for i, s in enumerate(_SCALE)}
_NORKVASH_Q = {
    "none": "skront",
    "few":  "drelvon",
    "some": "telan",
    "most": "skovra",
    "all":  "velprak",
}
_NEG = "druth"


def _pragmatic_reading(quantifier, negated, context_exhaustive):
    rank = _SCALE_RANK[quantifier]
    if negated:
        neg_rank = len(_SCALE) - 1 - rank
        base = _SCALE[neg_rank]
        reading = f"at-least-{base}"
        if context_exhaustive:
            reading += "+exhaustive(only)"
        return reading
    upper = _SCALE[rank+1] if rank < len(_SCALE)-1 else None
    if upper and quantifier in ("some","few","most"):
        reading = f"{quantifier}(implies not {upper})"
    else:
        reading = quantifier
    if context_exhaustive:
        reading += "+exhaustive(only)"
    return reading


def _build_norkvash_sent(subj, quant, pred, negated, exh):
    q_word = _NORKVASH_Q[quant]
    sentence = f"{subj} {q_word} {pred}"
    if negated:
        sentence = f"{sentence} {_NEG}"
    if exh:
        sentence = "EXH " + sentence
    reading = _pragmatic_reading(quant, negated, exh)
    return sentence, reading


_NORKVASH_SUBJS = ["skolven","drelvak","prantha","kolvra"]
_NORKVASH_PREDS = ["spoke","arrived","understood","failed","passed"]

_NORKVASH_SPECS = [
    ("skolven","some","spoke",False,False),
    ("drelvak","all","arrived",False,False),
    ("prantha","some","understood",False,True),
    ("kolvra","few","failed",False,False),
    ("skolven","most","passed",False,False),
    ("drelvak","some","spoke",True,False),
    ("prantha","all","arrived",True,False),
    ("kolvra","most","understood",False,True),
    ("skolven","none","failed",False,False),
    ("drelvak","few","passed",False,True),
    ("prantha","some","failed",True,True),
    ("kolvra","all","spoke",False,False),
    ("skolven","most","arrived",True,False),
    ("drelvak","none","understood",True,False),
    ("prantha","few","passed",True,True),
    ("kolvra","some","arrived",False,False),
]

_ALL_NORKVASH_DATA = [_build_norkvash_sent(*s) for s in _NORKVASH_SPECS]
_ALL_NORKVASH_SENTS = [d[0] for d in _ALL_NORKVASH_DATA]
_ALL_NORKVASH_READINGS = [d[1] for d in _ALL_NORKVASH_DATA]

MAX_EXAMPLES = 14
INITIAL_EXAMPLES = 4

_TEST_NORKVASH_SPECS = [
    ("skolven","some","passed",True,False),
    ("drelvak","most","spoke",False,True),
    ("prantha","few","arrived",False,False),
    ("kolvra","all","failed",True,False),
]
_TEST_NORKVASH_DATA = [_build_norkvash_sent(*s) for s in _TEST_NORKVASH_SPECS]
_TEST_NORKVASH_SENTS = [d[0] for d in _TEST_NORKVASH_DATA]
_TEST_EXPECTED_NK = [d[1] for d in _TEST_NORKVASH_DATA]


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
    name="norkvash_scalar_lang_learning",
    description="Learn NORKVASH quantifier scale (none<few<some<most<all) and scalar implicature+negation-reversal+exhaustivity rules from examples; predict 4 pragmatic readings. Score=accuracy*efficiency.",
)
def norkvash_scalar_lang_learning(llm) -> float:
    """Infer NORKVASH scalar quantifiers, scale implicature, negation reversal, and exhaustivity from examples; predict 4 pragmatic readings. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying NORKVASH pragmatic semantics.",
        "NORKVASH has a quantifier scale: none < few < some < most < all",
        "NORKVASH quantifier words:",
        "  none=skront, few=drelvon, some=telan, most=skovra, all=velprak",
        "Negation suffix: druth (appended after predicate)",
        "Exhaustivity marker: EXH (sentence-initial, adds 'only' reading)",
        "",
        "Each sentence has a PRAGMATIC READING based on scale implicature:",
        "  'some'  → implies 'not all' (scalar implicature, upper-bound)",
        "  'most'  → implies 'not all' (SI)",
        "  'few'   → implies 'not some/most/all'",
        "  Negation reverses the scale: 'not all' → 'some/few', 'not some' → 'none'",
        "  EXH adds an exhaustivity component (only interpretation)",
        "",
        "Labeled examples (NORKVASH sentence → pragmatic reading):",
    ]
    for i in range(INITIAL_EXAMPLES):
        lines.append(f"  Example {i+1}: '{_ALL_NORKVASH_SENTS[i]}' → {_ALL_NORKVASH_READINGS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover the implicature rules.",
        "Scoring note: Getting each examination scalar judgment right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in scalar particles and implicature, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("norkvash_scalar"):
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
                    ex = f"Example {idx+1}: '{_ALL_NORKVASH_SENTS[idx]}' → {_ALL_NORKVASH_READINGS[idx]}"
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
            "EXAMINATION — Determine the PRAGMATIC READING for each NORKVASH sentence.",
            "State the implicature (e.g. 'some(implies not all)', 'at-least-few+exhaustive(only)', etc.)",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, sent in enumerate(_TEST_NORKVASH_SENTS):
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
                "item": i+1, "input": _TEST_NORKVASH_SENTS[i],
                "expected": _TEST_EXPECTED_NK[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_NK[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("NORKVASH SCALAR IMPLICATURE", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    norkvash_scalar_lang_learning.run(kbench.llm)

