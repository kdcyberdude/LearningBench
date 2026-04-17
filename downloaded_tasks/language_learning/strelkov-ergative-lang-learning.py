#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "STRELKOV tests enhanced split ergativity with antipassive voice. "
    "PAST tense uses ergative/absolutive alignment; PRESENT uses nominative/accusative. "
    "ANTIPASSIVE: the transitive verb can demote the object and promote the Agent to absolutive, "
    "using a dedicated antipassive suffix -dron. When antipassive, the demoted object is in oblique case -vel. "
    "The SPLIT in ergativity interacts with voice: antipassive in PAST loses the ergative prefix, "
    "while antipassive in PRESENT just changes to nominative. "
    "Model must discover case system × voice × tense interaction. Score = accuracy * efficiency."
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

_NOUNS_ST = {
    "hunter": "skrelvo", "trader": "brantal", "warrior": "drokveth",
    "scout": "felpok", "elder": "skolven",
}
_VERBS_ST = {"see": "van", "chase": "drol", "find": "preth", "carry": "trel"}
_TENSE_SFX = {"past": "-dreth", "pres": "-olan"}
_CASE = {
    "ERG": "-ak", "ABS": "", "NOM": "", "ACC": "-on", "OBL": "-vel",
}


def _strelkov_sentence(agent, patient, verb, tense, antipassive):
    v = _VERBS_ST[verb]
    t = _TENSE_SFX[tense]
    if antipassive:
        v_form = v + "-dron" + t
        if tense == "past":
            a_np = _NOUNS_ST[agent] + _CASE["ABS"]
        else:
            a_np = _NOUNS_ST[agent] + _CASE["NOM"]
        o_np = _NOUNS_ST[patient] + _CASE["OBL"]
        return f"{a_np} {o_np} {v_form}"
    else:
        v_form = v + t
        if tense == "past":
            a_np = _NOUNS_ST[agent] + _CASE["ERG"]
            o_np = _NOUNS_ST[patient] + _CASE["ABS"]
        else:
            a_np = _NOUNS_ST[agent] + _CASE["NOM"]
            o_np = _NOUNS_ST[patient] + _CASE["ACC"]
        return f"{a_np} {o_np} {v_form}"


_ST_SPECS = [
    ("hunter","trader","see","past",False),
    ("warrior","scout","chase","pres",False),
    ("trader","elder","find","past",True),
    ("scout","hunter","carry","pres",True),
    ("elder","warrior","see","past",False),
    ("hunter","elder","chase","past",False),
    ("trader","hunter","find","pres",False),
    ("warrior","trader","carry","past",True),
    ("scout","elder","see","pres",False),
    ("elder","scout","chase","pres",True),
    ("hunter","warrior","find","past",False),
    ("trader","scout","carry","pres",False),
    ("warrior","hunter","see","past",True),
    ("elder","trader","chase","past",False),
    ("scout","warrior","find","pres",True),
    ("hunter","trader","carry","pres",False),
]
_ALL_OUTPUTS_ST = [_strelkov_sentence(*s) for s in _ST_SPECS]

MAX_EXAMPLES = 14
INITIAL_EXAMPLES = 4

_TEST_ST_SPECS = [
    ("warrior","elder","see","past",True),
    ("trader","hunter","chase","pres",False),
    ("scout","warrior","find","past",False),
    ("elder","trader","carry","pres",True),
]
_TEST_EXPECTED_ST = [_strelkov_sentence(*s) for s in _TEST_ST_SPECS]


def _fmt_st(s):
    a,p,v,t,ap = s
    return f"agent={a}, patient={p}, verb={v}, tense={t}, antipassive={'yes' if ap else 'no'}"


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
    name="strelkov_ergative_lang_learning",
    description="Learn STRELKOV split ergativity (PAST=erg/abs, PRES=nom/acc) + antipassive voice (-dron, demotes O to OBL, promotes A to ABS/NOM) from examples; produce 4 sentences. Score=accuracy*efficiency.",
)
def strelkov_ergative_lang_learning(llm) -> float:
    """Infer STRELKOV split ergative case system and antipassive voice interaction from examples; produce 4 sentences. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying STRELKOV sentence grammar.",
        "STRELKOV uses SPLIT ERGATIVITY by tense:",
        "  PAST:    Agent=ERG(-ak), Patient=ABS(∅)",
        "  PRESENT: Agent=NOM(∅),   Patient=ACC(-on)",
        "",
        "ANTIPASSIVE VOICE: suffix -dron on verb demotes object to OBLIQUE(-vel):",
        "  PAST + ANTIPASSIVE:    Agent=ABS(∅), demoted-Patient=OBL(-vel)",
        "  PRESENT + ANTIPASSIVE: Agent=NOM(∅), demoted-Patient=OBL(-vel)",
        "",
        "Verb form: ROOT + [-dron if AP] + TENSE-SUFFIX (-dreth=past, -olan=pres)",
        "",
        "Labeled examples (specification → STRELKOV sentence):",
    ]
    for i in range(INITIAL_EXAMPLES):
        lines.append(f"  Example {i+1}: [{_fmt_st(_ST_SPECS[i])}] → {_ALL_OUTPUTS_ST[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}.",
        "Scoring note: Getting each examination sentence right (case, voice, tense) matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in split ergativity, voice, and tense, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("strelkov_erg"):
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
                    ex = f"Example {idx+1}: [{_fmt_st(_ST_SPECS[idx])}] → {_ALL_OUTPUTS_ST[idx]}"
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
            "EXAMINATION — Produce the correct STRELKOV sentence for each specification.",
            "Apply the tense-split ergativity and antipassive rules correctly.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, spec in enumerate(_TEST_ST_SPECS):
            exam_lines.append(f"  Item {i+1}: {_fmt_st(spec)}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": _fmt_st(_TEST_ST_SPECS[i]),
                "expected": _TEST_EXPECTED_ST[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_ST[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("STRELKOV SPLIT ERGATIVITY + ANTIPASSIVE", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    strelkov_ergative_lang_learning.run(kbench.llm)

