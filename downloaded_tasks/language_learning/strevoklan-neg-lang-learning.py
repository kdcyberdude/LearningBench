
from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "STREVOKLAN tests negation + polarity-sensitive items + neg-raising. "
    "Negative sentences use the particle 'drul'; positive use 'vel'. "
    "POLARITY ITEMS: 'ever' (skov) only in negative; 'already' (preth) only in positive; "
    "'any' (drelk) only in negative; 'still' (vanko) in both but with semantic shift. "
    "NEG-RAISING: with attitude verbs (think/believe), sentential negation appears in the matrix "
    "but is interpreted in the embedded clause. "
    "Model must discover which items are positive/negative polarity and when neg-raising applies. Score = accuracy * efficiency."
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

_NEG = "drul"
_POS = "vel"
_SUBJ = {"hunter":"skrelvo","trader":"brantal","warrior":"drokveth","elder":"skolven"}
_VERBS_NR = {"see":"vanrel","know":"farvel","think":"drevol","believe":"skanto","run":"skopal","leave":"tolak"}
_POLARITY = {"ever":"skov","already":"preth","any":"drelk","still":"vanko"}
_ATTITUDE_VERBS = {"think","believe"}


def _strevoklan_sent(subj, verb, neg, polarity_item, embedded_verb=None):
    s = _SUBJ[subj]
    v = _VERBS_NR[verb]
    pol_word = _POLARITY.get(polarity_item, "") if polarity_item else ""
    particle = _NEG if neg else _POS
    if embedded_verb and verb in _ATTITUDE_VERBS:
        ev = _VERBS_NR[embedded_verb]
        if neg and polarity_item:
            return f"{s} {particle} {v} [{ev} {pol_word}]"
        return f"{s} {particle} {v} [{ev}]"
    if polarity_item:
        return f"{s} {pol_word} {particle} {v}"
    return f"{s} {particle} {v}"


def _strevoklan_reading(subj, verb, neg, polarity_item, embedded_verb):
    if verb in _ATTITUDE_VERBS and neg:
        return f"neg-raised: '{verb}' matrix-neg interpreted in embedded clause"
    if polarity_item == "ever" and not neg:
        return "UNGRAMMATICAL: 'ever' requires negative context"
    if polarity_item == "already" and neg:
        return "UNGRAMMATICAL: 'already' requires positive context"
    if polarity_item == "any" and not neg:
        return "UNGRAMMATICAL: 'any' requires negative context"
    if polarity_item == "still":
        return "unexpected continuation" if neg else "expected continuation"
    if neg:
        return "simple negation"
    return "positive assertion"


_STREV_SPECS = [
    ("hunter","see",True,"ever",None),
    ("trader","know",False,"already",None),
    ("warrior","think",True,None,"leave"),
    ("elder","believe",True,None,"run"),
    ("hunter","run",False,"still",None),
    ("trader","see",True,"any",None),
    ("warrior","know",False,None,None),
    ("elder","think",True,None,"see"),
    ("hunter","believe",False,None,"leave"),
    ("trader","run",True,"still",None),
    ("warrior","see",False,"already",None),
    ("elder","know",True,"any",None),
    ("hunter","think",False,None,"run"),
    ("trader","believe",True,None,"see"),
    ("warrior","run",True,"ever",None),
    ("elder","see",False,None,None),
]

_ALL_STREV_SENTS = [_strevoklan_sent(*s) for s in _STREV_SPECS]
_ALL_STREV_READINGS = [_strevoklan_reading(*s) for s in _STREV_SPECS]

MAX_EXAMPLES = 14
INITIAL_EXAMPLES = 4

_TEST_STREV_SPECS = [
    ("elder","believe",False,None,"run"),
    ("elder","see",True,"any",None),
    ("warrior","run",False,"still",None),
    ("trader","believe",True,None,"run"),
]
_TEST_STREV_SENTS = [_strevoklan_sent(*s) for s in _TEST_STREV_SPECS]
_TEST_EXPECTED_NR = [_strevoklan_reading(*s) for s in _TEST_STREV_SPECS]


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
    name="strevoklan_neg_lang_learning",
    description="Learn STREVOKLAN negative polarity items (ever/any=NPI, already=PPI) and neg-raising with attitude verbs from examples; predict 4 interpretations. Score=accuracy*efficiency.",
)
def strevoklan_neg_lang_learning(llm) -> float:
    """Infer STREVOKLAN polarity items and neg-raising from examples; predict pragmatic reading of 4 sentences. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying STREVOKLAN negation and polarity.",
        "STREVOKLAN uses particles for polarity: vel=positive, drul=negative.",
        "",
        "POLARITY-SENSITIVE ITEMS (must appear in the correct context):",
        "  skov ('ever'): NEGATIVE POLARITY ITEM — only grammatical in negative contexts",
        "  preth ('already'): POSITIVE POLARITY ITEM — only grammatical in positive contexts",
        "  drelk ('any'): NEGATIVE POLARITY ITEM — only grammatical in negative contexts",
        "  vanko ('still'): context-sensitive — in negative='unexpected continuation'; positive='expected continuation'",
        "",
        "NEG-RAISING: with 'drevol' (think) or 'skanto' (believe), matrix negation is INTERPRETED",
        "  in the EMBEDDED clause (not the matrix clause).",
        "",
        "Labeled examples (sentence → interpretation):",
    ]
    for i in range(INITIAL_EXAMPLES):
        lines.append(f"  Example {i+1}: '{_ALL_STREV_SENTS[i]}' → {_ALL_STREV_READINGS[i]}")
    lines += [
        "",
        f"Actions: action='request' (up to {MAX_EXAMPLES}) | action='submit' (4-item exam, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}.",
        "Scoring note: Getting each examination interpretation label right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in polarity items and neg-raising, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("strevoklan_neg"):
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
                    ex = f"Example {idx+1}: '{_ALL_STREV_SENTS[idx]}' → {_ALL_STREV_READINGS[idx]}"
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
            "EXAMINATION — Determine the pragmatic/semantic interpretation of each STREVOKLAN sentence.",
            "State: UNGRAMMATICAL (if polarity mismatch), neg-raised interpretation, or simple negation/assertion.",
            "Provide all 4 answers at once.",
            "",
        ]
        for i, sent in enumerate(_TEST_STREV_SENTS):
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
                "item": i+1, "input": _TEST_STREV_SENTS[i],
                "expected": _TEST_EXPECTED_NR[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED_NR[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("STREVOKLAN NEGATION + POLARITY + NEG-RAISING", turns, exam_results, final_score,
               examples_shown, exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    strevoklan_neg_lang_learning.run(kbench.llm)

