

from dataclasses import dataclass
import re
import unicodedata
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "DRALVEN tests the acquisition of a two-tier tonal grammar. "
    "Verb roots carry LEXICAL tones (H=acute, L=grave, HL=circumflex, LH=caron). "
    "TAM affixes impose a GRAMMATICAL floating tone that docks onto the final mora. "
    "CLASH RESOLUTION: same-tone clash creates contour (H+H→LH, L+L→HL); "
    "gram-tone dominates lex-tone when different; HL+H→super-H (macron). "
    "Model must infer both lexical melody AND clash-resolution rules from examples. "
    "Score = accuracy(0-4) * efficiency."
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
        print(f"\n[ASSISTANT — Exam]\n" + "\n".join(exam_raw))
    for r in exam_results:
        status = "CORRECT" if r["correct"] else "WRONG  "
        print(f"  Test {r['item']}: {status}  expected={r['expected']!r}  got={r['answer']!r}")
    correct = sum(1 for r in exam_results if r["correct"])
    print(f"\n  Examples used : {examples_used}/{MAX_EXAMPLES}  Exam: {correct}/{len(exam_results)}  Score: {final_score:.4f}\n{sep}\n")


NUM_TEST_ITEMS = 4
FREE_THRESHOLD = 3

_ACUTE = {"a":"á","e":"é","i":"í","o":"ó","u":"ú"}
_GRAVE = {"a":"à","e":"è","i":"ì","o":"ò","u":"ù"}
_CIRC  = {"a":"â","e":"ê","i":"î","o":"ô","u":"û"}
_CARON = {"a":"ǎ","e":"ě","i":"ǐ","o":"ǒ","u":"ǔ"}
_MACRO = {"a":"ā","e":"ē","i":"ī","o":"ō","u":"ū"}

def _mark(v, tone):
    return {"H":_ACUTE,"L":_GRAVE,"HL":_CIRC,"LH":_CARON,"HH":_MACRO}[tone][v]

def _base(c):
    return unicodedata.normalize("NFD", c)[0]

def _last_vowel(s):
    for i in range(len(s)-1,-1,-1):
        if _base(s[i]).lower() in "aeiou":
            return i
    return -1

def _lex_tone_of(c):
    nfd = unicodedata.normalize("NFD", c)
    comb = "".join(x for x in nfd if unicodedata.category(x)=="Mn")
    return {"\u0301":"H","\u0300":"L","\u0302":"HL","\u030c":"LH","\u0304":"HH"}.get(comb)

def _clash(lex, gram):
    if lex == gram:
        return "LH" if gram == "H" else "HL"
    if lex == "HL" and gram == "L":
        return "HL"
    if lex == "HL" and gram == "H":
        return "HH"
    return gram

def _surface(root, gram_tone):
    if gram_tone is None:
        return root
    chars = list(root)
    idx = _last_vowel(root)
    if idx < 0:
        return root
    c = chars[idx]
    base = _base(c)
    lex = _lex_tone_of(c)
    out = _clash(lex, gram_tone) if lex else gram_tone
    chars[idx] = _mark(base, out)
    return "".join(chars)

_TAM_SFXS = {"PRES":"an","PAST":"ov","FUT":"ul","PERF":"em","SUBJ":"ith","IMP":"ak"}
_TAM_GRAM  = {"PRES":None,"PAST":"H","FUT":"L","PERF":"H","SUBJ":"L","IMP":"H"}

_ROOTS = [
    ("trák","H"),("gèl","L"),("prân","HL"),("skǒr","LH"),
    ("flém","H"),("dròv","L"),("kwân","HL"),("blǎs","LH"),
    ("vrenák","H"),("skelòr","L"),("bralân","HL"),("drelvǒk","LH"),
    ("foltém","H"),("grosnòv","L"),("strevân","HL"),("klemǔs","LH"),
]

_ALL_PAIRS = [(root, tam) for root,_ in _ROOTS for tam in list(_TAM_SFXS)]

# Reorder so the FIRST 4 examples each demonstrate a different clash case:
#   1. H-root + PRES  → no gram-tone (baseline, no clash)
#   2. H-root + PAST  → H + H clash → LH (caron)
#   3. L-root + FUT   → L + L clash → HL (circumflex)
#   4. HL-root + IMP  → HL + H clash → HH (macron)
# This ensures models see every meaningful interaction in the initial set.
_PRIORITY_PAIRS = [
    ("trák", "PRES"),   # H root, no gram-tone — baseline
    ("trák", "PAST"),   # H root + H gram → H+H → LH (caron)
    ("gèl",  "FUT"),    # L root + L gram → L+L → HL (circumflex)
    ("prân", "IMP"),    # HL root + H gram → HL+H → HH (macron)
]
_REMAINING_PAIRS = [p for p in _ALL_PAIRS if p not in _PRIORITY_PAIRS]
_ALL_PAIRS = _PRIORITY_PAIRS + _REMAINING_PAIRS
_ALL_OUTPUTS = [_surface(root, _TAM_GRAM[tam]) + _TAM_SFXS[tam] for root, tam in _ALL_PAIRS]

MAX_EXAMPLES = 18
INITIAL_EXAMPLES = 6

_TEST_PAIRS  = [("flém","SUBJ"), ("dròv","PERF"), ("skǒr","SUBJ"), ("kwân","PAST")]
_TEST_EXPECTED = [_surface(r, _TAM_GRAM[t]) + _TAM_SFXS[t] for r,t in _TEST_PAIRS]


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
    name="dralven_tone_sandhi_lang_learning",
    description="Learn DRALVEN's lexical-vs-grammatical tone clash resolution from examples; produce 4 inflected verb forms. Score=accuracy*efficiency.",
)
def dralven_tone_sandhi_lang_learning(llm) -> float:
    """Infer DRALVEN tone clash rules (H+H→LH, L+L→HL, HL+H→macron) from examples; produce 4 exam forms. Score in [0,1]."""
    turns = []
    examples_shown = INITIAL_EXAMPLES

    lines = [
        "You are studying DRALVEN verbal morphology.",
        "Verb roots carry LEXICAL TONES marked on vowels:",
        "  á=H(high) à=L(low) â=HL(falling) ǎ=LH(rising)",
        "TAM suffixes impose a GRAMMATICAL floating tone onto the root's last vowel:",
        "  -an=PRES(no gram-tone) -ov=PAST(gram=H) -ul=FUT(gram=L)",
        "  -em=PERF(gram=H) -ith=SUBJ(gram=L) -ak=IMP(gram=H)",
        "When gram-tone docks, it may INTERACT with the existing lexical tone.",
        "",
        "Labeled examples (root-TAM → surface+suffix):",
    ]
    for i in range(INITIAL_EXAMPLES):
        root, tam = _ALL_PAIRS[i]
        lines.append(f"  Example {i+1}: {root} + {tam} → {_ALL_OUTPUTS[i]}")
    lines += [
        "",
        f"Actions: action='request' (get more examples, up to {MAX_EXAMPLES}) | "
        "action='submit' (enter 4-item examination, no feedback)",
        f"Seen {INITIAL_EXAMPLES}/{MAX_EXAMPLES}. Discover the clash-resolution rule.",
        "Scoring note: Getting each examination tonal surface form right matters most, but your score also depends on how efficiently you learn.",
        "You start with the initial labeled examples above at no efficiency cost; each extra example you request reduces the efficiency part of your score.",
        "Submitting too early and missing every exam item yields a score of 0; asking for far more examples than needed also lowers your score.",
        "Aim for the sweet spot: request only as many additional examples as you need to be confident in tone sandhi across morpheme boundaries, then use action='submit'.",
    ]
    next_prompt = "\n".join(lines)
    exam_results = []

    with kbench.chats.new("dralven_tone"):
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
                    root, tam = _ALL_PAIRS[idx]
                    ex = f"Example {idx+1}: {root} + {tam} → {_ALL_OUTPUTS[idx]}"
                    examples_shown += 1
                    turns.append(entry)
                    next_prompt = (f"{ex}\n\nSeen {examples_shown}/{MAX_EXAMPLES}.\n"
                                   "action='request' for more or action='submit' to examine.")
            elif action == "submit":
                turns.append(entry)
                break
            else:
                turns.append(entry)
                next_prompt = "Unknown action. Use 'request' or 'submit'."

        exam_lines = [
            "EXAMINATION — Produce the correct DRALVEN surface form for each verb.",
            "Include both the (tone-modified) root and the TAM suffix.",
            "",
        ]
        for i, (root, tam) in enumerate(_TEST_PAIRS):
            exam_lines.append(f"  Item {i+1}: root='{root}', TAM={tam}")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for i in range(NUM_TEST_ITEMS):
            ans = (raw_answers[i] or "").strip()
            exam_results.append({
                "item": i+1, "input": f"{_TEST_PAIRS[i][0]}+{_TEST_PAIRS[i][1]}",
                "expected": _TEST_EXPECTED[i], "answer": ans,
                "correct": _surface_equal(_TEST_EXPECTED[i], ans),
            })

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace("DRALVEN TONE SANDHI", turns, exam_results, final_score, examples_shown,
               exam_prompt=exam_prompt, exam_raw=exam_raw)
    return final_score


if __name__ == "__main__":
    dralven_tone_sandhi_lang_learning.run(kbench.llm)

