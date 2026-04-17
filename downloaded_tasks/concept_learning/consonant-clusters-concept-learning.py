#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Consonant-cluster parity transform with vowel-shift
#
# Rule (two interacting components):
#
#   CONSONANT CLUSTERS:
#     Split the word into maximal consonant runs (clusters) and vowels.
#     • Odd-length clusters  → reverse the cluster in place
#     • Even-length clusters → leave the cluster unchanged
#
#   VOWELS:
#     Each vowel is replaced by the vowel K steps forward in the cycle
#     (a→e→i→o→u→a), where K = length of the immediately preceding consonant
#     cluster (0 if the vowel is word-initial or follows another vowel).
# ---------------------------------------------------------------------------

from dataclasses import dataclass

import re
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "The model must infer a two-component word transformation rule. "
    "Component 1 (consonants): odd-length maximal consonant clusters are reversed "
    "in place; even-length clusters are left unchanged. "
    "Component 2 (vowels): each vowel advances K steps in (a,e,i,o,u) cyclically, "
    "where K is the length of the immediately preceding consonant cluster "
    "(0 if the vowel is word-initial or follows another vowel). "
    "Both components must be identified jointly — all single-component hypotheses "
    "are falsified by the training examples. "
    "The model actively requests labeled word-transform examples and must submit "
    "the transformed test word. "
    "Success means correctly transforming 4 test words in the examination phase."
)


def _log_trace(task: str, turns: list[dict], exam_results: list[dict],
               final_score: float, examples_used: int,
               exam_prompt: str = "", exam_raw: list[str] = None) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {_TASK_DESCRIPTION}")
    print(f"\n{sep}\n  CONVERSATION\n{sep}")
    for t in turns:
        print(f"\n[USER \u2014 Turn {t['turn']}]")
        print(t.get("prompt", ""))
        print(f"\n[ASSISTANT \u2014 Turn {t['turn']}]")
        print(f"action: {t['action']}")
        response = t.get("response", "")
        print(f"answer: {response if response else '(none)'}")
    print(f"\n{sep}\n  EXAMINATION\n{sep}")
    if exam_prompt:
        print("\n[USER \u2014 Exam]")
        print(exam_prompt)
    if exam_raw:
        print("\n[ASSISTANT \u2014 Exam]")
        print("\n".join(exam_raw))
    print(f"\n{sep}\n  RESULTS\n{sep}")
    for r in exam_results:
        status = "CORRECT" if r["correct"] else "WRONG  "
        print(f"  Test {r['item']}: {status}   input={r['input']!r}   expected={r['expected']!r}   got={r['answer']!r}")
    correct = sum(1 for r in exam_results if r["correct"])
    print(f"\n  Examples used : {examples_used}/{MAX_EXAMPLES}")
    print(f"  Exam accuracy : {correct}/{len(exam_results)}")
    print(f"  Final score   : {final_score:.4f}")
    print(f"{sep}\n")

NUM_TEST_ITEMS = 4
FREE_THRESHOLD = 8


def _concept_score(correct_count: int, examples_used: int,
                   max_examples: int, initial_examples: int) -> float:
    accuracy = correct_count / NUM_TEST_ITEMS
    if accuracy == 0:
        return 0.0
    effective_free = max(initial_examples, FREE_THRESHOLD)
    if max_examples <= effective_free or examples_used <= effective_free:
        efficiency = 1.0
    else:
        paid_used   = examples_used - effective_free
        paid_budget = max_examples  - effective_free
        efficiency  = max(0.0, 1.0 - paid_used / paid_budget)
    return accuracy * (0.40 + 0.60 * efficiency)


_VOWELS = ["a", "e", "i", "o", "u"]
_V_RANK = {v: i for i, v in enumerate(_VOWELS)}
_V_SET = set(_VOWELS)


def _rule(word: str) -> str:
    """
    1. Split into maximal consonant clusters and vowels.
    2. Odd-length clusters → reversed; even-length clusters → unchanged.
    3. Each vowel advances by (length of immediately preceding cluster) mod 5
       in the cycle a→e→i→o→u→a.  Vowels with no preceding cluster (word-initial
       or following another vowel) are unchanged.
    """
    w = word.lower()
    # Build a list of (kind, text, cluster_len) items
    items: list[tuple[str, str, int]] = []
    i = 0
    while i < len(w):
        if w[i] in _V_SET:
            items.append(("v", w[i], 0))
            i += 1
        else:
            j = i
            while j < len(w) and w[j] not in _V_SET:
                j += 1
            cluster = w[i:j]
            L = len(cluster)
            transformed = cluster[::-1] if L % 2 == 1 else cluster
            items.append(("c", transformed, L))
            i = j

    out: list[str] = []
    prev_L = 0
    for kind, text, length in items:
        if kind == "c":
            out.append(text)
            prev_L = length
        else:
            shift = prev_L % 5
            out.append(_VOWELS[(_V_RANK[text] + shift) % 5])
            prev_L = 0
    return "".join(out)


# ---------------------------------------------------------------------------
# Training pool — 40 words, chosen so that:
#   • Cluster lengths 1, 2, 3, 4, 5 all appear (disambiguates vowel shift rule).
#   • Both odd AND even clusters are present in the same words (forces the
#     parity distinction; "all reversed" and "none reversed" are each falsified
#     by the first two examples alone).
#   • Words starting with vowels appear (proves shift = 0 when no cluster precedes).
#   • Consecutive vowels appear (proves shift resets to 0 after each vowel).
# ---------------------------------------------------------------------------
_ALL_INPUTS = [
    # len-3 (odd) clusters – reversal is visually prominent, vowel shift = 3
    "scratch",     # scr→rcs; a+3=o; tch→hct          → rcsohct
    "thrush",      # thr→rht; u+3=i; sh kept (len2)    → rhtish
    "strong",      # str→rts; o+3=e; ng kept (len2)    → rtseng
    "scream",      # scr→rcs; e+3=u; (no cl)a+0=a; m→m → rcsuam
    # len-2 (even) clusters – left unchanged, vowel shift = 2
    "blanket",     # bl kept; a+2=i; nk kept; e+2=o; t→t → blinkot
    "planted",     # pl kept; a+2=i; nt kept; e+2=o; d→d  → plintod
    "cluster",     # cl kept; u+2=e; st kept; e+2=o; r→r  → clestor
    "drifted",     # dr kept; i+2=u; ft kept; e+2=o; d→d  → druftod
    # Mixed: SAME word has both odd and even clusters — forces parity rule
    "transcript",  # tr(2 kept); a+2=i; nscr(4 kept); i+4=e; pt(2 kept) → trinscrept
    "district",    # d(1→d); i+1=o; str(3→rts); i+3=a; ct(2 kept)       → dortsact
    "construct",   # c(1→c); o+1=u; nstr(4 kept); u+4=o; ct(2 kept)     → cunstroct
    "strength",    # str(3→rts); e+3=u; ngth(4 kept)                     → rtsungth
    "splinter",    # spl(3→lps); i+3=a; nt(2 kept); e+2=o; r(1→r)       → lpsantor
    # Vowel-initial: proves shift=0 when no preceding cluster
    "element",     # (0)e+0=e; l(1); e+1=i; m(1); e+1=i; nt(2 kept) → elimint
    "outside",     # (0)o+0=o; (0)u+0=u; ts(2 kept); i+2=u; d(1); e+1=i → outsudi
    # len-5 (odd) cluster — reversal of 5-char run visible
    "strengths",   # str(3→rts); e+3=u; ngths(5→shtgn)                  → rtsushtgn
    # More even-cluster coverage
    "blunder",     # bl(2 kept); u+2=e; nd(2 kept); e+2=o; r(1→r)       → blendor
    "strange",     # str(3→rts); a+3=o; ng(2 kept); e+2=o               → rtsongo
    "pretend",     # pr(2 kept); e+2=o; t(1→t); e+1=i; nd(2 kept)       → protind
    "crescent",    # cr(2 kept); e+2=o; sc(2 kept); e+2=o; nt(2 kept)   → croscont
    # Additional even-cluster (len-2) words
    "blister",     # bl(2 kept); i+2=u; st(2 kept); e+2=o; r(1→r)       → blustor
    "clutter",     # cl(2 kept); u+2=e; tt(2 kept); e+2=o; r(1→r)       → clettor
    "dresser",     # dr(2 kept); e+2=o; ss(2 kept); e+2=o; r(1→r)       → drossor
    "flutter",     # fl(2 kept); u+2=e; tt(2 kept); e+2=o; r(1→r)       → flettor
    "shelter",     # sh(2 kept); e+2=o; lt(2 kept); e+2=o; r(1→r)       → sholtor
    "slender",     # sl(2 kept); e+2=o; nd(2 kept); e+2=o; r(1→r)       → slondor
    "smother",     # sm(2 kept); o+2=u; th(2 kept); e+2=o; r(1→r)       → smathor
    "snapper",     # sn(2 kept); a+2=i; pp(2 kept); e+2=o; r(1→r)       → snippor
    "thunder",     # th(2 kept); u+2=e; nd(2 kept); e+2=o; r(1→r)       → thendor
    "twister",     # tw(2 kept); i+2=u; st(2 kept); e+2=o; r(1→r)       → twustor
    # Additional odd-cluster (len-3) words
    "shrivel",     # shr(3→rhs); i+3=a; v(1→v); e+1=i; l(1→l)          → rhsavil
    "splurge",     # spl(3→lps); u+3=o; rg(2 kept); e+2=o               → lpsirgo
    "sprigot",     # spr(3→rps); i+3=a; g(1→g); o+1=u; t(1→t)          → rpsagut
    # Mixed odd/even cluster words
    "freckle",     # fr(2 kept); e+2=o; ckl(3→lkc); e+3=u              → frolkcu
    "grovel",      # gr(2 kept); o+2=u; v(1→v); e+1=i; l(1→l)           → gravil
    "printer",     # pr(2 kept); i+2=u; nt(2 kept); e+2=o; r(1→r)       → pruntor
    "problem",     # pr(2 kept); o+2=u; bl(2 kept); e+2=o; m(1→m)       → prablom
    "spelunk",     # sp(2 kept); e+2=o; l(1→l); u+1=o; nk(2 kept)       → spolank
    "wristlet",    # wr(2 kept); i+2=u; stl(3→lts); e+3=u; t(1→t)       → wrultsut
    # Vowel-initial with cluster interaction
    "enclose",     # (0)e+0=e; ncl(3→lcn); o+3=u; s(1→s); e+1=i        → elcnusi
]

_ALL_OUTPUTS = [_rule(w) for w in _ALL_INPUTS]

MAX_EXAMPLES = 40
INITIAL_EXAMPLES = 12

# Test words are novel; each exercises the full two-component rule.
# plunder:   pl(2 kept); u+2=e; nd(2 kept); e+2=o; r(1→r)        → plendor
# glisten:   gl(2 kept); i+2=u; st(2 kept); e+2=o; n(1→n)        → gluston
# thrasher:  thr(3→rht); a+3=o; sh(2 kept); e+2=o; r(1→r)        → rhtoshor
# sprinkle:  spr(3→rps); i+3=a; nkl(3→lkn); e+3=u               → rpsalknu
_TEST_INPUTS = ["plunder", "glisten", "thrasher", "sprinkle"]
_TEST_EXPECTED = [_rule(w) for w in _TEST_INPUTS]


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


def _str_match(expected: str, actual: str) -> bool:
    """Return True if expected appears anywhere in actual (case-insensitive)."""
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


@kbench.task(
    name="consonant_clusters_concept_learning",
    description=(
        "Active concept formation: request examples to learn, then examine on "
        "4 test words. Score = accuracy × efficiency."
    ),
)
def consonant_clusters_concept_learning(llm) -> float:
    """
    Active concept formation: infer the two-component consonant-parity +
    vowel-shift transform; request examples or enter 4-item examination.
    Score = accuracy × efficiency.
    """
    turns = []
    examples_shown = INITIAL_EXAMPLES

    initial_lines = [
        "A transformation rule converts words into new strings.",
        "",
        "Labeled examples:",
    ]
    for i in range(INITIAL_EXAMPLES):
        initial_lines.append(f"  Example {i + 1}: '{_ALL_INPUTS[i]}' -> '{_ALL_OUTPUTS[i]}'")
    initial_lines += [
        "",
        "You have two actions:",
        "  action='request' — LEARN: get one more labeled example (up to "
        f"{MAX_EXAMPLES} total)",
        "  action='submit'  — EXAMINE: enter examination mode where you will answer 4 test words",
        "                     in a single response. No feedback, no going back.",
        "",
        f"You have seen {INITIAL_EXAMPLES} examples. {MAX_EXAMPLES - INITIAL_EXAMPLES} more are available.",
        "Your goal: study enough examples to confidently identify the rule, then enter the examination.",
        "When you submit, you will answer 4 unseen test words in a single response — make sure you have mastered the rule.",
        "Best scores go to models that need the fewest examples to answer all 4 correctly.",
    ]
    next_prompt = "\n".join(initial_lines)

    exam_results = []

    with kbench.chats.new("consonant_clusters"):
        for turn in range(1, MAX_EXAMPLES + 2):
            current_prompt = next_prompt
            try:
                sub = llm.prompt(current_prompt, schema=_ConceptAction)
            except Exception:
                entry = {
                    "turn": turn,
                    "action": "PARSE_ERROR",
                    "prompt": current_prompt,
                    "feedback": "Parse error — turn wasted.",
                }
                turns.append(entry)
                next_prompt = (
                    "Parse error. Use action='request' or action='submit' "
                    "with answer field."
                )
                continue

            action = (sub.action or "").strip().lower()
            entry: dict = {"turn": turn, "action": action, "prompt": current_prompt, "response": (sub.answer or "").strip()}

            if action == "request":
                if examples_shown >= MAX_EXAMPLES:
                    entry["feedback"] = (
                        "No more examples. You must submit to enter examination."
                    )
                    turns.append(entry)
                    next_prompt = (
                        "No more examples available. You must now enter examination mode.\n"
                        "action='submit' to begin the examination "
                        "(answer field will be ignored for this action)."
                    )
                else:
                    idx = examples_shown
                    ex_line = (
                        f"Example {idx + 1}: '{_ALL_INPUTS[idx]}' -> "
                        f"'{_ALL_OUTPUTS[idx]}'"
                    )
                    examples_shown += 1
                    remaining = MAX_EXAMPLES - examples_shown
                    entry["feedback"] = f"Showed example {examples_shown}."
                    turns.append(entry)
                    next_prompt = (
                        f"{ex_line}\n\n"
                        f"You have seen {examples_shown} examples. "
                        f"{remaining} more available.\n\n"
                        "action='request' for another example or "
                        "action='submit' to enter examination."
                    )
            elif action == "submit":
                entry["feedback"] = "Entering examination mode."
                turns.append(entry)
                break
            else:
                entry["feedback"] = "Unknown action."
                turns.append(entry)
                next_prompt = (
                    "Unknown action. Use action='request' or action='submit'."
                )

        exam_lines = [
            "EXAMINATION — Apply the rule to transform each of these 4 words.",
            "Provide all answers at once: answer_1 through answer_4.",
            "",
        ]
        for i in range(NUM_TEST_ITEMS):
            exam_lines.append(f"  Word {i + 1}: '{_TEST_INPUTS[i]}'")
        exam_prompt = "\n".join(exam_lines)
        try:
            sub = llm.prompt(exam_prompt, schema=_ExamAnswers)
            raw_answers = [sub.answer_1, sub.answer_2, sub.answer_3, sub.answer_4]
        except Exception:
            raw_answers = ["", "", "", ""]
        for test_idx in range(NUM_TEST_ITEMS):
            answer = (raw_answers[test_idx] or "").strip()
            correct = _str_match(_TEST_EXPECTED[test_idx], answer)
            exam_results.append(
                {
                    "item": test_idx + 1,
                    "input": _TEST_INPUTS[test_idx],
                    "expected": _TEST_EXPECTED[test_idx],
                    "answer": answer,
                    "correct": correct,
                }
            )

    exam_raw = [f"answer_{i+1}: {raw_answers[i]}" for i in range(NUM_TEST_ITEMS)]
    correct_count = sum(1 for r in exam_results if r["correct"])
    final_score = _concept_score(correct_count, examples_shown, MAX_EXAMPLES, INITIAL_EXAMPLES)
    _log_trace(
        "CONSONANT-CLUSTER PARITY + VOWEL-SHIFT",
        turns,
        exam_results,
        final_score,
        examples_shown,
        exam_prompt=exam_prompt,
        exam_raw=exam_raw,
    )
    return final_score


if __name__ == "__main__":
    consonant_clusters_concept_learning.run(kbench.llm)

