#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import re
import copy
import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Hard retroactive credit-ledger revaluation. "
    "Six players (Aven, Brek, Coss, Drix, Emul, Fave) accumulate a numeric balance "
    "across 12 matches under a novel 'differential credit' rule: "
    "  solo WIN +3, solo LOSS -3; "
    "  pair WIN: lower-balance player +3, higher-balance +1 (tie: each +2); "
    "  pair LOSS: higher-balance player -3, lower-balance -1 (tie: each -2). "
    "Final rating: balance >= 5 → HIGH, <= -5 → LOW, else NEUTRAL. "
    "Seven structurally distinct questions: "
    "  Q1 final rating (Aven → HIGH), "
    "  Q2 final rating (Coss → LOW), "
    "  Q3 retroactive balance query (Drix after M07, before M08 → -7), "
    "  Q4 intra-match credit allocation (who gained more in M08 → DRIX with +3), "
    "  Q5 balance at specific match mid-point (Emul after M10 → 5), "
    "  Q6 blame allocation (Fave's change in M09 → -3), "
    "  Q7 counterfactual revaluation (Drix rating if M07=WIN → NEUTRAL). "
    "All answers are uniquely determined by the stated rule; "
    "no background knowledge of learning theory is required or applicable."
)

# ─── Ground truth derivation ─────────────────────────────────────────────────
#
# Rule (stated in prompt):
#   solo WIN: +3    solo LOSS: -3
#   pair WIN:  if bal[p1] < bal[p2]: p1+3, p2+1
#              if bal[p1] > bal[p2]: p2+3, p1+1
#              tie (equal): p1+2, p2+2
#   pair LOSS: if bal[p1] > bal[p2]: p1-3, p2-1
#              if bal[p1] < bal[p2]: p2-3, p1-1
#              tie (equal): p1-2, p2-2
#   Rating: >= 5 → HIGH,  <= -5 → LOW,  else NEUTRAL
#
# Step-by-step trace:
#   start:  Aven=0  Brek=0  Coss=0  Drix=0  Emul=0  Fave=0
#   M01 WIN(Aven,Brek): tie(0,0) → each+2 → Aven=2 Brek=2
#   M02 WIN(Aven,Brek): tie(2,2) → each+2 → Aven=4 Brek=4
#   M03 LOSS(Drix,Coss): tie(0,0) → each-2 → Drix=-2 Coss=-2
#   M04 LOSS(Drix,Coss): tie(-2,-2) → each-2 → Drix=-4 Coss=-4
#   M05 WIN(Emul,Fave): tie(0,0) → each+2 → Emul=2 Fave=2
#   M06 solo WIN(Aven): Aven=4+3=7
#   M07 solo LOSS(Drix): Drix=-4-3=-7          ← Q3 answer: -7
#   [snapshot before M08: Aven=7 Brek=4 Coss=-4 Drix=-7 Emul=2 Fave=2]
#   M08 WIN(Brek,Drix): Drix(-7)<Brek(4) → Drix+3=-4, Brek+1=5
#                                           ← Q4 answer: DRIX gained +3
#   M09 LOSS(Coss,Fave): Fave(2)>Coss(-4) → Fave-3=-1, Coss-1=-5
#                                           ← Q6 answer: Fave -3
#   M10 WIN(Aven,Emul): Emul(2)<Aven(7) → Emul+3=5, Aven+1=8
#                                           ← Q5 answer: Emul=5
#   M11 LOSS(Aven,Brek): Aven(8)>Brek(5) → Aven-3=5, Brek-1=4
#   M12 LOSS(Drix,Coss): Drix(-4)>Coss(-5) → Drix-3=-7, Coss-1=-6
#   Final: Aven=5 Brek=4 Coss=-6 Drix=-7 Emul=5 Fave=-1
#   Ratings: Aven=HIGH Brek=NEUTRAL Coss=LOW Drix=LOW Emul=HIGH Fave=NEUTRAL
#
# Q1: Aven final rating → HIGH (balance=5, threshold=5, boundary case)
# Q2: Coss final rating → LOW  (balance=-6)
# Q3: Drix balance after M07, before M08 → -7
# Q4: In M08, which player's balance increased more? → DRIX (+3 vs Brek's +1)
# Q5: Emul's balance immediately after M10 → 5
#     (M10: Emul is lower at 2 vs Aven at 7 → Emul+3=5)
# Q6: By how much did Fave's balance change in M09 → -3
#     (M09: Fave(2)>Coss(-4) → Fave is higher → Fave-3)
# Q7: Counterfactual — if M07 had been a WIN, Drix final rating → NEUTRAL
#     Trace: M07=WIN → Drix=-4+3=-1
#     M08 WIN(Brek,Drix): Brek(4)>Drix(-1) → Drix+3=2, Brek+1=5
#     M09: unchanged (Fave=-1, Coss=-5 ... wait, Coss=-4 not -5 yet)
#     Full CF trace:
#       Drix=-1 after M07
#       M08 WIN(Brek,Drix): Brek(4)>Drix(-1) → Drix+3=2, Brek+1=5
#       M09 LOSS(Coss,Fave): Fave(2)>Coss(-4) → Fave-3=-1, Coss-1=-5
#       M10 WIN(Aven,Emul): Emul(2)<Aven(7) → Emul+3=5, Aven+1=8
#       M11 LOSS(Aven,Brek): Aven(8)>Brek(5) → Aven-3=5, Brek-1=4
#       M12 LOSS(Drix,Coss): Coss(-5)<Drix(2) i.e. Drix(2)>Coss(-5) → Drix-3=-1, Coss-1=-6
#     CF final: Drix=-1 → NEUTRAL   ✓
# ─────────────────────────────────────────────────────────────────────────────


def _log_trace(task: str, description: str, prompt: str, answers: dict, expected: dict, score: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    print(f"\n  RESPONSES:")
    for key in expected:
        actual = answers.get(key, "?")
        exp = expected[key]
        match = "✓" if _str_match(str(exp), str(actual)) else "✗"
        print(f"    {key}: got={actual!r}  expected={exp!r}  {match}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _str_match(expected: str, actual: str) -> bool:
    """Return True if expected appears as an isolated token in actual (case-insensitive).

    Uses lookaheads instead of \\b so that negative integers like '-7' are matched
    correctly: \\b fails before '-' because the minus sign is not a word character.
    """
    token = re.escape(expected.strip())
    m = re.search(rf"(?<!\w){token}(?!\w)", actual.strip(), re.IGNORECASE)
    if not m:
        return False
    prefix_words = re.findall(r"\w+", actual[: m.start()])[-3:]
    negations = {"not", "isn't", "isnt", "never", "no", "cannot", "can't", "cant", "neither", "without"}
    return not any(w.lower() in negations for w in prefix_words)


@dataclass
class RetroRevaluationAnswer:
    q_1: str   # Aven's final rating (HIGH / NEUTRAL / LOW)
    q_2: str   # Coss's final rating (HIGH / NEUTRAL / LOW)
    q_3: str   # Drix's balance after M07 and before M08 (integer as string)
    q_4: str   # Which player gained more in M08 (AVEN / BREK / COSS / DRIX / EMUL / FAVE)
    q_5: str   # Emul's balance immediately after M10 (integer as string)
    q_6: str   # Fave's balance change in M09 (integer as string, negative)
    q_7: str   # Drix's final rating if M07 had been a WIN (HIGH / NEUTRAL / LOW)


_RETRO_EXPECTED = {
    "q_1": "HIGH",
    "q_2": "LOW",
    "q_3": "-7",
    "q_4": "DRIX",
    "q_5": "5",
    "q_6": "-3",
    "q_7": "NEUTRAL",
}


@kbench.task(
    name="retrospective_revaluation_assoc_learning",
    description=(
        "H-02: 7-question differential-credit ledger revaluation — "
        "novel pair/solo scoring rule, 12 matches, 6 players; "
        "final ratings, retroactive balance queries, intra-match "
        "credit allocation, and a counterfactual revaluation."
    ),
)
def retrospective_revaluation_assoc_learning(llm) -> float:
    """
    Differential-credit ledger task: infer numeric balances and ratings
    from a novel scoring rule applied across 12 matches.
    Returns fraction of 7 questions answered correctly.
    """

    prompt = "\n".join([
        "Six players — Aven, Brek, Coss, Drix, Emul, Fave — compete across a series of",
        "matches. Every player begins with a balance of 0. Balances change after each",
        "match according to the following rule:",
        "",
        "  SOLO MATCH (one player listed):",
        "    WIN  → that player's balance increases by 3.",
        "    LOSS → that player's balance decreases by 3.",
        "",
        "  PAIR MATCH (exactly two players listed):",
        "    WIN  → compare the two players' current balances at the start of the match:",
        "             • If their balances are equal: each player gains 2.",
        "             • If one player has a lower balance: that player gains 3,",
        "               the other gains 1.",
        "    LOSS → compare the two players' current balances at the start of the match:",
        "             • If their balances are equal: each player loses 2.",
        "             • If one player has a higher balance: that player loses 3,",
        "               the other loses 1.",
        "",
        "  RATING (evaluated at end of all matches):",
        "    balance ≥  5  →  HIGH",
        "    balance ≤ −5  →  LOW",
        "    otherwise     →  NEUTRAL",
        "",
        "Players not listed in a match are unaffected by that match.",
        "Each match is resolved in chronological order; use the balances as they stand",
        "at the start of each match, not any future state.",
        "",
        "Match log (chronological):",
        "",
        "  Phase 1 — Pair matches:",
        "    M01:  Aven + Brek  →  WIN",
        "    M02:  Aven + Brek  →  WIN",
        "    M03:  Drix + Coss  →  LOSS",
        "    M04:  Drix + Coss  →  LOSS",
        "    M05:  Emul + Fave  →  WIN",
        "",
        "  Phase 2 — Solo matches:",
        "    M06:  Aven (solo)  →  WIN",
        "    M07:  Drix (solo)  →  LOSS",
        "",
        "  Phase 3 — Pair matches:",
        "    M08:  Brek + Drix  →  WIN",
        "    M09:  Coss + Fave  →  LOSS",
        "    M10:  Aven + Emul  →  WIN",
        "    M11:  Aven + Brek  →  LOSS",
        "    M12:  Drix + Coss  →  LOSS",
        "",
        "No further matches are played after M12.",
        "",
        "Answer the following seven questions. Apply the rule above exactly as stated;",
        "do not import any external scoring system.",
        "",
        "  Q1: What is Aven's final rating after all 12 matches?",
        "      Choose exactly one of: HIGH, NEUTRAL, LOW.",
        "",
        "  Q2: What is Coss's final rating after all 12 matches?",
        "      Choose exactly one of: HIGH, NEUTRAL, LOW.",
        "",
        "  Q3: What is Drix's balance immediately after M07 has been resolved",
        "      and before M08 begins?",
        "      Give a single integer (e.g. −4 or 2).",
        "",
        "  Q4: In match M08, which of the two players received the larger balance",
        "      increase?",
        "      Choose exactly one of: BREK, DRIX.",
        "",
        "  Q5: What is Emul's balance immediately after M10 has been resolved?",
        "      Give a single integer.",
        "",
        "  Q6: By how much does Fave's balance change as a result of M09?",
        "      Give a single integer (negative if the balance decreases).",
        "",
        "  Q7: COUNTERFACTUAL — suppose M07 had been a WIN for Drix instead of a LOSS,",
        "      while every other match (M01–M06 and M08–M12) remained exactly as shown.",
        "      Under that hypothetical, what would Drix's final rating be?",
        "      Choose exactly one of: HIGH, NEUTRAL, LOW.",
        "",
    ])

    result = llm.prompt(prompt, schema=RetroRevaluationAnswer)
    assertions = kbench.assertions
    correct = 0
    total = 7

    for key, expn in _RETRO_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must be {expn}.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _RETRO_EXPECTED}
    _log_trace("retrospective_revaluation", _TASK_DESCRIPTION, prompt, answers, _RETRO_EXPECTED, score)
    return score


if __name__ == "__main__":
    retrospective_revaluation_assoc_learning.run(kbench.llm)

