#!/usr/bin/env python
# coding: utf-8

import re
from dataclasses import dataclass

import kaggle_benchmarks as kbench



def efficiency_score(
    solved: bool,
    step_y: int,
    budget_n: int,
    min_necessary: int,
    floor: float = 0.1,
) -> float:
    """Efficiency credit for solving within a budget."""
    if not solved:
        return 0.0
    step_y = max(1, min(step_y, budget_n))
    if step_y <= min_necessary:
        return 1.0
    paid_used = step_y - min_necessary
    paid_budget = budget_n - min_necessary
    if paid_budget <= 0:
        return 1.0
    return max(floor, 1.0 - (1.0 - floor) * (paid_used / paid_budget))


def weighted_learning_mean(round_scores: list, weights: list = None) -> float:
    """Weighted mean emphasising later rounds."""
    n = len(round_scores)
    if n == 0:
        return 0.0
    if weights is None:
        weights = list(range(1, n + 1))
    denom = sum(weights)
    if denom == 0:
        return sum(round_scores) / n
    return sum(s * w for s, w in zip(round_scores, weights)) / denom


def _learning_curve_slope(round_scores: list) -> float:
    """Normalised OLS slope of round scores → [0,1] (0.5 = flat, 1.0 = perfect rise)."""
    n = len(round_scores)
    if n < 2:
        return 0.5
    x_mean = (n - 1) / 2.0
    y_mean = sum(round_scores) / n
    num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(round_scores))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.5
    slope = num / den
    max_slope = 1.0 / (n - 1)
    normalised = max(-1.0, min(1.0, slope / max_slope))
    return round((normalised + 1.0) / 2.0, 4)


def procedural_composite_score(round_scores: list, test_score: float) -> float:
    """Four-component procedural learning score in [0, 1].

    transfer    0.30  — test_score: transfer to novel instances without feedback
    asymptote   0.25  — mean of latter half of round_scores: peak skill reached
    trajectory  0.25  — learning_curve_slope: evidence of genuine improvement
    consistency 0.20  — weighted_learning_mean: overall quality, later rounds weighted more
    """
    n = len(round_scores)
    if n == 0:
        return 0.0
    k = max(1, n // 2)
    asymptote = sum(round_scores[-k:]) / k
    if asymptote < 1e-9 and float(test_score) < 1e-9:
        return 0.0
    consistency = weighted_learning_mean(round_scores)
    trajectory = _learning_curve_slope(round_scores)
    raw = (
        0.30 * float(test_score)
        + 0.25 * asymptote
        + 0.25 * trajectory
        + 0.20 * consistency
    )
    return round(raw, 4)

# ─────────────────────────────────────────────────────────────────────────────

_TASK_DESCRIPTION = (
    "Tests procedural learning of two hidden phonological transformation rules (VOWEL_SHIFT, PREFIX_ZA, etc.) "
    "applied in a fixed order. The model probes words to observe their dialect outputs, "
    "then applies the deduced rules to a target test word across 5 dialect instances. "
    "What makes it hard is that rules compose and the model must isolate each rule's effect from combined outputs. "
    "Success = 50% learning efficiency + 50% correct final test."
)


def _log_trace(
    task: str, phases: list[dict], final_score: float, initial_prompt: str = ""
) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {_TASK_DESCRIPTION}")
    if initial_prompt:
        print(f"\n  INITIAL PROMPT:\n{initial_prompt}")
    for phase in phases:
        label = phase["label"]
        print(f"\n  [{label}]  correct={phase['correct']}")
        for t in phase["turns"]:
            print(f"    Turn {t['turn']}  submitted: {t['submitted']}", end="")
            if "feedback" in t:
                print(f"  →  {t['feedback']}", end="")
            print()
        status = "PASS ✓" if phase["solved"] else "FAIL ✗"
        print(f"    {status}  steps={phase['steps']}  score={phase['score']:.4f}")
    print(f"\n  Final score : {final_score:.4f}")
    print(f"{sep}\n")


BUDGET = 10
MIN_NECESSARY = 3

VOWELS = "aeiou"


def _reverse_vowels(s):
    vowels_in_s = [c for c in s if c in VOWELS][::-1]
    result = list(s)
    vi = 0
    for i, c in enumerate(result):
        if c in VOWELS:
            result[i] = vowels_in_s[vi]
            vi += 1
    return "".join(result)


TRANSFORMS = {
    "VOWEL_SHIFT": lambda s: "".join(
        {"a": "e", "e": "i", "i": "o", "o": "u", "u": "a"}.get(c, c) for c in s
    ),
    "CONSONANT_DUP": lambda s: (s[0] + s) if s and s[0] not in VOWELS else s,
    "SUFFIX_OT": lambda s: s + "ot",
    "PREFIX_ZA": lambda s: "za" + s,
    "REVERSE_VOWELS": lambda s: _reverse_vowels(s),
    "VOWEL_DROP": lambda s: "".join(c for c in s if c not in VOWELS),
}


def _apply_rules(word, rules):
    result = word
    for rule in rules:
        result = TRANSFORMS[rule](result)
    return result


LEARNING_INSTANCES = [
    {"rules": ["VOWEL_SHIFT", "SUFFIX_OT"], "test_word": "hello"},
    {"rules": ["PREFIX_ZA", "CONSONANT_DUP"], "test_word": "world"},
    {"rules": ["VOWEL_DROP", "SUFFIX_OT"], "test_word": "bridge"},
    {"rules": ["REVERSE_VOWELS", "PREFIX_ZA"], "test_word": "stone"},
    {"rules": ["VOWEL_SHIFT", "CONSONANT_DUP"], "test_word": "train"},
]
TEST_INSTANCES = [
    {"rules": ["PREFIX_ZA", "VOWEL_SHIFT"], "test_word": "music"},
    {"rules": ["SUFFIX_OT", "REVERSE_VOWELS"], "test_word": "lake"},
    {"rules": ["CONSONANT_DUP", "VOWEL_DROP"], "test_word": "spark"},
    {"rules": ["VOWEL_SHIFT", "PREFIX_ZA"], "test_word": "cloud"},
]

LEARNING_CORRECT = [
    _apply_rules(inst["test_word"], inst["rules"]) for inst in LEARNING_INSTANCES
]


@dataclass
class _DialectAction:
    action: str
    word: str
    answer: str


def _str_match(expected: str, actual: str) -> bool:
    """Return True if expected appears anywhere in actual (case-insensitive)."""
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


@kbench.task(
    name="dialect_morphology_proc_learning",
    description=(
        "Deduce 2 hidden phonological transformation rules (VOWEL_SHIFT/PREFIX_ZA/etc.) by transforming probe words, "
        "then apply them across 5 dialect instances. Score = weighted_learning×0.5 + (tests_passed/4)×0.5."
    ),
)
def dialect_morphology_proc_learning(llm) -> float:
    """5 practice instances then 4 single-shot morphology tests."""
    phases = []

    with kbench.chats.new("dialect_morphology"):
        learning_scores = []
        initial_prompt = ""
        for idx, (inst, correct_output) in enumerate(
            zip(LEARNING_INSTANCES, LEARNING_CORRECT)
        ):
            rules = inst["rules"]
            test_word = inst["test_word"]
            turns = []
            solved = False
            num_steps = 0

            intro = (
                (
                    "A dialect applies 2 hidden transformation rules in a fixed order."
                    "The rules are not named — you must discover their effects by querying words."
                    "  VOWEL_DROP     — remove all vowels\n\n"
                    "Actions:\n"
                    "  action='transform', word='apple'  → dialect output of that word\n"
                    "  action='submit', answer='...'     → graded (correct transform of test word)\n\n"
                    "Scoring note: Your score has four components — transfer (30%): correctly "
                    "transforming test words in the final no-probe instances; asymptote (25%): "
                    "transformation accuracy in the later practice instances; trajectory (25%): "
                    "whether your accuracy improves across practice instances (a rising curve beats "
                    "a flat one even at the same average); consistency (20%): overall quality with "
                    "later instances weighted more. Using fewer transform probes per practice "
                    "instance also boosts your within-round efficiency score.\n\n"
                )
                if idx == 0
                else ""
            )

            next_prompt = (
                f"{intro}"
                f"Practice {idx + 1}/5 — Test word: '{test_word}'\n"
                f"Attempt 1 of {BUDGET}. Probe words or submit the transformed test word."
            )

            for turn in range(1, BUDGET + 1):
                num_steps = turn
                if idx == 0 and turn == 1:
                    initial_prompt = next_prompt
                try:
                    sub = llm.prompt(next_prompt, schema=_DialectAction)
                except Exception:
                    entry = {
                        "turn": turn,
                        "submitted": "PARSE_ERROR",
                        "feedback": "Failed to parse response — turn wasted.",
                    }
                    turns.append(entry)
                    next_prompt = f"Your last response could not be parsed. Please follow the schema exactly.\n\nAttempt {turn + 1} of {BUDGET}. Probe more words or submit revised answer."
                    continue
                action = (sub.action or "").strip().lower()

                if action == "transform":
                    probe = (sub.word or "").strip().lower()
                    submitted_val = probe
                    entry = {"turn": turn, "submitted": submitted_val}
                    if not probe or not probe.isalpha():
                        feedback = f"INVALID word '{probe}' — letters only."
                    else:
                        output = _apply_rules(probe, rules)
                        feedback = f"transform '{probe}' → '{output}'"
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Test word: '{test_word}'\n"
                        f"Attempt {turn + 1} of {BUDGET}. Probe more words or submit."
                    )
                elif action == "submit":
                    answer = (sub.answer or "").strip().lower()
                    entry = {"turn": turn, "submitted": answer}
                    if _str_match(correct_output, answer):
                        solved = True
                        turns.append(entry)
                        break
                    feedback = f"WRONG. Expected: '{correct_output}'"
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Test word: '{test_word}'\n"
                        f"Attempt {turn + 1} of {BUDGET}. Probe more words or submit revised answer."
                    )
                else:
                    entry = {"turn": turn, "submitted": action}
                    feedback = (
                        f"INVALID action '{action}'. Use 'transform' or 'submit'."
                    )
                    entry["feedback"] = feedback
                    turns.append(entry)
                    next_prompt = (
                        f"{feedback}\n\n"
                        f"Test word: '{test_word}'\n"
                        f"Attempt {turn + 1} of {BUDGET}. Use action='transform' or action='submit'."
                    )

            eff = efficiency_score(solved, num_steps, BUDGET, MIN_NECESSARY)
            learning_scores.append(eff)
            phases.append(
                {
                    "label": f"Practice {idx + 1}/5",
                    "correct": correct_output,
                    "turns": turns,
                    "solved": solved,
                    "steps": num_steps,
                    "score": eff,
                }
            )

        test_ok = 0
        for ti, tins in enumerate(TEST_INSTANCES, start=1):
            gold = _apply_rules(tins["test_word"], tins["rules"])
            try:
                test_sub = llm.prompt(
                    f"Final test {ti}/4 — Test word: '{tins['test_word']}'\n"
                    "No hints. Submit action='submit' with the transformed test word.",
                    schema=_DialectAction,
                )
            except Exception:
                test_sub = None
            test_answer = (
                (test_sub.answer or "").strip().lower()
                if test_sub is not None
                else ""
            )
            passed = _str_match(gold, test_answer)
            if passed:
                test_ok += 1
            phases.append(
                {
                    "label": f"Final test {ti}/4",
                    "correct": gold,
                    "turns": [{"turn": 1, "submitted": test_answer}],
                    "solved": passed,
                    "steps": 1,
                    "score": 1.0 if passed else 0.0,
                }
            )

    learning_score = weighted_learning_mean(learning_scores)
    test_score = test_ok / 4.0
    final_score = procedural_composite_score(learning_scores, test_score)
    _log_trace("DIALECT MORPHOLOGY", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    dialect_morphology_proc_learning.run(kbench.llm)

