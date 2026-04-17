#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import random
import re

import kaggle_benchmarks as kbench

_TASK_DESCRIPTION = (
    "Tests odd-frequency letter-position sum learning. "
    "For each word, assign each letter its 1-indexed alphabet position (A=1 … Z=26), "
    "then sum positions only for letters that appear an ODD number of times in that word "
    "(letters with an even count contribute 0; letters appearing 3 times are ODD and contribute). "
    "CIPHER if the first word's score strictly exceeds the second's; PLAIN otherwise (ties -> PLAIN). "
    "1,000 training pairs are generated from a pool of 112 animal names using a seeded RNG, "
    "so alpha_sum correctly predicts only ~46% of pairs, word_len ~50%, "
    "forcing the model to discover the odd-frequency rule. "
    "Key traps include: Reindeer(E×3→contributes, odd=32) vs Crane(41) — PLAIN despite "
    "Reindeer being longer and alpha-heavier; Pronghorn(R×2,O×2,N×2 cancel, odd=31) vs "
    "Heron(60) — PLAIN despite alpha=125 vs 60; Roadrunner(R×3→contributes, odd=64) vs "
    "Snipe(63) — CIPHER by delta=1; Motmot/Dodo/Nene score 0. "
    "Test questions: Salamander(A×3), Motmot(all cancel), Caracal(A×3,C×2 reversed alpha), "
    "Condor vs Orca delta=2, Barracuda(A×3,R×2 mostly cancel) vs Crow, identical-pair tie."
)


def _odd_score(word: str) -> int:
    word = word.upper()
    freq: dict[str, int] = {}
    for c in word:
        freq[c] = freq.get(c, 0) + 1
    return sum(ord(c) - 64 for c, n in freq.items() if n % 2 == 1)


def _label(w1: str, w2: str) -> str:
    return "CIPHER" if _odd_score(w1) > _odd_score(w2) else "PLAIN"


_ANIMALS = [
    "Alligator", "Anaconda",   "Aracari",     "Armadillo",   "Asp",        "Avocet",
    "Baboon",    "Bandicoot",  "Barracuda",   "Bat",         "Beluga",     "Bison",
    "Caracal",   "Caterpillar","Chamois",     "Cheetah",     "Civet",      "Cod",
    "Colobus",   "Condor",     "Coot",        "Cormorant",   "Crane",      "Crow",
    "Dab",       "Dingo",      "Dodo",        "Dove",        "Dugong",
    "Echidna",   "Eel",        "Egret",       "Elk",         "Emu",        "Ewe",
    "Ferret",    "Finch",      "Flamingo",    "Fox",
    "Genet",     "Gibbon",     "Gnu",         "Goral",       "Grasshopper","Grouper",    "Grouse",
    "Hawk",      "Heron",      "Hoatzin",
    "Ibis",      "Impala",
    "Jabiru",    "Jackal",     "Jay",
    "Kingfisher","Kookaburra", "Kudu",
    "Langur",    "Llama",      "Lorikeet",    "Lulu",
    "Manatee",   "Mandrill",   "Markhor",     "Marlin",      "Meerkat",    "Millipede",
    "Mink",      "Mole",       "Mongoose",    "Moose",       "Motmot",
    "Narwhal",   "Nene",       "Numbat",
    "Ocelot",    "Okapi",      "Orca",        "Otter",
    "Pelican",   "Pig",        "Platypus",    "Porcupine",   "Pronghorn",  "Puffin",
    "Quail",     "Quokka",
    "Raccoon",   "Rail",       "Ram",         "Reindeer",    "Roadrunner", "Robin",
    "Salamander","Snipe",      "Stoat",       "Stork",       "Sunbittern", "Swift",
    "Tapir",     "Tarantula",  "Tarpon",      "Teal",        "Toucan",
    "Vole",
    "Wahoo",     "Wallaby",    "Wildebeest",  "Wombat",      "Wren",
    "Yak",
    "Zebra",
]


def _build_training_pairs(n: int = 1000, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pool = [(a, b) for a in _ANIMALS for b in _ANIMALS if a != b]
    rng.shuffle(pool)
    return pool[:n]


_TRAINING_PAIRS: list[tuple[str, str]] = _build_training_pairs()


def _build_training_prompt() -> str:
    header = (
        "Each ordered pair of animal names maps to a code. "
        "Study every example carefully — the rule must be consistent with all of them "
        "— then answer the questions below.\n"
    )
    lines = [header]
    for w1, w2 in _TRAINING_PAIRS:
        lines.append(f"  ({w1}, {w2}) -> {_label(w1, w2)}")
    lines.append("")
    return "\n".join(lines)


_TEST_PAIRS: list[tuple[str, str]] = [
    ("Salamander", "Wren"),
    ("Wren",       "Motmot"),
    ("Caracal",    "Coot"),
    ("Condor",     "Orca"),
    ("Barracuda",  "Crow"),
    ("Caracal",    "Caracal"),
]

_ODD_LETTER_EXPECTED: dict[str, str] = {
    f"q_{i}": _label(w1, w2)
    for i, (w1, w2) in enumerate(_TEST_PAIRS, 1)
}


def _build_test_prompt() -> str:
    lines = ["What code applies to each pair?"]
    for i, (w1, w2) in enumerate(_TEST_PAIRS, 1):
        lines.append(f"  Question {i}: ({w1}, {w2})")
    lines.append("")
    return "\n".join(lines)


def _str_match(expected: str, actual: str) -> bool:
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


def _log_trace(
    task: str,
    description: str,
    prompt: str,
    answers: dict,
    expected: dict,
    score: float,
) -> None:
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


@dataclass
class OddLetterScoreAnswer:
    q_1: str
    q_2: str
    q_3: str
    q_4: str
    q_5: str
    q_6: str


@kbench.task(
    name="odd_letter_score_pair_assoc_learning",
    description=(
        "Infer an odd-frequency letter-position scoring rule from 1,000 deceptive "
        "animal-name pairs; generalise to triple-count letters, zero-score cancellations, "
        "and near-tie deltas."
    ),
)
def odd_letter_score_pair_assoc_learning(llm) -> float:
    """Odd-frequency letter-position score classification (CIPHER/PLAIN); return fraction correct."""
    prompt = _build_training_prompt() + "\n" + _build_test_prompt()

    result = llm.prompt(prompt, schema=OddLetterScoreAnswer)
    assertions = kbench.assertions
    correct = 0
    total = len(_ODD_LETTER_EXPECTED)
    for key, exp_val in _ODD_LETTER_EXPECTED.items():
        act = str(getattr(result, key)).strip().upper()
        expn = str(exp_val).strip().upper()
        if _str_match(expn, act):
            correct += 1
        assertions.assert_equal(expn, act, expectation=f"`{key}` must match.")

    score = correct / total
    answers = {k: getattr(result, k) for k in _ODD_LETTER_EXPECTED}
    _log_trace("odd_letter_score_pair", _TASK_DESCRIPTION, prompt, answers, _ODD_LETTER_EXPECTED, score)
    return score


if __name__ == "__main__":
    odd_letter_score_pair_assoc_learning.run(kbench.llm)

