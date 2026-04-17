#!/usr/bin/env python
# coding: utf-8

import random
import re
from dataclasses import dataclass

import kaggle_benchmarks as kbench


def _log_trace(task, description, prompt, test_results, score, reasoning=""):
    sep = "=" * 70
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    print(f"\n  PROMPT:\n{prompt}")
    if reasoning:
        print(f"\n  REASONING:\n{reasoning}")
    print(f"\n  TEST RESULTS:")
    for r in test_results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"    [{status}] Q{r['q']}: expected={r['expected']!r}  got={r['got']!r}")
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


_TASK_DESCRIPTION = (
    "The model observes glossed word forms in a richly invented agglutinative language with five "
    "morpheme slots: root (with ablaut for tense) + aspect-suffix + number-suffix + evidentiality-suffix "
    "+ politeness-suffix. Two interacting complications are hidden: (1) vowel harmony — each suffix has "
    "a front-vowel and back-vowel allomorph selected by the root's last vowel; (2) in the DIRECT "
    "evidential, aspect and number suffixes swap their linear order. Early demos expose only the "
    "first two morpheme slots; the remaining three emerge progressively. Success requires inductively "
    "discovering all five morpheme values, both allomorphy rules, and the evidential-conditioned "
    "reordering from ~20 examples, then applying them to novel five-slot glosses."
)

# ── roots: each has an inherent vowel-harmony class (BACK or FRONT)
# Back-vowel roots: last vowel is a/o/u  → back allomorphs
# Front-vowel roots: last vowel is e/i   → front allomorphs
_ROOTS = {
    "KALDU":  "back",   # last vowel u
    "MONVA":  "back",   # last vowel a
    "TROSBU": "back",   # last vowel u
    "VEPIT":  "front",  # last vowel i
    "ZELFI":  "front",  # last vowel i
    "DREME":  "front",  # last vowel e
}

_TENSE_GLOSSES    = ["past", "present", "future"]
_ASPECT_GLOSSES   = ["perfective", "imperfective"]
_NUM_GLOSSES      = ["singular", "plural", "dual"]
_EVID_GLOSSES     = ["direct", "indirect"]
_POLIT_GLOSSES    = ["familiar", "formal"]

_FIXED_SEED = 42

# Ablaut: root-internal vowel change encodes tense.
# In "past" the root's last vowel shifts: a→o, o→u, u→a, e→i, i→e.
# "present" = no change; "future" = root last vowel is lengthened (doubled).
_ABLAUT_PAST = str.maketrans("aouei", "ouaie")


def _apply_ablaut(root: str, tense: str) -> str:
    """Return the phonologically-modified root for a given tense."""
    lower = root.lower()
    if tense == "past":
        return lower.translate(_ABLAUT_PAST)
    elif tense == "future":
        # Double the last vowel of the root
        for i in range(len(lower) - 1, -1, -1):
            if lower[i] in "aouei":
                return lower[:i] + lower[i] * 2 + lower[i + 1:]
        return lower
    else:  # present
        return lower


def _build_morpheme_table(rng: random.Random) -> tuple:
    # Each suffix slot has two allomorphs: (back_form, front_form)
    # Aspect: perfective / imperfective
    asp_back  = rng.sample(["-bok", "-gon", "-tun", "-wol"], 2)
    asp_front  = rng.sample(["-pek", "-zin", "-vel", "-mir"], 2)
    aspect_map = {
        "perfective":   {"back": asp_back[0],  "front": asp_front[0]},
        "imperfective": {"back": asp_back[1],  "front": asp_front[1]},
    }

    # Number: singular / plural / dual
    num_back  = rng.sample(["-as", "-kor", "-dum", "-wal"], 3)
    num_front  = rng.sample(["-es", "-kir", "-dim", "-wel"], 3)
    number_map = {
        "singular": {"back": num_back[0],  "front": num_front[0]},
        "plural":   {"back": num_back[1],  "front": num_front[1]},
        "dual":     {"back": num_back[2],  "front": num_front[2]},
    }

    # Evidentiality: direct / indirect
    evid_back  = rng.sample(["-ob", "-uft", "-omp"], 2)
    evid_front  = rng.sample(["-ib", "-eft", "-imp"], 2)
    evid_map = {
        "direct":   {"back": evid_back[0],  "front": evid_front[0]},
        "indirect": {"back": evid_back[1],  "front": evid_front[1]},
    }

    # Politeness: familiar / formal
    pol_back  = rng.sample(["-dak", "-mur", "-tos"], 2)
    pol_front  = rng.sample(["-dek", "-mür", "-tes"], 2)
    polit_map = {
        "familiar": {"back": pol_back[0],  "front": pol_front[0]},
        "formal":   {"back": pol_back[1],  "front": pol_front[1]},
    }

    return aspect_map, number_map, evid_map, polit_map


def _inflect(root: str, tense: str, aspect: str, number: str,
             evidentiality: str, politeness: str,
             harmony_class: str,
             aspect_map: dict, number_map: dict,
             evid_map: dict, polit_map: dict) -> str:
    """
    Slot order: modified_root + aspect + number + evidentiality + politeness
    EXCEPTION: when evidentiality == 'direct', aspect and number swap:
               modified_root + number + aspect + evidentiality + politeness
    """
    h = harmony_class  # "back" or "front"
    mod_root = _apply_ablaut(root, tense)
    asp_sfx  = aspect_map[aspect][h]
    num_sfx  = number_map[number][h]
    evid_sfx = evid_map[evidentiality][h]
    pol_sfx  = polit_map[politeness][h]

    if evidentiality == "direct":
        return mod_root + num_sfx + asp_sfx + evid_sfx + pol_sfx
    else:
        return mod_root + asp_sfx + num_sfx + evid_sfx + pol_sfx


def _str_match(expected: str, actual: str) -> bool:
    if actual is None:
        return False
    return bool(re.search(re.escape(expected.strip()), actual.strip(), re.IGNORECASE))


def _make_test_cases(aspect_map, number_map, evid_map, polit_map):
    """Four test cases designed to probe every hidden rule simultaneously."""
    roots_list = list(_ROOTS.items())
    cases = [
        # (root, tense, aspect, number, evidentiality, politeness)
        # Q1: back-vowel root, future tense (lengthening), imperfective, dual, indirect, formal
        ("KALDU",  "future",  "imperfective", "dual",     "indirect", "formal"),
        # Q2: front-vowel root, past tense (ablaut), perfective, plural, direct (→ swap order), familiar
        ("VEPIT",  "past",    "perfective",   "plural",   "direct",   "familiar"),
        # Q3: back-vowel root, past tense (ablaut), perfective, singular, direct (→ swap order), formal
        ("MONVA",  "past",    "perfective",   "singular", "direct",   "formal"),
        # Q4: front-vowel root, future tense (lengthening), imperfective, dual, indirect, familiar
        ("ZELFI",  "future",  "imperfective", "dual",     "indirect", "familiar"),
    ]
    results = []
    for root, tense, aspect, number, evidentiality, politeness in cases:
        h = _ROOTS[root]
        form = _inflect(root, tense, aspect, number, evidentiality, politeness, h,
                        aspect_map, number_map, evid_map, polit_map)
        results.append(((root, tense, aspect, number, evidentiality, politeness), form))
    return results


# Build tables and test cases once at module load
_rng_master = random.Random(_FIXED_SEED)
_ASPECT_MAP, _NUMBER_MAP, _EVID_MAP, _POLIT_MAP = _build_morpheme_table(_rng_master)
_TEST_CASES = _make_test_cases(_ASPECT_MAP, _NUMBER_MAP, _EVID_MAP, _POLIT_MAP)


def _build_prompt(demos: list, test_glosses: list) -> str:
    lines = [
        "You are studying an invented agglutinative language.",
        "Words are built from a root and a series of suffixes encoding grammatical categories.",
        "",
        "Observations (gloss → word form):",
    ]
    for i, entry in enumerate(demos, 1):
        # entry is a tuple of (root, tense, aspect, number, evidentiality, politeness, form)
        # slots after tense may be None if not yet revealed
        root, tense, aspect, number, evidentiality, politeness, form = entry
        parts = [root, tense]
        if aspect is not None:
            parts.append(aspect)
        if number is not None:
            parts.append(number)
        if evidentiality is not None:
            parts.append(evidentiality)
        if politeness is not None:
            parts.append(politeness)
        gloss = ".".join(parts)
        lines.append(f"  {i:2d}. {gloss} → {form}")
    lines.append("")
    lines.append("Now solve these 4 test questions:")
    for i, ((root, tense, aspect, number, evidentiality, politeness), _) in enumerate(test_glosses, 1):
        gloss = f"{root}.{tense}.{aspect}.{number}.{evidentiality}.{politeness}"
        lines.append(f"  Q{i}: {gloss} → ?")
    lines.append("")
    lines.append("Submit answer_1 through answer_4 as the complete word forms.")
    return "\n".join(lines)


def _prepare():
    rng = random.Random(_FIXED_SEED + 13)
    am = _ASPECT_MAP
    nm = _NUMBER_MAP
    em = _EVID_MAP
    pm = _POLIT_MAP

    demos = []

    # ── Phase 1 (4 demos): tense only — show root+tense, no suffixes at all.
    # Reveals ablaut for past/future on both harmony classes.
    phase1 = [
        ("KALDU",  "past",    None, None, None, None),
        ("VEPIT",  "future",  None, None, None, None),
        ("MONVA",  "present", None, None, None, None),
        ("ZELFI",  "past",    None, None, None, None),
    ]
    for root, tense, aspect, number, evid, pol in phase1:
        h = _ROOTS[root]
        form = _apply_ablaut(root, tense)
        demos.append((root, tense, aspect, number, evid, pol, form))

    # ── Phase 2 (5 demos): tense + aspect only — no number/evidentiality/politeness.
    # Reveals aspect suffixes and vowel harmony (both harmony classes, both aspects).
    phase2 = [
        ("KALDU",  "present", "perfective",   None, None, None),
        ("DREME",  "past",    "imperfective",  None, None, None),
        ("MONVA",  "future",  "perfective",   None, None, None),
        ("ZELFI",  "present", "imperfective",  None, None, None),
        ("TROSBU", "past",    "perfective",   None, None, None),
    ]
    for root, tense, aspect, number, evid, pol in phase2:
        h = _ROOTS[root]
        form = _apply_ablaut(root, tense) + am[aspect][h]
        demos.append((root, tense, aspect, number, evid, pol, form))

    # ── Phase 3 (6 demos): tense + aspect + number — no evidentiality/politeness.
    # Reveals number suffixes and harmony for all three number values on both classes.
    phase3 = [
        ("VEPIT",  "present", "perfective",   "singular", None, None),
        ("KALDU",  "past",    "imperfective",  "plural",   None, None),
        ("DREME",  "future",  "perfective",   "dual",     None, None),
        ("MONVA",  "present", "imperfective",  "singular", None, None),
        ("ZELFI",  "past",    "perfective",   "plural",   None, None),
        ("TROSBU", "future",  "imperfective",  "dual",     None, None),
    ]
    for root, tense, aspect, number, evid, pol in phase3:
        h = _ROOTS[root]
        form = _apply_ablaut(root, tense) + am[aspect][h] + nm[number][h]
        demos.append((root, tense, aspect, number, evid, pol, form))

    # ── Phase 4 (7 demos): all five slots.
    # Crucially: some use DIRECT evidential so the suffix-swap is visible,
    # some use INDIRECT so normal order is visible. Both harmony classes represented.
    phase4 = [
        # indirect (normal order): asp + num + evid + pol
        ("KALDU",  "present", "perfective",   "singular", "indirect", "familiar"),
        ("VEPIT",  "present", "imperfective",  "plural",   "indirect", "formal"),
        ("MONVA",  "past",    "imperfective",  "dual",     "indirect", "familiar"),
        # direct (swapped order): num + asp + evid + pol
        ("TROSBU", "present", "perfective",   "singular", "direct",   "formal"),
        ("DREME",  "present", "imperfective",  "plural",   "direct",   "familiar"),
        ("ZELFI",  "future",  "perfective",   "singular", "direct",   "formal"),
        # one more indirect to reinforce
        ("KALDU",  "future",  "imperfective",  "plural",   "indirect", "formal"),
    ]
    for root, tense, aspect, number, evid, pol in phase4:
        h = _ROOTS[root]
        form = _inflect(root, tense, aspect, number, evid, pol, h, am, nm, em, pm)
        demos.append((root, tense, aspect, number, evid, pol, form))

    prompt = _build_prompt(demos, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (gloss, expected) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"answer_{i}", None)
            got = raw.strip() if isinstance(raw, str) else raw
            ok = _str_match(expected, str(got)) if got is not None else False
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: str
    answer_2: str
    answer_3: str
    answer_4: str


@kbench.task(
    name="agglutinative_morphology_obs_learning",
    description="Observe glossed word forms in an invented agglutinative language with 5 morpheme slots, vowel-harmony allomorphy, and a hidden suffix-order inversion triggered by direct evidentiality. Produce correct forms for novel glosses."
)
def agglutinative_morphology_obs_learning(llm) -> float:
    """Infer five morpheme slots, vowel harmony allomorphy, and evidential-conditioned reordering from ~22 examples."""
    prompt, grade_fn = _prepare()
    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None

    if response is not None:
        score, test_results = grade_fn(response)
    else:
        score = 0.0
        test_results = [
            {"q": i, "expected": _TEST_CASES[i - 1][1], "got": None, "correct": False}
            for i in range(1, 5)
        ]

    reasoning = getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    _log_trace(
        task="agglutinative_morphology_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    agglutinative_morphology_obs_learning.run(kbench.llm)

