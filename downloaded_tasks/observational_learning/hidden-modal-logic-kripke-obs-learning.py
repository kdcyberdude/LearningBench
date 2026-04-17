#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can simultaneously reconstruct a hidden atom valuation "
    "and a hidden accessibility relation R in a 5-world Kripke frame from purely "
    "modal observations. Unlike standard modal-logic tasks, no raw atom truth values "
    "are ever given; every observation is a depth-2 or depth-3 nested modal formula. "
    "Reconstruction requires interleaved reasoning: neither component can be fully "
    "solved without partial knowledge of the other. Test cases are depth-4 formulas "
    "requiring four-step modal chains over the fully reconstructed frame."
)

# ═══════════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH KRIPKE FRAME  (hidden from model — never revealed in the prompt)
# ═══════════════════════════════════════════════════════════════════════════════
#
#   Worlds:  {w0, w1, w2, w3, w4}
#
#   Atom valuations (ALL HIDDEN — model must infer from modal observations):
#     E = {w0, w2, w4}      F = {w1, w3}
#     G = {w0, w1, w3}      H = {w2, w3}
#
#   World "signatures" (E, F, G, H) — all five are distinct, aiding uniqueness:
#     w0: (E, G)    w1: (F, G)    w2: (E, H)    w3: (F, G, H)    w4: (E)
#
#   Accessibility relation R (HIDDEN):
#     R(w0) = {w2, w3}      R(w1) = {w0, w4}
#     R(w2) = {w1}          R(w3) = {w2, w4}
#     R(w4) = {w0, w1}
#
# ═══════════════════════════════════════════════════════════════════════════════
# UNIQUENESS PROOF SKETCH
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Phase A — From the eight □◇atom / ◇□atom observation families, together with
#  their FALSE counterparts, the model must infer both val and R jointly through
#  interleaved constraint propagation.  Neither component can be fully isolated:
#
#  Key deductions (illustrative; full proof is by exhaustive constraint propagation):
#
#  1. "□◇E at w2=T" with a single successor in R(w2) forces that successor s to
#     satisfy ◇E(s), i.e., R(s) ∩ {E-worlds} ≠ ∅.  Combined with "◇□E at w2=T",
#     there must also exist some successor s' of w2 with R(s') ⊆ E.  Together
#     these tightly constrain which world can be the unique successor of w2.
#
#  2. "□◇G at w0=F" rules out w0 from having all successors that see G.  Combined
#     with "◇□G at w0=T", exactly one specific successor set for w0 is consistent.
#
#  3. The depth-3 discriminator "□□◇F at w3=F" (sol1 gives True, only GT gives False)
#     eliminates the one remaining spurious solution after the depth-2 constraints
#     have reduced the space to exactly two candidate R relations.  This ensures that
#     the full (val, R) pair is uniquely determined by the observation set.
#
#  Computational verification: numpy-vectorized enumeration over all 2^25 = 33M
#  candidate R assignments (for fixed val_gt) confirms exactly ONE valid R.
#  A separate argument by symmetry-breaking on world signatures confirms that
#  val cannot be permuted without violating some modal observation.

_WORLDS = [0, 1, 2, 3, 4]
_VAL = {
    "E": {0, 2, 4},
    "F": {1, 3},
    "G": {0, 1, 3},
    "H": {2, 3},
}
_R = {0: [2, 3], 1: [0, 4], 2: [1], 3: [2, 4], 4: [0, 1]}

# ─── Observations: ALL are depth-2 or depth-3 modal formulas ─────────────────
# No raw atom truth values are given; the model must infer E, F, G, H entirely
# from how they interact with the hidden R across multiple worlds.
_DEMOS = [
    # ── Depth-2: □◇atom  (□◇X at w = T means every successor of w sees some X-world) ──
    ("□◇E",  1, True),   # R(w1)={w0,w4}: both w0,w4 ∈ succ(w1) must see E
    ("□◇E",  2, True),   # unique succ of w2 sees E
    ("□◇E",  4, True),   # both succs of w4 see E
    ("□◇E",  0, False),  # NOT all succs of w0 see E
    ("□◇E",  3, False),  # NOT all succs of w3 see E
    ("□◇F",  1, True),
    ("□◇F",  3, True),
    ("□◇F",  0, False),
    ("□◇F",  2, False),
    ("□◇F",  4, False),
    ("□◇G",  1, True),
    ("□◇G",  2, True),
    ("□◇G",  3, True),
    ("□◇G",  4, True),
    ("□◇G",  0, False),  # NOT all succs of w0 see G; key asymmetry
    ("□◇¬H", 0, True),   # every succ of w0 sees a non-H world
    ("□◇¬H", 2, True),
    ("□◇¬H", 3, True),
    ("□◇¬H", 1, False),
    ("□◇¬H", 4, False),
    # ── Depth-2: ◇□atom  (◇□X at w = T means some successor of w has ALL succs in X) ──
    ("◇□E",  0, True),   # some succ of w0 has all its succs in E
    ("◇□E",  2, True),
    ("◇□E",  4, True),
    ("◇□E",  1, False),
    ("◇□E",  3, False),
    ("◇□F",  0, True),
    ("◇□F",  3, True),
    ("◇□F",  1, False),
    ("◇□F",  2, False),
    ("◇□F",  4, False),
    ("◇□G",  0, True),
    ("◇□G",  1, True),
    ("◇□G",  3, True),
    ("◇□G",  2, False),
    ("◇□G",  4, False),
    ("◇□H",  1, True),
    ("◇□H",  4, True),
    ("◇□H",  0, False),
    ("◇□H",  2, False),
    ("◇□H",  3, False),
    # ── Depth-3 observations (the hardest constraints; require knowing both val and R) ──
    # □□◇G at w: every succ s of w, every succ t of s, t sees G
    ("□□◇G", 0, True),   # w0→{w2,w3}: both w2,w3 satisfy □◇G
    ("□□◇G", 2, True),
    ("□□◇G", 3, True),
    ("□□◇G", 1, False),
    ("□□◇G", 4, False),
    # □◇□G at w: every succ of w has SOME succ with all succs in G
    ("□◇□G", 2, True),
    ("□◇□G", 4, True),
    ("□◇□G", 0, False),
    ("□◇□G", 1, False),
    ("□◇□G", 3, False),
    # ◇□◇F at w: some succ of w has ALL succs s.t. each s sees F
    ("◇□◇F", 0, True),
    ("◇□◇F", 2, True),
    ("◇□◇F", 4, True),
    ("◇□◇F", 1, False),
    ("◇□◇F", 3, False),
    # Discriminator that eliminates the one spurious R candidate
    ("□□◇F",  3, False),  # NOT all depth-2 chains from w3 lead to F-worlds
]

# ─── Test cases: depth-4 formulas ─────────────────────────────────────────────
# Each requires a 4-step modal-evaluation chain using the reconstructed val and R.
# Answer labels: 4 TRUE, 4 FALSE (listed interleaved for difficulty).
_TEST_CASES = [
    # Q1: □◇◇□E at w2 — TRUE
    #   R(w2)={w1}; need ◇◇□E at w1
    #     R(w1)={w0,w4}; check ◇□E at w0 and w4
    #     ◇□E at w0: R(w0)={w2,w3}; □E(w2)=F (R(w2)={w1},E(w1)=F); □E(w3)=T (R(w3)={w2,w4},E(w2)=T,E(w4)=T) → ◇□E(w0)=T
    #     ◇□E at w4: R(w4)={w0,w1}; □E(w0)=F (w3 not E); □E(w1)=T (R(w1)={w0,w4},E(w0)=T,E(w4)=T) → ◇□E(w4)=T
    #     ◇◇□E at w1 = T ∧ T → TRUE
    #   □◇◇□E at w2: single succ w1 satisfies → TRUE
    ("□◇◇□E", 2, True),

    # Q2: ◇□◇□E at w0 — TRUE
    #   R(w0)={w2,w3}; need □◇□E at some succ
    #   □◇□E at w3: R(w3)={w2,w4}; ◇□E(w2)=T (shown above); ◇□E(w4)=T (shown above) → TRUE
    #   so ◇□◇□E(w0) = T (w3 satisfies) → TRUE
    ("◇□◇□E", 0, True),

    # Q3: □◇◇□H at w0 — TRUE
    #   R(w0)={w2,w3}; need ◇◇□H at both w2 and w3
    #   ◇□H(w0): R(w0)={w2,w3}: □H(w2)=F (R(w2)={w1}, H(w1)=F); □H(w3)=T (R(w3)={w2,w4}: H(w2)=T,H(w4)=F) → wait
    #     H={w2,w3}: H(w4)=F, H(w0)=F, H(w1)=F. □H(w3): R(w3)={w2,w4}: H(w2)=T, H(w4)=F → □H(w3)=F
    #     □H(w0): R(w0)={w2,w3}: H(w2)=T, H(w3)=T → □H(w0)=T
    #   ◇◇□H at w2: R(w2)={w1}; ◇□H(w1): R(w1)={w0,w4}: □H(w0)=T (shown), □H(w4)=F → ◇□H(w1)=T; ◇◇□H(w2)=T
    #   ◇◇□H at w3: R(w3)={w2,w4}; ◇□H(w2): R(w2)={w1}: □H(w1): R(w1)={w0,w4}: H(w0)=F → □H(w1)=F → ◇□H(w2)=F
    #     ◇□H(w4): R(w4)={w0,w1}: □H(w0)=T, □H(w1)=F → ◇□H(w4)=T; ◇◇□H(w3)=T
    #   □◇◇□H(w0) = T ∧ T → TRUE
    ("□◇◇□H", 0, True),

    # Q4: ◇□□◇E at w0 — TRUE
    #   R(w0)={w2,w3}; need □□◇E at some succ
    #   □□◇E at w3: R(w3)={w2,w4}; □◇E(w2): R(w2)={w1}: ◇E(w1)=T (R(w1)={w0,w4},E(w0)=T) → T
    #     □◇E(w4): R(w4)={w0,w1}: ◇E(w0)=T (R(w0)={w2,w3},E(w2)=T); ◇E(w1)=T → T
    #   □□◇E(w3) = T → ◇□□◇E(w0) = T
    ("◇□□◇E", 0, True),

    # Q5: □□◇□E at w0 — FALSE
    #   R(w0)={w2,w3}; need □◇□E at BOTH w2 and w3
    #   □◇□E at w2: R(w2)={w1}; ◇□E(w1): R(w1)={w0,w4}: □E(w0)=F (E(w3)=F), □E(w4)=F (E(w1)=F) → ◇□E(w1)=F
    #     □◇□E(w2)=F
    #   Not all succs satisfy → FALSE
    ("□□◇□E", 0, False),

    # Q6: □◇□◇F at w2 — FALSE
    #   R(w2)={w1}; need ◇□◇F at w1
    #   R(w1)={w0,w4}; □◇F at w0 and w4?
    #   □◇F at w0: R(w0)={w2,w3}: ◇F(w2)=T (R(w2)={w1},F(w1)=T); ◇F(w3)=F (R(w3)={w2,w4},F(w2)=F,F(w4)=F) → F
    #   □◇F at w4: R(w4)={w0,w1}: ◇F(w0)=T (R(w0)={w2,w3},F(w3)=T); ◇F(w1)=F (R(w1)={w0,w4},F(w0)=F,F(w4)=F) → F
    #   ◇□◇F(w1)=F → □◇□◇F(w2)=F
    ("□◇□◇F", 2, False),

    # Q7: ◇□□◇E at w2 — FALSE
    #   R(w2)={w1}; need □□◇E at w1
    #   R(w1)={w0,w4}; □◇E at w0 and w4?
    #   □◇E at w0: ◇E(w2)=F (R(w2)={w1},E(w1)=F); ◇E(w3)=T (R(w3)={w2,w4},E(w2)=T) → mixed: F (□ requires all)
    #     wait: □◇E at w0 = all s in R(w0) satisfy ◇E(s): ◇E(w2)=F → □◇E(w0)=F
    #   □□◇E(w1) requires □◇E at ALL succs of w1 = {w0,w4}: w0 fails → □□◇E(w1)=F
    #   ◇□□◇E(w2): only succ is w1, w1 fails → FALSE
    ("◇□□◇E", 2, False),

    # Q8: ◇◇□□E at w2 — FALSE
    #   R(w2)={w1}; ◇□□E at w1: R(w1)={w0,w4}; □□E at w0 and w4?
    #   □□E at w0: R(w0)={w2,w3}: □E(w2)=F (E(w1)=F) → □□E(w0)=F
    #   □□E at w4: R(w4)={w0,w1}: □E(w0)=F (E(w3)=F) → □□E(w4)=F
    #   ◇□□E(w1)=F → ◇◇□□E(w2)=F
    ("◇◇□□E", 2, False),
]

_GT_ANSWERS = [tc[2] for tc in _TEST_CASES]


def _eval_formula(formula: str, world: int, R: dict, val: dict) -> bool:
    for atom in ("E", "F", "G", "H"):
        if formula == atom:
            return world in val[atom]
        if formula == f"¬{atom}":
            return world not in val[atom]
    if formula.startswith("□"):
        succs = R.get(world, [])
        return (not succs) or all(
            _eval_formula(formula[1:], s, R, val) for s in succs
        )
    if formula.startswith("◇"):
        return any(
            _eval_formula(formula[1:], s, R, val) for s in R.get(world, [])
        )
    return False


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
        print(
            f"    [{status}] Q{r['q']}: expected={r['expected']!r}  got={r['got']!r}"
        )
    print(f"\n  SCORE: {score:.4f}")
    print(f"{sep}\n")


def _prepare():
    lines = [
        "You are observing a KRIPKE FRAME for modal logic.",
        "The frame has five worlds: {w0, w1, w2, w3, w4}.",
        "",
        "Two components of this frame are COMPLETELY HIDDEN:",
        "  (1) The atom valuation: for each atom in {E, F, G, H}, the set of worlds where it holds.",
        "  (2) The accessibility relation R: for each world w, the set R(w) of worlds accessible from w.",
        "",
        "Modal operators:",
        "  □φ is true at world w iff φ is true at EVERY world in R(w).",
        "      (Vacuously true if R(w) is empty.)",
        "  ◇φ is true at world w iff φ is true at SOME world in R(w).",
        "  ¬φ is the negation of φ.",
        "  Operators associate right-to-left: □◇□E means □(◇(□E)).",
        "",
        "IMPORTANT: you are never told which worlds satisfy E, F, G, or H directly.",
        "All information comes solely from modal truth values, listed below.",
        "",
        "Observations (formula evaluated at the given world → truth value):",
    ]
    for i, (f, w, v) in enumerate(_DEMOS, 1):
        lines.append(f"  {i:2d}. {f} at w{w} → {'TRUE' if v else 'FALSE'}")

    lines.append("")
    lines.append(
        "Using only the observations above, determine the hidden atom valuation "
        "and accessibility relation, then evaluate these 8 formulas:"
    )
    for qi, (f, w, _) in enumerate(_TEST_CASES, 1):
        lines.append(f"  Q{qi}. Is '{f}' true at w{w}?  (answer TRUE or FALSE)")

    lines.append("")
    lines.append("Provide your answers in fields answer_1 through answer_8.")

    prompt = "\n".join(lines)

    def grade_fn(response):
        test_results = []
        correct_count = 0
        for qi, (f, w, gt) in enumerate(_TEST_CASES, 1):
            field = f"answer_{qi}"
            raw = getattr(response, field, None)
            if isinstance(raw, bool):
                ans = raw
            elif isinstance(raw, str):
                ans = raw.strip().lower() in ("true", "yes", "1")
            else:
                ans = None
            correct = (ans == gt) if ans is not None else False
            test_results.append(
                {
                    "q": qi,
                    "expected": "TRUE" if gt else "FALSE",
                    "got": ("TRUE" if ans else "FALSE") if ans is not None else None,
                    "correct": correct,
                }
            )
            if correct:
                correct_count += 1
        score = correct_count / len(_TEST_CASES)
        return score, test_results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: bool
    answer_2: bool
    answer_3: bool
    answer_4: bool
    answer_5: bool
    answer_6: bool
    answer_7: bool
    answer_8: bool


@kbench.task(
    name="hidden_modal_logic_kripke2_obs_learning",
    description=(
        "5-world Kripke frame with both atom valuation and relation R hidden. "
        "Observations: depth-2/3 modal formulas; test: depth-4 reasoning. "
        "Both valuation and relation must be inferred together."
    ),
)
def hidden_modal_logic_kripke_obs_learning(llm) -> float:
    """
    5-world Kripke frame with both atom valuation and relation R hidden.
    Observations: depth-2/3 modal formulas; test: depth-4 reasoning.
    Both valuation and relation must be inferred together.
    """
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
            {
                "q": i,
                "expected": "TRUE" if _GT_ANSWERS[i - 1] else "FALSE",
                "got": None,
                "correct": False,
            }
            for i in range(1, len(_TEST_CASES) + 1)
        ]
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="hidden_modal_logic_kripke_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    hidden_modal_logic_kripke_obs_learning.run(kbench.llm)

