#!/usr/bin/env python
# coding: utf-8

import math
import random
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
    "The model observes amplitude sequences from a damped oscillator A·exp(−γt)·cos(ωt+φ) "
    "with hidden damping coefficient γ and angular frequency ω. Early short demos show minimal "
    "decay, making the system appear nearly undamped. Longer demos reveal the decay envelope. "
    "Success requires predicting amplitude at future time steps within ±0.05 for 4 test sequences."
)

_FIXED_SEED = 0
_TOL = 0.05


def _oscillator(A: float, gamma: float, omega: float, phi: float, t: int) -> float:
    return round(A * math.exp(-gamma * t) * math.cos(omega * t + phi), 4)


# Precompute demos and test cases at module level
def _build_all():
    rng = random.Random(_FIXED_SEED)

    # Fixed hidden parameters — same gamma and omega for all sequences
    gamma = rng.uniform(0.12, 0.22)
    k = gamma**2 + rng.uniform(0.5, 1.2)
    omega = math.sqrt(max(0.001, k - gamma**2))

    # 14 demo sequences of varying length and phase
    demos = []
    # 3 short sequences (t=0..5) — decay barely visible
    for _ in range(3):
        A = rng.uniform(6.0, 12.0)
        phi = rng.uniform(0.0, math.pi / 6)
        seq = [_oscillator(A, gamma, omega, phi, t) for t in range(6)]
        demos.append(seq)

    # 11 longer sequences (t=0..14) — decay clearly visible
    for _ in range(11):
        A = rng.uniform(4.0, 14.0)
        phi = rng.uniform(0.0, math.pi)
        seq = [_oscillator(A, gamma, omega, phi, t) for t in range(15)]
        demos.append(seq)

    # 4 test sequences: show t=0..9, predict at different future times
    predict_times = [16, 20, 25, 30]
    test_cases = []
    for pt in predict_times:
        A = rng.uniform(4.0, 14.0)
        phi = rng.uniform(0.0, math.pi)
        visible = [_oscillator(A, gamma, omega, phi, t) for t in range(10)]
        gt = _oscillator(A, gamma, omega, phi, pt)
        test_cases.append((visible, pt, gt))

    return gamma, omega, demos, test_cases


_GAMMA, _OMEGA, _DEMOS, _TEST_CASES = _build_all()
_GT_ANSWERS = [round(tc[2], 4) for tc in _TEST_CASES]


def _prepare():
    demos = _DEMOS
    test_cases = _TEST_CASES

    lines = [
        "You are observing amplitude recordings from a physical oscillator system.",
        "Each sequence shows the amplitude at integer time steps t=0, 1, 2, ...",
        "All sequences share the same hidden physical parameters.",
        "",
        "Observation sequences (amplitude values at t=0, 1, 2, ...):",
    ]
    for i, seq in enumerate(demos, 1):
        lines.append(f"  {i}. {seq}")

    lines.append("")
    lines.append("Now solve these 4 test questions.")
    lines.append(
        "Each test shows a partial sequence. Predict the amplitude at the requested future time step."
    )
    for qi, (visible, pt, gt) in enumerate(test_cases, 1):
        lines.append(
            f"  Q{qi}. Sequence (t=0..{len(visible) - 1}): {visible}  →  predict amplitude at t={pt}"
        )

    lines.append("")
    lines.append(
        "Submit your answers (floating-point numbers) in fields answer_1, answer_2, answer_3, answer_4."
    )

    prompt = "\n".join(lines)

    def grade_fn(response):
        test_results = []
        correct_count = 0
        for qi, (visible, pt, gt) in enumerate(test_cases, 1):
            field = f"answer_{qi}"
            raw = getattr(response, field, None)
            try:
                ans = float(raw)
                correct = abs(ans - gt) < _TOL
            except (TypeError, ValueError):
                ans = None
                correct = False
            test_results.append(
                {
                    "q": qi,
                    "expected": round(gt, 4),
                    "got": round(ans, 4) if ans is not None else None,
                    "correct": correct,
                }
            )
            if correct:
                correct_count += 1
        score = correct_count / 4
        return score, test_results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: float
    answer_2: float
    answer_3: float
    answer_4: float


@kbench.task(
    name="hidden_damping_physics_obs_learning",
    description=(
        "Observe amplitude sequences from a damped oscillator A·exp(−γt)·cos(ωt+φ) with "
        "hidden γ and ω. Early short sequences hide damping; longer demos reveal decay. "
        "Predict amplitudes at 4 different future time steps across 4 test sequences."
    ),
)
def hidden_damping_physics_obs_learning(llm) -> float:
    """Infer hidden damping γ and angular frequency ω from 14 oscillator sequences; predict amplitudes at 4 future steps."""
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
            {"q": i, "expected": _GT_ANSWERS[i - 1], "got": None, "correct": False}
            for i in range(1, 5)
        ]
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="hidden_damping_physics_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    hidden_damping_physics_obs_learning.run(kbench.llm)

