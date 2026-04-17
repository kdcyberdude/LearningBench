#!/usr/bin/env python
# coding: utf-8

import random
from dataclasses import dataclass

import kaggle_benchmarks as kbench


_TASK_DESCRIPTION = (
    "Tests whether a model can infer the hidden tap positions of an 8-bit LFSR from observed "
    "bit sequences generated from different seeds. Early demos use short sequences consistent "
    "with a simpler 2-tap model; longer demos require all 3 taps. Four test questions ask "
    "for the next bits from four different seeds, probing different parts of the LFSR cycle."
)

_FIXED_SEED = 0
_W = 8


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


def _lfsr_step(state: int, taps: list) -> tuple:
    out_bit = 0
    for t in taps:
        out_bit ^= (state >> t) & 1
    new_state = ((state >> 1) | (out_bit << (_W - 1))) & ((1 << _W) - 1)
    return new_state, out_bit


def _run_lfsr(seed: int, taps: list, length: int) -> list:
    state = seed & ((1 << _W) - 1)
    if state == 0:
        state = 1
    bits = []
    for _ in range(length):
        state, bit = _lfsr_step(state, taps)
        bits.append(bit)
    return bits


def _bits_to_str(bits):
    return "".join(str(b) for b in bits)


def _prepare():
    rng = random.Random(_FIXED_SEED)

    # Fixed taps chosen to be interesting: positions 1, 5, 7 (0-indexed from LSB)
    # This gives a non-trivial LFSR that is NOT consistent with any 2-tap pair on short sequences
    taps = [1, 5, 7]

    # Demo seeds (14 demos total: 3 short + 11 full)
    all_seeds = list(range(1, 256))
    rng.shuffle(all_seeds)

    # Test seeds: fixed for reproducibility, different from demo seeds
    test_seeds = [0xAB, 0x3F, 0xD7, 0x62]

    used_seeds = set(test_seeds)
    demo_seeds = []
    for s in all_seeds:
        if s not in used_seeds and len(demo_seeds) < 14:
            demo_seeds.append(s)
            used_seeds.add(s)

    demos = []
    # 3 short demos (length 10): consistent with many 2-tap combos to create initial ambiguity
    for s in demo_seeds[:3]:
        bits = _run_lfsr(s, taps, 10)
        demos.append((s, bits))
    # 11 longer demos (length 20): force the 3-tap solution
    for s in demo_seeds[3:14]:
        bits = _run_lfsr(s, taps, 20)
        demos.append((s, bits))

    # Each test: show 16 bits, predict next 8 bits
    SHOW = 16
    PREDICT = 8
    test_cases = []
    for ts in test_seeds:
        full = _run_lfsr(ts, taps, SHOW + PREDICT)
        shown = full[:SHOW]
        expected = _bits_to_str(full[SHOW : SHOW + PREDICT])
        test_cases.append((ts, shown, expected))

    lines = [
        f"You are observing a linear feedback shift register of width {_W}.",
        "The tap positions are hidden. Each entry shows a starting seed and the emitted bit sequence.",
        "",
        "Observations (seed → bit sequence):",
    ]
    for i, (s, bits) in enumerate(demos, 1):
        lines.append(f"  {i:2d}. seed=0x{s:02X} → {_bits_to_str(bits)}")
    lines.append("")
    lines.append(
        "Now solve these 4 test questions (each shows 16 continuation bits, predict the next 8):"
    )
    for q, (ts, shown, _) in enumerate(test_cases, 1):
        lines.append(f"  Q{q}: seed=0x{ts:02X}, first 16 bits: {_bits_to_str(shown)}")
        lines.append(f"       Predict bits 17-24.")
    lines += [
        "",
        "Submit the 8-bit prediction strings as bits_1, bits_2, bits_3, bits_4.",
        "Each must be exactly 8 characters of '0' and '1'.",
    ]
    prompt = "\n".join(lines)

    def grade_fn(response):
        results = []
        correct = 0
        for q_idx, (ts, shown, exp) in enumerate(test_cases, 1):
            raw = getattr(response, f"bits_{q_idx}", None)
            got = str(raw).strip().replace(" ", "") if raw is not None else None
            is_correct = got == exp
            results.append(
                {
                    "q": q_idx,
                    "expected": exp,
                    "got": got,
                    "correct": is_correct,
                }
            )
            if is_correct:
                correct += 1
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    bits_1: str
    bits_2: str
    bits_3: str
    bits_4: str


@kbench.task(
    name="linear_feedback_shift_register_obs_learning",
    description=(
        "Observe LFSR (width=8) bit sequences from 14 seeds with hidden tap positions. "
        "Short early sequences fit simpler 2-tap models; longer ones reveal the full 3-tap "
        "rule. Infer the taps and predict the next 8 bits for 4 new seeds."
    ),
)
def linear_feedback_shift_register_obs_learning(llm) -> float:
    """
    8-bit LFSR with 3 hidden taps (positions 1, 5, 7). First 3 demos (length 10) fit 2-tap models; next 11 (length 20) reveal the true rule. Returns fraction of 4 test cases answered correctly.
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
        # Reconstruct expected answers for logging
        taps = [1, 5, 7]
        test_seeds = [0xAB, 0x3F, 0xD7, 0x62]
        SHOW, PREDICT = 16, 8
        test_results = []
        for q_idx, ts in enumerate(test_seeds, 1):
            full = _run_lfsr(ts, taps, SHOW + PREDICT)
            exp = _bits_to_str(full[SHOW : SHOW + PREDICT])
            test_results.append(
                {"q": q_idx, "expected": exp, "got": None, "correct": False}
            )

    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="linear_feedback_shift_register_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )

    return score


if __name__ == "__main__":
    linear_feedback_shift_register_obs_learning.run(kbench.llm)

