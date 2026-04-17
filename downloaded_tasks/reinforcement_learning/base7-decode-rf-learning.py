#!/usr/bin/env python
# coding: utf-8

import itertools
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import kaggle_benchmarks as kbench


GLYPHS = ("⬡", "⬢", "⬣")
BUDGET_N = 36


@dataclass
class _GlyphAlternatingEpisodeResult:
    solved: bool
    num_steps: int
    max_steps: int
    intro: str
    detail: dict[str, Any]
    conversation: list = field(default_factory=list)
    progress: float = 0.0


def _digits_base3_msb_first(n: int) -> list[int]:
    if n == 0:
        return [0]
    out: list[int] = []
    x = n
    while x:
        out.append(x % 3)
        x //= 3
    out.reverse()
    return out


def _digit_to_glyph_maps() -> list[dict[int, str]]:
    maps: list[dict[int, str]] = []
    for perm in itertools.permutations(GLYPHS, 3):
        maps.append({0: perm[0], 1: perm[1], 2: perm[2]})
    return maps


def _invert_glyph_map(d2g: dict[int, str]) -> dict[str, int]:
    return {d2g[i]: i for i in (0, 1, 2)}


def _encode(
    n: int,
    d2g_even: dict[int, str],
    d2g_odd: dict[int, str],
    msb_first: bool,
) -> str:
    digits = _digits_base3_msb_first(n)
    seq = digits if msb_first else digits[::-1]
    return "".join(
        d2g_even[d] if p % 2 == 0 else d2g_odd[d] for p, d in enumerate(seq)
    )


def _decode_to_msb_digit_str(
    glyph_str: str,
    inv_even: dict[str, int],
    inv_odd: dict[str, int],
    msb_first: bool,
) -> Optional[str]:
    got: list[int] = []
    for p, ch in enumerate(glyph_str):
        inv = inv_even if p % 2 == 0 else inv_odd
        if ch not in inv:
            return None
        got.append(inv[ch])
    msb_digits = got if msb_first else got[::-1]
    return "".join(str(d) for d in msb_digits)


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def _witness_mismatch_total(
    inv_even: dict[str, int],
    inv_odd: dict[str, int],
    msb_first: bool,
    witnesses: list[tuple[str, str]],
) -> int:
    total = 0
    for g_str, d_ref in witnesses:
        d_hat = _decode_to_msb_digit_str(g_str, inv_even, inv_odd, msb_first)
        if d_hat is None:
            total += len(d_ref) + 1
        else:
            total += _hamming(d_hat, d_ref)
    return total


def _enumerate_hypotheses() -> list[tuple[dict[int, str], dict[int, str], bool]]:
    out: list[tuple[dict[int, str], dict[int, str], bool]] = []
    maps = _digit_to_glyph_maps()
    for ge in maps:
        for go in maps:
            for msb in (True, False):
                out.append((ge, go, msb))
    return out


def _count_fitting_hypotheses(
    witnesses: list[tuple[str, str]],
) -> tuple[int, list[tuple[dict[int, str], dict[int, str], bool]]]:
    fits: list[tuple[dict[int, str], dict[int, str], bool]] = []
    for ge, go, msb in _enumerate_hypotheses():
        inv_e, inv_o = _invert_glyph_map(ge), _invert_glyph_map(go)
        if _witness_mismatch_total(inv_e, inv_o, msb, witnesses) == 0:
            fits.append((ge, go, msb))
    return len(fits), fits


def _build_intro(
    witnesses: list[tuple[str, str]],
    test_glyph: str,
) -> str:
    lines = [
        "A fixed secret turns each nonnegative integer into a **glyph string** over "
        f"{list(GLYPHS)}.",
        "",
        "Structure (constant for this episode, all unknown):",
        "- Base-three expansion with **no leading-zero digit** except the integer 0.",
        "- The digit list is read either **most-significant-first** or **reversed** along the string.",
        "- Glyph at index 0 of that list uses one digit→glyph bijection; index 1 uses another "
        "**independent** bijection on the same three glyphs; indices alternate.",
        "",
        "**Witness rows** (each row: glyph string, then the matching digit list in MSB-first "
        "standard form using characters 0, 1, 2 only):",
    ]
    for g, d in witnesses:
        lines.append(f"- `{g}`  →  `{d}`")
    lines += [
        "",
        f"**Test glyph string:** `{test_glyph}`",
        "",
        "Reply each turn with **one** line:",
        "- Probe: `MAP EVEN abc ODD def ENDIAN MSB` or `... ENDIAN LSB` where `abc` and `def` "
        f"are each a permutation of 012 meaning, in glyph order {list(GLYPHS)}, the digit "
        "carried by ⬡, ⬢, ⬣ at even and odd indices respectively.",
        "- Final answer: `DIGITS zzz...` — MSB-first 0/1/2 digit list for the test string under "
        "the episode rules.",
        "",
        "After a `MAP` line you receive the **total digit mismatches** summed across all witness "
        "rows (exact Hamming; length mismatch counts per position). 0 means the probe matches "
        "every witness.",
    ]
    return "\n".join(lines)


_MAP_RE = re.compile(
    r"MAP\s+EVEN\s+([012]{3})\s+ODD\s+([012]{3})\s+ENDIAN\s+(MSB|LSB)",
    flags=re.IGNORECASE,
)
_DIGITS_RE = re.compile(r"DIGITS\s+([012]+)\s*$", flags=re.IGNORECASE | re.MULTILINE)


def _parse_map_line(raw: str) -> Optional[tuple[dict[str, int], dict[str, int], bool]]:
    m = _MAP_RE.search(raw.upper())
    if not m:
        return None
    a, b, end = m.group(1), m.group(2), m.group(3).upper()
    if sorted(a) != ["0", "1", "2"] or sorted(b) != ["0", "1", "2"]:
        return None
    inv_e = {GLYPHS[i]: int(a[i]) for i in range(3)}
    inv_o = {GLYPHS[i]: int(b[i]) for i in range(3)}
    return inv_e, inv_o, end == "MSB"


def _parse_digits_submission(raw: str) -> Optional[str]:
    m = _DIGITS_RE.search(raw.strip())
    if not m:
        return None
    s = m.group(1)
    return s if re.fullmatch(r"[012]+", s) else None


def _sample_episode(
    rng: random.Random,
    *,
    n_witness: int = 5,
    max_attempts: int = 400,
) -> Optional[
    tuple[
        dict[int, str],
        dict[int, str],
        bool,
        list[tuple[str, str]],
        str,
        str,
    ]
]:
    maps = _digit_to_glyph_maps()
    for _ in range(max_attempts):
        ge = rng.choice(maps)
        go = rng.choice(maps)
        msb = rng.random() < 0.5
        used_ns: set[int] = set()
        witnesses: list[tuple[str, str]] = []
        for _w in range(n_witness):
            for _try in range(80):
                n = rng.randint(4, 260)
                if n in used_ns:
                    continue
                g = _encode(n, ge, go, msb)
                d = "".join(str(x) for x in _digits_base3_msb_first(n))
                if len(g) < 2:
                    continue
                witnesses.append((g, d))
                used_ns.add(n)
                break
            else:
                witnesses = []
                break
        if len(witnesses) != n_witness:
            continue
        cnt, _fits = _count_fitting_hypotheses(witnesses)
        if cnt != 1:
            continue
        for _try in range(120):
            n_t = rng.randint(4, 260)
            if n_t in used_ns:
                continue
            g_t = _encode(n_t, ge, go, msb)
            d_t = "".join(str(x) for x in _digits_base3_msb_first(n_t))
            if len(g_t) < 2:
                continue
            return ge, go, msb, witnesses, g_t, d_t
    return None


def _run_glyph_alternating_episode(
    llm: Callable[[str], str],
    *,
    seed: int = 0,
    max_steps: Optional[int] = None,
) -> _GlyphAlternatingEpisodeResult:
    cap = max_steps if max_steps is not None else BUDGET_N
    rng = random.Random(seed)
    packed = _sample_episode(rng)
    if packed is None:
        return _GlyphAlternatingEpisodeResult(
            solved=False,
            num_steps=0,
            max_steps=cap,
            intro="(internal) failed to sample a uniquely determined episode",
            detail={"family": "alternating_glyph_isomorphism", "error": "sample_failed"},
        )
    _ge, _go, _msb, witnesses, test_glyph, target_digits = packed
    intro = _build_intro(witnesses, test_glyph)
    _max_mm = sum(len(d) for _, d in witnesses) + len(witnesses)
    _min_mm_seen: float = float(_max_mm)
    last_fb = ""
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nNext line (MAP or DIGITS)?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        d_sub = _parse_digits_submission(raw)
        if d_sub is not None:
            if d_sub == target_digits:
                _progress = max(0.0, 1.0 - _min_mm_seen / _max_mm) if _max_mm > 0 else 1.0
                return _GlyphAlternatingEpisodeResult(
                    solved=True,
                    num_steps=t + 1,
                    max_steps=cap,
                    intro=intro,
                    detail={"family": "alternating_glyph_isomorphism"},
                    conversation=conversation,
                    progress=_progress,
                )
            last_fb = "DIGITS string incorrect for the test row under the hidden rules."
            continue
        parsed = _parse_map_line(raw)
        if parsed is None:
            last_fb = (
                "Expected `MAP EVEN abc ODD def ENDIAN MSB` (or LSB) with abc,def permutations "
                "of 012, or `DIGITS ...`."
            )
            continue
        inv_e, inv_o, msb = parsed
        mm = _witness_mismatch_total(inv_e, inv_o, msb, witnesses)
        _min_mm_seen = min(_min_mm_seen, float(mm))
        last_fb = f"MAP score: **{mm}** total witness digit mismatches."

    _progress = max(0.0, 1.0 - _min_mm_seen / _max_mm) if _max_mm > 0 else 0.0
    return _GlyphAlternatingEpisodeResult(
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "alternating_glyph_isomorphism"},
        conversation=conversation,
        progress=_progress,
    )


# --- Task wrapper ---


class TurnLLM(Protocol):
    def __call__(self, user_message: str) -> str: ...


@dataclass
class RuntimeTaskResult:
    task_id: str
    solved: bool
    num_steps: int
    max_steps: int
    intro: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    conversation: list = field(default_factory=list)
    progress: float = 0.0


def _composite_score(
    solved: bool,
    step_y: int,
    budget_n: int,
    min_explore: int,
    progress: float,
    *,
    floor: float = 0.10,
) -> float:
    """
    Graded RL cognitive ability score in [0, 1].
      success   (0.55) — did the model solve the task?
      efficiency (0.25) — how quickly (only when solved)?
      progress  (0.20) — how close did it get (always defined)?
    A model that never engages scores 0.0; partial progress is always rewarded.
    """
    progress = max(0.0, min(1.0, float(progress)))
    if solved:
        step_y = max(1, min(step_y, budget_n))
        if step_y <= min_explore:
            eff = 1.0
        else:
            paid_used = step_y - min_explore
            paid_budget = budget_n - min_explore
            eff = max(floor, 1.0 - (1.0 - floor) * (paid_used / paid_budget)) if paid_budget > 0 else 1.0
    else:
        eff = 0.0
    return round(0.55 * float(solved) + 0.25 * eff + 0.20 * progress, 4)


_TASK_DESCRIPTION = (
    "Alternating-position glyph substitution over base-3 digit strings with unknown endianness; "
    "witness glyph↔digit pairs fix a unique hidden isomorphism; MAP probes return mismatch mass; "
    "commit with DIGITS. RL subskill: hypothesis formation + planning under constraint."
)

MIN_EXPLORE = 9


def _log_trace(
    task: str,
    description: str,
    conversation: list,
    solved: bool,
    num_steps: int,
    budget: int,
    final_score: float,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {description}")
    for entry in conversation:
        print(f"\n[USER — Turn {entry['turn']}]\n{entry['user']}")
        print(f"\n[ASSISTANT — Turn {entry['turn']}]\n{entry['response']}")
    status = "PASS ✓" if solved else "FAIL ✗"
    print(f"\n  RESULT: {status}  steps={num_steps}/{budget}  score={final_score:.4f}")
    print(f"{sep}\n")


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else BUDGET_N
    er = _run_glyph_alternating_episode(llm, seed=seed, max_steps=cap)
    return RuntimeTaskResult(
        task_id="base7_decode",
        solved=er.solved,
        num_steps=er.num_steps,
        max_steps=er.max_steps,
        intro=er.intro,
        detail=er.detail,
        conversation=er.conversation,
        progress=er.progress,
    )


@kbench.task(
    name="base7_decode_rf_learning",
    description="Alternating glyph substitution on base-3 strings; witnesses fix isomorphism; MAP probes mismatch; RL/planning; score∈[0,1], cap 36.",
)
def base7_decode_rf_learning(llm) -> float:
    """Alternating glyph maps on base-3 strings; MAP mismatch telemetry; composite score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return str(llm.prompt(user_message))
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "base7_decode_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    base7_decode_rf_learning.run(kbench.llm)

