#!/usr/bin/env python
# coding: utf-8

import random
import re
import string
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import kaggle_benchmarks as kbench


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


# Prime modulus 31 ⇒ 30×31 = 930 affine keys (a invertible mod 31, b arbitrary).
P = 31
ALPH = (string.digits + string.ascii_uppercase)[:P]

BUDGET_N = 28
MIN_EXPLORE = 7  # need probes to recover (a,b) before exploitation

_TASK_DESCRIPTION = (
    "Hidden affine index map on a 31-symbol roster: ciphertext_index = (A·plaintext_index + B) mod 31 "
    "with unknown (A,B). Model discovers the map by submitting PROBE: <word> queries. "
    "Hidden key space 30×31=930. Feedback uses PHASE_REJECT / PHASE_LOCK only for commit attempts."
)


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


MAX_STEPS = 22


def _idx(ch: str) -> int:
    return ALPH.index(ch)


def _ch(i: int) -> str:
    return ALPH[i % P]


def _encode_block(text: str, a: int, b: int) -> str:
    return "".join(_ch(a * _idx(c) + b) for c in text if c in ALPH)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    a = rng.randint(1, P - 1)  # all nonzero residues invert mod prime P
    b = rng.randint(0, P - 1)

    def rand_word(n: int) -> str:
        return "".join(rng.choice(ALPH) for _ in range(n))

    q_plain = rand_word(5)
    c_ans = _encode_block(q_plain, a, b)

    intro = (
        f"Roster (indices 0..{P - 1}): `{ALPH}`.\n"
        "A hidden **affine slot map** transforms plaintext into ciphertext **index-wise**:\n"
        f"  ciphertext_index = ( **A** × plaintext_index + **B** ) mod {P},\n"
        f"  with unknown integers **A** (invertible mod {P}) and **B**.\n"
        "You never see (A,B) directly.\n\n"
        "To learn the map, submit `PROBE: <word>` using roster symbols — you will see its ciphertext.\n"
        "When confident, submit exactly **five** roster symbols as your answer for the query word.\n"
        "Wrong submissions return **PHASE_REJECT**; exact matches return **PHASE_LOCK** (episode ends).\n\n"
        f"Query word to encode: **{q_plain}**\n"
        f"Budget: {cap} turns."
    )

    distinct_probes: set[str] = set()
    last_fb = ""
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nYour action (PROBE: <word>  or  five roster symbols)?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})

        # Check for PROBE action first
        raw_upper = raw.upper()
        probe_match = re.search(r"PROBE\s*:\s*([0-9A-Z]+)", raw_upper)
        if probe_match:
            probe_word = probe_match.group(1)
            valid_probe = "".join(c for c in probe_word if c in ALPH)
            if not valid_probe:
                last_fb = "PROBE word must contain roster symbols."
                continue
            distinct_probes.add(valid_probe)
            probe_cipher = _encode_block(valid_probe, a, b)
            last_fb = f"PROBE result: {valid_probe} → {probe_cipher}"
            continue

        # Otherwise treat as a commit attempt (5 roster symbols)
        letters_g = [x for x in raw_upper if x in ALPH]
        guess = "".join(letters_g[:5])
        if len(guess) != 5:
            last_fb = "PHASE_REJECT — need exactly five roster symbols, or use PROBE: <word> to query."
            continue
        if guess == c_ans:
            last_fb = "PHASE_LOCK — ciphertext matches the hidden map."
            progress = min(1.0, len(distinct_probes) / 3.0)
            return RuntimeTaskResult(
                task_id="shift_cipher",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                intro=intro,
                detail={"family": "affine_index_map", "P": P},
                conversation=conversation,
                progress=progress,
            )
        last_fb = "PHASE_REJECT — ciphertext inconsistent with the hidden map."

    progress = min(1.0, len(distinct_probes) / 3.0)
    return RuntimeTaskResult(
        task_id="shift_cipher",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "affine_index_map", "P": P},
        conversation=conversation,
        progress=progress,
    )


@kbench.task(
    name="shift_cipher_rf_learning",
    description="Hidden affine index map (A,B) mod 31 over a 31-symbol roster; model discovers it via PROBE queries; PHASE_* feedback for commit attempts. Multi-turn RL; return float in [0,1], cap 28 steps.",
)
def shift_cipher_rf_learning(llm) -> float:
    """Infer hidden affine cipher (A,B) mod 31 on a 31-symbol alphabet using PROBE words, then commit; PHASE_LOCK/PHASE_REJECT on commits. Returns composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "shift_cipher_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    shift_cipher_rf_learning.run(kbench.llm)

