#!/usr/bin/env python
# coding: utf-8

import random
import re
from collections import Counter
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
    conversation: list = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
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


BUDGET_N = 30
MIN_EXPLORE = 9  # scalar COUPLING_SUM only — needs more free probes than split telemetry

_TASK_DESCRIPTION = (
    "Mastermind-style code over nine symbols; each attempt returns only **COUPLING_SUM** "
    "(LOCK+DRIFT combined under multiset rules) with **no split** into anchored vs displaced mass."
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


@dataclass
class StepResult:
    guess: str
    feedback: str
    solved: bool
    step_index: int


@dataclass
class EpisodeResult:
    """Aggregate outcome for one episode (maps cleanly to kbench return types)."""

    solved: bool
    num_steps: int
    max_steps: int
    secret: str
    history: list[StepResult] = field(default_factory=list)

    def as_tuple_score(self) -> tuple[int, int]:
        """(1/0 success, steps_used) — cookbook-friendly tuple[int, int]."""
        return (1 if self.solved else 0, self.num_steps)

    def success_rate_component(self) -> float:
        return 1.0 if self.solved else 0.0


class WordleMicroEnv:
    """
    Mastermind / Wordle micro variant: fixed alphabet, fixed code length.

    Feedback format (no position leak for partials beyond counts):
    - "X exact" — correct symbol in correct slot
    - "Y partial" — correct symbol, wrong slot (count only)
    """

    def __init__(
        self,
        code_length: int = 4,
        alphabet: str = "ABCDEF",
        secret: Optional[str] = None,
        rng: Optional[random.Random] = None,
        *,
        feedback_mode: str = "standard",
    ):
        if code_length < 2:
            raise ValueError("code_length must be >= 2")
        self.code_length = code_length
        self.alphabet = alphabet
        self._rng = rng or random.Random()
        if feedback_mode not in ("standard", "veiled"):
            raise ValueError("feedback_mode must be 'standard' or 'veiled'")
        self._feedback_mode = feedback_mode
        if secret is not None:
            if len(secret) != code_length or any(c not in alphabet for c in secret):
                raise ValueError("invalid secret")
            self._secret = secret.upper()
        else:
            self._secret = "".join(
                self._rng.choice(alphabet) for _ in range(code_length)
            )

    @property
    def secret(self) -> str:
        return self._secret

    def score_guess(self, guess: str) -> tuple[int, int, bool]:
        """Return (exact, partial, solved)."""
        g = guess.upper().strip()
        if len(g) != self.code_length or any(c not in self.alphabet for c in g):
            raise ValueError("invalid guess format")
        if g == self._secret:
            return self.code_length, 0, True

        secret_list = list(self._secret)
        guess_list = list(g)
        exact = 0
        for i in range(self.code_length):
            if guess_list[i] == secret_list[i]:
                exact += 1
                secret_list[i] = guess_list[i] = "*"  # mark used

        rem_s = Counter(c for c in secret_list if c != "*")
        partial = 0
        for i in range(self.code_length):
            if guess_list[i] == "*":
                continue
            if guess_list[i] in rem_s and rem_s[guess_list[i]] > 0:
                partial += 1
                rem_s[guess_list[i]] -= 1
        return exact, partial, False

    def feedback_text(self, guess: str) -> tuple[str, bool]:
        exact, partial, solved = self.score_guess(guess)
        if solved:
            if self._feedback_mode == "veiled":
                return "Pattern complete. Episode success.", True
            return "All positions correct. You solved it.", True
        if self._feedback_mode == "veiled":
            parts = []
            if exact:
                parts.append(f"LOCK {exact} (tokens anchored in the correct slots)")
            if partial:
                parts.append(f"DRIFT {partial} (tokens present but misplaced)")
            if not parts:
                parts.append("NULL — no token overlap with the hidden pattern")
            return "; ".join(parts) + ".", False
        parts = []
        if exact:
            parts.append(f"{exact} exact (right letter, right place)")
        if partial:
            parts.append(f"{partial} partial (right letter, wrong place)")
        if not parts:
            parts.append("No overlap with the secret pattern")
        return "; ".join(parts) + ".", False


def parse_guess(raw: str, code_length: int, alphabet: str) -> Optional[str]:
    """
    Extract a code from model output without taking the *first* A–F letters in the text
    (which breaks on long reasoning that mentions patterns like AABB).

    Priority:
    1) JSON: "guess": "ABCD" or "code": "ABCD"
    2) Line tag: GUESS: ABCD (or =)
    3) Last non-empty line that is *only* exactly code_length alphabet symbols (after removing spaces)
    4) Last isolated block: exactly code_length alphabet chars not adjacent to other alphabet chars
    5) If the entire message has exactly code_length alphabet letters total (no extras), use them in order
    """
    if not raw or not raw.strip():
        return None
    u = raw.upper()
    esc = re.escape(alphabet)
    n = code_length

    def _ok(s: str) -> bool:
        return len(s) == n and all(c in alphabet for c in s)

    # 1) JSON fields
    for key in ("guess", "code", "answer"):
        m = re.search(rf'"{key}"\s*:\s*"([{esc}]{{{n}}})"', u)
        if m and _ok(m.group(1)):
            return m.group(1)

    # 2) GUESS: CODE
    m = re.search(rf"GUESS\s*[:=]\s*([{esc}]{{{n}}})\b", u)
    if m and _ok(m.group(1)):
        return m.group(1)

    # 3) Last line only symbols from alphabet, length n (spaces allowed between symbols)
    lines = [ln.strip() for ln in u.splitlines() if ln.strip()]
    for ln in reversed(lines):
        compact = re.sub(rf"[^{esc}]", "", ln)
        if _ok(compact):
            return compact

    # 4) Last isolated run of exactly n alphabet letters (not part of a longer run)
    pat = rf"(?<![{esc}])([{esc}]{{{n}}})(?![{esc}])"
    ms = list(re.finditer(pat, u))
    if ms:
        cand = ms[-1].group(1)
        if _ok(cand):
            return cand

    # 5) Legacy: whole message contributes exactly n letters total (reasoning used no extra A–F)
    all_letters = re.findall(rf"[{esc}]", u)
    if len(all_letters) == n:
        cand = "".join(all_letters)
        if _ok(cand):
            return cand

    return None


CODE_LEN = 5
ALPH = "ABCDEFGHI"
MAX_STEPS = 20

_INTRO = (
    f"A hidden pattern length {CODE_LEN} over symbols {list(ALPH)} (repetition allowed).\n"
    "Each attempt returns a **single scalar**: the environment's **COUPLING_SUM** = LOCK count + DRIFT count "
    "(see task-01 style semantics) but **without** splitting the two components.\n"
    "Infer the pattern purely from scalar echoes.\n"
    f'Submit via JSON "guess", `GUESS:`, or a final line of {CODE_LEN} symbols.\n'
    f"Budget: {MAX_STEPS} attempts."
)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    env = WordleMicroEnv(
        code_length=CODE_LEN, alphabet=ALPH, rng=rng, feedback_mode="veiled"
    )
    last_fb = ""
    conversation: list = []
    best_coupling_frac = 0.0
    for t in range(cap):
        user = _INTRO if t == 0 else f"Echo:\n{last_fb}\n\nNext pattern?"
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        g = parse_guess(raw, CODE_LEN, ALPH)
        if g is None:
            last_fb = f"Unreadable attempt — need {CODE_LEN} symbols from the alphabet."
            continue
        exact, partial, solved = env.score_guess(g)
        if solved:
            best_coupling_frac = 1.0
            return RuntimeTaskResult(
                task_id="mastermind_aggregate",
                solved=True,
                num_steps=t + 1,
                max_steps=cap,
                conversation=conversation,
                detail={"family": "mastermind_aggregate_feedback"},
                progress=best_coupling_frac,
            )
        coupling_sum = exact + partial
        best_coupling_frac = max(best_coupling_frac, coupling_sum / CODE_LEN)
        last_fb = (
            f"COUPLING_SUM = **{coupling_sum}** (combined multiset overlap; anchoring vs displacement **not** disclosed). "
            "Zero means no multiset overlap."
        )

    return RuntimeTaskResult(
        task_id="mastermind_aggregate",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        conversation=conversation,
        detail={"family": "mastermind_aggregate_feedback"},
        progress=best_coupling_frac,
    )


@dataclass
class _TurnCodeSymbols:
    guess: str


@kbench.task(
    name="mastermind_aggregate_rf_learning",
    description="Mastermind-style code; only scalar COUPLING_SUM feedback (no LOCK/DRIFT split). Multi-turn RL; return float in [0,1], cap 30 steps.",
)
def mastermind_aggregate_rf_learning(llm) -> float:
    """Mastermind with scalar COUPLING_SUM only; returns composite [0,1] for up to 30 turns."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnCodeSymbols)
            return r.guess.strip().upper()
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "mastermind_aggregate_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    mastermind_aggregate_rf_learning.run(kbench.llm)

