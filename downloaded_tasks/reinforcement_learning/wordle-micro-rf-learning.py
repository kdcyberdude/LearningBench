#!/usr/bin/env python
# coding: utf-8

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

import kaggle_benchmarks as kbench


class TurnLLM(Protocol):
    def __call__(self, user_message: str) -> str: ...


@dataclass
class RuntimeTaskResult:
    task_id: str
    solved: bool
    num_steps: int
    max_steps: int
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


BUDGET_N = 22
MIN_EXPLORE = 7  # ≥ ceil(6 * 1.2) to probe hidden ANCHOR weights

CODE_LEN = 6

_TASK_DESCRIPTION = (
    "A Mastermind-style code-breaking episode: a hidden 6-symbol pattern over KLMNPQRS with repetition. "
    "Feedback uses veiled telemetry (ANCHOR/SWAY/VOID) with hidden per-slot ANCHOR weights. "
    "The model must infer the pattern from repeated attempts within the step budget. "
    "Success means submitting the exact secret code."
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
    conversation: list = field(default_factory=list)

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
        lock_weights: Optional[list[int]] = None,
    ):
        if code_length < 2:
            raise ValueError("code_length must be >= 2")
        self.code_length = code_length
        self.alphabet = alphabet
        self._rng = rng or random.Random()
        if feedback_mode not in ("standard", "veiled"):
            raise ValueError("feedback_mode must be 'standard' or 'veiled'")
        self._feedback_mode = feedback_mode
        if lock_weights is not None and len(lock_weights) != code_length:
            raise ValueError("lock_weights length must match code_length")
        self._lock_weights = lock_weights
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
        g_norm = guess.upper().strip()
        exact, partial, solved = self.score_guess(guess)
        if solved:
            if self._feedback_mode == "veiled":
                return "Pattern complete. Episode success.", True
            return "All positions correct. You solved it.", True
        if self._feedback_mode == "veiled":
            if self._lock_weights is not None:
                anchored_idx = [
                    i
                    for i in range(self.code_length)
                    if g_norm[i] == self._secret[i]
                ]
                anchor = min(
                    self.code_length,
                    sum(self._lock_weights[i] for i in anchored_idx),
                )
            else:
                anchor = exact
            parts = []
            if anchor:
                parts.append(
                    f"ANCHOR {anchor} (weighted mass on correctly aligned slots)"
                )
            if partial:
                parts.append(f"SWAY {partial} (tokens present but displaced)")
            if not parts:
                parts.append("VOID — no token overlap with the hidden pattern")
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


PromptStyle = Literal["stateless_cumulative", "chat_incremental"]


@dataclass
class EpisodeConfig:
    max_steps: int = 8
    code_length: int = 4
    alphabet: str = "ABCDEF"
    # stateless_cumulative: one giant user blob per turn (OK for single-message APIs).
    # chat_incremental: one user message per turn; rely on chat history (OpenRouter / kbench).
    prompt_style: PromptStyle = "stateless_cumulative"
    # standard: classic exact/partial wording; veiled: ANCHOR/SWAY lexicon (no Wordle naming).
    feedback_mode: str = "standard"
    # Optional per-slot multipliers (length code_length) that reshape reported ANCHOR mass.
    lock_weights: Optional[list[int]] = None


def build_instruction(cfg: EpisodeConfig) -> str:
    if getattr(cfg, "feedback_mode", "standard") == "veiled":
        return (
            "You interact with a hidden **pattern** of length "
            f"{cfg.code_length} over symbols {list(cfg.alphabet)} (repetition allowed). "
            "Each attempt returns environment telemetry only:\n"
            "- **ANCHOR k** — weighted mass on correctly aligned slots (k aggregates hidden per-slot weights).\n"
            "- **SWAY k** — k symbols from your attempt appear in the pattern but in wrong slots.\n"
            "- **VOID** — no symbol overlap.\n"
            "Infer the pattern online; you never see it directly.\n"
            "Your **committed attempt** must appear as either:\n"
            f'- JSON field "guess": "{cfg.alphabet[: cfg.code_length]}" (exactly {cfg.code_length} symbols), or\n'
            f"- a line `GUESS: {'X' * cfg.code_length}`, or\n"
            f"- a **final line** with exactly {cfg.code_length} symbols from the alphabet.\n"
            "Avoid embedding extra alphabet symbols mid-reasoning so the parser does not mis-read you."
        )
    return (
        "You are playing a code-breaking game. A secret code of length "
        f"{cfg.code_length} was chosen using only these symbols: {list(cfg.alphabet)}. "
        "Symbols may repeat. After each guess you receive feedback:\n"
        "- 'exact' = correct symbol in the correct position.\n"
        "- 'partial' = correct symbol but wrong position (count only).\n"
        "You do NOT see the secret. Win by matching the code exactly.\n"
        "You may think out loud, but your **committed guess** must appear as either:\n"
        f'- a JSON field "guess": "{cfg.alphabet[: cfg.code_length]}" (exactly {cfg.code_length} symbols), or\n'
        f"- a line `GUESS: {'X' * cfg.code_length}` (only those symbols on that line), or\n"
        f"- a **final line** that contains nothing but exactly {cfg.code_length} symbols from the alphabet "
        "(optional spaces between symbols).\n"
        "Do not scatter extra symbols from the alphabet in the middle of long reasoning, or the parser may "
        "mis-read your intent."
    )


def run_episode(
    llm: TurnLLM,
    cfg: Optional[EpisodeConfig] = None,
    secret: Optional[str] = None,
    seed: Optional[int] = None,
) -> EpisodeResult:
    cfg = cfg or EpisodeConfig()
    rng = random.Random(seed)
    env = WordleMicroEnv(
        code_length=cfg.code_length,
        alphabet=cfg.alphabet,
        secret=secret,
        rng=rng,
        feedback_mode=getattr(cfg, "feedback_mode", "standard"),
        lock_weights=getattr(cfg, "lock_weights", None),
    )
    history: list[StepResult] = []
    transcript: list[str] = [
        build_instruction(cfg),
        "Game start. Enter your first guess.",
    ]
    last_fb = ""
    conversation: list = []
    for t in range(cfg.max_steps):
        if cfg.prompt_style == "stateless_cumulative":
            user_message = "\n\n".join(transcript)
        elif t == 0:
            user_message = (
                build_instruction(cfg)
                + '\n\nGame start. Submit your first guess (last line / GUESS: / JSON "guess").'
            )
        else:
            user_message = (
                f"Feedback after your last attempt:\n{last_fb}\n\n"
                f"Enter your next guess ({cfg.code_length} symbols from {cfg.alphabet}): "
                'put the code alone on the **last line**, or use `GUESS: ....`, or JSON "guess".'
            )
        raw = llm(user_message)
        conversation.append({"turn": t + 1, "user": user_message, "response": raw})
        guess = parse_guess(raw, cfg.code_length, cfg.alphabet)
        if guess is None:
            fb = (
                f"Invalid guess format at step {t + 1}. "
                f"Send exactly {cfg.code_length} symbols from {cfg.alphabet}."
            )
            solved = False
        else:
            fb, solved = env.feedback_text(guess)
        last_fb = fb
        history.append(
            StepResult(
                guess=guess or raw[:40], feedback=fb, solved=solved, step_index=t
            )
        )
        if solved:
            return EpisodeResult(
                solved=True,
                num_steps=t + 1,
                max_steps=cfg.max_steps,
                secret=env.secret,
                history=history,
                conversation=conversation,
            )
        transcript.append(f"Your output:\n{raw}\n\nFeedback:\n{fb}")
    return EpisodeResult(
        solved=False,
        num_steps=cfg.max_steps,
        max_steps=cfg.max_steps,
        secret=env.secret,
        history=history,
        conversation=conversation,
    )


MAX_STEPS = 16

_INTRO = (
    "You interact with a hidden **pattern** of length 6 over symbols ['K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S'] "
    "(repetition allowed). Each attempt returns environment telemetry only:\n"
    "- **ANCHOR k** — weighted mass on correctly aligned slots (hidden per-slot weights reshape k).\n"
    "- **SWAY k** — k symbols from your attempt appear in the pattern but in wrong slots.\n"
    "- **VOID** — no symbol overlap.\n"
    "Infer the pattern online; you never see it directly.\n"
    'Your **committed attempt** must appear as a JSON field "guess", a line `GUESS: XXXXXX`, '
    "or a **final line** with exactly 6 symbols from the alphabet."
)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    lock_weights = [rng.randint(1, 3) for _ in range(CODE_LEN)]
    cfg = EpisodeConfig(
        max_steps=cap,
        code_length=CODE_LEN,
        alphabet="KLMNPQRS",
        prompt_style="chat_incremental",
        feedback_mode="veiled",
        lock_weights=lock_weights,
    )
    r = run_episode(llm, cfg, seed=seed)

    # Compute best_anchor_frac from episode history
    best_anchor_frac = 0.0
    if r.solved:
        best_anchor_frac = 1.0
    else:
        for sr in r.history:
            m = re.search(r"ANCHOR\s+(\d+)", sr.feedback)
            if m:
                anchor_count = int(m.group(1))
                best_anchor_frac = max(best_anchor_frac, anchor_count / CODE_LEN)

    return RuntimeTaskResult(
        task_id="wordle_micro",
        solved=r.solved,
        num_steps=r.num_steps,
        max_steps=r.max_steps,
        detail={"secret": r.secret, "family": "symbol_feedback"},
        conversation=r.conversation,
        progress=best_anchor_frac,
    )


@dataclass
class _TurnCodeSymbols:
    guess: str


@kbench.task(
    name="wordle_micro_rf_learning",
    description="Mastermind-style 6-symbol pattern with ANCHOR/SWAY/VOID telemetry and hidden ANCHOR weights. Multi-turn RL; return float in [0,1], cap 22 steps.",
)
def wordle_micro_rf_learning(llm) -> float:
    """Break a hidden 6-symbol code over KLMNPQRS with ANCHOR/SWAY/VOID telemetry and hidden per-slot weights. Returns composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnCodeSymbols)
            return r.guess.strip().upper()
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "wordle_micro_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    wordle_micro_rf_learning.run(kbench.llm)

