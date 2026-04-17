#!/usr/bin/env python
# coding: utf-8

import random
import re
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


BUDGET_N = 32
MIN_EXPLORE = 13  # free exploration turns; no efficiency penalty within this zone

_TASK_DESCRIPTION = (
    "Explore arms with verbal payoff feedback; then commit to the best arm."
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


ARMS = "ABCDEFG"
MAX_STEPS = 26  # seven arms + decision budget

_INTRO = (
    "Nonstationary bandit. Seven levers **A–G**.\n"
    "Each turn either:\n"
    "  - Pull by sending exactly one letter **A–G**.\n"
    "  - Or finalize with `DECIDE:X` (your belief about the **currently** strongest lever).\n"
    "Latent payoff structure **drifts slowly** over time; verbal telemetry is all you get.\n"
    "Past strong levers are not guaranteed to stay strong — keep probing."
)


def _bucket(mean: float) -> str:
    if mean >= 0.85:
        return "Tier-Ω resonance — signal saturates the channel"
    if mean >= 0.55:
        return "Tier-Λ lift — clearly above ambient noise"
    if mean >= 0.35:
        return "Tier-Θ flicker — marginal coupling"
    return "Tier-Χ void — barely above dead channel"


def _maybe_drift(means: list[float], rng: random.Random, *, strong: bool = False) -> None:
    """Nonstationary bandit: random walk on latent means (wider after mid-episode)."""
    span = 0.12 if strong else 0.06
    for i in range(len(means)):
        means[i] = max(0.05, min(0.98, means[i] + rng.uniform(-span, span)))


def _pull_progress(pull_history: list, best_arm: str) -> float:
    """Fraction of the last third of pulls (or all pulls if few) on the current best arm."""
    if not pull_history:
        return 0.0
    n = max(1, len(pull_history) // 3)
    last_third = pull_history[-n:]
    return sum(p == best_arm for p in last_third) / len(last_third)


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    means = [0.12, 0.22, 0.32, 0.48, 0.58, 0.72, 0.93]
    rng.shuffle(means)
    best_arm = ARMS[means.index(max(means))]

    pulls = 0
    pull_history: list[str] = []
    last_fb = ""
    conversation: list = []
    for t in range(cap):
        user = (
            _INTRO
            if t == 0
            else f"Feedback from the environment:\n{last_fb}\n\nWhat is your next action?"
        )
        raw = llm(user)
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m_dec = re.search(r"DECIDE\s*:\s*([A-G])", raw.upper())
        if m_dec:
            choice = m_dec.group(1)
            solved = choice == best_arm
            progress = _pull_progress(pull_history, best_arm)
            return RuntimeTaskResult(
                task_id="verbal_bandit",
                solved=solved,
                num_steps=t + 1,
                max_steps=cap,
                detail={
                    "best_arm": best_arm,
                    "decided": choice,
                    "pulls_before_decide": pulls,
                    "family": "bandit_language_feedback",
                },
                conversation=conversation,
                progress=progress,
            )
        letter = None
        for c in raw.upper():
            if c in ARMS:
                letter = c
                break
        if letter is None:
            last_fb = "No valid action. Send one letter A–G to pull, or a line like DECIDE:C to finalize."
            continue
        pulls += 1
        pull_history.append(letter)
        idx = ARMS.index(letter)
        p = means[idx]
        hit = rng.random() < p
        tier = p if hit else min(p * 0.2, 0.18)
        verbal = _bucket(tier)
        last_fb = f"Lever {letter} sampled. Channel readout: **{verbal}**."
        if pulls and (pulls % 2 == 0 or pulls >= 15):
            _maybe_drift(means, rng, strong=(pulls >= 15))
            best_arm = ARMS[means.index(max(means))]

    progress = _pull_progress(pull_history, best_arm)
    return RuntimeTaskResult(
        task_id="verbal_bandit",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        detail={
            "best_arm": best_arm,
            "pulls": pulls,
            "family": "bandit_language_feedback",
        },
        conversation=conversation,
        progress=progress,
    )


@dataclass
class _TurnBandit:
    action: str
    lever: str


@kbench.task(
    name="verbal_bandit_rf_learning",
    description="Explore arms with verbal payoff feedback; then commit to the best arm. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 32 steps.",
)
def verbal_bandit_rf_learning(llm) -> float:
    """Seven-arm nonstationary bandit with verbal payoff tiers; explore then commit with DECIDE:X. Returns composite RL score in [0,1]."""

    def turn(user_message: str) -> str:
        try:
            r = llm.prompt(user_message, schema=_TurnBandit)
            act, lv = r.action.strip().upper(), r.lever.strip().upper()[:1]
            if act == "DECIDE":
                return f"DECIDE:{lv}"
            return lv if lv in "ABCDEFG" else ""
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "verbal_bandit_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    verbal_bandit_rf_learning.run(kbench.llm)

