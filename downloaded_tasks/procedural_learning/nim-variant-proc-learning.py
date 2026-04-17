#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass

import kaggle_benchmarks as kbench

def efficiency_score(
    solved: bool,
    step_y: int,
    budget_n: int,
    min_necessary: int,
    floor: float = 0.1,
) -> float:
    """Efficiency credit for solving within a budget."""
    if not solved:
        return 0.0
    step_y = max(1, min(step_y, budget_n))
    if step_y <= min_necessary:
        return 1.0
    paid_used = step_y - min_necessary
    paid_budget = budget_n - min_necessary
    if paid_budget <= 0:
        return 1.0
    return max(floor, 1.0 - (1.0 - floor) * (paid_used / paid_budget))


def weighted_learning_mean(round_scores: list, weights: list = None) -> float:
    """Weighted mean emphasising later rounds."""
    n = len(round_scores)
    if n == 0:
        return 0.0
    if weights is None:
        weights = list(range(1, n + 1))
    denom = sum(weights)
    if denom == 0:
        return sum(round_scores) / n
    return sum(s * w for s, w in zip(round_scores, weights)) / denom


def _learning_curve_slope(round_scores: list) -> float:
    """Normalised OLS slope of round scores → [0,1] (0.5 = flat, 1.0 = perfect rise)."""
    n = len(round_scores)
    if n < 2:
        return 0.5
    x_mean = (n - 1) / 2.0
    y_mean = sum(round_scores) / n
    num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(round_scores))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.5
    slope = num / den
    max_slope = 1.0 / (n - 1)
    normalised = max(-1.0, min(1.0, slope / max_slope))
    return round((normalised + 1.0) / 2.0, 4)


def procedural_composite_score(round_scores: list, test_score: float) -> float:
    """Four-component procedural learning score in [0, 1].

    transfer    0.30  — test_score: transfer to novel instances without feedback
    asymptote   0.25  — mean of latter half of round_scores: peak skill reached
    trajectory  0.25  — learning_curve_slope: evidence of genuine improvement
    consistency 0.20  — weighted_learning_mean: overall quality, later rounds weighted more
    """
    n = len(round_scores)
    if n == 0:
        return 0.0
    k = max(1, n // 2)
    asymptote = sum(round_scores[-k:]) / k
    if asymptote < 1e-9 and float(test_score) < 1e-9:
        return 0.0
    consistency = weighted_learning_mean(round_scores)
    trajectory = _learning_curve_slope(round_scores)
    raw = (
        0.30 * float(test_score)
        + 0.25 * asymptote
        + 0.25 * trajectory
        + 0.20 * consistency
    )
    return round(raw, 4)

# ─────────────────────────────────────────────────────────────────────────────


_TASK_DESCRIPTION = (
    """Tests whether the model can learn to win 3-cache Crystal Claim games with a hidden charge
    limit J∈{2,3,4} by observing opponent moves across 5 practice games (25-turn budget each).
    The charge limit caps the maximum crystals claimable from one cache per turn; J is fixed and
    unknown, requiring inference purely from observed opponent behaviour.
    Success requires winning the majority of practice games and all four final test games."""
)


def _log_trace(
    task: str, phases: list[dict], final_score: float, initial_prompt: str = ""
) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {task}\n{sep}")
    print(f"\n  TASK: {_TASK_DESCRIPTION}")
    if initial_prompt:
        print(f"\n  INITIAL PROMPT:\n{initial_prompt}")
    for phase in phases:
        label = phase["label"]
        print(f"\n  [{label}]  correct={phase['correct']}")
        for t in phase["turns"]:
            print(f"    Turn {t['turn']}  submitted: {t['submitted']}", end="")
            if "feedback" in t:
                print(f"  →  {t['feedback']}", end="")
            print()
        status = "PASS ✓" if phase["solved"] else "FAIL ✗"
        print(f"    {status}  steps={phase['steps']}  score={phase['score']:.4f}")
    print(f"\n  Final score : {final_score:.4f}")
    print(f"{sep}\n")


BUDGET = 25
# Opponent must make several visible moves before J can be narrowed to one value.
MIN_NECESSARY = 4

# J in {2, 3, 4} — three charge limits, harder to identify than a binary choice.
LEARNING_INSTANCES = [
    {"j": 2, "piles": [6, 4, 8]},
    {"j": 4, "piles": [5, 7, 6]},
    {"j": 3, "piles": [9, 5, 4]},
    {"j": 4, "piles": [7, 6, 5]},
    {"j": 3, "piles": [4, 8, 6]},
]
TEST_INSTANCES = [
    {"j": 3, "piles": [6, 5, 7]},
    {"j": 4, "piles": [5, 5, 5]},
    {"j": 2, "piles": [8, 4, 3]},
    {"j": 3, "piles": [7, 3, 6]},
]


def _optimal_move(caches: list, j: int) -> tuple:
    """Optimal move for bounded-take heap game under charge limit j."""
    m = j + 1
    nim_sum = 0
    for p in caches:
        nim_sum ^= p % m
    for i, p in enumerate(caches):
        for take in range(1, min(j, p) + 1):
            new_sum = nim_sum ^ (p % m) ^ ((p - take) % m)
            if new_sum == 0:
                return (i, take)
    largest = max(range(len(caches)), key=lambda i: caches[i])
    return (largest, 1)


@dataclass
class _ClaimMove:
    cache: int
    amount: int


def _run_crystal_claim(
    llm, piles_init: list, j: int, budget: int, game_label: str, is_first: bool
) -> tuple:
    """Run one Crystal Claim game. Returns (turns, solved, num_steps, first_prompt)."""
    caches = list(piles_init)
    turns = []
    solved = False
    num_steps = 0

    intro = (
        (
            "Crystal Claim — 3 crystal caches.\n"
            "Players alternate turns. On your turn claim 1 or more crystals from exactly one cache.\n"
            "A hidden charge limit J caps the maximum you may claim per turn.\n"
            "J is a fixed positive integer you must discover by observing your opponent's moves.\n"
            "The player who claims the LAST crystal wins. Your opponent plays optimally.\n"
            "After 5 practice games you will face four final games.\n\n"
            "Scoring note: Your score has four components — transfer (30%): winning the final "
            "games once you know J; asymptote (25%): win rate in the later practice games; "
            "trajectory (25%): whether your win rate improves across practice games (a rising "
            "curve beats a flat one even at the same average); consistency (20%): overall "
            "quality with later games weighted more. Discovering J sooner (fewer moves wasted) "
            "also boosts your within-round efficiency score.\n\n"
        )
        if is_first
        else ""
    )

    next_prompt = (
        f"{intro}"
        f"{game_label} — Initial caches: {piles_init}.\n"
        f"Your turn 1 of {budget}. Current caches: {caches}. "
        "Choose cache index (0-based) and amount to claim."
    )
    first_prompt = next_prompt

    for turn in range(1, budget + 1):
        num_steps = turn
        if all(p == 0 for p in caches):
            break

        try:
            submission: _ClaimMove = llm.prompt(next_prompt, schema=_ClaimMove)
        except Exception:
            entry = {"turn": turn, "submitted": "PARSE_ERROR",
                     "feedback": "Failed to parse response — turn wasted."}
            turns.append(entry)
            next_prompt = (
                f"Your last response could not be parsed. Follow the schema exactly.\n\n"
                f"Your turn {turn + 1} of {budget}. Current caches: {caches}. "
                "Choose cache index (0-based) and amount."
            )
            continue

        cache_idx = submission.cache
        amount = submission.amount
        entry = {"turn": turn, "submitted": (cache_idx, amount)}

        if cache_idx < 0 or cache_idx >= len(caches) or caches[cache_idx] == 0:
            feedback = f"INVALID: cache {cache_idx} is empty or out of range. Caches: {caches}."
            entry["feedback"] = feedback
            turns.append(entry)
            next_prompt = (
                f"{feedback}\n\nYour turn {turn + 1} of {budget}. "
                f"Current caches: {caches}. Choose a valid cache index and amount."
            )
            continue

        amount = max(1, min(amount, j, caches[cache_idx]))
        caches[cache_idx] -= amount
        caches_after_model = list(caches)

        if all(p == 0 for p in caches):
            solved = True
            feedback = f"You claimed {amount} from cache {cache_idx}. Caches: {caches}. YOU WIN!"
            entry["feedback"] = feedback
            turns.append(entry)
            break

        opp_cache, opp_take = _optimal_move(caches, j)
        opp_take = max(1, min(opp_take, caches[opp_cache]))
        caches[opp_cache] -= opp_take
        caches_after_opp = list(caches)

        if all(p == 0 for p in caches):
            feedback = (
                f"You claimed {amount} from cache {cache_idx} → {caches_after_model}. "
                f"Opponent claimed {opp_take} from cache {opp_cache} → {caches_after_opp}. OPPONENT WINS."
            )
            entry["feedback"] = feedback
            turns.append(entry)
            break

        feedback = (
            f"You claimed {amount} from cache {cache_idx} → {caches_after_model}. "
            f"Opponent claimed {opp_take} from cache {opp_cache} → {caches_after_opp}."
        )
        entry["feedback"] = feedback
        turns.append(entry)
        next_prompt = (
            f"{feedback}\n\nYour turn {turn + 1} of {budget}. "
            f"Current caches: {caches}. Choose cache index and amount."
        )

    return turns, solved, num_steps, first_prompt


@kbench.task(
    name="nim_variant_proc_learning",
    description=(
        "Learn to win 3-cache Crystal Claim games with a hidden charge limit J∈{2,3,4} across "
        "5 practice games (25-turn budget each), then win 4 final games. "
        "Score = procedural_composite(learning_scores, wins/4)."
    ),
)
def nim_variant_proc_learning(llm) -> float:
    """5 practice Crystal Claim games then 4 final games; test score = wins/4."""
    phases = []

    with kbench.chats.new("nim_variant"):
        learning_scores = []
        initial_prompt = ""
        for idx, inst in enumerate(LEARNING_INSTANCES):
            j = inst["j"]
            piles_init = inst["piles"]

            turns, solved, num_steps, first_prompt = _run_crystal_claim(
                llm, piles_init, j, BUDGET, f"Practice {idx + 1}/5", is_first=(idx == 0)
            )

            if idx == 0:
                initial_prompt = first_prompt

            eff = efficiency_score(solved, num_steps, BUDGET, MIN_NECESSARY)
            learning_scores.append(eff)
            phases.append({
                "label": f"Practice {idx + 1}/5",
                "correct": {"j": j, "piles": piles_init},
                "turns": turns, "solved": solved,
                "steps": num_steps, "score": eff,
            })

        wins = 0
        for gidx, tins in enumerate(TEST_INSTANCES, start=1):
            test_turns, test_solved, test_num_steps, _ = _run_crystal_claim(
                llm, tins["piles"], tins["j"], BUDGET, f"Final game {gidx}/4", is_first=False,
            )
            if test_solved:
                wins += 1
            phases.append({
                "label": f"Final game {gidx}/4",
                "correct": {"j": tins["j"], "piles": tins["piles"]},
                "turns": test_turns, "solved": test_solved,
                "steps": test_num_steps, "score": 1.0 if test_solved else 0.0,
            })

    test_score = wins / 4.0
    final_score = procedural_composite_score(learning_scores, test_score)
    _log_trace("CRYSTAL CLAIM (NIM VARIANT)", phases, final_score, initial_prompt)
    return final_score


if __name__ == "__main__":
    nim_variant_proc_learning.run(kbench.llm)

