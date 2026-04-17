#!/usr/bin/env python
# coding: utf-8

import re
import random
import kaggle_benchmarks as kbench
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


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
    "A complete weighted graph on 5 nodes has unknown positive integer edge weights. "
    "The model can query up to 12 individual edge weights via EDGE commands, then must "
    "submit the shortest-path distance from node 0 to node 4 via DIST. "
    "CAUTION: Each edge query has a 20% chance of returning the true weight ±1 (noisy oracle). "
    "The model must use strategic edge selection and handle uncertainty. "
    "Success means computing and submitting the exact shortest path within the step budget."
)

BUDGET_N = 18
MIN_EXPLORE = 5  # free exploration turns; no efficiency penalty within this zone


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


N = 5
MAX_STEPS = 22


def _all_pairs() -> list[tuple[int, int]]:
    out = []
    for u in range(N):
        for v in range(u + 1, N):
            out.append((u, v))
    return out


def _floyd(w: dict[tuple[int, int], int]) -> list[list[int]]:
    inf = 10**9
    d = [[inf] * N for _ in range(N)]
    for i in range(N):
        d[i][i] = 0
    for (a, b), wt in w.items():
        d[a][b] = d[b][a] = min(d[a][b], wt)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def run(
    llm: TurnLLM, *, seed: int = 0, max_steps: Optional[int] = None
) -> RuntimeTaskResult:
    cap = max_steps if max_steps is not None else MAX_STEPS
    rng = random.Random(seed)
    wmap: dict[tuple[int, int], int] = {}
    for u, v in _all_pairs():
        wmap[(u, v)] = rng.randint(1, 9)
    dist = _floyd(wmap)
    target = dist[0][4]
    queries = 0
    edges_queried = 0
    last_fb = ""
    intro = (
        "Five sites labeled 0..4 with an unknown **positive integer weight** on every undirected pair (complete graph).\n"
        "Commands:\n"
        "  `EDGE u v` (u≠v) reveals that edge weight — limited telemetry budget.\n"
        "  NOTE: The telemetry oracle is **noisy** — each EDGE reading has a 20% chance of returning the true weight ±1.\n"
        "  `DIST g` submits an integer guess for the **shortest-path total** from 0 to 4.\n"
        "Win only via correct DIST.\n"
        f"You may issue at most **12** EDGE queries; overall **{cap}** commands."
    )
    conversation: list = []
    for t in range(cap):
        user = intro if t == 0 else f"{last_fb}\n\nEDGE or DIST?"
        raw = llm(user).upper()
        conversation.append({"turn": t + 1, "user": user, "response": raw})
        m_d = re.search(r"DIST\s+(-?\d+)", raw)
        if m_d:
            g = int(m_d.group(1))
            if g == target:
                return RuntimeTaskResult(
                    task_id="graph_shortest_path",
                    solved=True,
                    num_steps=t + 1,
                    max_steps=cap,
                    intro=intro,
                    detail={"family": "graph_edge_probes"},
                    conversation=conversation,
                    progress=min(1.0, edges_queried / 10.0),
                )
            last_fb = "DIST wrong."
            continue
        m_e = re.search(r"EDGE\s+(\d+)\s+(\d+)", raw)
        if not m_e:
            last_fb = "Use `EDGE u v` or `DIST g`."
            continue
        if queries >= 12:
            last_fb = "EDGE budget exhausted."
            continue
        u, v = int(m_e.group(1)), int(m_e.group(2))
        if u == v or not (0 <= u < N and 0 <= v < N):
            last_fb = "Invalid endpoints."
            continue
        a, b = (u, v) if u < v else (v, u)
        queries += 1
        edges_queried += 1
        true_w = wmap[(a, b)]
        # 20% probability: return weight ±1 (noisy oracle)
        if rng.random() < 0.20:
            reported_w = true_w + rng.choice([-1, 1])
            reported_w = max(1, reported_w)
        else:
            reported_w = true_w
        last_fb = f"w({u},{v}) = **{reported_w}** ({queries}/12 EDGE uses)."

    return RuntimeTaskResult(
        task_id="graph_shortest_path",
        solved=False,
        num_steps=cap,
        max_steps=cap,
        intro=intro,
        detail={"family": "graph_edge_probes"},
        conversation=conversation,
        progress=min(1.0, edges_queried / 10.0),
    )


@kbench.task(
    name="graph_shortest_path_rf_learning",
    description="Hidden weighted complete graph on 5 nodes; noisy edge queries (20% ±1 noise); guess shortest path 0→4. Multi-turn RL: model only sees environment/user text each turn; return float in [0,1] (higher = fewer steps to succeed), cap 18 steps.",
)
def graph_shortest_path_rf_learning(llm) -> float:
    """Complete weighted 5-node graph; noisy EDGE queries; submit shortest-path distance 0 to 4 via DIST. Score in [0,1]; BUDGET_N=18, MIN_EXPLORE=5."""

    def turn(user_message: str) -> str:
        try:
            return llm.prompt(user_message)
        except Exception:
            return ""

    result = run(turn, seed=0, max_steps=BUDGET_N)
    score = _composite_score(result.solved, result.num_steps, BUDGET_N, MIN_EXPLORE, result.progress)
    _log_trace(
        "graph_shortest_path_rf_learning",
        _TASK_DESCRIPTION,
        result.conversation,
        result.solved,
        result.num_steps,
        BUDGET_N,
        score,
    )
    return score


if __name__ == "__main__":
    graph_shortest_path_rf_learning.run(kbench.llm)

