#!/usr/bin/env python
# coding: utf-8

import random
from dataclasses import dataclass
from itertools import permutations

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
    "Each path-graph flow network has edges with integer capacities and tags from {A,B,C,D}. "
    "A hidden tag priority ordering governs the max flow: sort edges by (tag_priority ASC, "
    "capacity ASC) and return the capacity at sorted position 1 (0-indexed)."
)

_FIXED_SEED = 42

# Hidden total ordering (index = priority rank, 0 = lowest).
# True order: C(0) < A(1) < D(2) < B(3)
_TAG_ORDER = ["C", "A", "D", "B"]
_TAG_RANK  = {tag: i for i, tag in enumerate(_TAG_ORDER)}
_TAGS      = ["A", "B", "C", "D"]


def _max_flow(edges: list, rank: dict = None) -> int:
    """
    edges: list of (capacity, tag).
    Sort by (tag_priority ASC, capacity ASC); return the capacity of the element
    at position 1 (0-indexed) — the 'second in priority order'.
    This is a total order so the result is always deterministic.
    """
    if rank is None:
        rank = _TAG_RANK
    # Stable secondary sort: (rank, cap, original_index) for full determinism
    ranked = sorted(
        enumerate(edges),
        key=lambda ie: (rank[ie[1][1]], ie[1][0], ie[0]),
    )
    return ranked[1][1][0]


def _format_edges(edges: list) -> str:
    return "[" + ", ".join(f"(cap={c}, tag={t})" for c, t in edges) + "]"


def _is_deceptive(edges: list) -> bool:
    """True iff the position-1 element also has the global minimum raw capacity."""
    return _max_flow(edges) == min(c for c, _ in edges)


def _is_revealing(edges: list) -> bool:
    return not _is_deceptive(edges)


def _make_demos(rng: random.Random) -> list:
    """
    Generate 14 demo examples:
      - 4 deceptive: position-1 edge coincidentally has min raw capacity.
      - 10 revealing: position-1 edge does NOT have min raw capacity.

    Each network has 4–6 edges; tags drawn from {A,B,C,D} with repeats allowed.

    We verify after generation that all 14 demos together uniquely determine the
    hidden ordering (24 permutations checked).

    Demo order: 2 deceptive, 4 revealing, 2 deceptive, 6 revealing.
    """
    def _sample_edges():
        n = rng.randint(4, 6)
        caps = [rng.randint(1, 20) for _ in range(n)]
        tags = [rng.choice(_TAGS) for _ in range(n)]
        return list(zip(caps, tags))

    seen: set = set()
    deceptive: list = []
    revealing:  list = []
    attempts = 0

    while (len(deceptive) < 4 or len(revealing) < 10) and attempts < 200_000:
        attempts += 1
        edges = _sample_edges()
        key = tuple(sorted(edges))
        if key in seen:
            continue
        seen.add(key)

        if _is_deceptive(edges) and len(deceptive) < 4:
            deceptive.append((edges, _max_flow(edges)))
        elif _is_revealing(edges) and len(revealing) < 10:
            revealing.append((edges, _max_flow(edges)))

    # Interleave: 2 deceptive, 4 revealing, 2 deceptive, 6 revealing
    ordered = (
        deceptive[:2]
        + revealing[:4]
        + deceptive[2:4]
        + revealing[4:10]
    )
    return ordered


def _verify_uniqueness(demos: list) -> tuple:
    """
    Enumerate all 24 permutations of {A,B,C,D}. Return (is_unique, valid_orders).
    """
    valid_orders = []
    for perm in permutations(_TAGS):
        rank = {tag: i for i, tag in enumerate(perm)}
        if all(_max_flow(e, rank) == f for e, f in demos):
            valid_orders.append(list(perm))
    return len(valid_orders) == 1, valid_orders


def _make_test_cases(rng: random.Random) -> list:
    """
    Generate 4 test networks. Each must be revealing (position-1 edge ≠ min-cap edge)
    so the answer depends strictly on knowing the priority ordering.
    Tags drawn from {A,B,C,D} with repeats; 4–6 edges each.
    """
    def _sample_edges():
        n = rng.randint(4, 6)
        caps = [rng.randint(1, 20) for _ in range(n)]
        tags = [rng.choice(_TAGS) for _ in range(n)]
        return list(zip(caps, tags))

    cases: list = []
    seen: set = set()
    attempts = 0
    while len(cases) < 4 and attempts < 200_000:
        attempts += 1
        edges = _sample_edges()
        key = tuple(sorted(edges))
        if key in seen:
            continue
        seen.add(key)
        if _is_revealing(edges):
            cases.append((edges, _max_flow(edges)))
    return cases


# ── Module-level precomputation ───────────────────────────────────────────────

_DEMOS      = _make_demos(random.Random(_FIXED_SEED))
_TEST_CASES = _make_test_cases(random.Random(_FIXED_SEED + 999))

_unique, _valid_orders = _verify_uniqueness(_DEMOS)
assert _unique, (
    f"Hidden ordering not uniquely determined by demos! "
    f"{len(_valid_orders)} consistent orderings found: {_valid_orders}"
)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(demos: list, test_cases: list) -> str:
    lines = [
        "You are observing a series of flow networks. Each network is a path graph",
        "s → v1 → v2 → ... → t. Every edge has a raw integer capacity and a tag",
        "drawn from {A, B, C, D}. A hidden rule involving the tags determines the",
        "maximum flow for each network.",
        "",
        "Study these examples carefully:",
        "",
    ]
    for i, (edges, flow) in enumerate(demos, 1):
        lines.append(f"  Example {i}:")
        lines.append(f"    Edges    : {_format_edges(edges)}")
        lines.append(f"    Max flow : {flow}")
        lines.append("")

    lines += [
        "Using the same hidden rule, compute the maximum flow for each network below:",
        "",
    ]
    for i, (edges, _) in enumerate(test_cases, 1):
        lines.append(f"  Q{i}: Edges = {_format_edges(edges)}")

    lines += [
        "",
        "Submit your answers as flow_1, flow_2, flow_3, flow_4.",
    ]
    return "\n".join(lines)


# ── Prepare ───────────────────────────────────────────────────────────────────

def _prepare():
    prompt = _build_prompt(_DEMOS, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (edges, expected) in enumerate(_TEST_CASES, 1):
            got = getattr(response, f"flow_{i}", None)
            ok = False
            try:
                ok = int(got) == expected
            except (TypeError, ValueError):
                ok = False
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


# ── Answer schema ─────────────────────────────────────────────────────────────

@dataclass
class _Answer:
    flow_1: int
    flow_2: int
    flow_3: int
    flow_4: int


# ── Task entry point ──────────────────────────────────────────────────────────

@kbench.task(
    name="flow_network_capacity_obs_learning",
    description=(
        "Path-graph flow networks: edges have capacities and tags {A,B,C,D}. "
        "Hidden tag priority governs max flow: sort edges by "
        "(tag_priority ASC, capacity ASC); return capacity at position 1. "
        "Infer the ordering and predict max flow for 4 networks."
    ),
)
def flow_network_capacity_obs_learning(llm) -> float:
    """Infer a hidden 4-tag priority ordering from 14 path-network flow demos; apply it to 4 new networks."""
    prompt, grade_fn = _prepare()
    try:
        response = llm.prompt(prompt, schema=_Answer)
    except Exception:
        response = None
    score, test_results = (
        grade_fn(response)
        if response is not None
        else (
            0.0,
            [
                {
                    "q": i,
                    "expected": _TEST_CASES[i - 1][1],
                    "got": None,
                    "correct": False,
                }
                for i in range(1, 5)
            ],
        )
    )
    reasoning = (
        getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    )
    _log_trace(
        task="flow_network_capacity_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    flow_network_capacity_obs_learning.run(kbench.llm)

