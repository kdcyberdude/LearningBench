#!/usr/bin/env python
# coding: utf-8

import random
from collections import Counter, OrderedDict
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
    "The model observes access sequences and the resulting cache contents for a CPU cache of "
    "capacity 4 that uses a hidden LFU (Least Frequently Used) eviction policy. Ties in "
    "frequency are broken by recency (LFU-LRU: among equally-least-frequent, evict the "
    "least recently used). Early demos use working sets that fit entirely in cache so no "
    "eviction occurs and all policies look identical. Success requires distinguishing LFU-LRU "
    "from LRU and FIFO on 4 test sequences where the policies diverge."
)

_CACHE_SIZE = 4
_FIXED_SEED = 0


def _lfu_lru_cache(accesses: list) -> list:
    """LFU with LRU tie-breaking: among min-frequency items, evict the least recently used."""
    cache: list = []       # ordered by insertion/last-access for tie-breaking
    freq: Counter = Counter()
    last_access: dict = {}  # item -> step of last access

    for step, item in enumerate(accesses):
        freq[item] += 1
        last_access[item] = step
        if item in cache:
            continue
        if len(cache) < _CACHE_SIZE:
            cache.append(item)
        else:
            min_freq = min(freq[c] for c in cache)
            candidates = [c for c in cache if freq[c] == min_freq]
            # Among ties, evict least recently used
            evict = min(candidates, key=lambda c: last_access[c])
            cache.remove(evict)
            cache.append(item)
    return sorted(cache)


def _lru_cache_sim(accesses: list) -> list:
    cache: OrderedDict = OrderedDict()
    for item in accesses:
        if item in cache:
            cache.move_to_end(item)
        else:
            if len(cache) >= _CACHE_SIZE:
                cache.popitem(last=False)
            cache[item] = True
    return sorted(cache.keys())


def _make_test_cases():
    # Four hand-crafted sequences where LFU-LRU diverges from LRU
    # Cache capacity = 4, items drawn from {1..6}
    test_seqs = [
        # Freq: 1→3, 2→3, 3→2, 4→2, 5→1, 6→1 → LFU evicts 5 or 6 first
        [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2],
        # Heavy repetition of 1,2,3 then introduce 4,5,6
        [1, 1, 2, 2, 3, 3, 4, 5, 6, 1, 4, 5],
        # Introduce 5 new items requiring multiple evictions
        [1, 2, 3, 4, 5, 1, 2, 5, 6, 1, 2, 6],
        # LRU vs LFU diverge on item 3 vs 4
        [3, 4, 1, 2, 3, 4, 5, 3, 4, 6, 1, 5],
    ]
    cases = [(seq, _lfu_lru_cache(seq)) for seq in test_seqs]
    return cases


_TEST_CASES = _make_test_cases()


def _build_prompt(demos: list, test_seqs: list) -> str:
    lines = [
        f"You are observing a CPU cache with capacity {_CACHE_SIZE} entries.",
        "When the cache is full and a new item arrives, one item is evicted according to a hidden policy.",
        "",
        "Observations (access sequence → final cache contents, sorted):",
    ]
    for i, (seq, cache) in enumerate(demos, 1):
        lines.append(f"  {i:2d}. accesses={seq} → cache={cache}")
    lines.append("")
    lines.append("Now solve these 4 test questions:")
    for i, (seq, _) in enumerate(test_seqs, 1):
        lines.append(f"  Q{i}: accesses={seq} → cache=?")
    lines.append("")
    lines.append("Submit answer_1 through answer_4 as sorted lists of integers, e.g. '[1, 3, 4, 6]'.")
    return "\n".join(lines)


def _prepare():
    rng = random.Random(_FIXED_SEED)

    demos = []

    # Phase 1: working sets that fit in cache — all policies identical
    fit_seqs = [
        [1, 2, 3, 4, 1, 2, 3, 4],
        [2, 3, 4, 1, 2, 4, 3, 1],
        [1, 3, 2, 4, 3, 1, 2, 4],
    ]
    for seq in fit_seqs:
        demos.append((seq, _lfu_lru_cache(seq)))

    # Phase 2: 5-item working set forcing eviction — LFU visible but not yet tie-breaking
    phase2_seqs = [
        [1, 2, 3, 4, 5, 1, 2, 3, 1, 2],
        [1, 1, 2, 3, 4, 5, 2, 1, 3, 5],
        [2, 2, 3, 3, 1, 4, 5, 2, 3, 4],
        [1, 2, 2, 3, 4, 5, 1, 2, 4, 3],
    ]
    for seq in phase2_seqs:
        lfu = _lfu_lru_cache(seq)
        lru = _lru_cache_sim(seq)
        demos.append((seq, lfu))

    # Phase 3: sequences where LFU-LRU tie-breaking is needed — differ from plain LFU
    phase3_seqs = [
        [1, 2, 3, 4, 1, 2, 5, 3, 6, 1, 2, 5],
        [3, 3, 2, 2, 1, 1, 4, 5, 6, 3, 2, 5],
        [1, 2, 3, 4, 5, 1, 3, 5, 6, 2, 4, 6],
        [2, 4, 1, 3, 2, 4, 5, 6, 1, 2, 4, 5],
        [1, 1, 2, 2, 3, 4, 5, 6, 1, 2, 3, 6],
        [5, 3, 1, 2, 5, 3, 4, 6, 5, 3, 1, 4],
        [1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 4, 5],
    ]
    for seq in phase3_seqs:
        if len(demos) < 14:
            demos.append((seq, _lfu_lru_cache(seq)))

    prompt = _build_prompt(demos, _TEST_CASES)

    def grade_fn(response):
        results = []
        correct = 0
        for i, (seq, expected) in enumerate(_TEST_CASES, 1):
            raw = getattr(response, f"answer_{i}", None)
            got = None
            ok = False
            if isinstance(raw, list):
                try:
                    got = sorted(int(x) for x in raw)
                    ok = (got == expected)
                except (TypeError, ValueError):
                    pass
            elif isinstance(raw, str):
                try:
                    cleaned = raw.strip().strip("[]")
                    got = sorted(int(x.strip()) for x in cleaned.split(",") if x.strip())
                    ok = (got == expected)
                except (TypeError, ValueError):
                    pass
            if ok:
                correct += 1
            results.append({"q": i, "expected": expected, "got": got, "correct": ok})
        return correct / 4, results

    return prompt, grade_fn


@dataclass
class _Answer:
    answer_1: list
    answer_2: list
    answer_3: list
    answer_4: list


@kbench.task(
    name="lfu_cache_eviction_obs_learning",
    description="Observe access sequences and cache states for a hidden LFU-LRU cache (capacity 4). LFU evicts least-frequently-used; ties broken by recency. Early demos fit in cache masking the policy. Predict final cache for 4 test sequences.")
def lfu_cache_eviction_obs_learning(llm) -> float:
    """Infer LFU-LRU eviction policy from 14 access-sequence examples; predict cache state for 4 new sequences."""
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
            {"q": i, "expected": _TEST_CASES[i - 1][1], "got": None, "correct": False}
            for i in range(1, 5)
        ]

    reasoning = getattr(response, "reasoning", "") or getattr(response, "thinking", "") or ""
    _log_trace(
        task="lfu_cache_eviction_obs_learning",
        description=_TASK_DESCRIPTION,
        prompt=prompt,
        test_results=test_results,
        score=score,
        reasoning=str(reasoning),
    )
    return score


if __name__ == "__main__":
    lfu_cache_eviction_obs_learning.run(kbench.llm)

