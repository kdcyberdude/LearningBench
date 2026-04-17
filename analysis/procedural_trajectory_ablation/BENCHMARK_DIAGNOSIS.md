# Procedural Learning Benchmark: Diagnosis Report

**Question:** Is the benchmark working correctly? Are models genuinely not learning, or is there a design or implementation problem?

**Short answer:** The benchmark is mostly sound. Most models genuinely are not learning. But there are structural confounds in 7 of 11 tasks that make the OLS slope an unreliable learning signal for those tasks, and one task has a confirmed query-evaluation bug.

---

## Task Classification by Learning Measurement Quality

### ✅ Type A — Clean learning design (same rule, different instances per round)

These tasks are correct for measuring learning. Each round presents a new instance of the same underlying problem, so improvement across rounds genuinely reflects rule acquisition.

| Task | What varies per round | What stays fixed |
|---|---|---|
| `adaptive-sort-rule` | The 6 numbers to sort | The hidden 3-key sort rule (odd-first, then descending digit-sum, then value) |
| `nim-variant` | Pile sizes and `j` value | The game variant (move constraint rule) |

**Evidence from logs:**

G-Pro on `adaptive-sort-rule`:
- Round 1: 0.836 (5 turns to figure out the rule)
- Round 2: 1.0 (2 turns — confirmed the rule)
- Rounds 3–5: 1.0 (1 turn each — applies the rule immediately)
- Round 2 reasoning: *"Based on the accepted list from Practice 1, the hidden sorting rule is: 1. Odd numbers first. 2. Within each group, descending digit-sum."*

Claude Haiku on `adaptive-sort-rule`:
- Rounds 1–5: 0.427, 0.509, 0.427, 0.754, 0.427 — oscillating, no trend
- Round 2 reasoning: *"This is Practice Round 2 with new numbers"* — no mention of the rule
- Round 5 Transfer: 0/4 correct — never extracted the rule at all

**Conclusion:** Rule abstraction is binary, not gradual. Models that learn do so by round 2 (step function). Models that don't learn show random oscillation driven by per-instance difficulty, not learning slope.

---

### ⚠️ Type B — Confounded (5 different problems, not 5 instances of the same problem)

These tasks present structurally different problems each round. The OLS slope conflates "model improved" with "round N+1 was easier than round N."

| Task | Round structure | Why it's confounded |
|---|---|---|
| `packet-filter` | 5 different AND-rules (src_port+protocol, dst_port+direction, etc.) | Round difficulty varies by how many attributes overlap with prior rounds |
| `grammar-induction` | G_SQUARE → G_ALTERNATING → G_EVENRUNS → G_BCPATTERN → G_BOOKEND | 5 structurally distinct grammars; prior grammars don't help with next ones |
| `boolean-circuit` | 5 different truth-table patterns | Different circuits have different structure; no transferable rule |
| `lights-out-variant` | 5 different initial board states | Each board requires solving from scratch; different solutions |
| `opponent-strategy` | RULE_ALPHA → RULE_BETA → RULE_GAMMA → RULE_DELTA → RULE_EPSILON | 5 different opponent strategies; each must be learned from scratch |
| `voting-protocol` | SCHEMA_APEX → SCHEMA_CHAIN → SCHEMA_MEDIAN → SCHEMA_CASCADE → SCHEMA_APEX | 5 different voting aggregation schemas; different structural properties |
| `state-machine-password` | 5 DFAs with different transition tables (seeds 101–505) | Each DFA is unique; skill is "search the DFA" not "apply the same rule" |

**What OLS slope measures for these tasks:** Whether later rounds happen to be easier or harder, not whether the model learned. For `state-machine-password`, OLS slope is negative for most models because seed 505 (round 5) is systematically harder (requires backtracking that most models don't do).

**What these tasks DO test:** Single-episode efficiency — can the model solve this type of puzzle in N turns or fewer? That IS meaningful, but it's not procedural learning in the sense of cross-episode improvement.

---

### ⚠️ Type C — Shortcut susceptible

| Task | What the shortcut is | Why it matters |
|---|---|---|
| `dialect-morphology` | During practice, model can call `transform(test_word)` and directly get the answer | Practice scores are inflated by shortcut; transfer reveals this |

**Evidence:** Haiku on dialect-morphology: 0.228, 1.0, 1.0, 1.0, 1.0 in practice — then 0.0 on all 4 transfer tests. The model never learned the morphological rule; it queried its way to 100% practice scores. The benchmark's transfer component catches this; the practice slope does not.

---

## Bug Report

### `sql-reverse-engineering` — Query evaluation bug

**File:** `downloaded_tasks/procedural_learning/sql-reverse-engineering-proc-learning.py`

**Function:** `_parse_simple_where(clause, row)`

**Bug:** The function applies `.replace("=", "==")` to convert SQL equality to Python equality. This corrupts comparison operators:
- `id <= 4` → `id <== 4` (syntax error → returns False, reported as `[]`)
- `salary >= 60000` → `salary >== 60000` (syntax error → returns False)

**Impact:** When a model submits `id <= 4` as a *query* (to probe the table), it gets back `matches rows: []` instead of the correct `[1, 2, 3, 4]`. This gives the model systematically wrong feedback and makes round 4 (`id < 5`) effectively unsolvable for models that reason about SQL correctly. Models that happen to use strict inequality (`id < 5` instead of `id <= 4`) are unaffected.

**Evidence from DeepSeek round 4 log:**
```
Turn 1  submitted: id <= 4  →  QUERY 'id <= 4' → matches rows: []   ← BUG: should be [1,2,3,4]
Turn 4  submitted: id IN (1, 2, 3, 4)  →  QUERY '...' → matches rows: []  ← also broken
Turn 10  submitted: dept != 'ops' AND salary < 80000  →  WRONG. false_positives=[7,8,10], false_negatives=[4]
FAIL  score=0.0000
```

**Fix:** Replace the `=` → `==` conversion with a proper SQL-to-Python translation:
```python
import re
py = re.sub(r'(?<![<>!])=(?!=)', '==', clause)
```
Or handle each operator explicitly.

---

## `state-machine-password` round 5 — Not a bug, a search-depth test

12 of 14 models score 0 on practice round 5 (seed 505).

**Is the DFA solvable?** Yes. Brute-force simulation finds `CAAAAAAB` as an accepting sequence.

**Why do models fail?** The DFA at state 0 accepts only `C` (all other symbols are TRAP). After building `CCCCCC` (staying in intermediate states), the model exhausts remaining budget varying only position 7-8. It never backtracks to try `CAAAA__`. With 10 turns total, once a model commits to the `CCCCCC*` branch and finds all 4 completions fail, it has no turns left to backtrack.

**What this reveals:** Models cannot maintain and navigate a partial search tree in a linear conversation buffer. This IS a real finding and a feature of the benchmark, not a defect.

---

## Summary: What the benchmark correctly shows

| Claim | Supported by data? | Evidence |
|---|---|---|
| Median OLS slope is 0.000 (flat) | ✅ Yes | Ablation study across 158 task-model pairs |
| Most models do not improve over rounds | ✅ Yes for Type A tasks | Haiku adaptive-sort logs confirm no rule abstraction |
| Rule abstraction is binary (step function by round 2, or never) | ✅ Yes | G-Pro, Opus, GLM-5 on adaptive-sort-rule |
| Models with high practice scores may have learned nothing | ✅ Yes | Haiku dialect-morphology: 100% practice, 0% transfer |
| Models cannot backtrack under search pressure | ✅ Yes | State-machine-password round 5: DFA solvable but 12/14 fail |
| "Models that start poorly tend to deteriorate" | ❌ Not confirmed | Deterioration is more common in models that start moderate-high |
| "This is true for G-Pro and GLM-5" | ❌ Incorrect | G-Pro has positive mean slope (+0.022); GLM-5's slope ≈ 0 but high scores reflect starting-level competence |

## Recommended updates

1. **Fix the `sql-reverse-engineering` parse bug** before publishing or re-running.
2. **Update `procedural_learning.md`** to remove the "deteriorate" claim and the G-Pro/GLM-5 blanket statement. ✅ Done.
3. **Consider separating Type A and Type B task results** in the benchmark summary — they measure different things.
4. **The `dialect-morphology` shortcut** should be documented as a known measurement artifact; the practice-slope for this task is not meaningful.
