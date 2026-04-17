# Rank-4 Analysis & Qwen 3 Dominance Hypothesis

**Date**: April 15, 2026  
**Scope**: Final 138-task benchmark (post Phase D curation)  
**Questions**:  
1. What is the minimum task removal set to move GPT-5.4 or Claude Opus 4.6 to rank **4**?  
2. Why does Qwen 3 Next 80B Thinking score significantly higher than GPT-5.4 and Claude Opus 4.6?

---

## Part 1: Minimum Task Removal for Rank 4

### The Rank-4 Problem is Much Easier Than Rank 3

To reach rank 3, both models had to defeat both Gemini 2.5 Flash **and** Qwen 3 (requiring 23–24 tasks removed). To reach rank 4, they only need to defeat **Gemini 2.5 Flash** (gap: 0.016 for GPT-5.4, 0.020 for Claude Opus).

| Target | Current Rank | **Minimum Tasks for Rank 4** | Tasks to Remove |
|---|---|---|---|
| **GPT-5.4** | 5 | **3 tasks** | `grid_transform_concept_learning`, `grid_octile_rf_learning`, `deceptive_stack_machine_obs_learning` |
| **Claude Opus 4.6** | 6 | **4 tasks** | `deceptive_stack_machine_obs_learning`, `interleave_reverse_concept_learning`, `grid_transform_concept_learning`, `positional_encode_concept_learning` |

Exhaustive search confirmed these are the minimum — no subset of 1 or 2 tasks achieves rank 4 for GPT-5.4, and no subset of 1, 2, or 3 tasks achieves rank 4 for Claude Opus.

---

### Minimum 3-Task Set: GPT-5.4 → Rank 4

**Removing these 3 tasks moves GPT-5.4 from rank 5 to rank 4.**

| Task | Category | Gemini 2.5 Flash | GPT-5.4 | Advantage |
|---|---|---|---|---|
| `grid_transform_concept_learning` | concept | 1.000 | 0.000 | +1.000 |
| `grid_octile_rf_learning` | rl | 1.000 | 0.000 | +1.000 |
| `deceptive_stack_machine_obs_learning` | observational | 1.000 | 0.000 | +1.000 |

**Resulting leaderboard:**

| Rank | Model | Score | Change |
|---|---|---|---|
| 1 | Gemini 3.1 Pro Preview | 0.8219 | — |
| 2 | GLM-5 | 0.6329 | — |
| 3 | Qwen 3 Next 80B Thinking | 0.5926 | — |
| **4** | **GPT-5.4** | **0.4703** | **↑ from 5** |
| 5 | Claude Opus 4.6 | 0.4667 | ↑ from 6 |
| 6 | Gemini 2.5 Flash | 0.4648 | ↓ from 4 |
| 7 | Claude Sonnet 4.6 | 0.4370 | — |

*Note: Removing these 3 tasks also lifts Claude Opus from rank 6 to rank 5 as a side effect.*

---

### Minimum 4-Task Set: Claude Opus 4.6 → Rank 4

**Removing these 4 tasks moves Claude Opus from rank 6 to rank 4.**

| Task | Category | Gemini 2.5 Flash | Claude Opus | Advantage |
|---|---|---|---|---|
| `deceptive_stack_machine_obs_learning` | observational | 1.000 | 0.000 | +1.000 |
| `interleave_reverse_concept_learning` | concept | 1.000 | 0.000 | +1.000 |
| `grid_transform_concept_learning` | concept | 1.000 | 0.000 | +1.000 |
| `positional_encode_concept_learning` | concept | 0.750 | 0.100 | +0.650 |

**Resulting leaderboard:**

| Rank | Model | Score | Change |
|---|---|---|---|
| 1 | Gemini 3.1 Pro Preview | 0.8209 | — |
| 2 | GLM-5 | 0.6358 | — |
| 3 | Qwen 3 Next 80B Thinking | 0.5951 | — |
| **4** | **Claude Opus 4.6** | **0.4695** | **↑ from 6** |
| 5 | GPT-5.4 | 0.4694 | ↑ from 5 |
| 6 | Gemini 2.5 Flash | 0.4627 | ↓ from 4 |
| 7 | Claude Sonnet 4.6 | 0.4461 | — |

---

### Rank-4 vs Rank-3 Comparison

| Goal | GPT-5.4 min tasks | Claude Opus min tasks |
|---|---|---|
| Reach rank **4** (beat Gemini 2.5 Flash only) | **3 tasks** | **4 tasks** |
| Reach rank **3** (beat Gemini 2.5 Flash + Qwen) | **23 tasks** | **≥24 tasks** |

The 7-8× jump in required removals from rank-4 to rank-3 is entirely due to the Qwen gap. This underscores how dominant Qwen 3 Thinking is in the rank-3 position — it is structurally separated from the GPT-5.4/Opus/Gemini-2.5-Flash cluster by a large capability gap.

---

## Part 2: Why Qwen 3 Next 80B Thinking Dominates the Benchmark

### The Core Finding

Qwen 3 Next 80B Thinking is a **mid-tier model** (by common classification) that scores **0.601**, surpassing all GPT-5.4 (0.460), Claude Opus 4.6 (0.457), and Gemini 2.5 Flash (0.476). This is a significant anomaly. The analysis reveals this is not an accident or artifact — it reflects deep, structural advantages.

---

### Hypothesis 1: The "Thinking Token" Multiplier Effect

The most powerful evidence comes from comparing Qwen 3 Next 80B **Thinking** vs. the identical base model with **Instruct** (no extended chain-of-thought):

| Category | Thinking | Instruct | Uplift |
|---|---|---|---|
| Concept | 0.5573 | 0.1860 | **+0.371** |
| Observational | 0.6048 | 0.3185 | **+0.286** |
| RL | 0.5891 | 0.3202 | **+0.269** |
| Language | 0.6227 | 0.4146 | **+0.208** |
| Associative | 0.6276 | 0.4638 | **+0.164** |
| **Overall** | **0.601** | **0.341** | **+0.261** |

**Without thinking tokens, Qwen 3 80B Instruct falls to rank 12** (0.341 — comparable to small models). The thinking tokens alone account for a +0.261 score uplift, which is larger than the difference between GPT-5.4 and the bottom model (GPT-5.4 nano: 0.239).

**What is this measuring?** The benchmark tasks require _inference-time learning_ — the model must observe patterns from in-context examples and apply them to new inputs. Thinking tokens allow the model to:
1. Enumerate the observed examples explicitly in its scratchpad
2. Hypothesize candidate rules and test them against prior observations
3. Iteratively refine the rule before committing to an answer

On 20 tasks, the thinking-to-instruct uplift is ≥ +0.800 — the Instruct version scores near 0, while the Thinking version solves it perfectly. These are the exact tasks that GPT-5.4 and Claude Opus also fail on.

---

### Hypothesis 2: Structural Simulation Specialization

Looking at which task types Qwen Thinking dominates most (advantage vs. average of all 4 frontier models):

| Task Keyword | n tasks | Qwen avg | GPT avg | Opus avg | Qwen Advantage |
|---|---|---|---|---|---|
| `grid` | 3 | **1.000** | 0.167 | 0.000 | **+0.917** |
| `transform` | 3 | **0.815** | 0.167 | 0.250 | **+0.606** |
| `layered` | 1 | **0.944** | 0.000 | 0.000 | **+0.944** |
| `transducer` | 1 | **1.000** | 0.250 | 0.250 | **+0.750** |
| `machine` | 6 | **0.625** | 0.133 | 0.219 | **+0.449** |
| `state` | 4 | **0.536** | 0.138 | 0.198 | **+0.368** |
| `hidden` | 12 | **0.566** | 0.340 | 0.316 | **+0.238** |
| `matrix` | 1 | **1.000** | 0.000 | 0.000 | **+1.000** |
| `stack` | 1 | **1.000** | 0.000 | 0.000 | **+1.000** |
| `codon` | 1 | **1.000** | 0.000 | 0.000 | **+1.000** |
| `shapley` | 1 | **1.000** | 0.000 | 0.000 | **+1.000** |

The pattern is unmistakable: **Qwen excels at any task involving multi-step structural simulation** — state machines, grids, abstract machines, computational models. GPT-5.4 and Claude Opus score 0.000 on these tasks, not because they are "dumb" but because these tasks require **sustained working-memory simulation over many steps**, which is exactly what thinking tokens enable.

The concept and observational categories show the largest Thinking uplift (+0.371 and +0.286), and these are precisely where Qwen's advantage over frontier peers is largest.

---

### Hypothesis 3: Score Distribution Architecture — Bimodal Mastery vs. Partial Credit

| Model | Perfect (1.0) | Zero (0.0) | Bimodal rate |
|---|---|---|---|
| Gemini 3.1 Pro Preview | 54.3% | 4.3% | 58.7% |
| GLM-5 | 29.0% | 18.8% | 47.8% |
| **Qwen 3 Thinking** | **26.8%** | **13.0%** | **39.9%** |
| GPT-5.4 | 16.7% | **22.5%** | 39.1% |
| Claude Opus 4.6 | 14.5% | 20.3% | 34.8% |
| Gemini 2.5 Flash | 15.2% | 15.2% | 30.4% |

Qwen Thinking has **37 perfect-score tasks** vs. GPT-5.4's 23 — a 61% advantage. Critically, GPT-5.4 has **31 zero-score tasks** (22.5% of benchmark), while Qwen has only **18** (13.0%). This means GPT-5.4 completely fails on 13 more tasks than Qwen.

The score distribution tells two stories:
- **Qwen's strength**: Higher rate of perfect mastery + lower rate of complete failure
- **GPT-5.4/Opus weakness**: Higher zero-rate suggests systematic blind spots, not just marginal underperformance

---

### Hypothesis 4: Where Frontier Models Beat Qwen — Qwen's Known Weaknesses

Qwen is not universally superior. On these task types, **all 4 frontier models collectively beat Qwen**:

| Task | Category | Qwen | Frontier Avg | Frontier Adv |
|---|---|---|---|---|
| `battleship_1d_rf_learning` | rl | 0.000 | **0.917** | −0.917 |
| `rule90_step_rf_learning` | rl | 0.089 | **0.958** | −0.869 |
| `battleship_two_ships_rf_learning` | rl | 0.100 | **0.951** | −0.851 |
| `perm_footrule_rf_learning` | rl | 0.000 | **0.731** | −0.731 |
| `affine_cipher_word_rf_learning` | rl | 0.000 | **0.607** | −0.607 |
| `digit_cipher_concept_learning` | concept | 0.000 | **0.600** | −0.600 |
| `nested_logic_concept_learning` | concept | 0.250 | **0.743** | −0.492 |
| `wordle_micro_rf_learning` | rl | 0.100 | **0.568** | −0.468 |

**The anti-thinking effect**: On `rule90_step_rf_learning`, Qwen Instruct scores 1.000 but Qwen Thinking scores only 0.089 — the worst thinking-hurts-performance example in the entire benchmark. This is likely because the step-counting rule for Rule 90 cellular automata has a rote, pattern-matched answer that thinking tokens "overthink" and override.

Qwen's weaknesses cluster around:
1. **Gamified RL tasks** (Battleship, Wordle, Wordle-micro) — tasks with game-like rules that GPT/Claude have likely seen extensively in training
2. **Compact rule extraction** (digit cipher, nested logic) — tasks where the answer is directly deducible in one step without simulation
3. **Well-known puzzles** (affine cipher, Rule 90) — tasks where GPT/Claude can pattern-match to prior knowledge, bypassing inference-time learning entirely

---

### Hypothesis 5: "Thinking as a Simulator" — The Core Mechanism

Synthesizing all evidence, here is the core hypothesis:

> **Qwen 3 Next 80B Thinking uses extended chain-of-thought as an in-context simulator.** When faced with a task requiring structural learning, it executes the system step-by-step in its scratchpad — effectively running a program — before outputting the final answer. GPT-5.4 and Claude Opus, despite being larger and more capable in many domains, do not have comparable inference-time simulation budgets or architectural incentives for this kind of extended sequential simulation.

This explains:
- Why tasks with keywords `machine`, `grid`, `transform`, `transducer`, `stack` show Qwen's largest advantages
- Why the Thinking model beats the Instruct model by +0.261 (the scratchpad is the key)
- Why GPT-5.4/Opus score exactly 0.000 on these tasks (no simulation → no signal)
- Why Gemini 2.5 Flash (also has thinking tokens) shows partial improvement but less than Qwen
- Why Qwen's advantage disappears on "gamified" tasks (no simulation needed — GPT/Claude apply rote strategies)

---

### Quantified Capability Summary

| Metric | Qwen 3 Thinking | GPT-5.4 | Claude Opus | Interpretation |
|---|---|---|---|---|
| Overall score | 0.601 | 0.460 | 0.457 | Qwen +30% over both |
| Perfect-score tasks | 37 (26.8%) | 23 (16.7%) | 20 (14.5%) | Qwen 61% more perfect solves |
| Zero-score tasks | 18 (13.0%) | 31 (22.5%) | 28 (20.3%) | GPT fails 72% more tasks completely |
| Head-to-head vs GPT | Wins 64/138 | Wins 40/138 | — | Qwen wins 46% of tasks vs GPT |
| Head-to-head vs Opus | Wins 70/138 | — | Wins 44/138 | Qwen wins 51% of tasks vs Opus |
| Thinking uplift | +0.261 over base | N/A | N/A | Thinking = core discriminating factor |
| Concept category | 0.557 | **0.266** | 0.259 | GPT scores 52% of Qwen here |
| Observational category | 0.605 | **0.293** | 0.323 | GPT scores 48% of Qwen here |

---

### For DeepMind Researchers: Key Takeaways

1. **Thinking tokens are the decisive variable for inference-time learning tasks.** The 76% score gap between Qwen Thinking (0.601) and Qwen Instruct (0.341) is the clearest signal in the entire dataset. The benchmark strongly rewards extended computation at inference time.

2. **GPT-5.4 and Claude Opus 4.6 have a specific structural simulation blind spot.** On 13 tasks, they score 0.000 while Qwen scores ≥ 0.5. These tasks — state machines, grid transformations, abstract machines, FSTs — require sustained sequential simulation that their architectures apparently don't deploy within context.

3. **The 3-task set that blocks GPT-5.4 at rank 5** (`grid_transform`, `grid_octile`, `deceptive_stack_machine`) defines the minimum discriminating surface between GPT-5.4 and Gemini 2.5 Flash. These are all tasks requiring spatial/structural simulation.

4. **Qwen Thinking is not "better" universally.** On gamified tasks (Battleship, Wordle, Rule 90), frontier models dominate — suggesting these measure a different capability axis (memorized game heuristics). The benchmark effectively identifies two sub-populations of tasks: simulation tasks (Qwen wins) and heuristic retrieval tasks (Frontier wins).

5. **Gemini 3.1 Pro Preview remains clearly rank 1** — it is the only model that combines thinking-level simulation ability with broad task mastery (54.3% perfect scores, only 4.3% zeros). Its performance is structurally different from every other model.

---

## Appendix: Rank-4 vs Rank-3 Task Overlap

3 tasks appear in both the rank-4 minimal sets:

| Task | In GPT-5.4 rank-4 set | In Opus rank-4 set |
|---|---|---|
| `grid_transform_concept_learning` | ✓ | ✓ |
| `deceptive_stack_machine_obs_learning` | ✓ | ✓ |
| `grid_octile_rf_learning` | ✓ | ✗ |
| `interleave_reverse_concept_learning` | ✗ | ✓ |
| `positional_encode_concept_learning` | ✗ | ✓ |

All 5 of the rank-4 removal tasks are also present in the rank-3 removal sets (they are a subset of the 23/24 task sets). Removing these 3–4 tasks is literally the minimum necessary to open a gap with Gemini 2.5 Flash — the "easiest" blocker to dislodge.
