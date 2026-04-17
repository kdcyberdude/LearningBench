This benchmark measures **adaptive hypothesis updating under feedback** — not single-shot inference. Every task places the model inside an unknown environment: a hidden target to reach, a rule to discover, a constraint to satisfy. The model takes actions, receives binary or graded feedback, and must converge on the solution within a fixed step budget. Failing to update a wrong hypothesis costs efficiency and, if persistent, costs the task entirely.

**30 tasks · Interactive · Score = `0.55 × solved + 0.25 × efficiency + 0.20 × partial progress`**

Efficiency = `1 − (steps_used / step_budget)`. Partial progress = fraction of sub-goals or partial credit awarded mid-task by the grader. A model that solves nothing earns a maximum of 0.20 from partial credit; a model that solves everything in minimum steps approaches 1.0.

| Task | What it tests |
|---|---|
| 1d-battleship | Hidden target on a 1D grid; binary hot/cold feedback; tests directional search |
| 2d-battleship | Hidden ships on a 2D grid; overlap-aware probing; tests spatial belief revision |
| binary-combination-lock | 6-bit combination; feedback is number of correct bits (not positions); tests combinatorial search |
| binary-search-tree | Hidden BST; infer structure from insertion-order probes; tests property-based reasoning |
| blackbox-logic | Hidden Boolean formula; query with input vectors, get T/F; tests active feature identification |
| chebyshev-localization | Hidden point; feedback is Chebyshev distance; tests geometric reasoning |
| coin-weighing | Balance scale with a single heavy/light coin; adaptive query selection |
| constraint-satisfaction | 4-variable CSP with hidden constraints; feedback is constraint violations count |
| counterfactual-simulator | Hidden SCM; model makes interventions and must estimate counterfactuals |
| decision-tree-induction | Hidden decision tree; query with feature vectors, get class; infer tree structure |
| escape-room | Multi-step puzzle; each action reveals one clue; tests chained constraint satisfaction |
| feature-localization | Hidden feature vector; feedback locates the first wrong bit; tests sequential isolation |
| firewall-rules-discovery | Hidden ACL; probe with packet tuples, get permit/deny; infer rule set |
| grid-transposition | 4×4 grid with hidden permutation; feedback shows number of correct positions |
| hidden-arithmetic-sequence | Hidden recurrence; query arbitrary positions; tests algebraic extrapolation |
| hidden-graph-traversal | Hidden directed graph; explore via outgoing-edge queries; infer reachability |
| hidden-modulus | Hidden modulus; query any integer, get remainder; tests structured numeric search |
| hidden-operator | Hidden infix operator; query expression values; tests function approximation |
| information-gain-game | Hidden state; model selects questions to maximize information; tests Bayesian query design |
| latin-square-solver | 4×4 Latin square with hidden values; row/column constraint feedback |
| lights-out | Hidden toggle-mask; model flips cells, sees resulting board state; tests dependency tracking |
| manhattan-localization | Hidden point on an integer grid; feedback is Manhattan distance |
| master-mind-4color | Mastermind with 4 colors, 4 positions; feedback is black/white pegs |
| multi-armed-bandit | 4-arm bandit with hidden reward distributions; tests exploration/exploitation balance |
| nim-variant | Nim with a hidden winning rule modification; must discover the variant by play |
| number-guessing | Hidden integer in [0, N]; feedback is higher/lower; baseline binary search task |
| pipeline-hazard | Hidden hazard rule in a simplified pipeline; query instruction sequences, get stall count |
| substitution-cipher-solving | Interactive Vigenère / substitution solver; known-plaintext queries allowed |
| tower-of-hanoi-constraint | Tower of Hanoi with one hidden illegal move; must discover constraint through attempted moves |
| truth-table-query | Hidden Boolean function over 4 variables; query any row, get output; infer function |

The step budget varies by task (8–20 steps). Partial progress rewards sub-goal completion for tasks with natural intermediate milestones.

---

## What this benchmark reveals

RL is the highest-scoring sub-ability for most models, and the reason is more informative than the score. Many tasks here have known algorithmic solutions — binary search, systematic grid sweeping, balanced weighing — that frontier models have encountered described and executed. What scores well on those tasks is not adaptive learning; it is the recall of a procedure that happens to fit. The tasks that actually distinguish models are the ones where the feedback is indirect or non-standard: not "you are wrong," but a distance signal that requires decomposition, a partial count that must be turned into a hypothesis. Most models cannot do this, and when they fail, they fail by repeating — failed runs consume over 4× more tokens than solved ones, because a model that cannot update its hypothesis keeps probing the same space until the budget expires.

What the benchmark measures underneath is whether a model can maintain a live hypothesis and revise it turn by turn. The models that perform well even on hard tasks are not smarter; they are updating. The ones that fail are not dumber; they are stuck. Token spend makes this visible: a solved run is efficient and converging; a failed run is thrashing. The benchmark treats token cost as a behavioral trace of cognitive flexibility, which is the thing it was actually built to measure.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11780821%2Fe15bfe5f5a0c0f6e77ae44095749986a%2Ffig_meta_calibration.png?generation=1776464036007614&alt=media)

**Capability rankings have systematic blind spots.** The benchmark surfaces multiple cross-family inversions in both directions. On `collatz-length`, Haiku (0.836) and Opus (0.928) outperform GPT-5.4 (0.020) and GLM-5 (0.060). On `affine-cipher-word`, the ranking reverses: GPT-5.4 (0.800) and Gemma (0.800) outperform Haiku (0.040) and Sonnet (0.040). These inversions are not noise — they appear on multiple tasks and in both directions, confirming that the benchmark captures task-specific learning capacities that overall capability scores do not proxy.