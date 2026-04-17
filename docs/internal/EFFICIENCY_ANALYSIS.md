# Token, Cost & Latency Analysis: What the Data Reveals About Model Behavior

*Generated from `all_task_runs.csv` — 1,835 task runs × 14 models × 131 tasks*

---

## Dataset Summary

| Metric | Value |
|---|---|
| Total runs | 1,835 |
| Models | 14 (excl. Gemini 3 Flash Preview, 1 stray row) |
| Tasks | 131 |
| Score coverage | 1,480 / 1,835 rows have a score (80.7%) |
| Missing scores by sub-ability | obs-learning (208), concept (78), lang (39), rf (30) |

---

## H1: Cost Efficiency — Score per Dollar

**Finding**: There is no positive correlation between price and quality. The most *cost-efficient* models (score per dollar) are the lightweight Efficient-tier models — not the expensive Frontier ones.

| Model | Median Score/$ | Total Spend (USD) | Mean Score |
|---|---|---|---|
| Gemini 3.1 Flash-Lite | 671.5 | $0.7 | 0.572 |
| Gemma 4 26B A4B | 438.4 | $0.5 | 0.467 |
| Qwen 3 Next 80B Instruct | 187.7 | $2.6 | 0.476 |
| GPT-5.4 mini | 156.7 | $0.6 | 0.455 |
| GPT-5.4 nano | 122.9 | $0.4 | 0.342 |
| GPT-5.4 | 92.8 | $2.0 | 0.592 |
| ... | | | |
| Gemini 3.1 Pro Preview | 6.8 | $27.7 | 0.862 |

**Interpretation**: Gemini 3.1 Pro Preview achieves the highest absolute score but at 100× the cost of Gemini Flash-Lite. Flash-Lite achieves comparable score at a fraction of the price — making it the dominant choice for budget-constrained deployments. The total cost to run the *entire* benchmark across all 14 models and 131 tasks was ~$115 USD.

---

## H2: Token Verbosity vs. Score

**Finding**: More tokens generally correlates with *lower* scores (Spearman r = −0.249, p < 0.001).

This is counter-intuitive but explainable: models that fail to learn within their allocated examples keep requesting more, inflating token counts. The relationship is most pronounced in:

- **RF-learning**: r = −0.49 (strongest signal — struggling models spiral into exploration without convergence)
- **Language learning**: r = −0.37 (models that don't grasp the phonological rule keep asking for more examples)
- **Concept formation**: r = −0.31

The exception is **Observational learning** (r = +0.13, p = 0.02) and **Associative learning** (r = +0.08, p = 0.20, not significant). These are fixed-input tasks where more output reflects more careful reasoning, not floundering.

**Key number**: Hard tasks (score ≤ 0.25) consume **7.0× more tokens** than easy tasks (score = 1.0) — mean 59,214 vs. 8,499 tokens. Token cost is a symptom of failure, not a cause.

---

## H3: Latency vs. Score

**Finding**: Latency has a small but significant positive correlation with score overall (r = +0.102, p < 0.001), but the relationship is subability-dependent.

| Sub-ability | Latency↔Score r | Interpretation |
|---|---|---|
| Associative | +0.33 *** | Slower responses reflect more careful reasoning |
| Concept Formation | +0.30 *** | Extended deliberation improves rule induction |
| Observational | +0.31 *** | Thinking time helps on inference tasks |
| Language Learning | −0.02 (ns) | No relationship — rule complexity not captured by time |
| RL | −0.03 (ns) | No relationship — RL relies on trial structure, not per-step time |

**Insight**: For "thinking" tasks (associative, concept, observational), taking longer per response is genuinely beneficial. For language learning and RL, latency is orthogonal to outcome — the bottleneck is structure and strategy, not deliberation time.

---

## H4: Provider Verbosity Signatures

Mean output tokens per task reveals distinct provider "fingerprints":

| Provider | Avg Output Tokens | Style |
|---|---|---|
| GLM-5 | 29,748 | Very verbose: chains-of-thought embedded in responses |
| Qwen 3 Next 80B Thinking | 41,368 | Highest output: extended reasoning traces |
| Qwen 3 Next 80B Instruct | 5,506 | Much lower than Thinking variant |
| DeepSeek V3.2 | 2,596 | Moderate; structured answers |
| Claude (all) | 1,684–2,327 | Concise; well-formatted answers |
| GPT-5.4 family | 223–776 | Very concise |
| Gemini Pro | 172 | Extremely concise |

**The thinking model paradox**: Qwen Thinking uses ~41K output tokens on average but achieves rank 3. GLM-5 uses ~30K output tokens and ranks 2nd. Yet GPT-5.4 uses only 237 output tokens and ranks 6th with 0.592 score. Token verbosity is not a reliable proxy for quality.

---

## H5: Token Waste on Easy vs. Hard Tasks

Easy tasks (score = 1.0): mean **8,499 total tokens**  
Hard tasks (score ≤ 0.25): mean **59,214 total tokens**

The 7× inflation on hard tasks is driven almost entirely by RL tasks, where models that cannot find the hidden state keep exploring:

| RF Score Quartile | Avg Input Tokens | Avg Output Tokens |
|---|---|---|
| Q1 (0–25%) | 102,272 | 20,380 |
| Q2 (25–50%) | 93,675 | 15,802 |
| Q3 (50–75%) | 54,480 | 14,956 |
| Q4 (75–100%) | 3,489 | 7,315 |

**Finding**: A model solving an RL task efficiently uses ~10K tokens. A model failing uses ~120K tokens. The token budget is exhausted in the search, not the solution.

---

## H6: Sub-Ability Token Profiles

RF-learning dominates all other sub-abilities in token consumption:

| Sub-ability | Avg Input | Avg Output | Avg Latency (s) |
|---|---|---|---|
| RL (RF) | 63,531 | 14,664 | 249 |
| Concept Formation | 13,754 | 5,304 | 94 |
| Language Learning | 11,277 | 5,699 | 86 |
| Associative | 5,889 | 2,129 | 33 |
| Observational | 1,150 | 3,305 | 58 |

**Observation**: Observational learning has the *lowest* input tokens but moderate output — models receive short demonstrations and produce structured predictions. RL produces the opposite: enormous input context (the exploration log) with moderate output (the guesses).

---

## H7: Anatomy of Failures

Zero-score runs (score = 0): only **14 cases total**, all in **Associative Learning**. These are tasks where models over-commit to a causal direction with no evidence — committing to YES or NO when the correct answer is UNKNOWN.

By model: GPT-5.4 nano (3), DeepSeek (2), GPT-5.4 (2), Gemma (2), Haiku (1), Opus (1), GLM-5 (1), GPT-5.4 mini (1), Flash-Lite (1).

Zero-score runs have mean input tokens of **1,903** and output tokens of just **60** — models that fail here are making confident but wrong short answers, not long confused ones.

---

## H8: Per-Task Difficulty vs. Cost

Harder tasks (lower mean score) cost significantly more to run:

- **Score vs. mean tokens per task**: Spearman r = −0.173 (p = 0.048)
- **Score vs. mean cost per task**: Spearman r = −0.372 (p < 0.001)

The strongest cost-difficulty signal is in RL tasks:

| Task (hardest) | Mean Score | Mean Total Tokens |
|---|---|---|
| nim-heap-rf-learning | 0.131 | 25,956 |
| digitwise-l1-rf-learning | 0.216 | 241,298 |
| bitstring-hamming-rf-learning | 0.256 | 279,944 |
| gray-hamming-rf-learning | 0.260 | 43,141 |

The `digitwise-l1` and `bitstring-hamming` tasks cost nearly **30× more** per run than the easiest tasks. This has a real budget implication: the 5 hardest RL tasks alone account for a disproportionate share of total benchmark spend.

---

## Charts Generated

All charts saved to `analysis/outputs/efficiency_charts/`:

1. **`cost_efficiency_scatter.png`** — Score vs mean cost per task (bubble = total spend)
2. **`score_per_dollar_bar.png`** — Cost efficiency ranking
3. **`token_verbosity_by_model.png`** — Stacked bar: avg input + output tokens per model
4. **`latency_vs_score.png`** — Latency vs score scatter per sub-ability (5-panel)
5. **`token_waste_heatmap.png`** — Heatmap: model × sub-ability avg tokens
6. **`rf_learning_token_quartiles.png`** — RF-learning: token use by score quartile
7. **`hard_easy_tasks.png`** — 12 hardest vs 12 easiest tasks
8. **`task_cost_vs_difficulty.png`** — Per-task mean tokens vs mean score scatter
9. **`provider_verbosity.png`** — Output token distributions by provider (violin)
10. **`efficiency_frontier.png`** — Pareto frontier: quality vs total spend

---

## Key Takeaways for Benchmark Design

1. **Token count is a failure signal, not a quality signal** — in interactive tasks (RL, Concept, Language), token explosion means the model is lost.
2. **Latency helps for static reasoning tasks** (associative, observational) but is orthogonal to RL and language learning performance.
3. **The efficiency axis in scoring is load-bearing** — without it, concept and language learning scores would reward models that waste examples.
4. **Cost varies 66× across models** for equivalent tasks. Running the benchmark on Flash-Lite costs the same as running 4 tasks on Gemini Pro.
5. **The hardest 5 RL tasks account for outsized cost** — future benchmark curation should cap token budgets per task to prevent cost concentration.
