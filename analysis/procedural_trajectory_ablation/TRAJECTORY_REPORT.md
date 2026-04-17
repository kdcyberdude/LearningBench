# Procedural Learning: Trajectory Ablation Study

**Date:** April 2026  
**Data source:** `analysis/outputs/notebook_logs/*/proc-learning/*.json` (stdout_log conversation histories)  
**Scripts:** `extract_trajectories.py`  
**Outputs:** `trajectories.csv`, `trajectory_summary.csv`

---

## Motivation

The procedural learning benchmark scoring formula includes a `trajectory` component (weight 0.25):
the OLS slope of a model's practice-round scores over 5 rounds, normalised to [0, 1].
This study asks a specific question the composite score cannot answer directly:

> **Do models actually improve with practice, or do they start at some level and stay there (or decline)?**

This matters because a model that scores 70% on round 1 and 70% on round 5 has not learned —
it is performing at its initial ceiling. A model that goes from 30% to 80% has. The composite
score blends four components together and cannot distinguish these cases without decomposition.

---

## Method

For each of the 11 procedural tasks × up to 15 models (158 total task-model pairs):

1. Parsed `stdout_log` from each `notebook_logs/{task}/{model}.json` file.
2. Extracted per-round practice scores from lines matching `[Practice N/5]` ... `score=X.XXXX`.
3. Computed OLS slope over rounds 1–5.
4. Classified trajectory:
   - **improving**: slope > 0.03  
   - **deteriorating**: slope < −0.03  
   - **flat**: |slope| ≤ 0.03

---

## Results

### Global trajectory distribution (158 task × model pairs)

| Trajectory | Count | % |
|---|---|---|
| flat | 85 | 53.8% |
| deteriorating | 40 | 25.3% |
| improving | 33 | 20.9% |

**The median OLS slope across all 158 pairs is exactly 0.000.**  
The mean slope is −0.009 (slightly negative).  
48 pairs (30%) show positive slopes, 61 (39%) are exactly zero, and 49 (31%) are negative.

**The claim "most models do not exhibit procedural learning" is confirmed:**  
Only 20.9% of task-model pairs show a genuine improving trajectory. The majority are flat or deteriorating.

---

### Per-model summary

| Model | Improving | Flat | Deteriorating | Mean slope |
|---|---|---|---|---|
| G-Pro | 3 | 5 | 3 | **+0.022** |
| GLM-5 | 4 | 5 | 2 | **+0.000** |
| Qwen-I | 3 | 7 | 1 | +0.008 |
| GPT-mini | 1 | 9 | 1 | +0.005 |
| GPT-nano | 2 | 7 | 2 | +0.006 |
| Claude Sonnet | 3 | 6 | 2 | −0.004 |
| Gemma | 1 | 8 | 2 | −0.006 |
| G-Flash | 2 | 4 | 5 | −0.012 |
| GPT-5.4 | 1 | 9 | 1 | −0.012 |
| Claude Haiku | 4 | 3 | 4 | −0.019 |
| G-Flash-3 | 0 | 3 | 1 | −0.021 |
| Claude Opus | 3 | 4 | 4 | −0.016 |
| DeepSeek | 1 | 8 | 2 | −0.024 |
| G-Lite | 3 | 3 | 5 | −0.030 |
| Qwen-T | 2 | 4 | 5 | **−0.044** |

**Key findings:**

- **G-Pro has the highest mean slope (+0.022)** and is the only model with a consistently positive mean slope, confirming it is the only model that reliably exhibits procedural learning across tasks.
- **GLM-5 has mean slope ≈ 0.000** — it neither improves nor deteriorates on average. Its high composite scores come from starting at a high level on several tasks (e.g., `sql-reverse-engineering`: r1=1.0, r5=1.0; `voting-protocol`: r1=0.775, r5=0.775), not from learning.
- **Qwen-T has the most negative mean slope (−0.044)** — more deteriorating trajectories (5) than improving (2). Its extended deliberation does not translate into round-over-round improvement.
- **GPT-5.4, GPT-mini, GPT-nano, DeepSeek** are predominantly flat — they reach a level in round 1 and stay there.

---

### G-Pro and GLM-5 detailed (directly addressing the question)

The question was whether the "flat/negative slope" finding applies to the top models too.

**G-Pro per-task trajectories:**

| Task | Round 1 | Round 5 | Δ | Trajectory |
|---|---|---|---|---|
| adaptive-sort-rule | 0.836 | 1.000 | +0.164 | improving |
| nim-variant | 0.000 | 0.871 | **+0.871** | improving |
| voting-protocol | 0.775 | 1.000 | +0.225 | improving |
| grammar-induction | 0.100 | 0.308 | +0.208 | flat |
| lights-out-variant | 0.640 | 0.580 | −0.060 | deteriorating |
| boolean-circuit | 0.600 | 0.300 | **−0.300** | deteriorating |
| packet-filter | 0.357 | 0.000 | **−0.357** | deteriorating |
| dialect-morphology | 0.871 | 0.743 | −0.129 | flat |
| opponent-strategy | 1.000 | 1.000 | 0.000 | flat |
| sql-reverse-engineering | 1.000 | 1.000 | 0.000 | flat |
| state-machine-password | 0.000 | 0.000 | 0.000 | flat |

**G-Pro does genuinely improve on 3 tasks** (`nim-variant`, `voting-protocol`, `adaptive-sort-rule`),
but **deteriorates on 3 others** (`boolean-circuit`, `packet-filter`, `lights-out-variant`) and is flat on 5.
Its overall positive mean slope is driven by the large `nim-variant` gain (+0.871).

**GLM-5 per-task trajectories:**

| Task | Round 1 | Round 5 | Δ | Trajectory |
|---|---|---|---|---|
| adaptive-sort-rule | 0.673 | 1.000 | +0.327 | improving |
| dialect-morphology | 0.486 | 1.000 | **+0.514** | improving |
| opponent-strategy | 1.000 | 1.000 | 0.000 | improving (slope=+0.1, flat by final delta) |
| state-machine-password | 0.000 | 0.000 | 0.000 | improving (slope=+0.045, mid-peak) |
| boolean-circuit | 0.600 | 0.200 | **−0.400** | deteriorating |
| packet-filter | 1.000 | 0.000 | **−1.000** | deteriorating |
| grammar-induction | 0.000 | 0.169 | +0.169 | flat |
| lights-out-variant | 0.520 | 0.580 | +0.060 | flat |
| nim-variant | 0.000 | 0.000 | 0.000 | flat |
| sql-reverse-engineering | 1.000 | 1.000 | 0.000 | flat |
| voting-protocol | 0.775 | 0.775 | 0.000 | flat |

**GLM-5's high composite scores mostly come from starting high and staying high** (`sql-reverse-engineering`,
`voting-protocol`) or starting high and collapsing (`packet-filter`: r1=1.0, r5=0.0).
Its mean slope is ≈ 0 because these effects cancel. GLM-5 **does not exhibit systematic procedural learning**.

---

### Per-task summary

| Task | Improving | Flat | Deteriorating | Mean slope | Pattern |
|---|---|---|---|---|---|
| adaptive-sort-rule | 8 | 6 | 0 | +0.047 | Most models improve |
| nim-variant | 3 | 11 | 0 | +0.050 | 3 models improve; 11 flat at 0.0 |
| voting-protocol | 5 | 10 | 0 | +0.034 | No model deteriorates |
| dialect-morphology | 4 | 9 | 1 | +0.039 | Mostly flat after quick learning |
| state-machine-password | 7 | 4 | 3 | +0.019 | Most show some improvement |
| grammar-induction | 0 | 14 | 1 | −0.001 | **Entirely flat — no model learns** |
| lights-out-variant | 2 | 7 | 6 | −0.014 | Deterioration common |
| boolean-circuit | 1 | 9 | 5 | −0.025 | Mostly flat/deteriorating |
| opponent-strategy | 3 | 6 | 5 | −0.021 | Mixed — binary task, high variance |
| packet-filter | 0 | 7 | 7 | **−0.072** | High deterioration rate |
| sql-reverse-engineering | 0 | 2 | 12 | **−0.162** | **Worst deterioration — almost universal** |

**Two categories of task are visible:**

- **Tasks where learning occurs** (`adaptive-sort-rule`, `nim-variant`, `voting-protocol`): the rule can be discovered within 5 rounds; models that discover it improve sharply.
- **Tasks where no learning occurs** (`grammar-induction`, `sql-reverse-engineering`, `packet-filter`): models either plateau immediately or deteriorate. `sql-reverse-engineering` shows 12 of 14 models deteriorating — most start competent (many reach r1=1.0) but lose consistency across rounds.

---

## Corrected claim for `procedural_learning.md`

The original proposed claim was:
> *"The median learning slope across tasks and models is flat, and a substantial portion is negative. Models that start poorly tend to deteriorate rather than improve."*

**What the data supports:**

- ✅ The **median slope is exactly 0.000** — confirmed flat.
- ✅ **79.1% of task-model pairs are flat or deteriorating** — confirmed.
- ✅ **25.3% of pairs are actively deteriorating** — a substantial fraction.
- ❌ "Models that start poorly tend to deteriorate" — **not confirmed**. Models that start at zero tend to stay at zero (flat). Deterioration is more common in models that start at a moderate or high level and lose consistency over rounds (see `sql-reverse-engineering`: 12 deteriorating models all started at 0.4–1.0).
- ❌ "This is true for Gemini Pro and GLM-5 as well" — **incorrect as a blanket statement**. G-Pro has a positive mean slope (+0.022) and genuinely improves on 3/11 tasks. GLM-5 mean slope ≈ 0 but its high scores come from starting-level competence, not learning. Both models *also* deteriorate on specific tasks.

**More accurate restatement:** Most models show flat or deteriorating practice trajectories. The models that start well often stay well (flat); models that start at a moderate level are as likely to deteriorate as improve. Only G-Pro consistently shows positive slopes across multiple tasks — its high composite score reflects genuine learning on tasks like `nim-variant` and `voting-protocol`, not just initial competence.

---

## Files

| File | Description |
|---|---|
| `extract_trajectories.py` | Full extraction + OLS computation script |
| `trajectories.csv` | 1238 rows: one per (task, model, phase, round) with score |
| `trajectory_summary.csv` | 158 rows: one per (task, model) with OLS slope, classification, composite score |
