# Strategic Plan: Winning the Learning Benchmark (Deepmind Hackathon)

**Created:** April 14, 2026
**Deadline:** April 16, 2026 (~48 hours)
**Goal:** Win the Measuring Progress Toward AGI — Learning track with a research-grade benchmark, airtight write-up, and deep analysis.

---

## Table of Contents

1. [Where We Stand Right Now](#1-where-we-stand-right-now)
2. [Forward Track — What Needs to Be Done](#2-forward-track--what-needs-to-be-done)
3. [Backward Track — Questions We Must Answer](#3-backward-track--questions-we-must-answer)
4. [Hypotheses to Test](#4-hypotheses-to-test)
5. [Analysis Pipeline](#5-analysis-pipeline)
6. [Task Curation — Keep, Remove, Investigate](#6-task-curation--keep-remove-investigate)
7. [Robustness & Ablation Checklist](#7-robustness--ablation-checklist)
8. [Write-Up Strategy](#8-write-up-strategy)
9. [Execution Plan — Sub-Agent Work Items](#9-execution-plan--sub-agent-work-items)
10. [Risk Register](#10-risk-register)
11. [Phase F: Research-Grade Polish (ACL 2025 Paper Insights)](#phase-f-research-grade-polish-acl-2025-paper-insights)

---

## 1. Where We Stand Right Now

### What We Have
- **157 tasks** across 5 categories (Associative: 20, Concept Formation: 19, Language Learning: 26, Observational: 42, RL: 50)
- **14 models** evaluated on all leaderboards
- **Leaderboard JSON data** for all 5 categories with per-task, per-model scores
- **Downloaded task source code** (.py files for most tasks)
- **Scoring formulas** fully documented in SCORING.md
- **Project master document** (PROJECT_MASTER.md) covering philosophy, methodology, results summary
- **Aggregate scores** for 4 of 5 categories (Associative Learning aggregate pending)

### What We're Missing
- [ ] Associative Learning aggregate scores (stated as "pending recalculation")
- [ ] Comprehensive per-task analysis (which tasks discriminate, which are broken)
- [ ] Statistical robustness tests (reliability, prompt sensitivity)
- [ ] Ablation study results (example count, efficiency scoring impact)
- [ ] The actual Kaggle writeup (1500 words max)
- [ ] The public analysis notebook
- [ ] Cover image
- [ ] Cross-category correlation analysis
- [ ] Task-removal/retention decisions backed by data
- [ ] Procedural learning — deferred, likely skipping

---

## 2. Forward Track — What Needs to Be Done

### Phase A: Data Extraction & Analysis Foundation (IMMEDIATE)
| Step | What | Output | Time Est |
|------|------|--------|----------|
| A1 | Parse all 5 leaderboard JSONs into a unified matrix: `model × task → score` | `analysis/score_matrix.csv` | 30 min |
| A2 | Compute Associative Learning aggregate scores per model (currently "pending recalculation" — simply the mean of per-task scores from the leaderboard JSON, same as the other categories) | Updated results table | 15 min |
| A3 | Generate per-task statistics: mean, std, min, max, % of models scoring 0, % scoring 1, **entropy** | `analysis/task_stats.csv` | 20 min |
| A4 | Generate per-model statistics: mean across categories, rank per category | `analysis/model_profiles.csv` | 20 min |
| A5 | Flag problematic tasks: all-zero, all-perfect, extreme bimodality, **low variance across tiers** | `analysis/flagged_tasks.csv` | 20 min |

> **Note on Associative Learning aggregates:** These are "pending" simply because we haven't yet computed them from the raw leaderboard JSON. The per-task scores exist for all 14 models × 20 tasks — we just need to average them. WI-01 (parse leaderboards) will resolve this automatically.

### Phase B: Deep Analysis (Hours 2-8)
| Step | What | Output |
|------|------|--------|
| B1 | Discriminatory power analysis — Item Response Theory (IRT) or simpler item-discrimination index for each task | Per-task discrimination scores |
| B2 | Cross-category correlation matrix — do models that are good at one type of learning also excel at others? | Correlation heatmap + interpretation |
| B3 | Scale-performance analysis — do bigger models consistently score higher? Where do inversions happen? | Scaling curves per category |
| B4 | Efficiency analysis — for interactive tasks (CF, LL, RL), separate accuracy from efficiency contributions | Accuracy-only vs composite score comparison |
| B5 | Failure mode taxonomy — categorize WHY models fail (format errors, wrong rule, partial rule, overcautious, random) | Failure mode table |
| B6 | Model "cognitive profile" radar charts — each model's strengths/weaknesses across 5 learning types | Visualization per model |
| B7 | **Entropy analysis per category** — compute Shannon entropy of score distributions for each task and each category. High-entropy tasks spread models evenly (good signal). Low-entropy tasks cluster models (poor discrimination). Also compute per-category aggregate entropy to compare how "informative" each learning type is. | Per-task entropy + category-level entropy comparison |
| B8 | **Domain diversity audit** — map each task to its underlying domain (linguistics, math, physics, CS theory, chemistry, game theory, abstract algebra, automata, etc.). Check coverage balance per category. Identify any domain gaps. | Domain coverage matrix + gap analysis |
| B9 | **Model provider analysis** — group models by provider (Anthropic: Opus/Sonnet/Haiku, OpenAI: GPT-5.4/mini/nano, Google: Gemini Pro/Flash/Flash-Lite/Gemma, Open-source: GLM-5/DeepSeek/Qwen) and analyze provider-level patterns. Do some providers have systematic strengths/weaknesses on certain learning types? | Provider-level performance breakdown |
| B10 | **Tier-level analysis** — beyond just plotting scores by tier, analyze WHY small/mid/frontier models differ. For each category, which specific tasks cause tier separations? Where do tiers overlap? | Tier discrimination report |

### Phase B (continued): Kernel Log Analysis (Hours 6-8)
| Step | What | Output |
|------|------|--------|
| B14 | **Kernel log download** — for each of the 14 models, retrieve the `run.json` from the Kaggle kernel where it is the current (latest) version. Extract `startTime`, `endTime`, `totalBackendLatencyMs`, `inputTokens`, `outputTokens`, cost. Build a flat timing table. | `analysis/outputs/kernel_logs_parsed.csv`, `analysis/outputs/kernel_logs/manifest.json` |
| B15 | **Timing & token hypothesis tests** — run H14–H18 on the extracted timing data: thinking-model token overhead, cost-performance correlation, provider verbosity patterns, cost-per-score-point efficiency, token-efficiency ranking reversal. | `analysis/outputs/timing_hypotheses_report.md`, charts `fig_h14_*` – `fig_h18_*` |

> **Scripts:** `analysis/scripts/17_download_kernel_logs.py` (parallel, ~10 workers), then `analysis/scripts/18_timing_hypotheses.py`
>
> **Architecture note:** Kaggle retains output files only for the *latest* version of each kernel. Each benchmark run re-uses the same kernel and overwrites the previous version's files. To get one representative timing sample per model, we scan all task kernels to find one where each model is the current (latest) version.

### Phase C: Robustness & Ablation (Hours 8-16)
| Step | What | Output |
|------|------|--------|
| C1 | Test-retest reliability estimate — score consistency across task variants within same category | Reliability coefficient |
| C2 | Prompt sensitivity check — do minor wording changes affect scores? (sample 3-5 tasks) | Sensitivity analysis |
| C3 | Example count ablation — how does performance change with fewer/more examples? (sample tasks) | Learning curves |
| C4 | Efficiency scoring ablation — what would rankings look like with accuracy-only? | Alternative ranking table |
| C5 | Task removal sensitivity — how do aggregate scores change if we drop flagged tasks? | Stability analysis |

### Phase D: Final Benchmark Curation (Hours 16-20)
| Step | What | Output |
|------|------|--------|
| D1 | Remove confirmed bad tasks (all-zero, all-perfect, ambiguous, **low-variance tasks that fail to discriminate across small/mid/frontier tiers**) | Finalized task list |
| D2 | Verify all remaining tasks have correct ground truth | Validation report |
| D3 | Confirm balanced difficulty distribution per category | Difficulty histogram |
| D4 | **Aggressively prune RL tasks** — we have 50, many unverified. Keep only the strongest, most diverse, best-discriminating ones. Having fewer verified tasks is better than many unverified. | Reduced RL task set (target: 20-30 from 50) |
| D5 | Final leaderboard recomputation with cleaned task set | Final results |

### Phase E: Deliverables (Hours 20-36)
| Step | What | Output |
|------|------|--------|
| E1 | Write the Kaggle writeup (1500 words max, scientific paper style) | Writeup document |
| E2 | Build the public analysis notebook | `.ipynb` with all analysis |
| E3 | Create cover image | Image file |
| E4 | Final benchmark package on Kaggle | Submitted benchmark |
| E5 | Review everything end-to-end | Final check |

---

## 3. Backward Track — Questions We Must Answer

These are the questions the judges will (implicitly or explicitly) ask. Each must have a clear, evidence-backed answer in our writeup or notebook.

### 3.1 Core Validity Questions

| # | Question | What We Need to Answer It | Status |
|---|----------|---------------------------|--------|
| Q1 | **Does this benchmark actually measure learning, not recall?** | Evidence that tasks are novel (anti-contamination), evidence that models can't solve without the provided examples | Need to formalize |
| Q2 | **Are the scores reliable?** (Would a model get the same score on a re-run?) | Test-retest data or argument from deterministic scoring | Need to test |
| Q3 | **Is there discriminatory power?** (Not all-0 or all-1) | Per-task score distributions, IRT analysis | Need to compute |
| Q4 | **Are the ground truth answers correct?** | Computed from rule functions + human verification | Partially done |
| Q5 | **Is the benchmark contamination-resistant?** | All synthetic, novel names, impossible to memorize | Argument exists, needs evidence |

### 3.2 Insight Questions (The 30% — Novelty & Discriminatory Power)

| # | Question | Hypothesis | Analysis Needed |
|---|----------|------------|-----------------|
| Q6 | **What can this benchmark tell us about model behavior that we couldn't see before?** | See sub-hypotheses below | Multi-pronged analysis |

**Q6 Sub-Hypotheses (Critical Question — Expand Into Multiple Angles):**

| # | Sub-Hypothesis | Test | Expected Finding |
|---|---------------|------|------------------|
| Q6a | Learning profiles are orthogonal to general capability rankings | Compare our rankings to Chatbot Arena / MMLU rankings for the same models | Models that rank high on general benchmarks may rank differently here |
| Q6b | Learning reveals hidden provider-level patterns | Group by provider (Anthropic, OpenAI, Google, Open-source) and compare across learning types | Providers may have systematic biases — e.g., Google models might excel at pattern learning, OpenAI at rule inference |
| Q6c | Efficiency (learning speed) is a completely new axis | Show that the efficiency-included ranking differs significantly from accuracy-only | Learning speed is invisible to all existing benchmarks and reshuffles model rankings |
| Q6d | Epistemic uncertainty handling is a measurable dimension | UNKNOWN-answer accuracy analysis | Models systematically over-commit — a failure mode invisible in standard evals |
| Q6e | Small models have unexpected strengths on specific learning types | Tier analysis with specific inversions | This benchmark reveals capability islands — specific tasks where a small model beats frontier |
| Q6f | Interactive learning (CF, LL, RL) reveals more than static evaluation | Compare CV and discrimination power of interactive vs static categories | The act of learning in-context is more differentiating than the outcome of learning |
| Q7 | **Why does Gemini 3.1 Pro dominate across all categories?** | Better in-context learning architecture, or better calibrated confidence | Score breakdown + efficiency analysis |
| Q8 | **Why do larger models sometimes score LESS on concept formation?** | Over-requesting examples (efficiency penalty) vs. genuine accuracy gap | Separate accuracy from efficiency scores |
| Q9 | **Why does Gemini 2.5 Flash outperform GPT-5.4 and Claude Opus on observational learning?** | Flash's architecture may be optimized for pattern extraction from dense input | Task-level breakdown of where Flash wins |
| Q10 | **Why does Gemma 4 26B (small) outperform GPT-5.4 and Claude Opus on RL?** | Explore-exploit behavior may not scale with model size | RL strategy analysis |
| Q11 | **Do models handle epistemic uncertainty correctly?** | Models over-commit rather than saying "UNKNOWN" | UNKNOWN-answer analysis in associative learning |
| Q12 | **Is there a speed-accuracy tradeoff in learning?** | Models that learn fast make more errors; models that are cautious learn correctly but slowly | Efficiency vs accuracy scatter plots |

### 3.3 Methodology Questions

| # | Question | Answer Needed |
|---|----------|---------------|
| Q13 | Why no static dataset? | Contamination risk, interaction protocols, computed ground truth (Section 3 of PROJECT_MASTER already covers this) |
| Q14 | Why synthetic tasks instead of real-world learning? | Any real-world task is in pre-training data |
| Q15 | Why these specific sub-abilities? | Grounded in DeepMind's cognitive framework |
| Q16 | Why these 14 models? | Span small/mid/frontier tiers across all major providers |
| Q17 | How were tasks validated? | 5-point validation process (Section 5) |
| Q18 | Why is efficiency part of the score? | Measures learning speed, not just learning outcome |

### 3.4 Presentation Questions

| # | Question | Needs |
|---|----------|-------|
| Q19 | Is the writeup clear, concise, and well-structured? | Draft and iterate |
| Q20 | Are the visualizations compelling? | Design and create |
| Q21 | Does the notebook reproduce all claims? | Build reproducible notebook |

---

## 4. Hypotheses to Test

Each hypothesis is a potential insight for the writeup. We test it with data and report the finding.

### H1: Learning Is Not Monolithic
- **Claim:** Performance on one learning sub-ability does not predict performance on another.
- **Test:** Compute Spearman rank correlation between model rankings across all 5 categories. If correlations are low (<0.7), the hypothesis holds.
- **Implication:** Existing benchmarks that measure "general intelligence" miss these orthogonal dimensions.
- **Analysis:** Cross-category rank correlation matrix.

### H2: Scale Does Not Linearly Improve Learning
- **Claim:** Bigger models are not always better learners. There exist capability inversions.
- **Test:** Plot score vs. model tier (Small/Mid/Frontier) per category. Identify specific inversions (e.g., Gemma 26B > GPT-5.4 on RL).
- **Implication:** Learning ability is partially independent of general capability, suggesting it's a distinct cognitive dimension.
- **Analysis:** Score-by-tier box plots + specific inversion examples.

### H3: Models Have a Speed-Accuracy Tradeoff in Learning
- **Claim:** For interactive tasks (CF, LL), models that request fewer examples have higher efficiency scores but potentially lower accuracy.
- **Test:** For each model, compute average accuracy-only vs. average composite score on CF and LL. Check if models with high efficiency sacrifice accuracy.
- **Implication:** Reveals metacognitive calibration — does the model know when it has learned enough?
- **Analysis:** Accuracy vs. efficiency scatter plot per model.

### H4: Efficiency Penalty Reverses Rankings
- **Claim:** The efficiency component in our scoring changes model rankings compared to accuracy-only.
- **Test:** Compute rankings with accuracy-only for CF and LL. Compare to current composite rankings.
- **Implication:** Validates that efficiency scoring captures a real, independent signal.
- **Analysis:** Side-by-side ranking tables.

### H5: Models Over-Commit on Uncertainty (Epistemic Failure)
- **Claim:** Models systematically fail on questions where "UNKNOWN" is the correct answer — they always commit to a definitive answer.
- **Test:** In associative learning, identify all questions where the correct answer is UNKNOWN. Compute model accuracy on those vs. non-UNKNOWN questions.
- **Implication:** Models lack epistemic humility — a critical failure mode for reliable AI.
- **Analysis:** UNKNOWN vs. definitive answer accuracy breakdown.

### H6: Gemini 3.1 Pro's Dominance Is Uniform
- **Claim:** Gemini 3.1 Pro doesn't just lead on aggregate — it leads on almost every individual task.
- **Test:** Count how many tasks each model ranks #1 on. Check if Gemini 3.1 Pro's dominance comes from being best on most tasks or from being consistently good (never lowest).
- **Implication:** Is this a general learning architecture advantage, or does it have hidden weaknesses?
- **Analysis:** Rank-1 counts + per-task rank distribution.

### H7: Bimodal Tasks Reveal Binary "Gets It or Doesn't" Pattern
- **Claim:** Some tasks show extreme bimodality: models either score ~1.0 or ~0.0, with nothing in between.
- **Test:** Identify tasks with bimodal score distributions. Analyze what cognitive demand separates "gets it" from "doesn't."
- **Implication:** Some learning challenges have a phase-transition quality — partial understanding is impossible.
- **Analysis:** Score histograms per task, identify bimodal ones.

### H8: Non-Linear Example Count Effects (Brittleness)
- **Claim:** Unlike humans who degrade gradually as task conditions change, models can suddenly drop to 0% — revealing brittle learning.
- **Test:** If we can run ablation with varying example counts on a sample of tasks, check for cliff-edge drops.
- **Implication:** Models have fragile representations that break catastrophically rather than degrading gracefully.
- **Analysis:** Performance vs. example count curves.

### H9: Interactive Tasks Reveal More Than Static Tasks
- **Claim:** Categories with interactive protocols (CF, LL, RL) produce higher-variance, more discriminating scores than static categories (Associative, Observational).
- **Test:** Compare coefficient of variation (CV) across categories.
- **Implication:** The learning process itself (how models gather evidence) is more differentiating than the learning outcome.
- **Analysis:** CV comparison across categories.

### H10: Thinking Models vs. Non-Thinking Models
- **Claim:** Models with explicit reasoning/thinking capabilities (Qwen Thinking, Gemini Pro) outperform their non-thinking counterparts.
- **Test:** Compare Qwen 3 Next 80B Thinking vs. Qwen 3 Next 80B Instruct across all categories.
- **Implication:** Chain-of-thought / thinking tokens improve learning ability at inference time.
- **Analysis:** Paired comparison on the Qwen pair.

### H11: Model Providers Have Systematic Learning Biases
- **Claim:** Different model providers (Anthropic, OpenAI, Google, Open-source) have distinct learning profiles. For example, Google models may systematically excel at pattern-based learning while OpenAI models may be better at rule inference.
- **Test:** Group models by provider. Compute per-provider mean on each category. Test with ANOVA or Kruskal-Wallis whether provider is a significant factor.
- **Implication:** Training approaches, data mixtures, and architecture choices create systematic biases in learning ability — visible only when learning is decomposed into sub-abilities.
- **Analysis:** Provider × category heatmap + statistical tests.

### H12: Task Entropy Separates Informative Categories from Noisy Ones
- **Claim:** Categories with higher average task entropy produce more informative model rankings (better discrimination). Categories with low entropy are either too easy or too hard uniformly.
- **Test:** Compute average Shannon entropy per category. Correlate with the category's ability to separate model tiers.
- **Implication:** Guides which learning sub-abilities are most useful for benchmarking and where we need better tasks.
- **Analysis:** Entropy comparison bar chart + entropy vs. discrimination correlation.

### H13: Benchmark Diversity Covers the Space of Learning Challenges
- **Claim:** Our tasks span diverse domains (linguistics, math, logic, science, game theory, abstract algebra, etc.) — not just variations on the same theme.
- **Test:** Classify all tasks by domain. Check whether each category has domain diversity and whether overall benchmark covers a wide space.
- **Implication:** Strengthens the argument that our benchmark measures general learning, not domain-specific skill.
- **Analysis:** Domain coverage matrix + sunburst/treemap visualization.

### H14: Thinking Models Incur Dramatically Higher Token Costs
- **Claim:** Models with explicit thinking/reasoning modes (Qwen3 Thinking, Gemini 3.1 Pro) generate significantly more output tokens per task than standard non-thinking models.
- **Test:** Compare mean output token counts between thinking and non-thinking models. Mann-Whitney U test.
- **Implication:** The learning advantage from explicit reasoning comes at a measurable cost — thinking tokens are the "compute price of cognition." Benchmarking must account for inference cost, not just accuracy.
- **Analysis:** `18_timing_hypotheses.py` H14.

### H15: Inference Cost Does Not Simply Predict Performance
- **Claim:** There is no strong positive correlation between model inference cost and task score — some cheap models punch above their weight.
- **Test:** Spearman correlation between log(cost_usd) and score across models on sampled tasks.
- **Implication:** Learning ability is not simply bought. A model's learning architecture matters more than its parameter count or API price.
- **Analysis:** `18_timing_hypotheses.py` H15.

### H16: Provider Training Style Predicts Output Verbosity
- **Claim:** Models from different providers (Anthropic, OpenAI, Google, Open-source) generate systematically different amounts of output tokens, reflecting different instruction-following and response-formatting training.
- **Test:** Kruskal-Wallis test across provider groups on output token counts.
- **Implication:** Verbosity is a training artifact, not just a task requirement. Providers have distinct "communication styles" that affect inference cost, latency, and potentially scoring.
- **Analysis:** `18_timing_hypotheses.py` H16.

### H17: Cost-Per-Point Reveals Hidden Efficiency Rankings
- **Claim:** When we normalize score by inference cost (cost per correct point), the model rankings change substantially — some frontier models are extremely cost-inefficient relative to their scores.
- **Test:** Compute cost_usd / score for each model. Compare ranking by cost-per-point to ranking by raw score.
- **Implication:** A practical benchmark should consider both capability and efficiency. A model that scores 10% better but costs 5× more may not be the right choice for learning-based agents.
- **Analysis:** `18_timing_hypotheses.py` H17.

### H18: Token Efficiency Ranking Flips the Leaderboard
- **Claim:** When models are ranked by tokens-per-score-point (score / output_tokens), the leaderboard changes, with thinking models dropping and efficient non-thinking models rising.
- **Test:** Spearman correlation between score rank and token-efficiency rank. Identify largest rank changes.
- **Implication:** Reveals that there are two valid dimensions of model quality: raw learning performance and learning efficiency. These are partially orthogonal, giving the benchmark a second axis of novelty.
- **Analysis:** `18_timing_hypotheses.py` H18.

### H19: Frontier Models Cannot Override Semantic Priors (Over-Specification)
- **Claim:** The strongest frontier models (Gemini 3.1 Pro, GPT-5.4) fail catastrophically on tasks that require learning a rule which contradicts their pre-trained semantic defaults. Smaller models succeed because they have weaker priors and can be updated by evidence.
- **Evidence:** `semantic_override_concept_learning` — Gemini 3.1 Pro=0.00, GPT-5.4=0.00 while Gemini 2.5 Flash=0.95, Gemma=0.90, Claude Opus=0.75.
- **Interpretation:** RLHF/instruction-tuning at scale creates "hardened priors" — strong models are over-specified toward human-intuitive answers. When examples contradict those defaults, they cannot update. This is a new form of alignment brittleness: not safety-related over-refusal, but semantic over-commitment.
- **Why novel:** No existing benchmark explicitly presents counter-intuitive rules as the ground truth. All known evals reward models for following semantic intuition. This exposes a learning failure that scale makes *worse*, not better.
- **Test:** Count tasks where top-3 models score < 0.10 while middle-tier models score > 0.70. Compute a "semantic rigidity index" per model.

### H20: Capability Is Non-Monotonic Within Model Families
- **Claim:** A larger model from a provider does not always outperform a smaller model from the same provider on specific learning tasks. The relationship between model scale and task performance is non-monotonic.
- **Evidence:** `manhattan_point_rf_learning` — Gemini 3.1 Flash-Lite=1.00, Gemini 2.5 Flash=0.15, Gemini 3.1 Pro=0.00. `grid_octile_rf_learning` — Claude Sonnet 4.6=1.00, Claude Opus 4.6=0.00.
- **Interpretation:** Fine-tuning choices, RLHF objective, and capability unlocking differ within provider families. A more capable foundation model can be trained in ways that suppress certain task-specific behaviors.
- **Why novel:** Benchmarks typically assume monotonic scaling. Our benchmark reveals specific task types where the larger sibling underperforms the smaller sibling — a testable, reproducible prediction about how these models are built.
- **Test:** For each task, compute the within-provider rank ordering. Flag tasks where the standard tier ordering is violated for ≥ 2 models from the same provider.

### H21: Provider-Specific RL Blind Spots Exist
- **Claim:** Different model providers have systematic blind spots in reinforcement-learning-style tasks. Specifically, OpenAI frontier models (GPT-5.4) fail at sequential constraint-inference tasks where Google and open-source models succeed.
- **Evidence:** `minesweeper_1d_rf_learning` — GPT-5.4=0.00, GPT-5.4 mini=0.00 while Gemma=0.83, Haiku=0.80. `verbal_bandit_rf_learning` — GPT-5.4=0.18, Opus=0.20, but Gemini Pro=1.00. `letter_overlap_word` — GPT-5.4=0.00, Gemini 2.5 Flash=1.00.
- **Interpretation:** OpenAI models may have been trained primarily on single-turn reasoning tasks with limited exposure to multi-step trial-and-error inference. The failure is systematic across task types within the RL category.
- **Why novel:** Existing benchmarks do not separate single-turn from multi-turn, feedback-driven learning. Our benchmark makes this failure mode visible and reproducible.
- **Test:** Within RL category, split tasks by whether they require multi-step trial-and-error vs. single-inference. Compare OpenAI vs. Google provider win rates on each subset.

### H22: Qwen Models Have a Hidden Priority-Ordering Advantage
- **Claim:** Qwen models (both thinking and instruct variants) have a systematic advantage on tasks requiring hidden priority ordering and multi-step observational inference that is not explained by their overall rank.
- **Evidence:** `hidden_priority_order_obs_learning` — Qwen3 80B Thinking=1.00, Qwen3 80B Instruct=1.00, Claude Opus=0.75, all other 11 models=0.00. This is not a fluke — both Qwen variants reproduce the result.
- **Interpretation:** Alibaba's training pipeline may have included strong emphasis on structured reasoning about priority/ordering problems. This is a narrow capability spike not visible in general benchmarks.
- **Why novel:** Qwen models rank 3rd-4th overall on our benchmark, yet uniquely dominate this specific task type. General benchmarks would show this as "Qwen is good," missing the specific capability structure entirely.
- **Test:** Identify all tasks requiring priority inference or hidden ordering. Compute Qwen models' relative rank on those tasks vs. their overall rank. Significance test vs. chance.

---

## 5. Analysis Pipeline

### 5.1 Analysis Strategy: Notebooks + Scripts

> **Decision:** All analysis will be done in **Jupyter notebooks** so they can be attached to the write-up and are directly visible to the judges. Each major analysis area gets its own notebook. Utility functions live in shared Python modules that notebooks import.
>
> For the final submission, we will likely submit a **single benchmark leaderboard** containing all tasks (rather than separate per-category leaderboards). This keeps it clean for the judges. We'll figure out the exact format at the end.

```
analysis/
├── parse_leaderboards.py     # A1: JSON → unified score matrix
├── task_statistics.py         # A3, A5: per-task stats + flagging
├── model_profiles.py          # A4: per-model profiles
├── discriminatory_power.py    # B1: item discrimination analysis
├── cross_category.py          # B2: correlation analysis
├── scaling_analysis.py        # B3: scale vs. performance
├── efficiency_ablation.py     # B4, C4: accuracy-only vs. composite
├── failure_modes.py           # B5: failure categorization
├── robustness_checks.py       # C1-C5: all robustness analyses
├── visualizations.py          # All charts and plots
└── writeup_stats.py           # Extract key numbers for writeup
├── scripts/                       # ← Each analysis is a standalone notebook
│   ├── 01_data_parsing.py        # A1-A2: Parse JSONs → score matrix
│   ├── 02_task_statistics.py     # A3-A5: Per-task stats + flagging + entropy
│   ├── 03_model_profiles.py      # A4: Per-model profiles + tier classification
│   ├── 04_cross_category.py      # B2: Correlation analysis (H1)
│   ├── 05_scaling_analysis.py    # B3, B10: Scale vs. performance + tier analysis
│   ├── 06_efficiency_ablation.py # B4: Accuracy-only vs. composite (H3, H4)
│   ├── 07_entropy_analysis.py    # B7: Task and category entropy
│   ├── 08_provider_analysis.py   # B9: Model provider breakdown
│   ├── 09_domain_diversity.py    # B8: Domain coverage audit
│   ├── 10_epistemic_analysis.py  # UNKNOWN accuracy (H5)
│   ├── 11_bimodality.py          # H7: Bimodal tasks
│   ├── 12_robustness_checks.py   # R1-R10: All robustness analyses
│   └── 13_final_visualizations.py # All publication-quality charts
├── utils/                           # Shared modules imported by notebooks
│   ├── data_loader.py               # Parse leaderboard JSONs, build score matrix
│   ├── stats.py                     # Statistical functions (entropy, discrimination, etc.)
│   └── viz.py                       # Plotting helpers for consistent style
└── outputs/                         # Generated CSVs, figures
    ├── score_matrix.csv
    ├── task_stats.csv
    ├── model_stats.csv
    ├── flagged_tasks.csv
    └── figures/
```

### 5.2 Key Visualizations Needed (Research Paper Quality)

All graphs should look like they belong in a top ML research paper — clean, informative, and visually compelling.

1. **Radar chart** — per-model cognitive profile across 5 categories
2. **Heatmap** — model × task score matrix (each category)
3. **Correlation matrix** — cross-category rank correlations
4. **Box plots** — score distributions by model tier per category
5. **Scatter plot** — accuracy vs. efficiency for interactive categories
6. **Bar chart** — overall model rankings (final leaderboard)
7. **Histogram** — per-task score distributions (identify bimodal tasks)
8. **Learning curve** — performance vs. example count (ablation)
9. **Rank comparison** — accuracy-only vs. composite rankings (side-by-side)
10. **UNKNOWN accuracy** — epistemic failure analysis bar chart
11. **Task discrimination** — sorted bar chart of item discrimination index
12. **Entropy bar chart** — per-category entropy comparison
13. **Provider breakdown** — grouped bar chart showing each provider's mean score per category
14. **Domain diversity sunburst/treemap** — visual of task domain coverage across categories
15. **Tier violin plots** — score distributions per tier (more expressive than box plots)
16. **Task retention waterfall** — showing how many tasks survive each curation filter

---

## 6. Task Curation — Keep, Remove, Investigate

### 6.1 Criteria for Removal
| Criterion | What It Means | Action |
|-----------|--------------|--------|
| All-zero | Every model scores 0.0 | Remove — task is broken or infeasible |
| All-perfect | Every model scores 1.0 | Remove — no discriminatory power |
| Near-uniform | Std dev < 0.05 across models | Investigate — may lack signal |
| **Low tier-variance** | **Scores don't vary meaningfully across small/mid/frontier tiers — the task can't distinguish model capability levels** | **Remove — not adding value to understanding model differences** |
| Low entropy | Score distribution is heavily concentrated (most models get same score) | Remove — no informational value |
| Extreme bimodality | Only 1-2 models score non-zero, rest are 0 | Investigate — may be too hard or have a trick |
| Redundant | Highly correlated (r > 0.95) with another task in same category | Remove the less discriminating one |
| Ambiguous ground truth | Multiple valid interpretations | Fix or remove |
| Format-dependent scoring | Score depends on output format, not learning | Fix scoring or remove |
| **Unverified (RL)** | **RL tasks that haven't been manually verified — we have 50, many are unverified** | **Remove aggressively — quality > quantity** |

### 6.2 Analysis Needed Per Task
For each of the 157 tasks, compute:
- Mean score across 14 models
- Standard deviation
- **Shannon entropy** of the score distribution (discretized into bins)
- Number of models scoring 0.0
- Number of models scoring 1.0
- Item discrimination index (correlation of task score with total category score)
- Bimodality test (Hartigan's dip test or visual inspection)
- **Tier discrimination** — is there a statistically meaningful difference in scores between small, mid, and frontier models?
- **Inter-task correlation** — how correlated is this task with others in the same category?

### 6.3 Expected Outcomes
- **Keep:** Tasks with mean between 0.15–0.85, std > 0.1, positive discrimination, meaningful tier variance
- **Remove:** Tasks at extremes (mean < 0.05 or mean > 0.95), near-zero variance, low entropy, unverified RL tasks
- **Investigate:** Bimodal tasks — understand why before deciding
- **RL specifically:** Start with 50, expect to prune to ~20-30 after removing unverified and low-signal tasks
- **Final count:** Likely 100-130 tasks after curation (rough estimate, down from 157)

---

## 7. Robustness & Ablation Checklist

### Must-Have (for judges)
- [ ] **R1: Ground truth verification** — Confirm all answers are computed from rule functions
- [ ] **R2: Score distribution analysis** — Show tasks produce a gradient, not binary
- [ ] **R3: Item discrimination** — Show individual tasks contribute meaningful signal
- [ ] **R4: Task removal sensitivity** — Show aggregate rankings are stable when dropping individual tasks (leave-one-out)
- [ ] **R5: Efficiency scoring ablation** — Show what changes with accuracy-only

### Nice-to-Have (strengthens submission)
- [ ] **R6: Test-retest reliability** — Run a few tasks twice, show consistency
- [ ] **R7: Prompt variation sensitivity** — Rephrase a few tasks, show robustness
- [ ] **R8: Example count ablation** — Vary training example count, show learning curves
- [ ] **R9: Cross-validation of task selection** — Bootstrap analysis of task subsets
- [ ] **R10: Random baseline** — Show random responding produces near-zero scores

---

## 8. Write-Up Strategy

### 8.1 Structure (1500 words max — every word counts)

| Section | Words | Content |
|---------|-------|---------|
| Title + Team | 20 | "Measuring How AI Learns: A Multi-Dimensional Benchmark for Inference-Time Learning" |
| Problem Statement | 150 | Why we need this. What's missing from existing evals. The learning gap. |
| Key Insight | 100 | Learning is not monolithic — it decomposes into distinct cognitive skills. Scale doesn't guarantee learning. |
| Benchmark Design | 250 | 5 sub-abilities, 157 tasks, interactive protocols, efficiency scoring. Cognitive science grounding. |
| Dataset | 150 | Synthetic, programmatic, contamination-immune. Why not static CSV. Per-category structure. |
| Technical Details | 200 | Scoring formulas, validation pipeline, anti-contamination measures. |
| Results | 300 | Key findings: Gemini dominance, scale inversions, epistemic failures, speed-accuracy tradeoff. Tables. |
| Insights & Novelty | 250 | What this reveals that no other benchmark can. The 3-4 strongest insights. |
| References | 80 | Key citations |

### 8.2 The "Killer Insights" We Lead With

These are the most surprising, novel findings that justify our benchmark's existence:

1. **Learning is multi-dimensional** — Models have distinct cognitive profiles across learning types. A top performer on one dimension can be mediocre on another.

2. **Scale inversions exist** — Small models can outperform frontier models on specific learning tasks (Gemma 26B > GPT-5.4 on RL, Gemini Flash > Claude Opus on observation).

3. **Models lack epistemic humility** — When evidence is insufficient, models commit to wrong answers instead of acknowledging uncertainty.

4. **Learning speed is an invisible dimension** — Models that achieve the same accuracy differ vastly in how many examples they need. This is invisible to existing benchmarks.

5. **Learning is brittle** — Models can drop from 100% to 0% with small task variations, unlike human gradual degradation.

### 8.3 What Makes Us Win (Competitive Analysis)

| Judge Criteria | Weight | Our Strength |
|----------------|--------|-------------|
| Dataset quality & task construction | 50% | 157 tasks, 5 categories, computed ground truth, 5-point validation, 14 models. Massive scope. |
| Writeup quality | 20% | Cognitive science grounding, clear methodology, compelling visualizations, scientific rigor. |
| Novelty, insights, discriminatory power | 30% | Multiple novel findings invisible to existing benchmarks. Clear performance gradients. Surprising inversions. |

### 8.4 Submission Format Strategy

> **Leaning towards:** A **single unified benchmark leaderboard** attached to the write-up (combining all categories), rather than separate per-category leaderboard submissions. This makes the submission cleaner and gives judges one leaderboard to look at. The analysis notebooks can show per-category breakdowns. Final decision at the end.
>
---

## 9. Execution Plan — Sub-Agent Work Items

Each item below is an independent unit of work that can be executed by a sub-agent.

### PRIORITY 1 — Data Foundation (Do First, Everything Depends On This)

#### WI-01: Parse Leaderboards Into Score Matrix
- **Input:** 5 leaderboard JSONs
- **Task:** Extract model × task score matrix for each category. Output unified CSV/DataFrame.
- **Output:** `analysis/score_matrix.csv` with columns: category, task_name, model, score
- **Dependencies:** None

#### WI-02: Compute Task-Level Statistics
- **Input:** Score matrix from WI-01
- **Task:** For each task: mean, std, min, max, count_zero, count_perfect, item_discrimination_index
- **Output:** `analysis/task_stats.csv` + list of flagged tasks
- **Dependencies:** WI-01

#### WI-03: Compute Model-Level Statistics
- **Input:** Score matrix from WI-01
- **Task:** For each model: mean per category, overall rank, rank per category, tier classification
- **Output:** `analysis/model_stats.csv`
- **Dependencies:** WI-01

### PRIORITY 2 — Key Analyses (The Insights)

#### WI-04: Cross-Category Correlation Analysis (H1)
- **Input:** Model scores per category
- **Task:** Spearman rank correlation between all pairs of categories. Test H1.
- **Output:** Correlation matrix + interpretation
- **Dependencies:** WI-03

#### WI-05: Scale vs. Performance Analysis (H2)
- **Input:** Model scores + tier labels
- **Task:** Score distributions by tier per category. Identify specific inversions.
- **Output:** Box plots + inversion examples
- **Dependencies:** WI-03

#### WI-06: Efficiency Ablation Analysis (H3, H4)
- **Input:** Leaderboard data for CF and LL (need per-task accuracy AND efficiency separately, or recompute)
- **Task:** Separate accuracy from efficiency. Compare rankings with vs. without efficiency.
- **Output:** Accuracy-only rankings + scatter plots
- **Dependencies:** WI-01, possibly need task source code

#### WI-07: Epistemic Uncertainty Analysis (H5)
- **Input:** Associative learning task source code + leaderboard scores
- **Task:** Identify questions where UNKNOWN is correct. Compute model accuracy on UNKNOWN vs. definitive questions.
- **Output:** UNKNOWN accuracy table
- **Dependencies:** WI-01

#### WI-08: Gemini Dominance Analysis (H6)
- **Input:** Score matrix
- **Task:** For each task, rank all models. Count how many tasks each model is #1. Analyze Gemini's dominance pattern.
- **Output:** Rank-1 distribution + task-level dominance analysis
- **Dependencies:** WI-01

#### WI-09: Bimodality Analysis (H7)
- **Input:** Score matrix
- **Task:** For each task, compute bimodality coefficient. Identify binary "gets it or doesn't" tasks.
- **Output:** Bimodality report + score distribution histograms
- **Dependencies:** WI-01

#### WI-10: Thinking vs. Non-Thinking Analysis (H10)
- **Input:** Qwen Thinking vs. Qwen Instruct scores
- **Task:** Paired comparison across all tasks and categories.
- **Output:** Thinking advantage table
- **Dependencies:** WI-01

#### WI-10B: Entropy Analysis Per Category (B7)
- **Input:** Score matrix from WI-01
- **Task:** Compute Shannon entropy per task and per category. Identify which categories produce the most informative (high entropy) score distributions. Low-entropy tasks are candidates for removal.
- **Output:** `scripts/07_entropy_analysis.py` — entropy tables + visualizations
- **Dependencies:** WI-01

#### WI-10C: Model Provider Analysis (B9)
- **Input:** Score matrix + provider labels
- **Task:** Group models by provider (Anthropic, OpenAI, Google DeepMind, Open-source). Analyze provider-level patterns: do certain providers have systematic strengths/weaknesses? Compare average scores per provider per category. Test multiple hypotheses (see Q6b).
- **Output:** `scripts/08_provider_analysis.py` — provider breakdown charts + statistical tests
- **Dependencies:** WI-03

#### WI-10D: Domain Diversity Audit (B8)
- **Input:** Task source code + task metadata
- **Task:** Classify each task by underlying domain/topic (linguistic, mathematical, logical, scientific, game theory, abstract algebra, etc.). Assess how diverse the benchmark is within and across categories. Identify any domain clustering or gaps.
- **Output:** `scripts/09_domain_diversity.py` — domain coverage matrix + sunburst visualization
- **Dependencies:** WI-01

#### WI-10E: Tier Deep-Dive (B10)
- **Input:** Score matrix + tier labels
- **Task:** For each category, identify which specific tasks cause the sharpest tier separations vs. which tasks have tier overlap. Explain WHY small/mid/frontier models differ on each category — not just that they do.
- **Output:** `scripts/05_scaling_analysis.py` (combined with WI-05) — tier discrimination tables + explanations
- **Dependencies:** WI-01, WI-02

#### WI-10F: Kernel Log Download + Timing Analysis (B14, B15)
- **Input:** All ~154 task kernels on Kaggle (scanned via Kaggle API)
- **Task:**
  1. For each of the 14 models, identify one task kernel where that model is the current (latest) version. Download its `run.json` output using `kaggle kernels output` CLI.
  2. Parse each `run.json` to extract: `startTime`, `endTime`, `totalBackendLatencyMs`, `inputTokens`, `outputTokens`, `cost_nanodollars`, `score`, and model slug.
  3. Run timing hypothesis tests H14–H18 on the extracted data.
- **Output:**
  - `analysis/outputs/kernel_logs/manifest.json`
  - `analysis/outputs/kernel_logs_parsed.csv`
  - `analysis/outputs/timing_hypotheses_report.md`
  - Charts: `fig_h14_thinking_tokens.png`, `fig_h15_cost_vs_score.png`, `fig_h16_provider_verbosity.png`, `fig_h17_cost_per_point.png`, `fig_h18_efficiency_rank.png`
- **Dependencies:** None (uses Kaggle CLI)
- **Scripts:** `17_download_kernel_logs.py` → `18_timing_hypotheses.py`

### PRIORITY 3 — Robustness Checks

#### WI-11: Task Removal Sensitivity (R4)
- **Input:** Score matrix + flagged tasks from WI-02
- **Task:** Leave-one-out analysis. How much do aggregate rankings change when dropping each task?
- **Output:** Stability analysis
- **Dependencies:** WI-02

#### WI-12: Random Baseline (R10)
- **Input:** Scoring formulas
- **Task:** Compute expected score from random responding for each category.
- **Output:** Random baseline scores
- **Dependencies:** None

#### WI-13: Ground Truth Spot-Check (R1)
- **Input:** Downloaded task source code
- **Task:** For 10-15 representative tasks, manually verify ground truth computation.
- **Output:** Verification report
- **Dependencies:** None

### PRIORITY 4 — Visualizations

#### WI-14: All Key Visualizations
- **Input:** Results from WI-01 through WI-13
- **Task:** Generate all 11 visualization types listed in Section 5.2
- **Output:** Publication-quality charts
- **Dependencies:** All analysis work items

### PRIORITY 5 — Deliverables

#### WI-15: Write the Kaggle Writeup
- **Input:** All analysis results, key findings
- **Task:** Draft 1500-word writeup following template
- **Output:** Writeup document
- **Dependencies:** WI-01 through WI-14

#### WI-16: Build Public Analysis Notebook
- **Input:** All analysis scripts and results
- **Task:** Assemble into a clean, reproducible Jupyter notebook
- **Output:** `analysis_notebook.ipynb`
- **Dependencies:** WI-01 through WI-14

#### WI-17: Create Cover Image
- **Input:** Key visualization (likely radar chart or multi-panel figure)
- **Task:** Design compelling cover image
- **Output:** Cover image file
- **Dependencies:** WI-14

#### WI-18: Final Benchmark Package
- **Input:** Curated task list, all scores, writeup, notebook
- **Task:** Package and submit on Kaggle
- **Output:** Submitted benchmark
- **Dependencies:** Everything

---

---

## Phase F: Research-Grade Polish (ACL 2025 Paper Insights)

> **Source:** AfriMed-QA (ACL 2025, Long Paper #96) — a pan-African medical benchmark from Google Research + 13 institutions. 30 models, 15k questions, multiple evaluation axes. Published methodology is the current gold standard for benchmark papers targeting top venues and research audiences.
>
> **Relevance:** We are targeting DeepMind researchers. The AfriMed-QA paper shows exactly what a research-grade benchmark paper looks like at ACL/NeurIPS level. Below are the concrete gaps between our current write-up and that standard, and the specific improvements to close them.

### F1: Benchmark Comparison Table (Critical — Judges Will Ask "How Does This Compare?")

AfriMed-QA opens with a **comparative feature table** (Table 1) benchmarking itself against 7 prior datasets across ~10 axes. This is standard practice in top benchmark papers — it immediately establishes novelty and positions the contribution.

**What we need:** Add a comparison table to the writeup showing LearningBench vs. existing evals (MMLU, BIG-Bench, ARC, HELM, Chatbot Arena, ARC-Challenge, IFEval) across axes that highlight our unique design:

| Feature | LearningBench | MMLU | BIG-Bench | ARC | HELM | IFEval |
|---|---|---|---|---|---|---|
| Measures inference-time learning | ✓ | ✗ | Partial | ✗ | ✗ | ✗ |
| Contamination-immune (synthetic) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Interactive protocols (active evidence request) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Efficiency scoring (learning speed) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Epistemic uncertainty axis (UNKNOWN) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Multi-dimensional cognitive profile | ✓ | ✗ | Partial | ✗ | Partial | ✗ |
| Programmatically verified ground truth | ✓ | Partial | Partial | Partial | Partial | ✗ |
| Number of models evaluated | 14 | Varies | Varies | Varies | Varies | Varies |
| Task count | 134 | 15,908 | 204 tasks | 1,119 | — | 500 |

> **Action:** Add this as Table 1 in the writeup (Section 2 or after Problem Statement). It takes ~50 words of prose to introduce and immediately signals that we know the field.

### F2: Multi-Metric Model Evaluation Table (Strengthens Rigour Claim)

AfriMed-QA evaluates each model on **multiple metrics simultaneously** (MCQ accuracy, BERTScore, ROUGE-Lsum, QuestEval) and reports all of them in a single comprehensive table (Table 3). This is 30 models × 8 metric columns. The sheer density of this table signals research-grade effort.

**What we have:** Our Table 2 (per-category breakdown) is good but reports only the composite score per category. We are missing the sub-metric breakdown that distinguishes our scoring.

**What to add:**
- For Concept Formation and Language Learning: show **accuracy-only score** alongside the **composite score** (the efficiency penalty component becomes a visible column)
- For RL: show the three sub-components separately: `solved_score`, `efficiency_score`, `progress_score`
- For Associative: show **UNKNOWN-accuracy** alongside overall accuracy

This table variant should appear in the supplementary/appendix section of the notebook (not necessarily in the 1500-word writeup) but should be referenced from the writeup.

> **Action:** Generate `analysis/outputs/per_model_sub_metric_breakdown.csv` and include as an extended table in the analysis notebook.

### F3: Quantitative Robustness of Automated Metrics (Addresses a Likely Judge Question)

AfriMed-QA explicitly discusses the **limitations of automated metrics** (Section 6.8) — BERTScore clusters too tightly, ROUGE-Lsum shows wider range, QuestEval best separates models. This candid discussion of metric limitations is a sign of intellectual honesty that judges value.

**Our equivalent:** We should add an explicit robustness discussion for our scoring formulas:
- What happens if the efficiency weight changes from 0.4 to 0.2 or 0.6? (We already have this from H4, but it's not written up as a metric robustness discussion.)
- Does changing the RL component weights substantially change rankings?
- The LOO analysis (ρ̄ = 0.9985) is already our strongest robustness argument — but it should be framed as "scoring is robust to individual task removal," which is the benchmark-level equivalent of metric robustness.

> **Action:** Add a "Scoring Robustness" paragraph to Section 5 (Discussion) in the writeup. Reference the efficiency ablation (H4) and LOO (Finding 5) together as converging evidence for robustness.

### F4: Effect of Prompting Strategy (Connects to a Published Debate)

AfriMed-QA studies the **effect of explanations on LLM accuracy** (Section 6.9 + Table 5) — finding that asking for explanations sometimes *hurts* accuracy due to post-processing extraction failures. They track post-processing error rates per model (Tables 12, 13).

**Our equivalent:** Our tasks use structured output requirements (e.g., models must respond with YES/NO/UNKNOWN, or with a specific label). Some models (Claude in particular) have known issues with format adherence.

**What to document:**
- Report which models had the highest format non-compliance rate on our tasks
- Mention whether we tried instruction-tuned prompts vs. base prompts (we likely used a single prompt design)
- This is a transparency/methodology point that judges will appreciate

> **Action:** Add a brief "Prompt Design" paragraph to Section 2 (Benchmark Construction) noting our prompt format, any format enforcement we applied, and whether models differed in format compliance rates.

### F5: Human Evaluation Framing (Even Without Human Eval, Discuss It)

AfriMed-QA runs a large-scale blind human evaluation (379 raters, 37,435 ratings) as a core contribution. We cannot do this in 48 hours. But judges will notice that human validation is a standard component of research-grade benchmarks.

**What to do (no new data needed):**
- Add a "Human Validation" paragraph to Section 2 explaining that our ground truth is *programmatically computed* (the rule function itself is the oracle), which is *stronger* than human annotation because it eliminates inter-annotator disagreement and annotation errors
- Explicitly state that all 134 tasks were manually verified by the authors for human solvability (6-phase pipeline, step 4)
- This reframes the absence of human eval as a *design choice* rather than a gap

> **Action:** Add a "Ground Truth Validation" paragraph to Section 2.3 that explicitly contrasts our programmatic ground truth with the annotation-based approach in prior work, positioning it as a methodological advantage.

### F6: Model Taxonomy Table (Enables Provider-Level Analysis)

AfriMed-QA evaluates 30 models and presents them in a structured taxonomy table (Table 3) with columns: Domain (General/Biomedical), Access (Open/Closed), Size, Type (Instruct/Pretrained/Finetuned). This makes provider- and architecture-level comparisons immediately obvious.

**Our equivalent:** Our leaderboard table has Tier but is missing:
- Provider column (Anthropic / OpenAI / Google / Open-source)
- Architecture type (Thinking / Standard Instruct)
- Open/Closed access designation

> **Action:** Add Provider, Access (Open/Closed), and Inference Mode (Thinking/Standard) columns to the leaderboard table. This immediately supports the provider-level analysis findings without requiring additional prose.

### F7: Performance vs. Specialty/Sub-Domain Breakdown (Our Equivalent: Per-Category Analysis)

AfriMed-QA breaks down LLM performance by specialty (Fig 4b), by country (Fig 5), and by question type (MCQ vs SAQ vs CQ). This multi-cut analysis is what moves a benchmark paper from "we tested models" to "we understand where models succeed and fail."

**Our equivalent (already partially done):** Our per-category breakdown (Table 2 in writeup) covers this. But we can push further:
- **Within-category difficulty gradient:** Show per-task score distributions within each category as a histogram or violin plot. This is Figure F7a.
- **Model-tier performance profile:** A grouped box plot showing score distributions by Frontier/Standard/Efficient tiers per category. This is Figure F7b.
- **Intra-provider variance:** For each provider, show the spread of scores across their models (e.g., the three Anthropic models: Opus, Sonnet, Haiku). High intra-provider variance is itself an insight.

> **Action:** Add to visualization list: (a) per-category task difficulty histogram, (b) tier-stratified box plots per category, (c) intra-provider score spread chart.

### F8: Temporal Performance Trend Discussion (Evidence of Progress)

AfriMed-QA explicitly discusses the GPT model series temporal trend (Section 6.4) — GPT-3.5 → GPT-4 → GPT-4o showing monotonic improvement on AfriMed-QA that is NOT attributable to memorization (since AfriMed-QA questions were not in training data). This temporal framing strengthens the contamination-resistance argument.

**Our equivalent:** We have multiple model generations/variants. We can make a similar argument:
- GPT-5.4 nano → GPT-5.4 mini → GPT-5.4 shows a clear within-family progression on most categories, confirming that our tasks are sensitive enough to detect generational improvements
- Claude Haiku → Sonnet → Opus shows a partially monotonic trend, with the Opus inversion on Concept Formation (0.259) being a notable deviation from the trend
- This temporal-analog argument strengthens the benchmark sensitivity claim

> **Action:** Add a "Benchmark Sensitivity" paragraph to Discussion (Section 5.1) showing that within-family model progressions produce detectable score differences on LearningBench, confirming our tasks are sensitive enough to discriminate model generations.

### F9: Ethical Considerations Section (Standard for Research Papers)

AfriMed-QA dedicates a full section (Section 8) to Ethical Considerations covering: data privacy, informed consent, DUA agreements, and licensing (CC-BY-NC-SA 4.0). For a medical dataset this is critical; for us it is less critical but still expected in ACL-style papers.

**What to add (minimal, 50-100 words):**
- License for the benchmark data (MIT or CC-BY preferred for open research)
- Note that all tasks are fully synthetic with no real personal data
- No third-party annotations — all ground truth is programmatically generated, eliminating annotation labor concerns
- Statement that tasks are designed for capability evaluation of LLM systems, not for deployment in high-stakes settings

> **Action:** Add a 1-paragraph "Ethics and Licensing" note to the writeup (can be at the end, before References). Brief but expected.

### F10: Limitations Section with Concrete Numbers (Not Just "Future Work")

AfriMed-QA's limitations (Section 9) are specific and honest: "over 60% of expert MCQ questions came from West Africa," "we plan to expand beyond English-only." These are concrete, specific, quantified limitations — not vague hedges.

**Compare to our current Limitations section (6.1):** We already have good limitations. But we can sharpen them:
- "Associative learning scores are only 1.8× above random" — keep this, it's a specific quantified limitation
- "Token and timing data coverage: 82 (task, model) pairs out of 134×14=1,876 possible" — add the fraction explicitly
- "Single-instance evaluation: each (task, model) pair evaluated exactly once with zero resampling" — add explicit statement
- **Add:** "Procedural and episodic learning sub-abilities are not covered, representing ~2 of 7 major cognitive learning categories in the Lake et al. (2017) framework"

> **Action:** Revise Section 6.1 to make all limitations quantified and specific. Replace any qualitative hedges with numbers.

### F11: Figure Gallery Plan (Visual Coverage Matches Paper Quality)

AfriMed-QA uses 6 main figures + 3 appendix figures. The main figures cover: methodology overview, model accuracy sorted bar chart, MCQ accuracy by specialty/dataset comparison, MCQ accuracy by country, human evaluation axis radar plots.

**Our existing visualization plan (Section 5.2):** Already lists 16 visualization types. But the AfriMed-QA figures suggest these specific additions that we may not have prioritized:

| Priority | Figure | Description | Maps to our work |
|---|---|---|---|
| HIGH | F1: Overview diagram | Methodology flow (like Fig 1 in AfriMed-QA) | Our 5-category cognitive framework + task pipeline |
| HIGH | F2: Leaderboard bar chart | Sorted by overall score, colored by tier | Already planned (#6) |
| HIGH | F3: Per-category breakdown heatmap | Model × category score heatmap | Already planned (#2) |
| HIGH | F4: Tier comparison box plots | Score distribution by tier per category | Already planned (#4, #15) |
| MED | F5: Model cognitive profiles | Radar chart per model or overlay | Already planned (#1) |
| MED | F6: Provider-level grouped bar | Mean scores by provider × category | Already planned (#13) |
| MED | F7: Scale inversion scatter | Score vs. model tier, with inversion annotations | Partial |
| LOW | F8: Benchmark comparison table | Feature table (see F1 above) | New |
| LOW | F9: Token usage comparison | Output tokens by model, colored by thinking/non-thinking | Partial (H14) |

> **Action:** Re-prioritize visualization queue. F1 (overview/methodology diagram) is missing from current plan and is the first figure in most top benchmark papers. Add it. Deprioritize low-value figures (bimodality histograms per-task) in favor of F1 and F8.

### F12: References — Add Benchmark Papers (Signals Field Knowledge)

AfriMed-QA cites 35+ references including domain benchmarks, evaluation frameworks, and methodology papers. Our writeup currently has 6 references. DeepMind judges will notice sparse references as a signal of insufficient engagement with the field.

**Minimum references to add:**
1. Hendrycks et al. (2020) MMLU — the dominant existing benchmark
2. Srivastava et al. (2022) BIG-Bench — largest general capability benchmark
3. Zheng et al. (2023) LMSYS Chatbot Arena — revealed model ranking instability
4. Guo et al. (2023) evaluating LLMs on reasoning — discusses brittle vs robust reasoning
5. Chollet (2019) ARC — the closest conceptual predecessor (measures fluid intelligence, not crystallized knowledge)
6. Lake et al. (2017) — already in our writeup, keep
7. Brown et al. (2020) GPT-3 — in-context learning seminal work; we build on ICL evaluation

**Framing:** Position LearningBench not just as a new benchmark but as a response to a specific gap in the ICL evaluation literature — models are evaluated for what they know via ICL, not for how well they learn via ICL.

> **Action:** Add 6-8 references to WRITEUP.md References section. Add 1 sentence connecting LearningBench to the ICL evaluation gap in the Problem Statement.

---

### Phase F Summary — Priority Queue

| Item | Impact | Effort | Do When |
|---|---|---|---|
| **F1: Benchmark comparison table** | HIGH — signals field knowledge | Low (30 min) | Before final writeup review |
| **F6: Model taxonomy columns in leaderboard** | HIGH — enables provider analysis | Low (15 min) | Before final writeup review |
| **F11: Add F1 overview figure** | HIGH — every top paper has this | Medium (1-2 hrs) | During visualization pass |
| **F5: Ground truth validation paragraph** | HIGH — addresses expected judge question | Low (20 min) | During writeup revision |
| **F12: Add 6-8 references** | MEDIUM — signals field knowledge | Low (30 min) | During writeup revision |
| **F8: Benchmark sensitivity paragraph** | MEDIUM — strengthens contamination-resistance | Low (20 min) | During writeup revision |
| **F3: Scoring robustness paragraph** | MEDIUM — reframes our LOO + efficiency ablation | Low (20 min) | During writeup revision |
| **F4: Prompt design paragraph** | MEDIUM — methodology transparency | Low (15 min) | During writeup revision |
| **F9: Ethics paragraph** | LOW — expected but quick | Low (10 min) | At end |
| **F10: Quantify limitations** | MEDIUM — intellectual honesty signal | Low (15 min) | During writeup revision |
| **F2: Sub-metric breakdown table** | LOW — strengthens technical appendix | Medium (1 hr) | In notebook |
| **F7: Additional visualization cuts** | MEDIUM — reinforces per-category story | Medium (1-2 hrs) | During visualization pass |

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **Time crunch** — 48 hours for everything | HIGH | HIGH | Prioritize ruthlessly. Must-haves first. Sub-agent parallelism. |
| **Bad tasks dilute quality** — tasks with no signal in final benchmark | MEDIUM | HIGH | WI-02 flags them. Remove aggressively. Quality > quantity. |
| **Efficiency data unavailable** — can't separate accuracy from efficiency for ablation | MEDIUM | MEDIUM | May need to recompute from task source code, or argue from formula alone. |
| **Missing Associative Learning aggregates** | LOW | MEDIUM | WI-01 will compute from raw data. |
| **Writeup too long / too dense** | MEDIUM | MEDIUM | Draft early, iterate. 1500 words is very tight — every sentence must earn its place. |
| **Judges want a static dataset** | LOW | MEDIUM | Section 3 of PROJECT_MASTER already explains why. Strengthen this argument. |
| **A competitor has a better benchmark** | UNKNOWN | HIGH | Focus on what we uniquely offer: multi-dimensional learning, efficiency scoring, interactive protocols, surprising insights. |
| **Reproducibility gaps** — notebook can't reproduce all claims | MEDIUM | HIGH | WI-16 must be runnable. All numbers from data, not from memory. |

---

## Appendix A: Key Numbers We Need for the Writeup

These specific numbers must be computed and ready:

- [ ] Total tasks after curation (currently 157, likely ~100-130 after cleaning) - cleaning logic will depend on each learning sub type
- [ ] Number of tasks per category (after curation)
- [ ] Average item discrimination index across tasks
- [ ] Cross-category correlation values (for H1)
- [ ] Specific scale inversions with exact numbers (for H2)
- [ ] UNKNOWN accuracy rate (for H5)
- [ ] How many tasks Gemini 3.1 Pro ranks #1 on (for H6)
- [ ] Number of bimodal tasks (for H7)
- [ ] Accuracy-only vs. composite ranking changes (for H4)
- [ ] Random baseline expected scores (for R10)
- [ ] Leave-one-out maximum ranking change (for R4)
- [ ] **Per-category entropy values (for H12)**
- [ ] **Provider-level mean scores per category (for H11)**
- [ ] **Number of unique domains covered (for H13)**
- [ ] **Number of RL tasks retained after pruning (from 50)**
- [ ] **Thinking model token overhead ratio vs non-thinking (H14)**
- [ ] **Cost-performance correlation coefficient (H15)**
- [ ] **Largest rank change from token-efficiency ranking (H18)**
- [ ] **Mean wall-clock time by tier (Bonus)**

## Appendix B: Information We Still Need to Gather

| What | Where to Get It | Why We Need It |
|------|----------------|----------------|
| Per-task accuracy separate from efficiency (CF, LL) | Recompute from task source code or leaderboard metadata | Efficiency ablation (H3, H4) |
| Per-question breakdown in associative learning | Task source code + model conversation logs | UNKNOWN analysis (H5) |
| Example request counts per model (CF, LL) | Task conversation logs or leaderboard metadata | Efficiency analysis |
| Model conversation logs / reasoning traces | Kaggle platform (if available) | Failure mode analysis (B5) |
| Competitor submissions | Browse Kaggle competition page | Competitive positioning |
| Exact model sizes / parameter counts | Public model documentation | Scale analysis precision |

## Appendix C: What Each Completed Analysis Gives Us

| Analysis | Feeds Into | Writeup Section |
|----------|-----------|-----------------|
| Score matrix (WI-01) | Everything | All sections |
| Task statistics (WI-02) | Task curation, quality argument | Dataset quality |
| Cross-category correlation (WI-04) | "Learning is not monolithic" insight | Key Insight #1 |
| Scale analysis (WI-05) | "Scale doesn't guarantee learning" insight | Key Insight #2 |
| Efficiency ablation (WI-06) | "Learning speed is invisible" insight | Key Insight #4 |
| UNKNOWN analysis (WI-07) | "Epistemic failure" insight | Key Insight #3 |
| Bimodality analysis (WI-09) | "Learning is brittle" insight | Key Insight #5 |
| Entropy analysis (WI-10B) | Task curation (remove low-entropy), category comparison | Dataset quality + design justification |
| Provider analysis (WI-10C) | "Provider-level behavioral patterns" insight | New insight for Results/Insights sections |
| Domain diversity (WI-10D) | "Benchmark covers diverse domains" argument | Dataset quality + methodology |
| Tier deep-dive (WI-10E) | "Why tier differences exist" narrative | Results section |
| **Kernel logs (WI-10F)** | **Timing, token cost, efficiency analysis** | **Technical details + novelty (learning speed axis)** |
| **Timing hypotheses (WI-10F)** | **H14–H18: thinking overhead, cost-effectiveness, provider verbosity** | **New "Learning Efficiency" section in writeup** |
| Task removal sensitivity (WI-11) | Robustness argument | Dataset quality |
| Random baseline (WI-12) | "Scores are not trivial" argument | Technical details |
