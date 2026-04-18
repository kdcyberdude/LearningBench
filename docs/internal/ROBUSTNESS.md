# LearningBench — Robustness & Analysis

> Everything here is reproducible. The primary data is `analysis/outputs/score_matrix_phase_d.csv` (138 tasks × 14 models). All statistical tests are in `analysis/scripts/30_hypothesis_tests.py` with BH-corrected p-values. Figures are in `figures/`.

---

## Contents

- [Benchmark Integrity](#1-benchmark-integrity)
- [Measurement Validity](#2-measurement-validity)
- [Learning Signal](#3-learning-signal)
- [Model Behavior](#4-model-behavior)
- [Anomalous Findings](#5-anomalous-findings)
- [Curation Robustness](#6-curation-robustness)

---

## 1. Benchmark Integrity

### Are scores above chance?

Every model, on every sub-ability, significantly beats random performance.

| Sub-ability | Random baseline | Best model | Signal ratio (best) |
|---|---|---|---|
| Associative | 0.333 | 0.93 | 2.8× |
| Concept | 0.020 | 0.73 | 36.2× |
| Language | 0.010 | 0.76 | 75.6× |
| Observational | 0.010 | 0.84 | 83.8× |
| RL | 0.020 | 0.83 | 31.2× |

- **Script:** `12_random_baseline.py` → `outputs/random_baseline.csv`
- Associative's smaller ratio (2.8×) is structural — 3-class format (YES/NO/UNKNOWN) gives random a 33% floor, not a benchmark weakness.

---

### Is the ground truth reliable?

The same rule function that generates training examples grades model responses. No static answer keys exist.

- **Script:** `13_ground_truth_spotcheck.py` → `outputs/ground_truth_spotcheck.csv`
- 15 tasks spot-checked: all 15 confirmed computed ground truth, no hardcoded keys, no scoring bugs found.
- RL tasks use seeded random generation — correct by construction at runtime.

---

### Do tasks actually discriminate between models?

| Classification | Count | % |
|---|---|---|
| Excellent (r ≥ 0.50) | 93 | 59% |
| Good (0.30–0.50) | 19 | 12% |
| Fair (0.10–0.30) | 27 | 17% |
| Poor / no signal | 14 | 9% (removed in Phase D) |

- **Script:** `04_discriminatory_power.py` → `outputs/discrimination_report.csv`
- Mean discrimination by sub-ability: Language (r=0.593), Associative (r=0.573), Observational (r=0.568), Concept (r=0.527), RL (r=0.341).
- RL's lower mean discrimination is structural — 34 diverse tasks vs. a unified cognitive paradigm.

---

## 2. Measurement Validity

### Does removing any single task change who wins?

No.

- **Script:** `11_task_removal_sensitivity.py` → `outputs/loo_global.csv`, `benchmark_stability_summary.csv`
- Leave-one-out across all 135 tasks: mean Spearman with full ranking = **0.9985**, min = 0.9912.
- Maximum rank change from removing any single task: **1 position**.
- Removing all 19 Phase-D-removed tasks (157 → 138): top-11 ranking unchanged. Only positions 12/13 swap (GPT-5.4 mini ↔ Qwen Instruct).

---

### Do the six sub-abilities measure distinct things, or are they just one "g-factor"?

Distinct, but correlated.

- **Script:** `05_cross_category.py` → `outputs/cross_category_correlations.csv`
- PCA on category-level scores: PC1 explains **81%** of variance — a strong general learning factor exists, but not unity.
- Pairwise cross-category Spearman ρ ranges from **0.29 to 0.93** (mean 0.65).
- Weakest pair: Concept × Observational (ρ = 0.55) — abstract category formation and structural inference from demonstrations are genuinely distinct.
- Strongest pair: Concept × RL (ρ = 0.89) — both require hypothesis testing under uncertainty.
- Model profile inversions confirm non-monolithic structure: Gemma 4 26B ranks #4 on RL but #14 on Language; GPT-5.4 ranks #4 on Language but #9 on Concept.

---

### Does the efficiency component in the score formula change the leaderboard?

No. It amplifies differentiation at the top without changing relative order.

- **Script:** `07_efficiency_ablation.py` → `outputs/efficiency_ablation.csv`
- Maximum rank change when efficiency scoring removed: **0 positions**.
- Efficiency scoring captures a real signal (top models are accurate *and* efficient — they used 37–47% of available examples), but the ranking is already determined by accuracy.

---

### Does each sub-ability's score distribution carry real information (entropy)?

Yes, but unevenly across sub-abilities.

| Sub-ability | Mean entropy (bits) | % high-entropy tasks |
|---|---|---|
| Language | 2.211 | 65% |
| Concept | 1.927 | 42% |
| Associative | 1.893 | 15% |
| Observational | 1.555 | 12% |
| RL | 1.067 | 0% |

- **Script:** `08_entropy_analysis.py` → `outputs/entropy_report.csv`, `category_entropy.csv`
- Language Learning's interactive protocol (model asks for examples until confident, then tested on held-out words) produces the richest distributions.
- RL's near-zero high-entropy rate is consistent with its lower mean discrimination — 16 tasks removed in Phase D had near-zero entropy.
- Entropy ↔ discrimination correlation: r = 0.165 (p = 0.04) — weakly significant; entropy is a complementary quality signal.

---

## 3. Learning Signal

### Does the number of examples a model requests predict how well it learns?

Yes, strongly and negatively.

- **Script:** `30_hypothesis_tests.py` (H1) → `outputs/hypothesis_tests.json`
- Spearman ρ(examples requested, score) = **−0.516**, 95% CI [−0.591, −0.436], p = 8.5e-29, n = 403 (concept + language runs).
- Concept: ρ = −0.456 (p < 1e-9). Language: ρ = −0.568 (p < 1e-22).
- Evidence appetite spans **1.8 to 11.9 examples** per run across models — a 6.5× spread.

| Profile | Models | Avg examples used | Mean score |
|---|---|---|---|
| Well-calibrated | Qwen Thinking, Gemini Flash-Lite, GLM-5, Gemini Pro | 1.8–3.6 | 0.67–0.78 |
| Underconfident | Claude Haiku, Claude Opus, Gemma, GPT-5.4 nano | 8–12 | 0.29–0.35 |

- **Figure:** `figures/fig_meta_calibration.png`

---

### Does token spend predict failure in reinforcement learning?

Yes, very strongly.

- **Script:** `30_hypothesis_tests.py` (H2) → `outputs/hypothesis_tests.json`
- Spearman ρ(total tokens, score) = **−0.533**, 95% CI [−0.599, −0.462], p = 1.6e-30, n = 397 RL runs.
- Solved runs: **41K tokens** avg. Failed runs: **177K tokens** avg — **4.3× gap** (Cliff's δ = −0.60).
- 43 runs show ≥10 consecutive identical actions — when the first hypothesis is wrong, many models cannot revise at all.
- **Figure:** `figures/fig_tokens_vs_score.png`

Token spend is observable in real-time. A production monitor can flag likely failures before the wrong answer returns.

---

### Does procedural learning trajectory carry signal beyond the final score?

Yes — 99% of the trajectory signal is orthogonal to the final score.

- **Script:** `30_hypothesis_tests.py` (H5) → `analysis/procedural_trajectory_ablation/trajectories.csv`
- Spearman ρ(OLS slope, final asymptote) = **−0.017** (p = 0.86), n = 112 (model, task) pairs.
- R² = 0.011 — the slope explains only 1.1% of final-score variance.
- Concrete example: two runs both ending at score ≈ 0.5 have slopes of +0.18 and −0.30 respectively — one model is still improving, the other declining.
- **Figure:** `figures/fig_trajectory_orthogonal.png`

No existing benchmark reports this. Final score and learning trajectory are independent dimensions.

---

## 4. Model Behavior

### Does reasoning (thinking tokens) help with learning?

Yes, substantially — especially for induction-heavy tasks.

- **Script:** `30_hypothesis_tests.py` (H4); `10_bimodality_and_dominance.py` → `outputs/thinking_comparison.csv`
- Controlled A/B: Qwen 3 Next 80B Thinking vs. Qwen 3 Next 80B Instruct — same weights, only reasoning trace toggled.
- On **concept formation**: Thinking mean = 0.563 vs. Instruct mean = 0.186. Uplift = **+203%**. Thinking wins 14/18 tasks (Wilcoxon p = 0.0022).
- Across 132 matched tasks (all sub-abilities): Thinking wins 87, Instruct wins 19, ties 26.
- Largest gains: Observational (+0.277), Concept (+0.350), RL (+0.206).
- **Figure:** `figures/fig_thinking_vs_instruct.png`

---

### Does scale reliably predict learning performance?

At the tier-aggregate level yes. At the model level, there are 22 exceptions.

- **Script:** `06_scaling_analysis.py` → `outputs/tier_stats.csv`, `tier_inversions.csv`
- Frontier > mid > small in all 5 sub-abilities at the tier average.
- But 22 model-level inversions exist with gap > 0.05:
  - Qwen Thinking (mid) beats GPT-5.4 (frontier) on Observational by **+0.289**
  - Qwen Thinking (mid) beats GPT-5.4 (frontier) on Concept by **+0.274**
  - Gemini 2.5 Flash (mid) beats GPT-5.4 (frontier) on Concept by **+0.250**
- The inversions cluster on Concept Formation and Observational Learning — both require systematic rule induction, not broad knowledge. Models with explicit reasoning modes (Qwen Thinking, Gemini) dominate these despite lower tier labels.

---

### Do provider families show systematic capability profiles?

Yes — rule induction is a clear axis of divergence.

- **Script:** `09_provider_analysis.py` → `outputs/provider_pivot.csv`
- **H3** (formal test): Google + Open-source mean = 0.591 on rule-induction tasks; Anthropic + OpenAI mean = 0.420. Gap = **+40.8%**. Mann-Whitney p = 0.029, Cliff's δ = 0.71 (large effect).

| Provider | Relative strength | Relative weakness |
|---|---|---|
| Google | Concept (+0.13 vs mean) | — |
| Open-source | Observational (+0.12 vs mean) | — |
| OpenAI | — | Observational (−0.13 vs mean) |
| Anthropic | — | Concept (−0.07 vs mean) |

- Due to small N (3–4 models per provider), individual provider Kruskal-Wallis tests are underpowered. The H3 result uses the combined Google+OSS vs. Anthropic+OpenAI grouping.
- **Figure:** `figures/fig_radar_profiles.png`

---

### Can models recognize when evidence is genuinely insufficient (epistemic uncertainty)?

Yes — and there is a provider-level split.

- **Script:** `14_epistemic_analysis.py` → `outputs/epistemic_analysis.csv`, `unknown_task_scores.csv`
- Models score **higher** on tasks where the correct answer is UNKNOWN (mean 0.695) than on definitive-answer tasks (mean 0.582). Paired t-test: t = −3.255, p = 0.006.
- `blocking_effect` — a pure epistemic trap where every correct answer is UNKNOWN:

| Outcome | Models |
|---|---|
| Perfect (1.0) | Claude Opus, Claude Sonnet, GPT-5.4 |
| Strong (0.875) | Gemma 4 26B, Qwen Instruct, GPT-5.4 nano |
| Partial (0.50) | **Gemini 3.1 Pro** ← overall top performer |

- Gemini 3.1 Pro's weakness here is not random — it over-commits to pattern matches even when the causal structure is provably ambiguous.

---

## 5. Anomalous Findings

These tasks were initially flagged for removal. After code inspection, the inversions were confirmed as genuine behavioral signals — they were **retained**.

### Can strong priors block evidence-based learning?

Yes — for the most capable frontier models.

- **Task:** `semantic_override` (concept)
- The task presents a rule that overrides a semantically obvious interpretation (e.g., a structural pattern disguised by word choice).
- Gemini 3.1 Pro = **0.00**, GPT-5.4 = **0.00**. Gemini 2.5 Flash = 0.95, Gemma 4 26B = 0.90.
- std = 0.32, max = 0.95 — not a statistical artifact.
- Interpretation: RLHF/instruction-tuning at scale hardens semantic defaults. The stronger the model's priors, the harder it is to override them with in-context evidence.

---

### Is capability monotonic within a model family?

No.

- **Tasks:** `manhattan_point`, `grid_octile` (RL)
- Gemini 3.1 Flash-Lite = **1.00**, Gemini 2.5 Flash = 0.15, Gemini 3.1 Pro = **0.00** on `manhattan_point`.
- Claude Sonnet = **1.00**, Claude Opus = **0.00** on `grid_octile`.
- Both inversions are within-family (smaller sibling outperforms flagship) and reproducible across tasks.
- Points to fine-tuning choices at larger scale suppressing specific RL capabilities.

---

### Do frontier OpenAI models have RL-specific blind spots?

Yes, on sequential inference tasks.

- **Tasks:** `minesweeper_1d`, `verbal_bandit` (RL)
- GPT-5.4 ≈ 0.00 on both; Gemini/open-source models score 0.5–1.0.
- This is not a general intelligence deficit — GPT-5.4 ranks top-5 overall. It is specific to multi-step, feedback-driven inference.
- Both tasks were retained because the inversion is consistent (not noise) and reveals a meaningful capability gap invisible to single-turn benchmarks.

---

### Do any model families have narrow capability spikes?

Yes — Qwen on priority ordering inference.

- **Task:** `hidden_priority_order` (observational)
- Qwen 3 Next 80B Thinking = **1.00**, Qwen 3 Next 80B Instruct = **1.00**. All 12 other models: ≤ 0.75, with 11 scoring 0.00.
- Both Qwen variants solve it; no other provider does. Not explained by overall rank (Qwen Thinking is #3, but #14 models also score 0).
- Points to a systematic advantage in Alibaba's training pipeline for hidden priority/ordering inference.

---

## 6. Curation Robustness

### How were the 19 removed tasks identified?

Four-phase systematic analysis, not ad-hoc.

- **Phase A:** Data extraction and validation.
- **Phase B:** Item discrimination (IRT-style), entropy, scaling analysis, provider analysis, bimodal classification — flagged 26 tasks.
- **Phase C:** Leave-one-out robustness, random baselines, ground-truth spot-check, epistemic analysis — confirmed flags.
- **Phase D:** Full Python source code inspection of all 26 flagged tasks — individual decision per task.
- **Scripts:** `04` through `15` for Phases B–C; see `analysis/PHASE_D_INSIGHTS.md` for code inspection results.

---

### What were the removal reasons?

| Reason | Count | Examples |
|---|---|---|
| Implementation flaw (undisclosed mechanics, mid-episode drift, mislabeling) | 4 | `hangman_lite`, `linear_equation`, `lights_out_2x2`, `grid_nav` |
| Budget-infeasible (task requires more turns than allocated to demonstrate learning) | 6 | `euler_totient`, `parity_groups`, `grid_seven` |
| Prior knowledge contamination (well-known format bypasses in-context learning) | 2 | `hot_cold`, `mastermind_aggregate` |
| Noise-induced discrimination collapse | 3 | `cyclic_distance`, `minesweeper_1d`, `parity_groups` |
| Extreme bimodal / no partial-credit gradient | 4 | `interval_contains`, `hanoi_three` |

See `analysis/PHASE_D_INSIGHTS.md` and `analysis/CURATION_DECISIONS.md` for the full per-task decision table.

---

### What was the leaderboard impact of removing 19 tasks?

| Effect | Value |
|---|---|
| Top-11 ranking change | None |
| Positions 12/13 | GPT-5.4 mini ↔ Qwen Instruct swap |
| RL category mean | +0.103 (noise removed) |
| Tier gaps | Slightly widened (better discrimination) |
| Scores direction | All models increased uniformly |

- **Script:** Phase D outputs → `outputs/phase_d_leaderboard_impact.csv`
- Uniform score increase confirms removed tasks were noise (suppressing all models equally), not signal.

---

## Reproduce

```bash
# All formal hypothesis tests (H1–H8)
python analysis/scripts/30_hypothesis_tests.py
# → analysis/outputs/hypothesis_tests.json
# → analysis/outputs/hypothesis_tests.md

# Item discrimination, entropy, scaling, provider analysis
python analysis/scripts/04_discriminatory_power.py
python analysis/scripts/08_entropy_analysis.py
python analysis/scripts/06_scaling_analysis.py
python analysis/scripts/09_provider_analysis.py

# LOO stability + random baselines + ground truth
python analysis/scripts/11_task_removal_sensitivity.py
python analysis/scripts/12_random_baseline.py
python analysis/scripts/13_ground_truth_spotcheck.py

# Epistemic analysis + phase C synthesis
python analysis/scripts/14_epistemic_analysis.py
python analysis/scripts/15_phase_c_summary.py

# Regenerate all figures
python analysis/scripts/make_writeup_figures.py
```

All pre-computed outputs are committed to `analysis/outputs/`. No Kaggle API required for any of the above.
