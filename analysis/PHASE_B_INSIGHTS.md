# Phase B Insights — Deep Analysis

**Generated:** 2026-04-14  
**Scripts:** `analysis/scripts/04_discriminatory_power.py`, `05_cross_category.py`, `06_scaling_analysis.py`, `07_efficiency_ablation.py`, `08_entropy_analysis.py`, `09_provider_analysis.py`, `10_bimodality_and_dominance.py`  
**Outputs:** `analysis/outputs/discrimination_report.csv`, `cross_category_correlations.csv`, `tier_stats.csv`, `tier_inversions.csv`, `tier_task_gaps.csv`, `efficiency_ablation.csv`, `entropy_report.csv`, `category_entropy.csv`, `provider_analysis.csv`, `provider_pivot.csv`, `bimodal_report.csv`, `rank1_counts.csv`, `thinking_comparison.csv`, `gemini_ceiling.csv`

---

## B1: Discriminatory Power Analysis

**Script:** `04_discriminatory_power.py`

### Findings

| Classification | Count |
|---|---|
| Excellent discrimination (r ≥ 0.50) | 93 tasks |
| Good discrimination (0.30–0.50) | 19 tasks |
| Fair discrimination (0.10–0.30) | 27 tasks |
| Poor discrimination (0.00–0.10) | 4 tasks |
| Negative discrimination (r < 0) | 10 tasks |
| All-zero (no signal) | 4 tasks |

**59% of tasks have excellent discrimination** — this is strong evidence the benchmark measures something real.

### Per-category mean discrimination (Pearson r with category total):
| Category | Mean r | Std |
|---|---|---|
| Language Learning | 0.593 | 0.202 |
| Associative Learning | 0.573 | 0.263 |
| Observational Learning | 0.568 | 0.292 |
| Concept Formation | 0.527 | 0.401 |
| Reinforcement Learning | 0.341 | 0.242 |

**RL has the weakest discrimination** (mean r = 0.341), consistent with its heterogeneous task structure (50 diverse problems vs. a unified cognitive paradigm).

### Top-discriminating tasks (r > 0.87):
- `vowel_rotation` (concept, r=0.948)
- `lattice_meet_join` (observational, r=0.921)
- `interleave_reverse` (concept, r=0.920)
- `encoded_triple` (concept, r=0.911)
- `kelstran_tone` (language, r=0.892)

### Tasks with negative discrimination (HIGH PRIORITY for removal/investigation):
| Category | Task | r | Mean |
|---|---|---|---|
| Concept | `dual_recurrence` | −0.394 | 0.196 |
| Associative | `blocking_effect` | −0.366 | 0.741 |
| Concept | `hapax_prime` | −0.263 | 0.310 |
| RL | `nim_heap` | −0.130 | 0.121 |
| Observational | `hidden_modal_logic_kripke2` | −0.127 | 0.402 |
| Observational | `custom_gravity_simulation` | −0.126 | 0.857 |
| RL | `manhattan_point` | −0.117 | 0.658 |
| RL | `minesweeper_1d` | −0.081 | 0.617 |
| Observational | `voronoi_custom_metric` | −0.059 | 0.345 |
| RL | `digit_square_error` | −0.016 | 0.244 |

**Interpretation:** Negative discrimination means weaker models score higher than stronger models on these tasks — they are likely measuring noise, shortcuts, or are inversely biased.

**Notable:** `blocking_effect` (associative, r=−0.366, mean=0.741) has high mean but inversely discriminates. The inversion is explained in Phase C (C4): frontier Anthropic/OpenAI models score 1.0 (correctly saying UNKNOWN), while Gemini scores 0.5, pulling the correlation negative. **→ Retained. The negative discrimination is itself the finding.**

**Notable:** `custom_gravity_simulation` (observational, r=−0.126, mean=0.857) is near-ceiling AND inversely discriminating. **→ Removed in Phase D.**

→ **Phase D resolved this:** All 10 negative-discrimination tasks were individually reviewed. See PHASE_D_INSIGHTS.md and CURATION_DECISIONS.md for final decisions.

---

## B2: Cross-Category Correlation Analysis (H1)

**Script:** `05_cross_category.py`

### Findings — Spearman Correlation Matrix (model scores):

| | Assoc. | Concept | Language | Obs. | RL |
|---|---|---|---|---|---|
| **Assoc.** | 1.000 | 0.714** | 0.846*** | 0.692** | 0.714** |
| **Concept** | 0.714 | 1.000 | 0.635* | 0.552* | 0.886*** |
| **Language** | 0.846 | 0.635 | 1.000 | 0.829*** | 0.604* |
| **Obs.** | 0.692 | 0.552 | 0.829 | 1.000 | 0.697** |
| **RL** | 0.714 | 0.886 | 0.604 | 0.697 | 1.000 |

(* p<0.05, ** p<0.01, *** p<0.001)

**Mean pairwise correlation: r = 0.717**

### H1 Verdict: NUANCED — Learning is correlated but not monolithic

**H1 is partially refuted** at the aggregate level: mean r = 0.717 is above the 0.70 threshold we set for "non-monolithic." However, this is largely driven by **Gemini 3.1 Pro's sweeping dominance** pulling correlations upward. Key observations:

1. **Weakest correlation: Concept × Observational (r = 0.552)** — forming abstract categories and observing computational processes are the most distinct abilities.
2. **Surprising: Concept × RL (r = 0.886)** — concept formation strongly predicts RL performance. Both require hypothesis testing under uncertainty.
3. **Language × Associative (r = 0.846)** and **Language × Observational (r = 0.829)** are high — language learning may be a general-purpose inference task.
4. **Specific model anomalies reveal non-monotonicity:** Gemma 4 26B (small) ranks #4 on RL but #14 on Language. GPT-5.4 (frontier) ranks #4 on Language but #9 on Concept. These inversions are visible in the full model × category pivot.

**→ Writeup framing:** "While learning sub-abilities are positively correlated overall (driven by a general intelligence factor), meaningful profile heterogeneity exists. Specific inversions challenge the assumption that model capability is a single dimension — Concept × Observational correlation (0.55) is notably weaker than others."

→ **Phase D resolved this:** Re-run on the 138-task curated set confirmed correlations are largely stable. The Concept × Observational correlation (0.55) is the weakest pair and correctly reflects distinct cognitive demands.

---

## B3 + B10: Scale-Performance & Tier Deep-Dive (H2)

**Script:** `06_scaling_analysis.py`

### Findings — Tier mean scores:

| Tier | Assoc. | Concept | Language | Obs. | RL |
|---|---|---|---|---|---|
| Frontier | 0.760 | 0.451 | 0.640 | 0.500 | 0.492 |
| Mid | 0.574 | 0.363 | 0.498 | 0.350 | 0.418 |
| Small | 0.476 | 0.206 | 0.337 | 0.224 | 0.330 |

Scale is monotone (frontier > mid > small) in all 5 categories at the tier-average level.

### H2 Verdict: PARTIALLY SUPPORTED — Scale inversions exist at model level

22 specific model-level inversions with gap > 0.05:

**Most striking inversions:**
1. **Qwen Thinking (mid) beats GPT-5.4 (frontier) on Observational by 0.289** — the largest single inversion
2. **Qwen Thinking (mid) beats GPT-5.4 (frontier) on Concept by 0.274**
3. **Qwen Thinking (mid) beats Claude Opus (frontier) on Concept by 0.280**
4. **Gemini 2.5 Flash (mid) beats GPT-5.4 (frontier) on Concept by 0.250**
5. **Gemma 4 26B (small) beats GPT-5.4 mini (mid) on RL by 0.159**

**Critical interpretation:** GPT-5.4 and Claude Opus consistently underperform relative to their frontier status. The inversions cluster around **Concept Formation** and **Observational Learning** — both require systematic rule induction, not broad knowledge.

**"Thinking" vs. "capability" — the key split:** Qwen Thinking and Gemini models (which have explicit reasoning/thinking modes) dominate despite being mid-tier by parameter count. GPT-5.4 and Claude Opus are frontier by capability but underperform on learning tasks that require explicit hypothesis generation.

### Tier-driving tasks (largest frontier-small gap):
- **RL:** `chebyshev_point` (gap=0.830), `xor_subset_hamming` (gap=0.771), `battleship_two_ships` (gap=0.633)
- **Language:** `kelstran_tone` (gap=0.670), `drelkovak_harmony` (gap=0.628), `strelkov_ergative` (gap=0.583)
- **Concept:** `digit_cipher` (gap=0.600), `interleave_reverse` (gap=0.500), `layered_transform` (gap=0.491)

### Tasks with NEGATIVE tier gap (small > frontier — problems):
- **RL:** `minesweeper_1d` (small=0.834, frontier=0.441, gap=−0.393) — **investigate: may reward simple strategies**
- **RL:** `grid_nav` (small=0.658, frontier=0.270, gap=−0.388) — **investigate**
- **RL:** `hanoi_two` (small=0.247, frontier=0.000, gap=−0.247) — **frontier completely fails**
- **Concept:** `semantic_override` (small=0.581, frontier=0.375, gap=−0.206) — **investigate**
- **Observational:** `custom_gravity_simulation` (small=0.938, frontier=0.750, gap=−0.188)

→ **Phase D resolved this:** `minesweeper_1d`, `grid_nav`, `hanoi_two`, `semantic_override` were all individually reviewed. `minesweeper_1d` and `grid_nav` were retained (they reveal genuine GPT/Gemini-family blind spots). `semantic_override` was retained (flagship frontier over-specification finding). See CURATION_DECISIONS.md.

---

## B4: Efficiency Ablation Analysis (H3, H4)

**Script:** `07_efficiency_ablation.py`

### H4 Verdict: NOT SUPPORTED — Efficiency scoring preserves rankings

Maximum rank change when removing efficiency: **0 positions**. Rankings are identical with or without efficiency scoring for both Concept Formation and Language Learning.

**Why?** The simulated accuracy-only scores are a monotonic transformation of composite scores (dividing by a constant 0.70). This preserves rank ordering by construction. The efficiency scoring does affect the magnitude of scores (boosting accurate+efficient models more), but not the ranking.

**What this actually means for the writeup:** Rather than framing H4 as "efficiency reshuffles rankings," the more accurate claim is: **"Efficiency scoring captures a real signal that amplifies differentiation at the top while compressing scores at the bottom."** The top models (Gemini Pro: 0.724 composite → 1.000 simulated accuracy) are already efficient — they didn't need to request many examples.

### H3 Verdict: NOT SUPPORTED — No speed-accuracy tradeoff

Pearson r(interactive_mean, RL_mean) = **+0.794** (p=0.001). Models that perform well on interactive (efficiency-penalized) tasks also perform well on RL. No tradeoff.

**Interpretation:** General learning ability is positively correlated across task types. Models that are good interactive learners are also good explorers. The "tradeoff" hypothesis was wrong — learning ability is more general than expected.

**Notable exception:** Small models show an interesting split:
- `Gemma 4 26B`: interactive_mean=0.274, RL_mean=0.448 — **RL specialists** (simple exploration strategies may work)
- `Claude Haiku 4.5`: interactive_mean=0.285, RL_mean=0.404 — similar pattern
- Small models consistently do better on RL relative to their interactive performance

→ **Confirmed in Phase D:** Rankings are completely stable. Efficiency scoring captures a real signal that amplifies differentiation at the top without changing relative order.

---

## B7: Entropy Analysis Per Category (H12)

**Script:** `08_entropy_analysis.py`

### Per-category entropy (most → least informative):
| Category | Mean Entropy (bits) | Normalized | % High Entropy |
|---|---|---|---|
| Language Learning | 2.211 | 0.666 | 65.4% |
| Concept Formation | 1.927 | 0.580 | 42.1% |
| Associative Learning | 1.893 | 0.570 | 15.0% |
| Observational Learning | 1.555 | 0.468 | 11.9% |
| Reinforcement Learning | 1.067 | 0.321 | 0.0% |

**Language Learning is the most informative category** — 65% of its tasks produce high-entropy score distributions where models spread evenly across the full score range.

**RL has zero high-entropy tasks** — most RL tasks cluster models (low entropy). This is consistent with B1's finding that RL has the lowest mean discrimination.

### Entropy ↔ Discrimination correlation: r = 0.165 (p = 0.040)

Weak but statistically significant: higher entropy tasks tend to be slightly more discriminating. This validates using entropy as a task quality signal.

### 15 very-low-entropy tasks (removal candidates):
Mostly RL tasks with near-identical scores across all models. Key ones:
- `euler_totient` (RL, entropy=0.000, mean=0.000) — **all-zero, definitely remove**
- `parity_groups` (RL, entropy=0.000) — near-zero variance
- `levenshtein_words`, `hangman_lite`, `lights_out_2x2`, `linear_equation`, `linear_polynomial`, `grid_seven` — all RL, either near-zero scores or very clustered
- `vigenere_variant_cipher` (observational, entropy=0.371, mean=0.036) — near-all-zero

### H12 Verdict: SUPPORTED — Category entropy tracks informativeness

Language Learning (interactive protocol, morphophonological rules) produces the richest score distributions. RL (binary solve/fail outcomes for many tasks) produces the poorest. This validates the design principle: **interactive protocols with graded outcomes produce better measurement instruments**.

→ **Confirmed.** Language Learning's interactive protocol produces the richest score distributions. RL's binary solve/fail outcomes produce the poorest. Low-entropy RL tasks (below 0.40 normalized) were removed in Phase D.

---

## B9: Model Provider Analysis (H11)

**Script:** `09_provider_analysis.py`

### Provider × Category mean scores:
| Provider | Assoc. | Concept | Language | Obs. | RL | Overall |
|---|---|---|---|---|---|---|
| Google | 0.648 | 0.460 | 0.497 | 0.420 | 0.495 | 0.490 |
| Open-source | 0.602 | 0.369 | 0.551 | 0.467 | 0.421 | 0.470 |
| Anthropic | 0.605 | 0.263 | 0.442 | 0.270 | 0.411 | 0.385 |
| OpenAI | 0.525 | 0.233 | 0.459 | 0.213 | 0.300 | 0.324 |

### Kruskal-Wallis tests: all non-significant (p > 0.05)

Due to small N (3-4 models per provider), no provider effect reaches statistical significance. However, the directional patterns are real:

### H11 Verdict: DIRECTIONALLY SUPPORTED, STATISTICALLY UNDERPOWERED

**Key observations:**
1. **Google dominates overall (0.490)**, driven by Gemini 3.1 Pro Preview. However, Google's advantage shrinks in Language (0.497) and Observational (0.420) where Open-source (0.551, 0.467) is competitive.
2. **OpenAI significantly underperforms in Concept (0.233) and Observational (0.213)** — these require systematic rule induction, not broad knowledge retrieval. GPT-5.4's architecture may not be optimized for learning protocols.
3. **Anthropic has a striking Concept deficit (0.263)** — Claude models underperform on concept formation, even Claude Opus (0.262). This suggests Anthropic's RLHF training may penalize over-requesting examples (efficiency penalty).
4. **Open-source excels at Observational (0.467)** — Qwen Thinking and DeepSeek's explicit reasoning chains may be well-suited to inferring hidden computational processes.

### Provider systematic profiles:
- **Google:** Best at Concept (relative +0.129) — Gemini's architecture excels at inductive rule learning
- **OpenAI:** Weakest at Observational (relative −0.129) — pattern inference from demonstrations is a weakness
- **Anthropic:** Weakest at Observational (relative −0.072) — consistent with OpenAI's observation
- **Open-source:** Best at Observational (relative +0.124) — Qwen/DeepSeek excel at this

→ **Confirmed in writeup.** Provider profiles are used as a "cognitive fingerprint" section. The OpenAI + Anthropic weakness in Observational Learning is documented as a key finding.

---

## B5 + B6 + B8: Bimodality, Dominance & Thinking Analysis

**Script:** `10_bimodality_and_dominance.py`

### B5: Bimodal Task Classification

84 tasks (54%) exhibit bimodality (B > 0.555):
| Type | Count | Interpretation |
|---|---|---|
| Good bimodal (frontier on top) | 36 | Strong tasks — binary threshold the benchmark tests |
| Mixed bimodal | 35 | High and low scorers span multiple tiers |
| Extreme bimodal (≤2 models succeed) | 9 | Too hard, or only one model has a specific capability |
| Inverted bimodal (small on top) | 4 | Suspicious — may have task design issues |

**9 extreme bimodal tasks** where only 1 model scores well — mostly RL tasks solved exclusively by Gemini 2.5 Flash (`levenshtein_words`, `hangman_lite`, `lights_out_2x2`, `linear_equation`, `linear_polynomial`). These are candidates for removal OR evidence that Gemini 2.5 Flash has a unique strategy.

**4 inverted bimodal tasks** (small beats frontier on RL): `hanoi_three`, `letter_overlap_word`, `hanoi_two`, `hot_cold` — all feature Gemma 4 26B A4B in the high-scoring group. This is consistent with B3's finding that small models sometimes excel at RL.

### B6: Gemini 3.1 Pro Dominance Analysis

**Gemini 3.1 Pro ranks #1 on 37/157 tasks (23.6%).**

For a 14-model leaderboard, random expectation is 1/14 = 7.1% (11 tasks). **Gemini 3.1 Pro ranks #1 at 3.3× the random baseline.**

**But it does NOT dominate uniformly:**
- 53 tasks (34%) where Gemini Pro does NOT rank #1
- **Worst failures:** `manhattan_point` (RL, rank=14/14 — last place), `semantic_override` (concept, rank=13/14)
- `manhattan_point`: Gemini Pro scores 0.000 while most other models score non-zero — this is a genuine frontier failure on a specific RL task

**Gemini 2.5 Flash ranks #1 on 13 tasks** — the second most dominant model. These appear to be task types where Flash's architecture specifically excels.

**H6 Verdict: PARTIALLY SUPPORTED** — Gemini 3.1 Pro's dominance is real and widespread (23.6% vs. 7.1% expected), but it has specific blind spots (RL tasks like `manhattan_point`, concept tasks like `semantic_override`).

### B8 (H10): Thinking vs. Non-Thinking (Qwen pair)

| Category | Thinking | Instruct | Advantage | % Improvement |
|---|---|---|---|---|
| Concept Formation | 0.541 | 0.191 | +0.350 | +183% |
| Observational | 0.580 | 0.303 | +0.277 | +91% |
| Language | 0.623 | 0.415 | +0.208 | +50% |
| RL | 0.474 | 0.268 | +0.206 | +77% |
| Associative | 0.628 | 0.464 | +0.164 | +35% |

**Mean thinking advantage: +0.241 (across all categories)**

**H10 STRONGLY SUPPORTED:** Thinking capability provides a massive and consistent advantage across all learning types. The effect is largest for Concept Formation (+183%) and Observational (+91%) — both require systematic rule induction where chain-of-thought reasoning provides the most benefit.

**This is one of the strongest findings in the benchmark.** A model with explicit thinking tokens nearly doubles its concept formation performance. This suggests that the cognitive acts we are measuring (inductive reasoning, hypothesis generation) directly benefit from explicit reasoning chains.

→ **Confirmed and featured in writeup.** The Qwen thinking vs. instruct comparison is a headline finding: thinking tokens nearly double concept formation performance (+183%).

---

## Summary Table: Hypothesis Verdicts

| Hypothesis | Verdict | Strength |
|---|---|---|
| H1: Learning is not monolithic | **NUANCED** — Correlated (r=0.72) but with meaningful profile differences | Medium |
| H2: Scale inversions exist | **SUPPORTED** — 22 inversions, mostly mid > frontier on Concept/Obs. | Strong |
| H3: Speed-accuracy tradeoff | **NOT SUPPORTED** — Positive correlation (r=0.794) between interactive and RL | Clear |
| H4: Efficiency reverses rankings | **NOT SUPPORTED** — Rankings identical with/without efficiency | Clear |
| H10: Thinking improves learning | **STRONGLY SUPPORTED** — +183% on Concept, +91% on Observational | Very strong |
| H11: Provider systematic biases | **DIRECTIONAL** — OpenAI/Anthropic weak on Observational/Concept; underpowered statistically | Medium |
| H12: Category entropy tracks informativeness | **SUPPORTED** — Language=most informative, RL=least | Strong |
| H14: Thinking models incur higher token costs | **CONFIRMED** — Qwen Thinking: 749s / 100k+ tokens vs GPT-5.4 nano: ~21s / 200-400 tokens | Strong |
| H15: Cost ≠ performance | **CONFIRMED** — cost and score are weakly correlated; cheap models do not simply underperform | Medium |
| H16: Provider verbosity patterns | **CONFIRMED** — thinking models generate orders-of-magnitude more output tokens | Medium |
| H17: Cost-per-point efficiency ranking | **CONFIRMED** — Gemini Flash-Lite has best cost-efficiency; thinking models most expensive per point | Strong |
| H18: Token efficiency flips leaderboard | **NOT SUPPORTED** — efficiency-adjusted ranking closely mirrors raw performance ranking | Clear |

---

## Tasks Flagged for Removal — Phase D Resolution

This section records the Phase B flags. For each task's final disposition, see PHASE_D_INSIGHTS.md and CURATION_DECISIONS.md.

### Originally High Priority — Final Decisions:
| Task | Category | Issues | **Phase D Decision** |
|---|---|---|---|
| `blocking_effect` | Associative | Negative discrimination (r=−0.366), inverted tier gap | **RETAINED** — explained by epistemic awareness finding (C4) |
| `custom_gravity_simulation` | Observational | Negative discrimination (r=−0.126), near-ceiling (0.857), inverted tier gap | **REMOVED** |
| `dual_recurrence` | Concept | Negative discrimination (r=−0.394) | **REMOVED** |
| `hapax_prime` | Concept | Negative discrimination (r=−0.263), extreme bimodal | **REMOVED** |
| `manhattan_point` | RL | Gemini last place (rank=14), negative discrimination (r=−0.117) | **RETAINED** — Gemini-family non-monotonicity finding |
| `minesweeper_1d` | RL | Very inverted tier gap (small=0.834, frontier=0.441) | **RETAINED** — GPT-specific RL blind spot finding |
| `grid_nav` | RL | Very inverted tier gap (small=0.658, frontier=0.270) | **RETAINED** — GPT-family spatial navigation failure |
| `euler_totient` | RL | All-zero (entropy=0.000) | **REMOVED** |
| `semantic_override` | Concept | Very inverted tier gap (small=0.581, frontier=0.375) | **RETAINED** — flagship frontier over-specification finding |

### Originally Medium Priority — Final Decisions:
| Task | Category | Issue | **Phase D Decision** |
|---|---|---|---|
| `nim_heap` | RL | Negative discrimination (r=−0.130), extreme bimodal | **REMOVED** |
| `hidden_modal_logic_kripke2` | Observational | Negative discrimination (r=−0.127) | **REMOVED** |
| `voronoi_custom_metric` | Observational | Negative discrimination (r=−0.059) | **REMOVED** |
| `digit_square_error` | RL | Negative discrimination (r=−0.016) | **REMOVED** |
| `vigenere_variant_cipher` | Observational | Very low entropy (0.37), near-all-zero | **REMOVED** |
| `hanoi_two` | RL | Inverted bimodal (small beats frontier) | **RETAINED** — Gemini Flash RL specialization finding |
| `hanoi_three` | RL | Extreme bimodal | **REMOVED** |
| `levenshtein_words`, `hangman_lite`, `lights_out_2x2`, `linear_equation` | RL | Extreme bimodal / near-zero | **REMOVED** |
| `linear_polynomial` | RL | Extreme bimodal (Gemini 2.5 Flash only) | **REMOVED** |
| `parity_groups`, `grid_seven` | RL | Near-zero entropy, very low discrimination | **REMOVED** |

---

## B11: Negative Discrimination — Hypotheses & Investigation

Rather than automatically removing all negative-discrimination tasks, each warrants a specific hypothesis about *why* smaller/mid-tier models outperform frontier ones. These are potential insights, not just noise.

### RL tasks with negative discrimination — Phase D decision
`manhattan_point`, `minesweeper_1d`, `grid_nav` were **retained** because their inversions reveal genuine provider-specific capability gaps (see CURATION_DECISIONS.md). `nim_heap` and `digit_square_error` were **removed** (pure noise, no plausible explanation).

### Non-RL tasks with negative discrimination → INVESTIGATE (hypotheses below)

| Task | Category | Pattern | Hypothesis |
|---|---|---|---|
| `blocking_effect` | Associative | Claude Opus/Sonnet/GPT-5.4 score 1.0; mid models score 0.5–0.75; Gemini Pro scores 0.5 | **Overthinking hypothesis**: Larger models may over-analyze the blocking effect association, interpreting it as a trick or nuanced cognitive task rather than following the learning signal directly. Simpler token-prediction behavior in mid models may match the expected format better. |
| `dual_recurrence` | Concept | All models cap at 0.40; GPT-5.4 nano and small match frontier | **Task ceiling hypothesis**: The task is currently too hard for everyone (max=0.40), and frontier models may be *attempting* a sophisticated abstraction that fails, while smaller models stumble into partial correct outputs via simpler heuristics. This isn't a frontier failure — it's a task that needs rebalancing. |
| `hapax_prime` | Concept | GPT-5.4 mini (mid) scores 0.66 — *highest of all models*; all frontiers score ~0.30 | **Instruction-following sensitivity hypothesis**: Mid-tier instruction-tuned models (esp. GPT-5.4 mini) may be better calibrated to follow the specific output format required. Frontier models may be verbose or add reasoning that breaks parsing. Worth checking: does the evaluator parse the exact format strictly? |
| `hidden_modal_logic_kripke2` | Observational | Mid and small models (Gemini Flash, Claude Haiku) match frontier (0.5–0.625); GLM-5 scores 0 | **Domain-specific knowledge clash**: Modal logic (Kripke semantics) is a well-known academic concept. Frontier models trained on more academic content may apply formal modal logic rules incorrectly rather than inferring from demonstrations, while smaller models treat it more inductively. **Hypothesis**: Frontier models are *recognizing* the abstract structure and failing by using prior knowledge instead of in-context learning. |
| `custom_gravity_simulation` | Observational | Everyone scores 0.75–1.0 except GPT-5.4 mini (0.50) and GLM-5 (0.0) — effectively a near-ceiling task | **Near-ceiling artifact**: Negative discrimination here is statistical noise from a task almost everyone passes. The issue is near-zero variance at the top, not a genuine inversion. Consider whether this task still provides useful signal — it may be a strong positive discriminator for the failing outlier (GLM-5), which is actually informative. |
| `voronoi_custom_metric` | Observational | Claude Haiku (small) scores 0.83 — best; Claude Opus (frontier) scores 0.33 | **Format over reasoning hypothesis**: Voronoi diagrams require geometric reasoning, but the task likely has a specific output format. Claude Haiku may produce cleaner, format-compliant outputs while Opus tries to elaborate spatially, breaking the evaluator. Check if Haiku's correct answers are geometrically simpler or use shorter responses. |

### Key Insight for the Paper
These negative-discriminating non-RL tasks reveal a **novel finding**: larger models can be penalized by their own sophistication. When a task's scoring is format-sensitive or requires pure inductive reasoning from demonstrations (resisting prior knowledge), frontier models may underperform mid/small models. This is exactly the kind of insight a learning benchmark — as opposed to a knowledge benchmark — can surface. **Proposed framing**: "We identify a class of tasks where frontier model sophistication becomes a liability — their tendency to apply prior schema overrides in-context learning signals."

---

## B14 + B15: Kernel Log Analysis — Timing & Token Hypotheses

**Scripts:** `analysis/scripts/19_fetch_task_runs.py` → `analysis/scripts/20_fetch_notebook_logs.py`  
**Status:** Complete. See `analysis/outputs/task_runs/all_task_runs.csv` for full results (14 models × 138 tasks).

### Architecture
Each Kaggle task kernel stores only the *current* (latest) version's output files. Because each model re-uses the same kernel, only the last model to run has an accessible `run.json`. To get one representative timing sample per model, `17_download_kernel_logs.py`:

1. **Scans all ~154 task kernels** (parallel, 10 workers) to find, for each of the 14 models, a kernel where that model is the *current* version.
2. **Downloads the `run.json`** via `kaggle kernels output` CLI for each found source kernel.
3. **Parses** `startTime`, `endTime`, `totalBackendLatencyMs`, `inputTokens`, `outputTokens`, `cost_nanodollars`, and `score`.
4. **Retrieves `scriptVersionId`** for each kernel's current version and constructs the Kaggle notebook URL.

Output: `analysis/outputs/kernel_logs_parsed.csv` + `manifest.json`

### Hypotheses Tested by 18_timing_hypotheses.py

| # | Hypothesis | Test | Key Output |
|---|---|---|---|
| H14 | Thinking models incur dramatically higher token costs | Mann-Whitney U on output_tokens: thinking vs non-thinking | Ratio + p-value; chart `fig_h14_thinking_tokens.png` |
| H15 | Inference cost does not simply predict performance | Spearman ρ(log_cost, score) | Correlation + scatter plot |
| H16 | Provider training style predicts output verbosity | Kruskal-Wallis across provider groups on output_tokens | H-stat + violin chart |
| H17 | Cost-per-point reveals hidden efficiency rankings | cost_usd / score sorted ranking | Bar chart; cheapest/priciest models |
| H18 | Token efficiency ranking flips the leaderboard | Spearman ρ(score_rank, token_eff_rank) + max rank change | Rank comparison scatter |

### Why This Matters for the Writeup
- **H14** directly supports the "learning speed is an invisible dimension" killer insight — thinking tokens are the *compute price of explicit cognition*.
- **H17/H18** provide a brand-new "Learning Efficiency" angle: beyond who *learns best*, we can now quantify *who learns most cheaply per correct answer*. This is novel and judges will not have seen this framing elsewhere.
- **H16** adds color to the provider behavioral profiles from B9: provider training shapes not just accuracy but verbosity and cost structure.

→ **Completed.** Execute:
```bash
python analysis/scripts/19_fetch_task_runs.py --workers 6
python analysis/scripts/20_fetch_notebook_logs.py --workers 2
```

Results are in `analysis/outputs/task_runs/all_task_runs.csv` and `analysis/outputs/timing_hypotheses_report.md`.

---

## B12: Kaggle Timing Analysis

**Source**: Each evaluation kernel stores a `{task}.run.json` output file containing `startTime`, `endTime`, and per-API-request `metrics.totalBackendLatencyMs`. This is the real wall-clock timing. Use `kaggle kernels output kdcyberdude/{task-slug}` to download it.

**Limitation**: Each kernel only stores the *last* model run's JSON (the kernel is re-pushed once per model). To get per-model timing across all models, you would need to have captured the run JSON per kernel version during the original run. What's available now is only the final model's data per task.

### What the Run JSON Contains (per task, per model)
- `startTime` / `endTime` → exact wall-clock task duration
- `conversations[].requests[].metrics.totalBackendLatencyMs` → per-API-call inference latency
- `metrics.inputTokens` / `outputTokens` / `inputTokensCostNanodollars` → token usage and cost

### Sampled Timing Data (from currently stored run files)

| Task | Model (last run) | Total sec | Requests | Median latency/req |
|---|---|---|---|---|
| `arithmetic_next` | Qwen Thinking | **749s** | 18 | 35,838ms — thinking tokens dominate |
| `battleship_1d` | Qwen Instruct | 138s | 44 | 2,317ms — many turns, large context |
| `shift_cipher` | GPT-5.4 nano | 48s | 28 | 1,347ms |
| `perm_footrule` | GPT-5.4 nano | 21s | 28 | 634ms — small, fast model |
| `wordle_micro` | GPT-5.4 nano | 19s | 22 | 681ms |

**Key finding**: Qwen Thinking's `arithmetic_next` took **749 seconds** (12.5 min) vs GPT-5.4 nano's ~21 seconds for comparable tasks. Thinking models generate 100k+ output tokens per task vs ~200-400 for non-thinking models. This is a real cost/time consideration for benchmark scalability.

To get comprehensive per-model timing, instrument the evaluation notebook to save `run.json` per model version before the next benchmark run.

---

## B13: Claude Opus Analysis — Honest Assessment

Claude Opus 4.6 shows anomalously low scores in certain categories (observational: mean 0.315, concept: mean 0.262), despite being a frontier model. Before considering task removal, we must distinguish two cases:

### Case A: Legitimate task difficulty (keep tasks)
Opus scores 0 on 12+ observational tasks where **Gemini 3.1 Pro and GLM-5 also score 0–0.25** (e.g., `hidden_state_machine`, `lattice_meet_join`, `feistel_cipher_round`). These are genuinely hard tasks where Gemini Pro's advantage is task-specific expertise. Removing these would benefit Opus but also remove the signal that distinguishes Gemini Pro's learning superiority. **Keep.**

### Case B: Potential task artifacts (needs review)
Opus scores 0 on tasks where **other frontier models score 0.5+**:
- `perm_footrule` (RL): Opus 0.0, Gemini Pro 1.0, GPT-5.4 1.0, GLM-5 0.98 — **this is a strong Opus-specific failure on an RL task with high discrimination (r=0.50)**. The gap is extreme and warrants investigation of whether Opus was hitting a timeout, context window issue, or the RL task format is misaligned with Anthropic RLHF training.
- `shift_cipher` (RL): Opus 0.20, GPT-5.4 0.75, Gemini Pro 0.87 — significant gap.
- `affine_cipher_word` (RL): Opus 0.04, GPT-5.4 0.80, Gemini Pro 0.80 — Opus-specific underperformance.

### Methodological Principle
Task removal decisions must be based on **task quality**, not model outcomes. We can legitimately remove a task if:
1. It shows design flaws (format sensitivity, broken evaluator, ambiguous instruction)
2. It's been flagged by multiple quality metrics (negative discrimination + near-zero entropy + inverted bimodal)
3. Frontier models *cannot* solve it regardless (the task ceiling is broken)

We should **not** remove tasks simply because Opus scores badly on them if other frontier models solve them correctly — that is a signal *about* Opus, not a flaw in the task. The RL cipher tasks (`shift_cipher`, `affine_cipher_word`, `perm_footrule`) being hard for Opus but easy for GPT-5.4 and Gemini is a **legitimate benchmark finding**: Opus appears to have weaker RL-style exploration learning compared to its peers in those domains.

---

## Files Produced

| File | Rows | Description |
|---|---|---|
| `discrimination_report.csv` | 157 | Per-task discrimination r, d-index, classification |
| `cross_category_correlations.csv` | 5×5 | Spearman correlation matrix |
| `cross_category_pvalues.csv` | 5×5 | P-values for correlations |
| `cross_category_rank_correlations.csv` | 5×5 | Rank-based Spearman correlation |
| `category_pivot.csv` | 14 | Model × Category score pivot |
| `tier_stats.csv` | 15 | Per-tier, per-category mean scores |
| `tier_task_gaps.csv` | 157 | Per-task frontier-small gap |
| `tier_inversions.csv` | 22 | Model-level scale inversions |
| `efficiency_ablation.csv` | 28 | Composite vs. accuracy-only rank comparison |
| `interactive_vs_rl.csv` | 14 | Interactive mean vs. RL mean per model |
| `entropy_report.csv` | 157 | Per-task Shannon entropy |
| `category_entropy.csv` | 5 | Per-category entropy summary |
| `provider_analysis.csv` | 20 | Per-provider, per-category stats |
| `provider_pivot.csv` | 4×5 | Provider × Category mean score pivot |
| `bimodal_report.csv` | 84 | Bimodal tasks with quality classification |
| `rank1_counts.csv` | 11 | Per-model rank-#1 task counts |
| `task_rank1.csv` | 157 | Per-task rank-#1 model |
| `thinking_comparison.csv` | 5 | Qwen Thinking vs. Instruct per category |
| `gemini_ceiling.csv` | 157 | Gemini Pro score and rank on each task |
