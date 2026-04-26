"""
compute_jaggedness.py — Sub-ability correlations and jaggedness index for LearningBench

Jaggedness index definition:
    Morris et al. (2026) "Characterizing Model Jaggedness Supports Safety and Usability",
    Google DeepMind, arXiv:2601.07573 (https://arxiv.org/abs/2601.07573).

    The paper defines the jaggedness index J as the standard deviation of per-domain
    z-scores for a model (Eq. 2 in the paper):

        z_i = (x_i - μ_human_i) / σ_human_i     # normalize to human baseline
        J   = std(z'_i)                           # std of Winsorized z-scores

    Because LearningBench tasks have no published human baseline, we replace the
    human mean/SD with the benchmark-internal mean/SD across 14 evaluated models
    (same normalization direction, different reference population).  The resulting
    J values are therefore inter-model jaggedness, not human-relative jaggedness —
    they measure how unevenly a model's strengths are distributed *relative to the
    other models on this benchmark*, not relative to humans.

    A J value of 0 means the model is uniformly strong or weak across all six
    sub-abilities.  A high J means the model excels on some sub-abilities and
    fails on others.

Usage:
    python analysis/compute_jaggedness.py

Outputs:
    - Pearson correlation matrix across the 6 sub-abilities (computed across 14 models)
    - Per-model jaggedness table
    - Key numerical findings cited in sub_benchmarks/learningbench_details.md
"""

import csv
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCORE_MATRIX_PATH = ROOT / "leaderboard" / "leaderboard_score_matrix.csv"

CATS = [
    "assoc-learning",
    "concept-learning",
    "lang-learning",
    "obs-learning",
    "rf-learning",
    "proc-learning",
]
CAT_LABELS = ["Assoc", "Concept", "Lang", "Observ", "RL", "Proc"]


def load_scores(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    models = [k for k in rows[0].keys() if k not in ("task_slug", "category")]
    return rows, models


def per_model_cat_means(rows, models, cats):
    """Return dict model -> cat -> mean score (excluding unknown category)."""
    result = {}
    for model in models:
        result[model] = {}
        for cat in cats:
            cat_rows = [r for r in rows if r["category"] == cat]
            scores = [float(r[model]) for r in cat_rows if r[model] != ""]
            result[model][cat] = np.mean(scores) if scores else np.nan
    return result


def compute_score_matrix(model_cat_scores, models, cats):
    return np.array([[model_cat_scores[m][c] for c in cats] for m in models])


def pearson_correlation_matrix(score_matrix, cat_labels):
    """Pearson r between sub-abilities, computed across the model population."""
    n = score_matrix.shape[1]
    r_matrix = np.eye(n)
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, p = stats.pearsonr(score_matrix[:, i], score_matrix[:, j])
            r_matrix[i, j] = r
            p_matrix[i, j] = p
    return r_matrix, p_matrix


def compute_jaggedness(score_matrix):
    """
    Jaggedness index per Morris et al. (2026) Eq. 2, adapted to benchmark-internal
    normalization (no human baseline available).

    Steps:
      1. For each sub-ability column, compute the cross-model mean and SD.
      2. z-score each model's domain score: z_i = (x_i - col_mean) / col_std
      3. J = std(z_i across domains) for each model
    """
    col_means = score_matrix.mean(axis=0)
    col_stds = score_matrix.std(axis=0)
    z_matrix = (score_matrix - col_means) / col_stds
    # Population std (ddof=0) matches Morris et al. formula
    return z_matrix.std(axis=1, ddof=0), z_matrix


def print_correlation_matrix(r_matrix, cat_labels):
    print("=== Pearson Correlation Matrix (across 14 models) ===")
    header = " " * 9 + " | ".join(f"{l:>7}" for l in cat_labels)
    print(header)
    for i, label in enumerate(cat_labels):
        vals = " | ".join(f"{r_matrix[i, j]:7.3f}" for j in range(len(cat_labels)))
        print(f"{label:>7}: {vals}")
    print()


def print_jaggedness_table(models, model_cat_scores, cats, jaggedness):
    print("=== Jaggedness Table ===")
    header = (
        f"{'Model':<35} | {'Overall':>7} | {'J':>5} | "
        + " | ".join(f"{l:>7}" for l in CAT_LABELS)
    )
    print(header)
    for i, m in enumerate(models):
        scores = [model_cat_scores[m][c] for c in cats]
        overall = np.mean(scores)
        score_cols = " | ".join(f"{s:7.3f}" for s in scores)
        print(f"{m:<35} | {overall:7.3f} | {jaggedness[i]:5.3f} | {score_cols}")
    print()


def print_key_findings(score_matrix, models, model_cat_scores, cats, r_matrix, jaggedness):
    print("=== Key Findings (numbers cited in learningbench_details.md) ===")

    rl_idx = cats.index("rf-learning")
    concept_idx = cats.index("concept-learning")
    obs_idx = cats.index("obs-learning")

    rl_mean = score_matrix[:, rl_idx].mean()
    concept_mean = score_matrix[:, concept_idx].mean()
    obs_mean = score_matrix[:, obs_idx].mean()
    print(f"RL mean across models:      {rl_mean:.3f}")
    print(f"Concept mean across models: {concept_mean:.3f}")
    print(f"RL - Concept gap:           {rl_mean - concept_mean:.3f}")
    print()

    concept_gt_obs = sum(
        1
        for m in models
        if model_cat_scores[m]["concept-learning"] > model_cat_scores[m]["obs-learning"]
    )
    print(f"Models where Concept > Obs: {concept_gt_obs}/{len(models)}")
    print(f"(Interactivity paradox: {len(models)-concept_gt_obs}/{len(models)} do better passively)")
    print()

    # Min/max off-diagonal correlations
    min_r, max_r = 1.0, -1.0
    min_pair, max_pair = None, None
    n = len(cats)
    for i in range(n):
        for j in range(i + 1, n):
            r = r_matrix[i, j]
            if r < min_r:
                min_r = r
                min_pair = (CAT_LABELS[i], CAT_LABELS[j])
            if r > max_r:
                max_r = r
                max_pair = (CAT_LABELS[i], CAT_LABELS[j])
    print(f"Weakest inter-domain correlation: {min_pair[0]} x {min_pair[1]}: r={min_r:.3f}")
    print(f"Strongest inter-domain correlation: {max_pair[0]} x {max_pair[1]}: r={max_r:.3f}")
    print()

    # Jaggedness summary
    min_j_idx = np.argmin(jaggedness)
    max_j_idx = np.argmax(jaggedness)
    print(f"Most uniform (lowest J): {models[min_j_idx]} — J={jaggedness[min_j_idx]:.3f}")
    print(f"Most jagged (highest J): {models[max_j_idx]} — J={jaggedness[max_j_idx]:.3f}")


def main():
    rows, models = load_scores(SCORE_MATRIX_PATH)
    model_cat_scores = per_model_cat_means(rows, models, CATS)
    score_matrix = compute_score_matrix(model_cat_scores, models, CATS)
    r_matrix, p_matrix = pearson_correlation_matrix(score_matrix, CAT_LABELS)
    jaggedness, z_matrix = compute_jaggedness(score_matrix)

    print_correlation_matrix(r_matrix, CAT_LABELS)
    print_jaggedness_table(models, model_cat_scores, CATS, jaggedness)
    print_key_findings(score_matrix, models, model_cat_scores, CATS, r_matrix, jaggedness)


if __name__ == "__main__":
    main()
