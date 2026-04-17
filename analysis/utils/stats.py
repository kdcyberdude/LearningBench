"""
Shared statistical functions.
"""

import numpy as np
import pandas as pd
from scipy import stats


def shannon_entropy(scores: np.ndarray, n_bins: int = 10) -> float:
    """Compute Shannon entropy of a score distribution.

    Discretises scores into n_bins equal-width bins, then computes H = -Σ p log2(p).
    High entropy → scores spread evenly (good discrimination).
    Low entropy  → scores clustered (poor discrimination).
    """
    if len(scores) == 0:
        return 0.0
    counts, _ = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def item_discrimination_index(task_scores: np.ndarray, total_scores: np.ndarray) -> float:
    """Point-biserial / Pearson correlation of task score with total category score.

    Values close to 1 → task tracks overall performance (good discrimination).
    Values near 0 or negative → task is uncorrelated with / inversely related to overall.
    """
    if len(task_scores) < 3:
        return float("nan")
    r, _ = stats.pearsonr(task_scores, total_scores)
    return float(r)


def bimodality_coefficient(scores: np.ndarray) -> float:
    """Sarle's bimodality coefficient B = (skewness^2 + 1) / kurtosis.

    B > 0.555 suggests bimodality.
    """
    if len(scores) < 4:
        return float("nan")
    n = len(scores)
    sk = float(stats.skew(scores))
    ku = float(stats.kurtosis(scores))  # excess kurtosis
    # Formula: B = (sk^2 + 1) / (ku + 3*(n-1)^2 / ((n-2)*(n-3)))
    denom = ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom == 0:
        return float("nan")
    return (sk**2 + 1) / denom


def tier_discrimination(scores_by_tier: dict[str, np.ndarray]) -> dict:
    """Kruskal-Wallis test across model tiers.

    Returns dict with:
      - kw_stat, kw_p: Kruskal-Wallis H-statistic and p-value
      - tier_means: dict of mean score per tier
      - discriminates: bool (p < 0.05)
    """
    groups = [v for v in scores_by_tier.values() if len(v) > 0]
    tier_means = {k: float(np.mean(v)) if len(v) > 0 else float("nan")
                  for k, v in scores_by_tier.items()}

    if len(groups) < 2:
        return {"kw_stat": float("nan"), "kw_p": float("nan"),
                "tier_means": tier_means, "discriminates": False}

    kw_stat, kw_p = stats.kruskal(*groups)
    return {
        "kw_stat": float(kw_stat),
        "kw_p": float(kw_p),
        "tier_means": tier_means,
        "discriminates": kw_p < 0.05,
    }
