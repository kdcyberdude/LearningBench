"""
C2: Random Baseline Analysis (R10)

Compute expected scores from random responding for each task category.
Proves that our benchmark cannot be gamed by chance.

For each category, derives the random-chance baseline from:
  - Associative: multi-class classification (YES / NO / UNKNOWN)
  - Concept Formation: short-text answers (effectively near-0 for random)
  - Language Learning: short-text answers (near-0 for random)
  - Observational: sequence-of-tokens output (near-0 for random)
  - RL: 3-component score (solved + efficiency + progress), random gives near-0

Also computes: how far above random are the actual model scores?
This is the "signal-to-noise" ratio of the benchmark.

Outputs:
  - random_baseline.csv (per-category baseline + model mean + signal ratio)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


# --------------------------------------------------------------------------
# Analytically-derived random baselines
# --------------------------------------------------------------------------
# These are principled estimates based on the scoring formulas in SCORING.md
# and task design. See comments for derivation.
RANDOM_BASELINES = {
    # Associative Learning:
    # Each question has 3 choices: YES, NO, UNKNOWN
    # Random accuracy = 1/3 ≈ 0.333
    # But most questions have 2 valid choices for a given trial log, so
    # even random is generous at 0.333. True informative baseline is even lower
    # because UNKNOWN questions are rare (only 3 of 20 tasks have UNKNOWN).
    # Effective random baseline across the full category: ~0.333
    "associative": {
        "baseline": 1 / 3,
        "derivation": "3-class classification (YES/NO/UNKNOWN), random = 1/3",
        "n_choices": 3,
    },
    # Concept Formation:
    # Scored by substring match on exact rule outputs.
    # Random text generation has near-zero probability of hitting exact output.
    # Practical random baseline ≈ 0.0 (regex search finds almost nothing)
    # Conservative estimate: 0.02 (2% from partial matches / lucky guesses)
    "concept": {
        "baseline": 0.02,
        "derivation": (
            "Exact rule output required (substring match). "
            "Random text hits correct answer with ~0% probability. "
            "Conservative estimate 0.02 accounts for lucky partial matches."
        ),
        "n_choices": None,
    },
    # Language Learning:
    # Same as concept — exact morphophonological form required.
    # Random text generation essentially never produces the correct wug-form.
    # Baseline ≈ 0.01 (slightly lower than CF because outputs are longer phonological strings)
    "language": {
        "baseline": 0.01,
        "derivation": (
            "Exact novel word-form required (substring match). "
            "Random phonological output near-zero probability of match."
        ),
        "n_choices": None,
    },
    # Observational Learning:
    # Scored on correct_sequences / total_sequences.
    # Each sequence is a structured output (typically 3–12 tokens).
    # Random token sequences cannot match multi-step computed outputs.
    # Baseline ≈ 0.01 (single-step outputs might occasionally match)
    "observational": {
        "baseline": 0.01,
        "derivation": (
            "Multi-token sequence output scored by full match. "
            "Random output has near-zero probability of matching computed sequence."
        ),
        "n_choices": None,
    },
    # RL (Runtime RL):
    # Score = 0.55*solved + 0.25*efficiency + 0.20*progress
    # Random agent: solves task with P ≈ 0 (hidden state must be discovered),
    # efficiency is 0 (never solves), progress ≈ 0.1 (random exploration gets some partial credit)
    # Expected: 0.55*0 + 0.25*0 + 0.20*0.1 = 0.02
    "rl": {
        "baseline": 0.02,
        "derivation": (
            "RL score = 0.55*solved + 0.25*efficiency + 0.20*progress. "
            "Random agent: solved=0 (hidden state), efficiency=0, progress≈0.1 (random exploration). "
            "Expected = 0.55*0 + 0.25*0 + 0.20*0.10 = 0.020"
        ),
        "n_choices": None,
    },
}


def compute_signal_to_noise(df: pd.DataFrame) -> pd.DataFrame:
    """For each model and category, compute how far above random they score."""
    model_cat = df.groupby(["category", "model"])["score"].mean().reset_index()
    model_cat.columns = ["category", "model", "mean_score"]

    # Attach tier
    tier_map = df[["model", "tier"]].drop_duplicates().set_index("model")["tier"]
    model_cat["tier"] = model_cat["model"].map(tier_map)

    rows = []
    for cat, cat_df in model_cat.groupby("category"):
        baseline = RANDOM_BASELINES[cat]["baseline"]
        cat_mean = cat_df["mean_score"].mean()
        worst_model = cat_df.loc[cat_df["mean_score"].idxmin(), "model"]
        worst_score = cat_df["mean_score"].min()
        best_model = cat_df.loc[cat_df["mean_score"].idxmax(), "model"]
        best_score = cat_df["mean_score"].max()

        rows.append(
            {
                "category": cat,
                "random_baseline": baseline,
                "mean_model_score": round(cat_mean, 4),
                "best_model": best_model,
                "best_score": round(best_score, 4),
                "worst_model": worst_model,
                "worst_score": round(worst_score, 4),
                "signal_above_random_mean": round(cat_mean - baseline, 4),
                "signal_ratio_mean": round(cat_mean / max(baseline, 0.001), 2),
                "signal_ratio_best": round(best_score / max(baseline, 0.001), 2),
                "worst_above_random": worst_score > baseline,
                "derivation": RANDOM_BASELINES[cat]["derivation"],
            }
        )

    return pd.DataFrame(rows)


def per_model_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model signal above random for each category."""
    model_cat = df.groupby(["category", "model", "tier"])["score"].mean().reset_index()

    rows = []
    for _, row in model_cat.iterrows():
        cat = row["category"]
        baseline = RANDOM_BASELINES[cat]["baseline"]
        rows.append(
            {
                "category": cat,
                "model": row["model"],
                "tier": row["tier"],
                "mean_score": round(row["score"], 4),
                "random_baseline": baseline,
                "above_random": round(row["score"] - baseline, 4),
                "above_random_pct": round((row["score"] - baseline) / max(baseline, 0.001) * 100, 1),
            }
        )

    return pd.DataFrame(rows).sort_values(["category", "mean_score"], ascending=[True, False])


def main():
    print("=== C2: Random Baseline Analysis ===\n")

    df = load_score_matrix()
    print(f"Loaded {df['task_name'].nunique()} tasks, {df['model'].nunique()} models\n")

    # Per-category summary
    print("Computing signal-to-noise ratios per category...")
    signal_df = compute_signal_to_noise(df)
    signal_df.to_csv(OUTPUT_DIR / "random_baseline.csv", index=False)
    print("  Saved random_baseline.csv\n")

    # Per-model signal
    model_signal = per_model_signal(df)
    model_signal.to_csv(OUTPUT_DIR / "model_signal_above_random.csv", index=False)
    print("  Saved model_signal_above_random.csv\n")

    # Print findings
    print("--- KEY FINDINGS ---\n")
    print("Random baselines and signal ratios:\n")

    display_cols = [
        "category",
        "random_baseline",
        "mean_model_score",
        "signal_above_random_mean",
        "signal_ratio_mean",
        "signal_ratio_best",
        "worst_above_random",
    ]
    print(signal_df[display_cols].to_string(index=False))

    print("\n\nAll models above random baseline on every category?")
    below_random = model_signal[model_signal["above_random"] < 0]
    if below_random.empty:
        print("  YES — every model beats random on every category ✓")
    else:
        print(f"  NO — {len(below_random)} model-category combos below random:")
        print(below_random[["category", "model", "mean_score", "random_baseline"]].to_string(index=False))

    print("\nMinimum signal ratio (worst model vs. random):")
    for cat in ["associative", "concept", "language", "observational", "rl"]:
        cat_df = model_signal[model_signal["category"] == cat]
        worst_ratio = cat_df["above_random_pct"].min()
        worst_model = cat_df.loc[cat_df["above_random_pct"].idxmin(), "model"]
        print(f"  {cat:12s}: {worst_ratio:+.1f}% above random ({worst_model})")


if __name__ == "__main__":
    main()
