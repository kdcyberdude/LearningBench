"""
C4: Epistemic Uncertainty / UNKNOWN Answer Analysis (H5)

Tests H5: "Models systematically fail on questions where UNKNOWN is the correct
answer — they always commit to a definitive answer."

We identify all tasks with UNKNOWN-answer questions (confirmed: blocking_effect,
xor_attribute_binding, second_order_extinction from Phase B investigation),
then analyze whether models answer UNKNOWN correctly or commit to definitive
(wrong) answers.

Since we only have aggregate task scores (not per-question), we:
1. Analyze aggregate scores on UNKNOWN-heavy tasks vs non-UNKNOWN tasks
2. Check if models that score well overall also do well on UNKNOWN tasks
3. Compare the difficulty of UNKNOWN tasks vs other associative learning tasks
4. Build a principled H5 assessment from the data we have

Outputs:
  - epistemic_analysis.csv  (UNKNOWN vs non-UNKNOWN task comparison)
  - unknown_task_scores.csv  (per-model scores on UNKNOWN tasks)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Tasks confirmed to have UNKNOWN as correct answer (from Phase B investigation)
# blocking_effect: ALL 4 questions have UNKNOWN as correct answer
# xor_attribute_binding: 1 of 6 questions is UNKNOWN (q_6)
# second_order_extinction (deep_second_order_extinction): 3 of 10 questions are UNKNOWN (q5, q7, q9)
UNKNOWN_TASKS = {
    "blocking_effect_assoc_learning": {
        "n_unknown_questions": 4,
        "total_questions": 4,  # ALL questions are UNKNOWN
        "pct_unknown": 1.0,
        "structure": "ALL questions are epistemic uncertainty traps — correct answer is UNKNOWN for every question",
    },
    "xor_attribute_binding_assoc_learning": {
        "n_unknown_questions": 1,
        "total_questions": 6,
        "pct_unknown": 1 / 6,
        "structure": "1/6 questions require UNKNOWN (P-ambiguity case)",
    },
    "deep_second_order_extinction_assoc_learning": {
        "n_unknown_questions": 3,
        "total_questions": 10,
        "pct_unknown": 0.3,
        "structure": "3/10 questions are UNKNOWN (blocking ambiguity, dependent uncertainty chain)",
    },
}

# Alternative task name variants from leaderboard
UNKNOWN_TASK_ALIASES = {
    "blocking_effect": "blocking_effect_assoc_learning",
    "xor_attribute_binding": "xor_attribute_binding_assoc_learning",
    "second_order_extinction": "deep_second_order_extinction_assoc_learning",
    "deep_second_order_extinction": "deep_second_order_extinction_assoc_learning",
}


def _normalize_task_name(name: str) -> str:
    """Strip category suffix for matching."""
    for suffix in ["_assoc_learning", "_rf_learning", "_concept_learning", "_obs_learning"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def find_unknown_tasks_in_matrix(df: pd.DataFrame) -> set:
    """Find which tasks from our UNKNOWN list are in the score matrix."""
    all_tasks = set(df["task_name"].unique())
    found = set()
    for t in all_tasks:
        norm = _normalize_task_name(t)
        if norm in {_normalize_task_name(k) for k in UNKNOWN_TASKS}:
            found.add(t)
        elif norm in UNKNOWN_TASK_ALIASES:
            found.add(t)
    return found


def analyze_unknown_vs_normal(df: pd.DataFrame, unknown_task_names: set) -> pd.DataFrame:
    """Compare model scores on UNKNOWN tasks vs non-UNKNOWN associative tasks."""
    assoc_df = df[df["category"] == "associative"].copy()

    assoc_df["has_unknown"] = assoc_df["task_name"].isin(unknown_task_names)

    unknown_scores = assoc_df[assoc_df["has_unknown"]].groupby("model")["score"].mean()
    normal_scores = assoc_df[~assoc_df["has_unknown"]].groupby("model")["score"].mean()
    overall_scores = assoc_df.groupby("model")["score"].mean()

    rows = []
    for model in sorted(df["model"].unique()):
        tier = df[df["model"] == model]["tier"].iloc[0]
        rows.append(
            {
                "model": model,
                "tier": tier,
                "unknown_task_score": round(unknown_scores.get(model, float("nan")), 4),
                "normal_task_score": round(normal_scores.get(model, float("nan")), 4),
                "overall_assoc_score": round(overall_scores.get(model, float("nan")), 4),
                "deficit_on_unknown": round(
                    normal_scores.get(model, 0) - unknown_scores.get(model, 0), 4
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("overall_assoc_score", ascending=False)


def blocking_effect_deep_dive(df: pd.DataFrame) -> dict:
    """
    blocking_effect has ALL UNKNOWN correct answers — it's a pure epistemic trap.
    Any non-zero score means the model got lucky guessing UNKNOWN.
    Analyze scores on this specific task.
    """
    # Find blocking effect task
    blocking = df[
        df["task_name"].str.contains("blocking_effect", case=False, na=False)
        & (df["category"] == "associative")
    ]

    if blocking.empty:
        return {"found": False}

    task_name = blocking["task_name"].iloc[0]
    scores = blocking.set_index("model")["score"].sort_values(ascending=False)

    # Perfect score on blocking_effect = model answered UNKNOWN to all 4 questions
    # Score 0.75 = got 3/4 UNKNOWN correct
    # Score 0.5  = got 2/4 UNKNOWN correct (could be random)
    # Score 0.25 = got 1/4 UNKNOWN correct
    # Score 0.0  = answered definitively wrong to all 4

    perfect = (scores == 1.0).sum()
    high = ((scores >= 0.75) & (scores < 1.0)).sum()
    partial = ((scores > 0.0) & (scores < 0.75)).sum()
    zero = (scores == 0.0).sum()

    # Random baseline: P(answer UNKNOWN) ≈ 1/3 per question
    # P(all 4 correct by random) = (1/3)^4 ≈ 0.012
    random_expected = 4 * (1 / 3)  # Expected correct = 1.33 / 4 = 0.333

    return {
        "found": True,
        "task_name": task_name,
        "all_questions_unknown": True,
        "scores": scores.to_dict(),
        "mean_score": float(scores.mean()),
        "random_baseline": round(random_expected, 4),
        "n_perfect": int(perfect),
        "n_high_75plus": int(high),
        "n_partial": int(partial),
        "n_zero": int(zero),
        "interpretation": (
            f"Perfect score requires all 4 UNKNOWN answers correct. "
            f"Mean score {scores.mean():.3f} vs random baseline {random_expected:.3f}. "
            f"{zero} models score 0 (never said UNKNOWN). "
            f"{perfect} models score 1.0 (always said UNKNOWN correctly)."
        ),
    }


def h5_assessment(df: pd.DataFrame, unknown_tasks: set, comparison: pd.DataFrame) -> dict:
    """
    Formal assessment of H5.

    H5 claims: Models systematically fail on UNKNOWN questions — they commit
    to definitive answers instead of acknowledging uncertainty.
    """
    if not unknown_tasks:
        return {
            "verdict": "CANNOT TEST",
            "reason": "No UNKNOWN tasks found in score matrix",
        }

    # Key metric: do models underperform on UNKNOWN tasks vs normal tasks?
    deficit = comparison["deficit_on_unknown"]
    mean_deficit = float(deficit.mean())
    pct_worse = float((deficit > 0.05).mean() * 100)

    # Statistical test: paired t-test of UNKNOWN vs normal scores
    valid = comparison.dropna(subset=["unknown_task_score", "normal_task_score"])
    if len(valid) > 2:
        t_stat, p_val = stats.ttest_rel(
            valid["normal_task_score"], valid["unknown_task_score"]
        )
    else:
        t_stat, p_val = float("nan"), float("nan")

    blocking_mean = comparison["unknown_task_score"].mean()
    normal_mean = comparison["normal_task_score"].mean()

    if mean_deficit > 0.05 and (np.isnan(p_val) or p_val < 0.05):
        verdict = "SUPPORTED"
        strength = "Strong" if mean_deficit > 0.15 else "Moderate"
    elif mean_deficit > 0.02:
        verdict = "WEAKLY SUPPORTED"
        strength = "Weak"
    else:
        verdict = "NOT SUPPORTED"
        strength = "No effect"

    return {
        "verdict": verdict,
        "strength": strength,
        "mean_deficit_on_unknown": round(mean_deficit, 4),
        "pct_models_worse_on_unknown": round(pct_worse, 1),
        "unknown_task_mean": round(blocking_mean, 4),
        "normal_task_mean": round(normal_mean, 4),
        "t_stat": round(t_stat, 3) if not np.isnan(t_stat) else None,
        "p_value": round(p_val, 4) if not np.isnan(p_val) else None,
        "n_unknown_tasks": len(unknown_tasks),
        "writeup_note": (
            "Note: We measure H5 via aggregate task scores (not per-question). "
            "blocking_effect is a pure UNKNOWN trap (all 4 correct answers = UNKNOWN). "
            "Deficits on UNKNOWN-heavy tasks indicate models commit to definitive answers. "
            "blocking_effect's negative discrimination (Phase B) is consistent with "
            "frontier models attempting sophisticated reasoning that avoids UNKNOWN, "
            "while mid/small models more likely to output the literal word UNKNOWN."
        ),
    }


def main():
    print("=== C4: Epistemic Uncertainty / UNKNOWN Analysis (H5) ===\n")

    df = load_score_matrix()
    print(f"Loaded {df['task_name'].nunique()} tasks\n")

    # Find UNKNOWN tasks
    unknown_tasks = find_unknown_tasks_in_matrix(df)
    print(f"UNKNOWN tasks found in matrix: {len(unknown_tasks)}")
    for t in sorted(unknown_tasks):
        norm = _normalize_task_name(t)
        if norm in UNKNOWN_TASKS:
            info = UNKNOWN_TASKS[norm]
        else:
            # Try alias
            for alias, canonical in UNKNOWN_TASK_ALIASES.items():
                if alias in t:
                    canonical_norm = _normalize_task_name(canonical)
                    info = UNKNOWN_TASKS.get(canonical_norm, {})
                    break
            else:
                info = {}
        print(f"  {t}: {info.get('structure', 'see task source')}")

    print()

    # Compare UNKNOWN vs normal associative tasks
    comparison = analyze_unknown_vs_normal(df, unknown_tasks)
    comparison.to_csv(OUTPUT_DIR / "unknown_task_scores.csv", index=False)
    print("Saved unknown_task_scores.csv\n")

    # Blocking effect deep dive
    blocking = blocking_effect_deep_dive(df)
    print(f"=== blocking_effect deep dive ===")
    if blocking["found"]:
        print(f"  Mean score: {blocking['mean_score']:.4f}")
        print(f"  Random baseline: {blocking['random_baseline']:.4f} (expected {1/3:.3f} per question by chance)")
        print(f"  Models scoring 0.0 (never said UNKNOWN): {blocking['n_zero']}")
        print(f"  Models scoring 1.0 (always UNKNOWN): {blocking['n_perfect']}")
        print(f"  Scores by model:")
        for m, s in sorted(blocking["scores"].items(), key=lambda x: -x[1]):
            print(f"    {m}: {s:.4f}")
    else:
        print("  Task not found in matrix")

    # H5 formal assessment
    print(f"\n=== H5 Formal Assessment ===")
    h5 = h5_assessment(df, unknown_tasks, comparison)
    for k, v in h5.items():
        if k != "writeup_note":
            print(f"  {k}: {v}")
    print(f"\n  Note: {h5['writeup_note']}")

    # Save epistemic analysis
    rows = []
    for task_name in sorted(df["task_name"].unique()):
        if df[df["task_name"] == task_name]["category"].iloc[0] != "associative":
            continue
        is_unknown = task_name in unknown_tasks
        task_scores = df[df["task_name"] == task_name]["score"]
        rows.append(
            {
                "task_name": task_name,
                "has_unknown_answers": is_unknown,
                "mean_score": round(task_scores.mean(), 4),
                "std_score": round(task_scores.std(), 4),
                "n_models_scoring_zero": int((task_scores == 0).sum()),
            }
        )
    epistemic_df = pd.DataFrame(rows).sort_values("has_unknown_answers", ascending=False)
    epistemic_df.to_csv(OUTPUT_DIR / "epistemic_analysis.csv", index=False)
    print("\nSaved epistemic_analysis.csv")

    print("\n--- MODEL COMPARISON: UNKNOWN vs Normal Tasks ---")
    print(comparison[
        ["model", "tier", "unknown_task_score", "normal_task_score", "deficit_on_unknown"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
