"""
Parse leaderboard JSONs into a unified score matrix.

Each leaderboard JSON has structure:
  {"rows": [
    {
      "modelVersionName": "...",
      "modelVersionSlug": "...",
      "taskResults": [
        {
          "benchmarkTaskName": "...",  # empty string = aggregate row
          "benchmarkTaskSlug": "...",
          "result": {"numericResult": {"value": float}}
        }, ...
      ]
    }, ...
  ]}

Returns a DataFrame with columns:
  category, task_name, model, score

------------------------------------------------------------
FAST PATH (recommended): Use the pre-computed CSV instead of
re-parsing Kaggle JSONs.

  from utils.data_loader import load_score_matrix_csv
  df = load_score_matrix_csv()            # all 157 tasks
  df = load_score_matrix_csv(curated=True) # 138 tasks (Phase D)

The raw leaderboard JSONs from Kaggle are NOT included in the
repository. If you need to re-parse them, download them from
the respective Kaggle benchmark pages and place them in:
  <repo_root>/leaderboards/
------------------------------------------------------------
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
LEADERBOARD_DIR = REPO_ROOT / "leaderboards"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

LEADERBOARD_FILES = {
    "associative": "kdcyberdude_associativelearning_leaderboard.json",
    "concept": "kdcyberdude_conceptlearning_leaderboard.json",
    "language": "kdcyberdude_languagelearning_leaderboard.json",
    "observational": "kdcyberdude_observationallearning_leaderboard.json",
    "rl": "kdcyberdude_rlbench_leaderboard.json",
}

CATEGORY_FULL_NAMES = {
    "associative": "Associative Learning",
    "concept": "Concept Formation",
    "language": "Language Learning",
    "observational": "Observational Learning",
    "rl": "Reinforcement Learning",
}

# Model tier classification — matches PROJECT_MASTER.md Section 15
MODEL_TIERS = {
    # Frontier
    "Gemini 3.1 Pro Preview": "frontier",
    "GLM-5": "frontier",
    "Claude Opus 4.6": "frontier",
    "GPT-5.4": "frontier",
    # Mid
    "Qwen 3 Next 80B Thinking": "mid",
    "Gemini 3.1 Flash-Lite Preview": "mid",
    "Gemini 2.5 Flash": "mid",
    "Claude Sonnet 4.6": "mid",
    "DeepSeek V3.2": "mid",
    "GPT-5.4 mini": "mid",
    # Small
    "Claude Haiku 4.5": "small",
    "GPT-5.4 nano": "small",
    "Qwen 3 Next 80B Instruct": "small",
    "Gemma 4 26B A4B": "small",
}

MODEL_PROVIDERS = {
    "Gemini 3.1 Pro Preview": "Google",
    "Gemini 2.5 Flash": "Google",
    "Gemini 3.1 Flash-Lite Preview": "Google",
    "Gemma 4 26B A4B": "Google",
    "GPT-5.4": "OpenAI",
    "GPT-5.4 mini": "OpenAI",
    "GPT-5.4 nano": "OpenAI",
    "Claude Opus 4.6": "Anthropic",
    "Claude Sonnet 4.6": "Anthropic",
    "Claude Haiku 4.5": "Anthropic",
    "Qwen 3 Next 80B Thinking": "Open-source",
    "Qwen 3 Next 80B Instruct": "Open-source",
    "DeepSeek V3.2": "Open-source",
    "GLM-5": "Open-source",
}


def _load_leaderboard(path: Path, category: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one leaderboard JSON.

    Returns:
        task_df  – rows for individual tasks (benchmarkTaskName non-empty)
        agg_df   – rows for aggregate scores (benchmarkTaskName empty)
    """
    with open(path) as f:
        data = json.load(f)

    task_rows = []
    agg_rows = []

    for model_entry in data["rows"]:
        model = model_entry["modelVersionName"]
        for task in model_entry["taskResults"]:
            result = task["result"]
            # Some results have no numeric value (missing/boolean-only results)
            if "numericResult" in result:
                score = result["numericResult"]["value"]
            elif "numericResultNullable" in result and result.get("hasNumericResult"):
                score = result["numericResultNullable"]["value"]
            else:
                score = None  # missing — will be filtered out
            task_name = task["benchmarkTaskName"]
            if task_name:
                task_rows.append(
                    {
                        "category": category,
                        "category_full": CATEGORY_FULL_NAMES[category],
                        "task_name": task_name,
                        "model": model,
                        "score": score,
                    }
                )
            else:
                agg_rows.append(
                    {
                        "category": category,
                        "category_full": CATEGORY_FULL_NAMES[category],
                        "model": model,
                        "aggregate_score": score,
                    }
                )

    return pd.DataFrame(task_rows), pd.DataFrame(agg_rows)


def load_score_matrix() -> pd.DataFrame:
    """Load all leaderboards → unified long-format DataFrame.

    Columns: category, category_full, task_name, model, score, tier, provider
    """
    all_task_rows = []

    for cat, fname in LEADERBOARD_FILES.items():
        path = LEADERBOARD_DIR / fname
        task_df, _ = _load_leaderboard(path, cat)
        all_task_rows.append(task_df)

    df = pd.concat(all_task_rows, ignore_index=True)

    # Drop rows where score is missing (tasks with no numeric result)
    n_before = len(df)
    df = df.dropna(subset=["score"]).copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"   Note: dropped {n_dropped} rows with no numeric score")

    # Attach tier and provider
    df["tier"] = df["model"].map(MODEL_TIERS).fillna("unknown")
    df["provider"] = df["model"].map(MODEL_PROVIDERS).fillna("unknown")

    return df


def load_aggregates() -> pd.DataFrame:
    """Load aggregate scores from JSON (the empty-name rows).

    Also computes aggregates from task means for categories where
    the JSON aggregate may differ (e.g., associative was 'pending').

    Returns DataFrame with columns:
      category, model, json_aggregate, computed_aggregate
    """
    json_aggs = []
    computed_aggs = []

    for cat, fname in LEADERBOARD_FILES.items():
        path = LEADERBOARD_DIR / fname
        task_df, agg_df = _load_leaderboard(path, cat)

        # JSON aggregate (the single empty-name row per model)
        for _, row in agg_df.iterrows():
            json_aggs.append(
                {
                    "category": cat,
                    "model": row["model"],
                    "json_aggregate": row["aggregate_score"],
                }
            )

        # Computed aggregate = mean of per-task scores
        computed = (
            task_df.groupby("model")["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "computed_aggregate"})
        )
        computed["category"] = cat
        computed_aggs.append(computed)

    json_df = pd.DataFrame(json_aggs)
    computed_df = pd.concat(computed_aggs, ignore_index=True)

    merged = json_df.merge(computed_df, on=["category", "model"])
    merged["tier"] = merged["model"].map(MODEL_TIERS).fillna("unknown")
    merged["provider"] = merged["model"].map(MODEL_PROVIDERS).fillna("unknown")

    return merged


# ── Fast path: load from pre-computed CSV (no Kaggle API needed) ─────────────

def load_score_matrix_csv(curated: bool = False) -> pd.DataFrame:
    """Load the pre-computed score matrix from CSV.

    This is the recommended way to load data if you are not re-parsing
    the raw Kaggle leaderboard JSONs.

    Args:
        curated: If True, load the Phase D curated matrix (138 tasks).
                 If False, load the full pre-curation matrix (157 tasks).

    Returns:
        DataFrame with columns:
            category, category_full, task_name, model, score, tier, provider
    """
    fname = "score_matrix_phase_d.csv" if curated else "score_matrix.csv"
    path = OUTPUTS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-computed CSV not found at {path}. "
            "Make sure you are running from the repository root, or that "
            "analysis/outputs/ is present."
        )
    df = pd.read_csv(path)
    return df
