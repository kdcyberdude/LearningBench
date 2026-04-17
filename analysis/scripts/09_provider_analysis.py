"""
B9: Model Provider Analysis (H11)
- Group models by provider (Google, OpenAI, Anthropic, Open-source)
- Compute per-provider mean score on each category
- Test whether provider is a significant factor (Kruskal-Wallis)
- Identify systematic provider strengths/weaknesses
- Outputs: provider_analysis.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_score_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
PROVIDERS = ["Google", "OpenAI", "Anthropic", "Open-source"]
CATEGORIES = ["associative", "concept", "language", "observational", "rl"]


def compute_provider_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-provider, per-category mean scores."""
    # Model mean per category
    model_cat = df.groupby(["model", "category", "provider", "tier"])["score"].mean().reset_index()

    rows = []
    for (prov, cat), sub in model_cat.groupby(["provider", "category"]):
        rows.append({
            "provider": prov,
            "category": cat,
            "n_models": len(sub),
            "mean_score": round(float(sub["score"].mean()), 4),
            "std_score": round(float(sub["score"].std()), 4),
            "min_score": round(float(sub["score"].min()), 4),
            "max_score": round(float(sub["score"].max()), 4),
            "models": ", ".join(sorted(sub["model"].tolist())),
        })
    return pd.DataFrame(rows)


def compute_provider_pivot(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Provider × Category mean score pivot."""
    return stats_df.pivot_table(
        values="mean_score", index="provider", columns="category"
    ).reindex(index=PROVIDERS)


def kruskal_wallis_by_provider(df: pd.DataFrame, cat: str) -> tuple:
    """KW test: does provider explain score variance for a category?"""
    cat_df = df[df["category"] == cat]
    model_means = cat_df.groupby(["model", "provider"])["score"].mean().reset_index()
    groups = [model_means[model_means["provider"] == p]["score"].values
              for p in PROVIDERS if len(model_means[model_means["provider"] == p]) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    if any(len(g) < 2 for g in groups):
        return np.nan, np.nan
    kw_stat, kw_p = stats.kruskal(*groups)
    return float(kw_stat), float(kw_p)


def compute_provider_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Overall mean per provider (across all categories)."""
    model_overall = df.groupby(["model", "provider"])["score"].mean().reset_index()
    prov_overall = model_overall.groupby("provider")["score"].agg(["mean", "std", "count"])
    prov_overall.columns = ["overall_mean", "overall_std", "n_models"]
    return prov_overall.sort_values("overall_mean", ascending=False)


def identify_provider_strengths(pivot: pd.DataFrame) -> dict:
    """For each provider, identify their strongest and weakest categories."""
    strengths = {}
    for prov in pivot.index:
        row = pivot.loc[prov].dropna()
        if len(row) == 0:
            continue
        # Relative to column means
        col_means = pivot.mean(axis=0)
        relative = row - col_means
        strengths[prov] = {
            "strongest_cat": relative.idxmax(),
            "weakest_cat": relative.idxmin(),
            "relative_strength": round(float(relative.max()), 3),
            "relative_weakness": round(float(relative.min()), 3),
        }
    return strengths


def print_summary(prov_stats: pd.DataFrame, pivot: pd.DataFrame,
                  overall: pd.DataFrame, strengths: dict, kw_results: dict) -> None:
    print("\n" + "=" * 60)
    print("B9: MODEL PROVIDER ANALYSIS (H11)")
    print("=" * 60)

    print("\n--- Provider × Category mean score pivot ---")
    print(pivot.round(3).to_string())

    print("\n--- Overall provider rankings ---")
    print(overall.round(3).to_string())

    print("\n--- Kruskal-Wallis test: does provider explain variance? ---")
    for cat, (kw_stat, kw_p) in kw_results.items():
        if np.isnan(kw_stat):
            print(f"  {cat:<15} KW test insufficient data")
        else:
            sig = "SIGNIFICANT" if kw_p < 0.05 else "not significant"
            print(f"  {cat:<15} H={kw_stat:.2f}  p={kw_p:.3f}  → {sig}")

    print("\n--- Provider systematic strengths/weaknesses ---")
    for prov, s in strengths.items():
        print(f"\n  {prov}:")
        print(f"    Strongest: {s['strongest_cat']} (relative advantage: +{s['relative_strength']:.3f})")
        print(f"    Weakest:   {s['weakest_cat']} (relative deficit: {s['relative_weakness']:.3f})")

    print("\n--- H11 verdict ---")
    any_sig = any(p < 0.05 for _, (_, p) in kw_results.items() if not np.isnan(p))
    if any_sig:
        print("  → H11 SUPPORTED: Provider is a significant factor in at least one category.")
        sig_cats = [cat for cat, (_, p) in kw_results.items() if not np.isnan(p) and p < 0.05]
        print(f"  Significant categories: {sig_cats}")
    else:
        print("  → H11 NOT CONFIRMED: Provider does not significantly explain variance (may be due to small N).")

    print("\n--- Notable observations ---")
    # Google dominance check
    if "Google" in pivot.index:
        google_row = pivot.loc["Google"].dropna()
        is_top = all(pivot[cat].max() == pivot.loc["Google", cat] for cat in google_row.index)
        if is_top:
            print("  Google leads in ALL categories (driven by Gemini 3.1 Pro).")
        else:
            non_top = [cat for cat in google_row.index if pivot[cat].max() != pivot.loc["Google", cat]]
            print(f"  Google leads in most categories, except: {non_top}")


def main():
    df = load_score_matrix()
    prov_stats = compute_provider_category_stats(df)
    pivot = compute_provider_pivot(prov_stats)
    overall = compute_provider_overall(df)
    strengths = identify_provider_strengths(pivot)
    kw_results = {cat: kruskal_wallis_by_provider(df, cat) for cat in CATEGORIES}

    print_summary(prov_stats, pivot, overall, strengths, kw_results)

    prov_stats.to_csv(OUTPUT_DIR / "provider_analysis.csv", index=False)
    pivot.to_csv(OUTPUT_DIR / "provider_pivot.csv")
    overall.to_csv(OUTPUT_DIR / "provider_overall.csv")
    print(f"\nSaved → {OUTPUT_DIR}/provider_*.csv")
    return prov_stats, pivot


if __name__ == "__main__":
    main()
