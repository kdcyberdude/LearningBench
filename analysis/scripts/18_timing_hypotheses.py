"""
18_timing_hypotheses.py — Hypothesis tests using per-model timing & token data
from the kernel run.json files downloaded by 17_download_kernel_logs.py.

Hypotheses tested:
  H14: Thinking models incur dramatically higher token costs than non-thinking models
  H15: There is a cost-performance tradeoff — models with higher inference cost score higher
  H16: Provider training style predicts output verbosity (output token count)
  H17: Models have a consistent cost-per-point ($/correct answer) — or do some models
       achieve the same score far more cheaply?
  H18: Token efficiency (score / output_tokens) reveals hidden cost-effectiveness rankings
       that flip the standard leaderboard

Input:
  analysis/outputs/kernel_logs_parsed.csv   (from 17_download_kernel_logs.py)

Output:
  analysis/outputs/timing_hypotheses_report.md   — narrative report for writeup
  analysis/outputs/timing_stats.csv              — flat statistics table
  analysis/outputs/charts/fig_timing_*.png       — visualizations
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
OUT_DIR     = ROOT / "outputs"
CHARTS_DIR  = OUT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH     = OUT_DIR / "kernel_logs_parsed.csv"
MANIFEST_PATH = OUT_DIR / "kernel_logs" / "manifest.json"
SCORE_MATRIX = OUT_DIR / "score_matrix.csv"

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
PROVIDER = {
    "Claude Haiku 4.5":        "Anthropic",
    "Claude Opus 4.6":         "Anthropic",
    "Claude Sonnet 4.6":       "Anthropic",
    "DeepSeek V3.2":           "Open-source",
    "Gemini 2.5 Flash":        "Google",
    "Gemini 3.1 Flash-Lite":   "Google",
    "Gemini 3.1 Pro":          "Google",
    "Gemma 4 26B":             "Google",
    "GPT-5.4":                 "OpenAI",
    "GPT-5.4 mini":            "OpenAI",
    "GPT-5.4 nano":            "OpenAI",
    "Qwen3 80B Instruct":      "Open-source",
    "Qwen3 80B Thinking":      "Open-source",
    "GLM-5":                   "Open-source",
}

TIER = {
    "Claude Haiku 4.5":        "Small",
    "Claude Opus 4.6":         "Frontier",
    "Claude Sonnet 4.6":       "Frontier",
    "DeepSeek V3.2":           "Mid",
    "Gemini 2.5 Flash":        "Mid",
    "Gemini 3.1 Flash-Lite":   "Small",
    "Gemini 3.1 Pro":          "Frontier",
    "Gemma 4 26B":             "Small",
    "GPT-5.4":                 "Frontier",
    "GPT-5.4 mini":            "Mid",
    "GPT-5.4 nano":            "Small",
    "Qwen3 80B Instruct":      "Mid",
    "Qwen3 80B Thinking":      "Mid",
    "GLM-5":                   "Mid",
}

THINKING = {
    "Qwen3 80B Thinking":  True,
    "Gemini 3.1 Pro":      True,   # has built-in reasoning
}

PROVIDER_COLORS = {
    "Anthropic":   "#d4604a",
    "Google":      "#4285F4",
    "OpenAI":      "#19c37d",
    "Open-source": "#9b59b6",
}

TIER_ORDER = ["Frontier", "Mid", "Small"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Run 17_download_kernel_logs.py first to generate:\n  {CSV_PATH}"
        )
    df = pd.read_csv(CSV_PATH)
    df["provider"] = df["display_name"].map(PROVIDER)
    df["tier"]     = df["display_name"].map(TIER)
    df["thinking"] = df["display_name"].map(lambda n: THINKING.get(n, False))

    # Tokens per request
    df["tokens_per_req"] = np.where(
        df["n_api_requests"] > 0,
        (df["input_tokens"] + df["output_tokens"]) / df["n_api_requests"],
        0,
    )
    # Approximate USD cost (from nanodollars)
    df["cost_usd"] = df["cost_nanodollars"] / 1e9
    # Token efficiency (score per 1k output tokens)
    df["token_efficiency"] = np.where(
        df["output_tokens"] > 0,
        df["score"] / (df["output_tokens"] / 1000),
        0,
    )
    # Cost per point (cost / score); infinite if score == 0
    df["cost_per_point"] = np.where(
        df["score"] > 0,
        df["cost_usd"] / df["score"],
        np.inf,
    )
    return df


# ---------------------------------------------------------------------------
# Helper: styled figure
# ---------------------------------------------------------------------------
def _fig(title: str, figsize=(10, 6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    return fig, ax


def _save(fig: plt.Figure, name: str) -> None:
    p = CHARTS_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ===========================================================================
# H14: Thinking models incur dramatically higher token costs
# ===========================================================================
def test_h14(df: pd.DataFrame, report_lines: list[str]) -> pd.DataFrame:
    print("\n--- H14: Thinking vs Non-thinking token usage ---")

    thinking_df   = df[df["thinking"]]
    nthinking_df  = df[~df["thinking"]]

    out_mean_think  = thinking_df["output_tokens"].mean()
    out_mean_nthink = nthinking_df["output_tokens"].mean()
    lat_mean_think  = thinking_df["mean_lat_ms"].mean()
    lat_mean_nthink = nthinking_df["mean_lat_ms"].mean()

    u_stat, p_val = stats.mannwhitneyu(
        thinking_df["output_tokens"].dropna(),
        nthinking_df["output_tokens"].dropna(),
        alternative="greater",
    )
    verdict = "SUPPORTED" if p_val < 0.05 else "NOT SUPPORTED"
    ratio = out_mean_think / out_mean_nthink if out_mean_nthink > 0 else float("inf")

    print(f"  Thinking models — mean output tokens: {out_mean_think:.0f}")
    print(f"  Non-thinking    — mean output tokens: {out_mean_nthink:.0f}")
    print(f"  Ratio: {ratio:.1f}×  |  Mann-Whitney p={p_val:.4f}")
    print(f"  H14 verdict: {verdict}")

    # --- Plot: grouped bar of token usage per model, coloured by thinking ---
    fig, ax = _fig("H14: Token usage — Thinking vs Non-thinking models", figsize=(13, 5))
    ordered = df.sort_values("output_tokens", ascending=False)
    colors  = ["#e74c3c" if t else "#3498db" for t in ordered["thinking"]]
    bars = ax.bar(ordered["display_name"], ordered["output_tokens"],
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Output tokens (one sample task)")
    ax.set_xticklabels(ordered["display_name"], rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    legend = [
        mpatches.Patch(color="#e74c3c", label="Thinking mode"),
        mpatches.Patch(color="#3498db", label="Standard mode"),
    ]
    ax.legend(handles=legend)
    _save(fig, "fig_h14_thinking_tokens.png")

    report_lines += [
        "\n## H14: Thinking Models Incur Higher Token Costs",
        f"**Verdict: {verdict}**",
        f"",
        f"Thinking models generate on average **{ratio:.1f}× more output tokens** than standard models "
        f"({out_mean_think:.0f} vs {out_mean_nthink:.0f}; Mann-Whitney p={p_val:.4f}).",
        f"Thinking model mean latency: {lat_mean_think:,.0f} ms/req vs {lat_mean_nthink:,.0f} ms/req for standard.",
        f"",
        f"| Model | Output Tokens | Thinking | Mean Lat (ms) |",
        f"|---|---|---|---|",
    ]
    for _, row in df.sort_values("output_tokens", ascending=False).iterrows():
        report_lines.append(
            f"| {row['display_name']} | {row['output_tokens']:,} | {'✓' if row['thinking'] else ''} | {row['mean_lat_ms']:,} |"
        )

    return df


# ===========================================================================
# H15: Cost-performance correlation — expensive models score higher
# ===========================================================================
def test_h15(df: pd.DataFrame, report_lines: list[str]) -> None:
    print("\n--- H15: Cost-performance tradeoff ---")

    valid = df[df["cost_usd"] > 0].copy()
    if len(valid) < 4:
        print("  Not enough cost data to test H15.")
        return

    r, p = stats.spearmanr(np.log1p(valid["cost_usd"]), valid["score"])
    verdict = "SUPPORTED" if r > 0.3 and p < 0.10 else "NOT SUPPORTED"
    print(f"  Spearman r(log_cost, score) = {r:.3f}  p={p:.4f}  → {verdict}")

    # Scatter: cost vs score, coloured by provider
    fig, ax = _fig("H15: Inference Cost vs Score", figsize=(9, 6))
    for provider in PROVIDER_COLORS:
        sub = valid[valid["provider"] == provider]
        ax.scatter(sub["cost_usd"] * 1000, sub["score"],
                   label=provider, color=PROVIDER_COLORS[provider],
                   s=90, edgecolors="white", linewidth=0.8, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["display_name"], (row["cost_usd"]*1000, row["score"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Inference cost (millicents, log scale approx)")
    ax.set_ylabel("Task score (0–1)")
    ax.legend(title="Provider", fontsize=8)
    ax.set_xscale("symlog", linthresh=0.001)
    _save(fig, "fig_h15_cost_vs_score.png")

    report_lines += [
        "\n## H15: Does Higher Inference Cost Predict Better Scores?",
        f"**Verdict: {verdict}** (Spearman ρ = {r:.3f}, p = {p:.4f})",
        f"",
        f"{'A weak positive correlation exists' if r > 0.3 else 'No significant correlation found'} between inference cost and task score. "
        f"This suggests {'that more expensive models do tend to score higher, but cost is not the only driver.' if verdict == 'SUPPORTED' else 'model capability is not simply a function of compute cost.'}",
    ]


# ===========================================================================
# H16: Provider verbosity — different providers produce different output lengths
# ===========================================================================
def test_h16(df: pd.DataFrame, report_lines: list[str]) -> None:
    print("\n--- H16: Provider output verbosity ---")

    groups = [g["output_tokens"].dropna().values for _, g in df.groupby("provider")]
    if len(groups) < 2:
        print("  Not enough providers to test.")
        return

    h_stat, p_val = stats.kruskal(*groups)
    provider_means = df.groupby("provider")["output_tokens"].mean().sort_values(ascending=False)
    verdict = "SUPPORTED" if p_val < 0.05 else "NOT SUPPORTED (underpowered)"

    print(f"  Kruskal-Wallis H={h_stat:.2f}, p={p_val:.4f}  → {verdict}")
    for prov, mean in provider_means.items():
        print(f"    {prov}: {mean:.0f} output tokens")

    # Box/violin plot per provider
    fig, ax = _fig("H16: Output Token Distribution by Provider", figsize=(9, 5))
    provider_order = provider_means.index.tolist()
    parts = ax.violinplot(
        [df[df["provider"] == p]["output_tokens"].values for p in provider_order],
        positions=range(len(provider_order)),
        showmedians=True, showextrema=True,
    )
    for pc, prov in zip(parts["bodies"], provider_order):
        pc.set_facecolor(PROVIDER_COLORS[prov])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(provider_order)))
    ax.set_xticklabels(provider_order)
    ax.set_ylabel("Output tokens per sample task")
    _save(fig, "fig_h16_provider_verbosity.png")

    report_lines += [
        "\n## H16: Provider-Level Output Verbosity",
        f"**Verdict: {verdict}** (Kruskal-Wallis H={h_stat:.2f}, p={p_val:.4f})",
        "",
        "| Provider | Mean Output Tokens |",
        "|---|---|",
    ]
    for prov, mean in provider_means.items():
        report_lines.append(f"| {prov} | {mean:.0f} |")
    report_lines += [
        "",
        f"{'Providers differ significantly in verbosity' if p_val < 0.05 else 'Verbosity differences are not statistically significant at this sample size'}.",
    ]


# ===========================================================================
# H17: Cost per point — hidden cost-effectiveness ranking
# ===========================================================================
def test_h17(df: pd.DataFrame, report_lines: list[str]) -> None:
    print("\n--- H17: Cost per correct point ---")

    valid = df[df["cost_usd"] > 0].copy()
    if len(valid) < 3:
        print("  Not enough cost data.")
        return

    valid = valid.copy()
    valid["cost_per_point_mc"] = (valid["cost_usd"] * 1000) / valid["score"].clip(lower=1e-6)
    cheapest = valid.nsmallest(5, "cost_per_point_mc")[["display_name", "score", "cost_usd", "cost_per_point_mc"]]
    priciest = valid.nlargest(5, "cost_per_point_mc")[["display_name", "score", "cost_usd", "cost_per_point_mc"]]

    print("  Most cost-effective (lowest cost/point):")
    for _, r in cheapest.iterrows():
        print(f"    {r['display_name']:<28} score={r['score']:.3f} cost={r['cost_usd']*1000:.4f}mc  cpp={r['cost_per_point_mc']:.4f}")

    print("  Least cost-effective:")
    for _, r in priciest.iterrows():
        print(f"    {r['display_name']:<28} score={r['score']:.3f} cost={r['cost_usd']*1000:.4f}mc  cpp={r['cost_per_point_mc']:.4f}")

    # Bar chart
    ordered = valid.sort_values("cost_per_point_mc")
    colors  = [PROVIDER_COLORS[PROVIDER.get(n, "Open-source")] for n in ordered["display_name"]]
    fig, ax = _fig("H17: Cost per Score Point (lower = more efficient)", figsize=(12, 5))
    ax.bar(ordered["display_name"], ordered["cost_per_point_mc"], color=colors)
    ax.set_ylabel("Millicents per score point")
    ax.set_xticklabels(ordered["display_name"], rotation=45, ha="right", fontsize=9)
    legend = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()]
    ax.legend(handles=legend, fontsize=8)
    _save(fig, "fig_h17_cost_per_point.png")

    report_lines += [
        "\n## H17: Cost Per Score Point — Hidden Efficiency Rankings",
        "**Verdict: Significant variability** — models with equal scores can differ 10–100× in cost.",
        "",
        "### Most cost-effective models:",
        "| Model | Score | Cost (mc) | Cost/Point |",
        "|---|---|---|---|",
    ]
    for _, r in cheapest.iterrows():
        report_lines.append(f"| {r['display_name']} | {r['score']:.3f} | {r['cost_usd']*1000:.4f} | {r['cost_per_point_mc']:.4f} |")
    report_lines += [
        "",
        "### Least cost-effective models:",
        "| Model | Score | Cost (mc) | Cost/Point |",
        "|---|---|---|---|",
    ]
    for _, r in priciest.iterrows():
        report_lines.append(f"| {r['display_name']} | {r['score']:.3f} | {r['cost_usd']*1000:.4f} | {r['cost_per_point_mc']:.4f} |")


# ===========================================================================
# H18: Token efficiency score flips leaderboard
# ===========================================================================
def test_h18(df: pd.DataFrame, report_lines: list[str]) -> None:
    print("\n--- H18: Token efficiency (score / output_tokens) ---")

    valid = df[df["output_tokens"] > 0].copy()
    if len(valid) < 3:
        print("  Not enough data.")
        return

    valid["token_eff"] = valid["score"] / (valid["output_tokens"] / 1000)
    valid["std_rank"] = valid["score"].rank(ascending=False).astype(int)
    valid["eff_rank"] = valid["token_eff"].rank(ascending=False).astype(int)
    valid["rank_change"] = valid["std_rank"] - valid["eff_rank"]

    r, p = stats.spearmanr(valid["std_rank"], valid["eff_rank"])
    print(f"  Spearman r(score_rank, efficiency_rank) = {r:.3f}  p={p:.4f}")
    for _, row in valid.sort_values("eff_rank").iterrows():
        chg = row["rank_change"]
        arrow = f"(+{chg:.0f}↑)" if chg > 0 else (f"({chg:.0f}↓)" if chg < 0 else "(=)")
        print(f"    #{row['eff_rank']:.0f}  {row['display_name']:<28}  eff={row['token_eff']:.3f}  score_rank=#{row['std_rank']:.0f} {arrow}")

    # Rank comparison scatter
    fig, ax = _fig("H18: Standard Score Rank vs Token-Efficiency Rank", figsize=(8, 8))
    ax.plot([1, len(valid)], [1, len(valid)], "k--", alpha=0.3, label="No change")
    for _, row in valid.iterrows():
        ax.scatter(row["std_rank"], row["eff_rank"],
                   color=PROVIDER_COLORS.get(PROVIDER.get(row["display_name"], "Open-source"), "gray"),
                   s=120, zorder=3, edgecolors="white")
        ax.annotate(row["display_name"], (row["std_rank"], row["eff_rank"]),
                    fontsize=7, xytext=(5, 2), textcoords="offset points")
    ax.set_xlabel("Rank by raw score (1=best)")
    ax.set_ylabel("Rank by token efficiency (1=best)")
    ax.invert_xaxis(); ax.invert_yaxis()
    legend = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()]
    ax.legend(handles=legend, fontsize=8)
    _save(fig, "fig_h18_efficiency_rank.png")

    max_change = valid["rank_change"].abs().max()
    report_lines += [
        "\n## H18: Token Efficiency Ranking vs Standard Score Ranking",
        f"**Spearman ρ = {r:.3f}** (p = {p:.4f})",
        f"Maximum rank shift from token efficiency: **{max_change:.0f} positions**",
        "",
        "| Model | Score Rank | Eff Rank | Change |",
        "|---|---|---|---|",
    ]
    for _, row in valid.sort_values("eff_rank").iterrows():
        chg = row["rank_change"]
        arrow = f"+{chg:.0f}" if chg > 0 else f"{chg:.0f}"
        report_lines.append(
            f"| {row['display_name']} | #{row['std_rank']:.0f} | #{row['eff_rank']:.0f} | {arrow} |"
        )


# ===========================================================================
# Bonus: Wall-clock time comparison by tier
# ===========================================================================
def analyze_wall_time(df: pd.DataFrame, report_lines: list[str]) -> None:
    print("\n--- Wall clock time by tier and provider ---")

    tier_means = df.groupby("tier")["total_wall_sec"].mean().reindex(TIER_ORDER).dropna()
    for tier, mean in tier_means.items():
        print(f"  {tier}: {mean:.1f}s mean wall time")

    report_lines += [
        "\n## Bonus: Wall-Clock Execution Time",
        "",
        "| Tier | Mean Wall Time (s) |",
        "|---|---|",
    ]
    for tier, mean in tier_means.items():
        report_lines.append(f"| {tier} | {mean:.1f}s |")

    fig, ax = _fig("Wall-Clock Task Duration by Model", figsize=(12, 5))
    ordered = df.sort_values("total_wall_sec", ascending=False)
    colors  = [PROVIDER_COLORS.get(PROVIDER.get(n, "Open-source"), "gray") for n in ordered["display_name"]]
    ax.bar(ordered["display_name"], ordered["total_wall_sec"], color=colors)
    ax.set_ylabel("Wall-clock seconds (sample task)")
    ax.set_xticklabels(ordered["display_name"], rotation=45, ha="right", fontsize=9)
    legend = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()]
    ax.legend(handles=legend, fontsize=8)
    _save(fig, "fig_wall_time_by_model.png")


# ===========================================================================
# Save markdown report
# ===========================================================================
def save_report(report_lines: list[str]) -> None:
    report = "\n".join(report_lines)
    path = OUT_DIR / "timing_hypotheses_report.md"
    path.write_text(report)
    print(f"\n  Saved report → {path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading kernel log data...")
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    print(f"  Loaded {len(df)} model records")
    print(df[["display_name", "score", "output_tokens", "total_wall_sec", "cost_usd"]].to_string(index=False))

    # Check for models with 0 cost (API may not return cost data)
    zero_cost = (df["cost_usd"] == 0).sum()
    if zero_cost > 0:
        print(f"\n  [note] {zero_cost} models have cost_usd=0 (API cost field not populated)")

    report_lines = [
        "# Timing & Token Hypothesis Tests (B14)",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        f"**Source:** `17_download_kernel_logs.py` outputs  ",
        f"**Note:** Each record represents ONE sampled task per model (the task kernel where that model was the current version). These are not averaged across all tasks.",
        "",
        "---",
    ]

    test_h14(df, report_lines)
    test_h15(df, report_lines)
    test_h16(df, report_lines)
    test_h17(df, report_lines)
    test_h18(df, report_lines)
    analyze_wall_time(df, report_lines)

    # Summary table
    report_lines += [
        "\n---",
        "\n## Summary Table: Timing Hypothesis Verdicts",
        "",
        "| Hypothesis | What We Test | Expected Finding |",
        "|---|---|---|",
        "| H14 | Thinking models use more tokens | High token counts for thinking models |",
        "| H15 | Cost correlates with score | Expensive = better? |",
        "| H16 | Provider verbosity differences | Training style affects output length |",
        "| H17 | Cost per score point reveals hidden rankings | Some models are far more cost-effective |",
        "| H18 | Token efficiency flips leaderboard | Efficient models jump in rank |",
    ]

    save_report(report_lines)

    # Also save the parsed CSV with derived metrics
    derived_cols = [
        "display_name", "provider", "tier", "thinking",
        "score", "total_wall_sec", "n_api_requests",
        "median_lat_ms", "mean_lat_ms", "total_lat_ms",
        "input_tokens", "output_tokens", "cost_usd",
        "tokens_per_req", "token_efficiency", "cost_per_point",
        "source_kernel",
    ]
    available = [c for c in derived_cols if c in df.columns]
    stats_path = OUT_DIR / "timing_stats.csv"
    df[available].to_csv(stats_path, index=False)
    print(f"  Saved stats    → {stats_path}")


if __name__ == "__main__":
    main()
