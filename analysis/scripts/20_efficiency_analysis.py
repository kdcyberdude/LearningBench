"""
20_efficiency_analysis.py — Token, cost, latency vs quality analysis.

Generates charts in analysis/outputs/efficiency_charts/:
  1. cost_efficiency_scatter.png   — score vs cost per task (bubble = total cost)
  2. score_per_dollar_bar.png      — score-per-dollar bar chart
  3. token_verbosity_by_model.png  — stacked bar: avg input+output tokens
  4. latency_vs_score.png          — scatter: latency vs score per subability
  5. token_waste_heatmap.png       — model x subability avg tokens heatmap
  6. rf_learning_token_quartiles.png — RF tokens by score quartile
  7. hard_easy_tasks.png           — hardest vs easiest tasks by mean score
  8. task_cost_vs_difficulty.png   — per-task mean tokens vs mean score
  9. provider_verbosity.png        — output tokens distribution by provider
 10. efficiency_frontier.png       — Pareto: score vs total cost bubble chart
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = PROJECT_ROOT / "analysis" / "outputs" / "task_runs" / "all_task_runs.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "efficiency_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (one per model, consistent across all charts) ────────────
MODEL_COLORS = {
    "Gemini 3.1 Pro Preview":       "#1565C0",
    "GLM-5":                        "#C62828",
    "Qwen 3 Next 80B Thinking":     "#6A1B9A",
    "Claude Opus 4.6":              "#E65100",
    "Claude Sonnet 4.6":            "#F57C00",
    "GPT-5.4":                      "#1B5E20",
    "Gemini 3.1 Flash-Lite Preview":"#0097A7",
    "Claude Haiku 4.5":             "#FF8F00",
    "DeepSeek V3.2":                "#AD1457",
    "GPT-5.4 mini":                 "#558B2F",
    "Gemini 2.5 Flash":             "#1976D2",
    "Gemma 4 26B A4B":              "#795548",
    "Qwen 3 Next 80B Instruct":     "#7B1FA2",
    "GPT-5.4 nano":                 "#388E3C",
}

TIER_COLORS = {
    "Frontier":  "#1565C0",
    "Standard":  "#E65100",
    "Efficient": "#2E7D32",
}

MODEL_TIERS = {
    "Gemini 3.1 Pro Preview":       "Frontier",
    "Claude Opus 4.6":              "Frontier",
    "GPT-5.4":                      "Frontier",
    "GLM-5":                        "Frontier",
    "Qwen 3 Next 80B Thinking":     "Standard",
    "Claude Sonnet 4.6":            "Standard",
    "Gemini 2.5 Flash":             "Standard",
    "DeepSeek V3.2":                "Standard",
    "Qwen 3 Next 80B Instruct":     "Standard",
    "Claude Haiku 4.5":             "Efficient",
    "GPT-5.4 mini":                 "Efficient",
    "Gemini 3.1 Flash-Lite Preview":"Efficient",
    "Gemma 4 26B A4B":              "Efficient",
    "GPT-5.4 nano":                 "Efficient",
}

SA_LABELS = {
    "assoc-learning":   "Associative",
    "concept-learning": "Concept Formation",
    "lang-learning":    "Language Learning",
    "obs-learning":     "Observational",
    "rf-learning":      "Reinforcement",
}

SA_COLORS = {
    "assoc-learning":   "#1565C0",
    "concept-learning": "#C62828",
    "lang-learning":    "#2E7D32",
    "obs-learning":     "#6A1B9A",
    "rf-learning":      "#E65100",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["score"] = df["score_value"].fillna(df["score_fraction"])
    df = df[df["model_display_name"] != "Gemini 3 Flash Preview"].copy()
    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
    df["total_latency_sec"] = df["total_latency_ms"] / 1000
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0)
    df["tier"] = df["model_display_name"].map(MODEL_TIERS)
    return df


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
    })


# ── Chart 1: Score vs mean cost per task (bubble = total cost) ──────────────
def chart_cost_efficiency_scatter(df: pd.DataFrame):
    model_stats = (
        df.dropna(subset=["score"])
        .groupby("model_display_name")
        .agg(mean_score=("score", "mean"), mean_cost=("cost_usd", "mean"),
             total_cost=("cost_usd", "sum"))
        .reset_index()
    )
    model_stats["tier"] = model_stats["model_display_name"].map(MODEL_TIERS)

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in model_stats.iterrows():
        c = TIER_COLORS.get(row["tier"], "#888")
        size = np.clip(row["total_cost"] * 300, 100, 3000)
        ax.scatter(row["mean_cost"], row["mean_score"], s=size,
                   color=c, alpha=0.8, edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(row["model_display_name"], (row["mean_cost"], row["mean_score"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                    color="#333", zorder=4)

    patches = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in ["Frontier", "Standard", "Efficient"]]
    ax.legend(handles=patches, loc="lower right", fontsize=9, title="Tier")
    ax.set_xlabel("Mean Cost per Task (USD)")
    ax.set_ylabel("Mean Score (0–1)")
    ax.set_title("Score vs. Cost per Task\n(bubble size = total benchmark spend)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.3f}"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cost_efficiency_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ cost_efficiency_scatter.png")


# ── Chart 2: Score per dollar bar chart ─────────────────────────────────────
def chart_score_per_dollar(df: pd.DataFrame):
    df_v = df.dropna(subset=["score"]).copy()
    df_v = df_v[df_v["cost_usd"] > 0]
    df_v["score_per_dollar"] = df_v["score"] / df_v["cost_usd"]
    spd = df_v.groupby("model_display_name")["score_per_dollar"].median().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [TIER_COLORS.get(MODEL_TIERS.get(m, ""), "#888") for m in spd.index]
    bars = ax.barh(range(len(spd)), spd.values, color=colors, alpha=0.85, edgecolor="white")

    for i, (m, v) in enumerate(spd.items()):
        ax.text(v + max(spd) * 0.01, i, f"{v:.0f}", va="center", fontsize=8.5)

    ax.set_yticks(range(len(spd)))
    ax.set_yticklabels(spd.index, fontsize=9)
    ax.set_xlabel("Median Score per Dollar (higher = more efficient)")
    ax.set_title("Cost Efficiency: Score per Dollar\n(median across all tasks)", fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in ["Frontier", "Standard", "Efficient"]]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "score_per_dollar_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ score_per_dollar_bar.png")


# ── Chart 3: Token verbosity stacked bar ────────────────────────────────────
def chart_token_verbosity(df: pd.DataFrame):
    tok = df.groupby("model_display_name").agg(
        avg_input=("input_tokens", "mean"),
        avg_output=("output_tokens", "mean"),
    ).reset_index().sort_values("avg_input", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    x = range(len(tok))
    c_input = "#1565C0"
    c_output = "#E65100"
    ax.bar(x, tok["avg_input"], label="Avg Input Tokens", color=c_input, alpha=0.8)
    ax.bar(x, tok["avg_output"], bottom=tok["avg_input"], label="Avg Output Tokens",
           color=c_output, alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tok["model_display_name"], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Average Tokens per Task")
    ax.set_title("Token Verbosity by Model\n(averaged across all tasks)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "token_verbosity_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ token_verbosity_by_model.png")


# ── Chart 4: Latency vs score scatter per subability ────────────────────────
def chart_latency_vs_score(df: pd.DataFrame):
    df_v = df.dropna(subset=["score"]).copy()
    df_v = df_v[(df_v["total_latency_sec"] > 0) & (df_v["total_latency_sec"] < 2000)]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    for ax, (sa, label) in zip(axes, SA_LABELS.items()):
        sub = df_v[df_v["subability_type"] == sa]
        if sub.empty:
            ax.set_visible(False)
            continue
        r, p = stats.spearmanr(sub["total_latency_sec"], sub["score"])
        ax.scatter(sub["total_latency_sec"], sub["score"],
                   color=SA_COLORS[sa], alpha=0.3, s=15, edgecolors="none")
        # trend line
        m, b, *_ = stats.linregress(sub["total_latency_sec"], sub["score"])
        xs = np.linspace(sub["total_latency_sec"].min(), sub["total_latency_sec"].max(), 100)
        ax.plot(xs, m * xs + b, color="black", linewidth=1.5, alpha=0.7)
        ax.set_title(f"{label}\nr={r:.2f}, p={'<0.001' if p<0.001 else f'{p:.3f}'}", fontsize=9)
        ax.set_xlabel("Latency (s)", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    axes[0].set_ylabel("Score (0–1)")
    fig.suptitle("Latency vs. Score by Sub-Ability", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "latency_vs_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ latency_vs_score.png")


# ── Chart 5: Token waste heatmap ────────────────────────────────────────────
def chart_token_waste_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(
        index="model_display_name", columns="subability_type",
        values="total_tokens", aggfunc="mean"
    )
    pivot.columns = [SA_LABELS.get(c, c) for c in pivot.columns]
    # Sort models by overall mean
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate with K values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                txt = f"{v/1000:.0f}K" if v >= 1000 else f"{v:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if v > pivot.values.max() * 0.6 else "#333")

    fig.colorbar(im, ax=ax, label="Avg Total Tokens", fraction=0.03, pad=0.02)
    ax.set_title("Token Consumption Heatmap\n(model × sub-ability)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "token_waste_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ token_waste_heatmap.png")


# ── Chart 6: RF-learning token quartiles ────────────────────────────────────
def chart_rf_token_quartiles(df: pd.DataFrame):
    rf = df[df["subability_type"] == "rf-learning"].dropna(subset=["score"]).copy()
    rf["score_bin"] = pd.cut(rf["score"], bins=[0, 0.25, 0.5, 0.75, 1.0],
                              labels=["0–25%", "25–50%", "50–75%", "75–100%"],
                              include_lowest=True)
    grp = rf.groupby("score_bin")[["input_tokens", "output_tokens"]].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(grp))
    w = 0.38
    ax.bar(x - w/2, grp["input_tokens"] / 1000, w, label="Avg Input Tokens (K)", color="#1565C0", alpha=0.85)
    ax.bar(x + w/2, grp["output_tokens"] / 1000, w, label="Avg Output Tokens (K)", color="#E65100", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Score {b}" for b in grp.index], fontsize=10)
    ax.set_ylabel("Avg Tokens (thousands)")
    ax.set_title("Reinforcement Learning Tasks:\nToken Use vs Score Quartile", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.0f}K", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rf_learning_token_quartiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ rf_learning_token_quartiles.png")


# ── Chart 7: Hardest vs easiest tasks ───────────────────────────────────────
def chart_hard_easy_tasks(df: pd.DataFrame):
    task_stats = (
        df.dropna(subset=["score"])
        .groupby(["task_slug", "subability_type"])
        .agg(mean_score=("score", "mean"), mean_tokens=("total_tokens", "mean"))
        .reset_index()
    )
    hardest = task_stats.nsmallest(12, "mean_score")
    easiest = task_stats.nlargest(12, "mean_score")

    def short_name(slug):
        parts = slug.replace("-rf-learning","").replace("-assoc-learning","") \
            .replace("-concept-learning","").replace("-obs-learning","") \
            .replace("-lang-learning","").split("-")
        return " ".join(p.capitalize() for p in parts[:4])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, cmap_key in [
        (ax1, hardest, "12 Hardest Tasks (lowest mean score)", False),
        (ax2, easiest, "12 Easiest Tasks (highest mean score)", True),
    ]:
        data = data.sort_values("mean_score", ascending=not cmap_key)
        colors = [SA_COLORS.get(sa, "#888") for sa in data["subability_type"]]
        bars = ax.barh(range(len(data)), data["mean_score"], color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([short_name(s) for s in data["task_slug"]], fontsize=8.5)
        ax.set_xlabel("Mean Score (0–1)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.05)
        for i, (_, row) in enumerate(data.iterrows()):
            ax.text(row["mean_score"] + 0.01, i, f"{row['mean_score']:.2f}", va="center", fontsize=8)

    patches = [mpatches.Patch(color=SA_COLORS[sa], label=SA_LABELS[sa]) for sa in SA_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               title="Sub-ability", bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "hard_easy_tasks.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ hard_easy_tasks.png")


# ── Chart 8: Per-task tokens vs difficulty ───────────────────────────────────
def chart_task_cost_vs_difficulty(df: pd.DataFrame):
    task_stats = (
        df.dropna(subset=["score"])
        .groupby(["task_slug", "subability_type"])
        .agg(mean_score=("score", "mean"), mean_tokens=("total_tokens", "mean"))
        .reset_index()
    )
    r, p = stats.spearmanr(task_stats["mean_score"], task_stats["mean_tokens"])

    fig, ax = plt.subplots(figsize=(10, 7))
    for sa in SA_LABELS:
        sub = task_stats[task_stats["subability_type"] == sa]
        ax.scatter(sub["mean_score"], sub["mean_tokens"] / 1000,
                   label=SA_LABELS[sa], color=SA_COLORS[sa], alpha=0.65, s=45)

    m, b, *_ = stats.linregress(task_stats["mean_score"], task_stats["mean_tokens"] / 1000)
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, m * xs + b, color="black", linewidth=1.5, linestyle="--", alpha=0.7, label=f"Trend (r={r:.2f})")

    ax.set_xlabel("Mean Score Across Models (task difficulty)", fontsize=11)
    ax.set_ylabel("Mean Total Tokens (thousands)", fontsize=11)
    ax.set_title(f"Task Difficulty vs Token Consumption\nSpearman r={r:.2f}, p={'<0.001' if p<0.001 else f'{p:.3f}'}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "task_cost_vs_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ task_cost_vs_difficulty.png")


# ── Chart 9: Provider verbosity distributions ────────────────────────────────
def chart_provider_verbosity(df: pd.DataFrame):
    df2 = df[df["output_tokens"] > 0].copy()
    providers = {
        "Anthropic": ["Claude Opus 4.6", "Claude Sonnet 4.6", "Claude Haiku 4.5"],
        "OpenAI":    ["GPT-5.4", "GPT-5.4 mini", "GPT-5.4 nano"],
        "Google":    ["Gemini 3.1 Pro Preview", "Gemini 2.5 Flash", "Gemini 3.1 Flash-Lite Preview", "Gemma 4 26B A4B"],
        "Others":    ["Qwen 3 Next 80B Thinking", "Qwen 3 Next 80B Instruct", "DeepSeek V3.2", "GLM-5"],
    }
    prov_colors = {"Anthropic": "#E65100", "OpenAI": "#1B5E20", "Google": "#1565C0", "Others": "#6A1B9A"}

    data_by_prov = {}
    for prov, models in providers.items():
        subset = df2[df2["model_display_name"].isin(models)]
        data_by_prov[prov] = np.log10(subset["output_tokens"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot([data_by_prov[p] for p in providers], positions=range(len(providers)),
                          showmedians=True, widths=0.7)
    for i, (pc_key, prov) in enumerate(zip(parts["bodies"], providers)):
        pc_key.set_facecolor(prov_colors[prov])
        pc_key.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(list(providers.keys()), fontsize=11)
    ax.set_ylabel("Log₁₀(Output Tokens)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"10^{v:.0f}"))
    ax.set_title("Output Token Distribution by Provider\n(violin = log scale)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "provider_verbosity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ provider_verbosity.png")


# ── Chart 10: Efficiency frontier (Pareto) ───────────────────────────────────
def chart_efficiency_frontier(df: pd.DataFrame):
    model_stats = (
        df.dropna(subset=["score"])
        .groupby("model_display_name")
        .agg(mean_score=("score", "mean"), total_cost=("cost_usd", "sum"))
        .reset_index()
    )
    model_stats["tier"] = model_stats["model_display_name"].map(MODEL_TIERS)

    # Pareto frontier
    ms_sorted = model_stats.sort_values("total_cost")
    pareto = []
    best_score = -1
    for _, row in ms_sorted.iterrows():
        if row["mean_score"] > best_score:
            best_score = row["mean_score"]
            pareto.append(row)
    pareto_df = pd.DataFrame(pareto)

    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in model_stats.iterrows():
        c = TIER_COLORS.get(row["tier"], "#888")
        ax.scatter(row["total_cost"], row["mean_score"], s=180, color=c,
                   alpha=0.85, edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(row["model_display_name"], (row["total_cost"], row["mean_score"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5, color="#333")

    if len(pareto_df) > 1:
        ax.step(pareto_df["total_cost"], pareto_df["mean_score"],
                where="post", color="#333", linewidth=1.5, linestyle="--",
                alpha=0.5, label="Pareto frontier", zorder=2)

    patches = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in ["Frontier", "Standard", "Efficient"]]
    ax.legend(handles=patches + [mpatches.Patch(color="#333", label="Pareto frontier")],
              loc="lower right", fontsize=9)
    ax.set_xlabel("Total Benchmark Cost (USD)", fontsize=11)
    ax.set_ylabel("Mean Score (0–1)", fontsize=11)
    ax.set_title("Efficiency Frontier: Quality vs Total Spend\n(all 131 tasks)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "efficiency_frontier.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ efficiency_frontier.png")


if __name__ == "__main__":
    print(f"Loading data from {CSV_PATH}...")
    df = load_data()
    print(f"Loaded {len(df):,} rows, {df['model_display_name'].nunique()} models, "
          f"{df['task_slug'].nunique()} tasks\n")

    apply_style()
    print("Generating charts...")
    chart_cost_efficiency_scatter(df)
    chart_score_per_dollar(df)
    chart_token_verbosity(df)
    chart_latency_vs_score(df)
    chart_token_waste_heatmap(df)
    chart_rf_token_quartiles(df)
    chart_hard_easy_tasks(df)
    chart_task_cost_vs_difficulty(df)
    chart_provider_verbosity(df)
    chart_efficiency_frontier(df)
    print(f"\nAll charts saved to {OUT_DIR}/")
