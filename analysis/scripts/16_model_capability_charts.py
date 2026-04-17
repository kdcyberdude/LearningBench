"""
Script 16: Model Performance by Learning Capability — Before & After Curation
Produces:
  - Fig 1: Radar chart (before)
  - Fig 2: Grouped bar chart (before)
  - Fig 3: Heatmap (before)
  - Fig 4: Same radar (after removing high+medium priority tasks)
  - Fig 5: Same grouped bar (after)
  - Fig 6: Side-by-side rank change table plot
  - Fig 7: Summary showing tasks-to-retain recommendation
All saved to analysis/outputs/charts/
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import seaborn as sns
from math import pi

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent.parent
OUT  = ROOT / "analysis" / "outputs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

SCORE_CSV = ROOT / "analysis" / "outputs" / "score_matrix.csv"
FLAGS_CSV = ROOT / "analysis" / "outputs" / "final_flagged_tasks.csv"

# ─── load data ────────────────────────────────────────────────────────────────
df    = pd.read_csv(SCORE_CSV)
flags = pd.read_csv(FLAGS_CSV)

# Ordered categories with readable labels
CAT_ORDER  = ["associative", "concept", "language", "observational", "rl"]
CAT_LABELS = {
    "associative":  "Associative\nLearning",
    "concept":      "Concept\nLearning",
    "language":     "Language\nLearning",
    "observational":"Observational\nLearning",
    "rl":           "Reinforcement\nLearning",
}

# Model display names & tier colours
TIER_COLOR = {
    "frontier": "#1a1a2e",
    "strong":   "#16213e",
    "mid":      "#0f3460",
    "budget":   "#533483",
}
MODEL_META = {
    "Gemini 3.1 Pro Preview":        {"short": "Gemini 3.1 Pro",     "tier": "frontier", "provider": "Google"},
    "GLM-5":                          {"short": "GLM-5",              "tier": "frontier", "provider": "Zhipu"},
    "Qwen 3 Next 80B Thinking":       {"short": "Qwen3 80B Think",   "tier": "strong",   "provider": "Alibaba"},
    "Claude Opus 4.6":                {"short": "Claude Opus 4.6",   "tier": "strong",   "provider": "Anthropic"},
    "Claude Sonnet 4.6":              {"short": "Claude Sonnet 4.6", "tier": "strong",   "provider": "Anthropic"},
    "Gemini 2.5 Flash":               {"short": "Gemini 2.5 Flash",  "tier": "strong",   "provider": "Google"},
    "GPT-5.4":                        {"short": "GPT-5.4",           "tier": "strong",   "provider": "OpenAI"},
    "DeepSeek V3.2":                  {"short": "DeepSeek V3.2",     "tier": "mid",      "provider": "DeepSeek"},
    "Gemini 3.1 Flash-Lite Preview":  {"short": "Gemini 3.1 F-Lite", "tier": "mid",      "provider": "Google"},
    "Claude Haiku 4.5":               {"short": "Claude Haiku 4.5",  "tier": "mid",      "provider": "Anthropic"},
    "Qwen 3 Next 80B Instruct":       {"short": "Qwen3 80B Inst",    "tier": "mid",      "provider": "Alibaba"},
    "GPT-5.4 mini":                   {"short": "GPT-5.4 mini",      "tier": "budget",   "provider": "OpenAI"},
    "Gemma 4 26B A4B":                {"short": "Gemma 4 26B",       "tier": "budget",   "provider": "Google"},
    "GPT-5.4 nano":                   {"short": "GPT-5.4 nano",      "tier": "budget",   "provider": "OpenAI"},
}

# Nice colour palette — one colour per model
MODEL_COLORS = {
    "Gemini 3.1 Pro Preview":        "#e63946",
    "GLM-5":                          "#f4a261",
    "Qwen 3 Next 80B Thinking":       "#2a9d8f",
    "Claude Opus 4.6":                "#457b9d",
    "Claude Sonnet 4.6":              "#1d3557",
    "Gemini 2.5 Flash":               "#f94144",
    "GPT-5.4":                        "#06d6a0",
    "DeepSeek V3.2":                  "#118ab2",
    "Gemini 3.1 Flash-Lite Preview":  "#ffd166",
    "Claude Haiku 4.5":               "#8338ec",
    "Qwen 3 Next 80B Instruct":       "#fb5607",
    "GPT-5.4 mini":                   "#3a86ff",
    "Gemma 4 26B A4B":                "#06a77d",
    "GPT-5.4 nano":                   "#adb5bd",
}

def get_pivot(data):
    """Return model × category mean score matrix, sorted by overall mean desc."""
    p = data.groupby(["model", "category"])["score"].mean().unstack()
    p = p[CAT_ORDER]
    p["overall"] = p.mean(axis=1)
    p = p.sort_values("overall", ascending=False)
    return p

def short(model): return MODEL_META.get(model, {}).get("short", model)

# ─── BUILD BEFORE / AFTER PIVOTS ─────────────────────────────────────────────
pivot_before = get_pivot(df)

# Remove high + medium priority tasks
remove_tasks = flags[flags["removal_priority"].isin(["high", "medium"])]["task_name"].tolist()
df_after = df[~df["task_name"].isin(remove_tasks)]
pivot_after = get_pivot(df_after)

n_before = df["task_name"].nunique()
n_after  = df_after["task_name"].nunique()
print(f"Tasks before: {n_before}, after: {n_after} (removed {n_before - n_after})")

# Per-category counts
cat_before = df.groupby("category")["task_name"].nunique()[CAT_ORDER]
cat_after  = df_after.groupby("category")["task_name"].nunique()[CAT_ORDER]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Radar chart (before curation)
# ═══════════════════════════════════════════════════════════════════════════════
def make_radar(pivot, title, filename, subtitle=""):
    cats = CAT_ORDER
    N = len(cats)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]           # close polygon

    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CAT_LABELS[c] for c in cats],
                       fontsize=11, fontweight="bold")

    # Y grid
    ax.set_rlabel_position(30)
    ax.yaxis.grid(True, color="grey", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
    ax.spines["polar"].set_visible(False)

    # Plot each model
    for model in pivot.index:
        vals = pivot.loc[model, cats].tolist()
        vals += vals[:1]
        color = MODEL_COLORS.get(model, "#888888")
        ax.plot(angles, vals, linewidth=1.8, linestyle="solid",
                label=short(model), color=color, alpha=0.85)
        ax.fill(angles, vals, alpha=0.05, color=color)

    # Legend
    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.42, 1.18),
        fontsize=8.5, frameon=True, framealpha=0.9,
        title="Models (ranked by overall mean)",
        title_fontsize=9
    )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color="#555555", pad=18)

    plt.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

make_radar(
    pivot_before.drop(columns="overall"),
    "Model Performance by Learning Capability — BEFORE Curation",
    "fig1_radar_before.png",
    f"All {n_before} tasks across 5 categories · scores = mean(0–1) per task"
)
make_radar(
    pivot_after.drop(columns="overall"),
    "Model Performance by Learning Capability — AFTER Curation",
    "fig4_radar_after.png",
    f"{n_after} retained tasks (removed {n_before - n_after} flagged) · scores = mean(0–1) per task"
)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Grouped bar chart (before) + FIGURE 5 (after)
# ═══════════════════════════════════════════════════════════════════════════════
def make_grouped_bar(pivot, title, filename, task_counts, subtitle=""):
    models = list(pivot.index)
    n_models = len(models)
    n_cats   = len(CAT_ORDER)
    x = np.arange(n_cats)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, model in enumerate(models):
        vals  = pivot.loc[model, CAT_ORDER].values
        xpos  = x - 0.4 + (i + 0.5) * width
        bars  = ax.bar(xpos, vals, width * 0.9,
                       color=MODEL_COLORS.get(model, "#888"),
                       alpha=0.85, label=short(model))

    # Category labels with task counts
    xlabels = [
        f"{CAT_LABELS[c].replace(chr(10), ' ')}\n(n={task_counts[c]})"
        for c in CAT_ORDER
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Score (0–1)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", pad=12)

    legend = ax.legend(
        loc="upper right", ncol=2, fontsize=8.5,
        frameon=True, framealpha=0.9,
        title="Models (ranked by overall mean)",
        title_fontsize=9
    )
    plt.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

make_grouped_bar(
    pivot_before.drop(columns="overall"),
    "Model Performance by Learning Capability — BEFORE Curation",
    "fig2_grouped_bar_before.png",
    cat_before,
    subtitle=f"All {n_before} tasks · sorted by overall rank (best → worst left to right within each group)"
)
make_grouped_bar(
    pivot_after.drop(columns="overall"),
    "Model Performance by Learning Capability — AFTER Curation",
    "fig5_grouped_bar_after.png",
    cat_after,
    subtitle=f"{n_after} retained tasks (removed {n_before - n_after} flagged)"
)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Heatmap (before + after, side by side)
# ═══════════════════════════════════════════════════════════════════════════════
def pivot_for_heatmap(pivot):
    p = pivot.drop(columns="overall").copy()
    p.index = [short(m) for m in p.index]
    p.columns = [CAT_LABELS[c].replace("\n", " ") for c in CAT_ORDER]
    return p

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, piv, label, task_n in [
    (axes[0], pivot_before, f"BEFORE — {n_before} tasks", cat_before),
    (axes[1], pivot_after,  f"AFTER — {n_after} tasks",  cat_after),
]:
    data = pivot_for_heatmap(piv)
    im = sns.heatmap(
        data, ax=ax, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 8.5}
    )
    ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Model", fontsize=10)
    # add task counts to column labels
    xlabs = [
        f"{CAT_LABELS[c].replace(chr(10), ' ')}\n(n={task_n[c]})"
        for c in CAT_ORDER
    ]
    ax.set_xticklabels(xlabs, rotation=15, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)

fig.suptitle(
    "Heatmap: Mean Score per Model × Category  (Before vs After Curation)",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
path = OUT / "fig3_heatmap_before_after.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Rank change table
# ═══════════════════════════════════════════════════════════════════════════════
rank_b = pivot_before["overall"].rank(ascending=False).astype(int)
rank_a = pivot_after["overall"].rank(ascending=False).astype(int)

rank_df = pd.DataFrame({
    "Model":          [short(m) for m in rank_b.index],
    "Score (before)": pivot_before["overall"].round(3).values,
    "Rank (before)":  rank_b.values,
    "Score (after)":  [pivot_after.loc[m, "overall"] if m in pivot_after.index else None
                       for m in rank_b.index],
    "Rank (after)":   [rank_a[m] if m in rank_a.index else None for m in rank_b.index],
}).sort_values("Rank (before)")
rank_df["Δ Rank"] = rank_df["Rank (before)"].values - rank_df["Rank (after)"].values.astype(int)

fig, ax = plt.subplots(figsize=(11, 7))
ax.axis("off")
col_labels = ["Model", "Score\n(before)", "Rank\n(before)", "Score\n(after)", "Rank\n(after)", "Δ Rank"]
table_data = [
    [
        r["Model"],
        f"{r['Score (before)']:.3f}",
        str(int(r["Rank (before)"])),
        f"{r['Score (after)']:.3f}",
        str(int(r["Rank (after)"])),
        f"{int(r['Δ Rank']):+d}" if r["Δ Rank"] != 0 else "—"
    ]
    for _, r in rank_df.iterrows()
]

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.3, 1.7)

# Colour header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2b2d42")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Colour rank change column
for i, (_, row) in enumerate(rank_df.iterrows(), start=1):
    delta = int(row["Δ Rank"])
    if delta > 0:
        tbl[i, 5].set_facecolor("#d4edda")
    elif delta < 0:
        tbl[i, 5].set_facecolor("#f8d7da")
    else:
        tbl[i, 5].set_facecolor("#f5f5f5")
    # Alternate row shading
    if i % 2 == 0:
        for j in range(5):
            tbl[i, j].set_facecolor("#f0f4f8")

ax.set_title(
    f"Overall Model Rank — Before ({n_before} tasks) vs After ({n_after} tasks) Curation\n"
    f"(Removed {n_before - n_after} flagged tasks: 4 high-priority + {n_before - n_after - 4} medium-priority)",
    fontsize=12, fontweight="bold", pad=10
)
plt.tight_layout()
path = OUT / "fig6_rank_change_table.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Task Retention Recommendation
# ═══════════════════════════════════════════════════════════════════════════════
RECOMMENDATIONS = {
    "associative": {
        "before": 20, "remove_high": 0, "remove_med": 0, "retain": 20,
        "rationale": (
            "Perfect category — zero flagged tasks.\n"
            "Excellent discrimination, diverse stimuli,\n"
            "consistent tier ordering. Keep all 20."
        ),
        "notes": ["Strongest discriminating category", "All tasks show positive r", "Retain 100%"],
    },
    "concept": {
        "before": 19, "remove_high": 0, "remove_med": 2, "retain": 17,
        "rationale": (
            "2 tasks flagged: hapax_prime (negative r=−0.26)\n"
            "and semantic_override (inverted tier gap).\n"
            "Both are measurement artefacts, not real signals.\n"
            "Remove both; keep 17."
        ),
        "notes": ["hapax_prime: negative discrimination", "semantic_override: inverted tiers", "Retain 89%"],
    },
    "language": {
        "before": 26, "remove_high": 0, "remove_med": 0, "retain": 26,
        "rationale": (
            "No flagged tasks — clean category.\n"
            "Tasks span phonology, syntax, semantics,\n"
            "analogy, morphology. Well-spread difficulty.\n"
            "Keep all 26."
        ),
        "notes": ["Second best overall discrimination", "Good tier spread", "Retain 100%"],
    },
    "observational": {
        "before": 42, "remove_high": 0, "remove_med": 3, "retain": 39,
        "rationale": (
            "3 tasks flagged: custom_gravity (negative r),\n"
            "vigenere_variant (near-zero scores / low entropy),\n"
            "voronoi_custom (negative r). Remove all 3.\n"
            "Largest category at 39 is still healthy."
        ),
        "notes": ["3 noisy tasks identified", "vigenere: too hard for all", "Retain 93%"],
    },
    "rl": {
        "before": 50, "remove_high": 4, "remove_med": 17, "retain": 29,
        "rationale": (
            "Most problematic category: 4 tasks score near-zero\n"
            "(all models fail — implementation bugs suspected),\n"
            "17 more show inverted tier gaps or bimodality.\n"
            "Removing 21 leaves 29 — still the largest category."
        ),
        "notes": ["4 broken tasks (all-zero)", "17 show inverted tier signals", "Retain 58%"],
    },
}

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 5, figure=fig, hspace=0.55, wspace=0.35)

cat_colors = {
    "associative":  "#2a9d8f",
    "concept":      "#e9c46a",
    "language":     "#f4a261",
    "observational":"#457b9d",
    "rl":           "#e76f51",
}

for col, cat in enumerate(CAT_ORDER):
    rec = RECOMMENDATIONS[cat]
    ax_bar  = fig.add_subplot(gs[0, col])
    ax_text = fig.add_subplot(gs[1, col])

    # Stacked bar: retain / remove_med / remove_high
    vals = [rec["retain"], rec["remove_med"], rec["remove_high"]]
    lbls = ["Retain", "Remove\n(medium)", "Remove\n(high)"]
    clrs = [cat_colors[cat], "#fcbf49", "#e63946"]

    bottom = 0
    for v, c, l in zip(vals, clrs, lbls):
        ax_bar.bar(0, v, bottom=bottom, color=c, width=0.5, label=l)
        if v > 0:
            ax_bar.text(0, bottom + v / 2, str(v),
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color="white")
        bottom += v

    ax_bar.set_xlim(-0.6, 0.6)
    ax_bar.set_ylim(0, max(50, rec["before"] + 5))
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("# Tasks", fontsize=9)
    pct = rec["retain"] / rec["before"] * 100
    ax_bar.set_title(
        f"{cat.upper()}\n{rec['retain']}/{rec['before']} retained ({pct:.0f}%)",
        fontsize=10, fontweight="bold"
    )

    # Text rationale
    ax_text.axis("off")
    ax_text.text(
        0.5, 1.0, rec["rationale"],
        transform=ax_text.transAxes,
        ha="center", va="top", fontsize=7.5,
        color="#2b2d42", wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=cat_colors[cat], alpha=0.15)
    )

# Overall summary bar at top
fig.suptitle(
    f"Task Retention Recommendation by Category\n"
    f"Before: {n_before} tasks  →  Recommended: {n_after} tasks  (Removed: {n_before - n_after})",
    fontsize=14, fontweight="bold", y=1.01
)

# Shared legend
patches = [
    mpatches.Patch(color="#2a9d8f", label="Retain"),
    mpatches.Patch(color="#fcbf49", label="Remove — medium priority"),
    mpatches.Patch(color="#e63946", label="Remove — high priority"),
]
fig.legend(handles=patches, loc="upper right",
           bbox_to_anchor=(1.12, 1.01), fontsize=9, frameon=True)

path = OUT / "fig7_retention_recommendation.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Print summary table to console
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("TASK RETENTION SUMMARY")
print("═"*70)
print(f"{'Category':<16} {'Before':>7} {'-High':>6} {'-Med':>5} {'Retain':>7} {'%':>6}")
print("─"*70)
total_before = total_high = total_med = total_retain = 0
for cat in CAT_ORDER:
    r = RECOMMENDATIONS[cat]
    total_before  += r["before"]
    total_high    += r["remove_high"]
    total_med     += r["remove_med"]
    total_retain  += r["retain"]
    pct = r["retain"] / r["before"] * 100
    print(f"{cat:<16} {r['before']:>7} {r['remove_high']:>6} {r['remove_med']:>5} {r['retain']:>7} {pct:>5.0f}%")
print("─"*70)
pct_total = total_retain / total_before * 100
print(f"{'TOTAL':<16} {total_before:>7} {total_high:>6} {total_med:>5} {total_retain:>7} {pct_total:>5.0f}%")
print("═"*70)

print("\n✅ All charts saved to:", OUT)
print("   fig1_radar_before.png")
print("   fig2_grouped_bar_before.png")
print("   fig3_heatmap_before_after.png")
print("   fig4_radar_after.png")
print("   fig5_grouped_bar_after.png")
print("   fig6_rank_change_table.png")
print("   fig7_retention_recommendation.png")
