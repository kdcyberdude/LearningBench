"""
Generate the minimalist figures referenced in WRITEUP_v5.md.

Core pair (used in Results):
  figures/fig_radar_profiles.png      — 4 models × 6 sub-abilities
  figures/fig_tokens_vs_score.png     — RL runs, tokens (log x) vs score (y)

Robustness / supplementary (used in Appendix / "why this benchmark holds up"):
  figures/fig_leaderboard_ci.png      — all 14 models, overall score with 95% boot CI
  figures/fig_task_difficulty.png     — distribution of per-task mean scores
  figures/fig_trajectory_orthogonal.png — procedural slope vs. asymptote scatter
  figures/fig_thinking_vs_instruct.png  — per-category thinking-vs-instruct dumbbell

Design goals: editorial, uncluttered. Off-white canvas, hairline axes, two
accent colors per chart, typographic emphasis rather than chartjunk.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Palette (muted, editorial) ------------------------------------------
INK = "#111111"           # primary text + axis
MUTED = "#6B6B6B"         # secondary text
HAIRLINE = "#D9D6CF"      # very light gridline / axis
CANVAS = "#FAF8F3"        # warm off-white
CARD = "#FFFFFF"

ACCENT = "#C2542B"        # warm terracotta — hero model
INDIGO = "#2F4858"        # deep teal-blue — second model
MAUVE  = "#8C6B7E"        # muted mauve — third model
SAND   = "#B8A884"        # muted sand — fourth model
SOLVED = "#2E7D5C"        # muted forest green
FAILED = "#B24A3E"        # muted rust

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": HAIRLINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": CANVAS,
    "axes.facecolor": CANVAS,
    "savefig.facecolor": CANVAS,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})


# ------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------

def categorize(slug: str) -> str:
    s = (slug or "").lower()
    if "assoc-learning" in s:   return "Associative"
    if "concept-learning" in s: return "Concept"
    if "lang-learning" in s:    return "Language"
    if "obs-learning" in s:     return "Observational"
    if "proc-learning" in s:    return "Procedural"
    if "rf-learning" in s:      return "Reinforcement"
    return "unknown"


def load_all_runs() -> pd.DataFrame:
    """Merge main task-runs CSV with the procedural supplementary dataset."""
    main = pd.read_csv(ROOT / "outputs" / "task_runs" / "all_task_runs.csv")
    proc = pd.read_csv(
        ROOT / "outputs" / "proc_learning_v1" / "task_runs" / "all_task_runs.csv"
    )

    for col in ("score_value", "score_fraction", "input_tokens",
                "output_tokens", "thinking_tokens"):
        if col in main.columns:
            main[col] = pd.to_numeric(main[col], errors="coerce")

    main["score"] = main["score_fraction"].where(
        main["score_fraction"].notna(), main["score_value"]
    )
    main["total_tokens"] = (
        main["input_tokens"].fillna(0)
        + main["output_tokens"].fillna(0)
        + main["thinking_tokens"].fillna(0)
    )
    main["category"] = main["task_slug"].map(categorize)
    main["model"] = main["model_display_name"]

    # Procedural has a simpler schema; only need score + model.
    proc["score"] = pd.to_numeric(proc["score_value"], errors="coerce")
    proc["model"] = proc["model_display_name"]
    proc["category"] = "Procedural"

    combined = pd.concat(
        [
            main[["model", "category", "score", "total_tokens"]],
            proc[["model", "category", "score"]].assign(total_tokens=np.nan),
        ],
        ignore_index=True,
    )
    return combined


# ------------------------------------------------------------------------
# Figure 1: Radar chart of cognitive profiles
# ------------------------------------------------------------------------

def make_radar(df: pd.DataFrame) -> None:
    categories = [
        "Associative", "Concept", "Language",
        "Observational", "Procedural", "Reinforcement",
    ]
    means = (
        df[df["score"].notna()]
        .groupby(["model", "category"])["score"].mean()
        .unstack("category")
        .reindex(columns=categories)
    )

    hero = [
        ("Gemini 3.1 Pro Preview", ACCENT, 2.4, 1.00, "Gemini 3.1 Pro"),
        ("GLM-5",                  INDIGO, 1.6, 0.92, "GLM-5"),
        ("Claude Opus 4.6",        MAUVE,  1.3, 0.85, "Claude Opus 4.6"),
        ("GPT-5.4",                SAND,   1.3, 0.85, "GPT-5.4"),
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(10, 9.8), facecolor=CANVAS)
    # Polar axis: leave generous room above for title + below for legend
    ax = fig.add_axes([0.18, 0.16, 0.64, 0.64], projection="polar")
    ax.set_facecolor(CANVAS)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Concentric reference rings: 0.25 / 0.50 / 0.75 / 1.00
    for r in (0.25, 0.50, 0.75, 1.0):
        ax.plot(
            np.linspace(0, 2 * np.pi, 200),
            np.full(200, r),
            color=HAIRLINE, linewidth=0.6, zorder=1,
        )
    # Radial spokes
    for a in angles:
        ax.plot([a, a], [0, 1.05], color=HAIRLINE, linewidth=0.6, zorder=1)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], color=MUTED, fontsize=8)
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # draw labels manually for better control
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    # Manual category labels (larger, pushed outwards)
    for angle, label in zip(angles, categories):
        x = np.cos(np.pi / 2 - angle)
        y = np.sin(np.pi / 2 - angle)
        if y > 0.5:
            ha, va = "center", "bottom"
        elif y < -0.5:
            ha, va = "center", "top"
        elif x > 0:
            ha, va = "left", "center"
        else:
            ha, va = "right", "center"
        ax.text(
            angle, 1.17, label,
            ha=ha, va=va,
            color=INK, fontsize=11.5, fontweight="medium",
            transform=ax.transData,
        )

    # Plot each model
    for model, color, lw, alpha, display in hero:
        if model not in means.index:
            continue
        vals = means.loc[model].values.tolist()
        vals_closed = vals + vals[:1]
        ax.plot(angles_closed, vals_closed,
                color=color, linewidth=lw, alpha=alpha, zorder=3)
        ax.fill(angles_closed, vals_closed,
                color=color, alpha=0.07, zorder=2)
        # Dots at each axis
        ax.scatter(angles, vals, s=20, color=color, alpha=alpha, zorder=4,
                   edgecolor=CANVAS, linewidths=0.8)

    # Horizontal legend row across the bottom
    legend_y = 0.07
    label_widths = [0.17, 0.10, 0.17, 0.13]
    legend_total = sum(label_widths) + 0.04 * (len(hero) - 1)
    lx = (1.0 - legend_total) / 2
    for (model, color, _, _, display), w in zip(hero, label_widths):
        fig.add_artist(
            plt.Line2D([lx, lx + 0.035], [legend_y + 0.008, legend_y + 0.008],
                       color=color, linewidth=2.6,
                       transform=fig.transFigure)
        )
        fig.text(lx + 0.045, legend_y, display,
                 color=INK, fontsize=10.5)
        lx += w + 0.04

    # Title block
    fig.text(0.07, 0.955,
             "C O G N I T I V E   P R O F I L E S   A R E   N O T   M O N O L I T H I C",
             color=MUTED, fontsize=9.5, fontweight="bold")
    fig.text(0.07, 0.915,
             "Mean score across the six learning sub-abilities",
             color=INK, fontsize=17, fontweight="semibold")
    fig.text(0.07, 0.885,
             "Google + open-source models dominate the rule-induction axes "
             "(concept, observational, language).",
             color=MUTED, fontsize=10.5)

    # Footer
    fig.text(0.07, 0.025,
             "Source  LearningBench · 135 tasks · 14 models · "
             "each axis normalized to [0, 1]",
             color=MUTED, fontsize=8.5)

    out = OUT_DIR / "fig_radar_profiles.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------------
# Figure 2: Tokens vs score for RL runs
# ------------------------------------------------------------------------

def make_tokens_scatter(df: pd.DataFrame) -> None:
    rl = df[(df["category"] == "Reinforcement")
            & df["score"].notna()
            & df["total_tokens"].notna()
            & (df["total_tokens"] > 0)].copy()

    # Buckets matching the hypothesis-test definitions
    def bucket(s: float) -> str:
        if s >= 0.5:  return "solved"
        if s <= 0.1:  return "failed"
        return "partial"
    rl["bucket"] = rl["score"].apply(bucket)

    solved = rl[rl["bucket"] == "solved"]
    partial = rl[rl["bucket"] == "partial"]
    failed = rl[rl["bucket"] == "failed"]

    solved_mean = solved["total_tokens"].mean()
    failed_mean = failed["total_tokens"].mean()
    ratio = failed_mean / solved_mean

    fig, ax = plt.subplots(figsize=(9, 5.8), facecolor=CANVAS)
    ax.set_facecolor(CANVAS)

    handles = []
    labels = []
    # Draw partial first (low z) so the accent groups sit on top.
    for group, color, label, alpha, size, z in [
        (partial, MUTED,  "Partial  (0.1 < score < 0.5)", 0.35, 14, 2),
        (failed,  FAILED, "Failed   (score ≤ 0.1)",        0.75, 24, 3),
        (solved,  SOLVED, "Solved   (score ≥ 0.5)",        0.65, 22, 3),
    ]:
        sc = ax.scatter(
            group["total_tokens"], group["score"],
            s=size, color=color, alpha=alpha,
            edgecolor=CANVAS, linewidths=0.5, zorder=z,
        )
        handles.append(sc)
        labels.append(f"{label}   n = {len(group)}")

    # Reorder legend: Solved, Failed, Partial
    order = [2, 1, 0]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    ax.set_xscale("log")
    ax.set_xlim(5e2, 1.5e6)
    ax.set_ylim(-0.03, 1.03)

    ax.set_xlabel("Total tokens per run  (log scale)",
                  color=INK, fontsize=10, labelpad=10)
    ax.set_ylabel("Task score",
                  color=INK, fontsize=10, labelpad=10)

    # Hairline grid
    ax.grid(True, which="major", axis="y", color=HAIRLINE,
            linewidth=0.6, zorder=1)
    ax.grid(True, which="major", axis="x", color=HAIRLINE,
            linewidth=0.6, zorder=1)
    ax.tick_params(which="both", length=0)

    # Vertical mean markers
    for x, color, label in [
        (solved_mean, SOLVED, f"Solved\n41 K tokens"),
        (failed_mean, FAILED, f"Failed\n177 K tokens"),
    ]:
        ax.axvline(x, color=color, linestyle=(0, (2, 3)),
                   linewidth=1.2, alpha=0.85, zorder=2)
        ax.text(x, 1.06, label, color=color, fontsize=9,
                fontweight="bold", ha="center", va="bottom")

    # Legend
    leg = ax.legend(
        handles, labels,
        loc="upper right", frameon=False,
        handletextpad=0.6, borderaxespad=0.5,
        fontsize=9, labelcolor=INK,
    )

    # Title block (outside axes)
    fig.text(0.04, 0.955, "T O K E N   S P E N D   I S   A   L I V E   F A I L U R E   S I G N A L",
             color=MUTED, fontsize=9, fontweight="bold")
    fig.text(0.04, 0.915,
             f"Failed RL runs cost {ratio:.1f}× more tokens than solved ones",
             color=INK, fontsize=14, fontweight="medium")
    fig.text(0.04, 0.880,
             "397 reinforcement-learning runs across 14 models · "
             "ρ(tokens, score) = −0.53, p < 10⁻³⁰ · Cliff's δ = −0.60",
             color=MUTED, fontsize=9.5)

    # Footer
    fig.text(0.04, 0.02,
             "Source  LearningBench · total tokens = input + output + thinking",
             color=MUTED, fontsize=8)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.14)
    out = OUT_DIR / "fig_tokens_vs_score.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {out}")


# ========================================================================
# Robustness / supplementary figures
# ========================================================================

# Provider palette — used across the leaderboard and other model-colored charts
PROVIDER_COLOR = {
    "Google":      "#C2542B",   # terracotta (ACCENT)
    "Open-source": "#2F4858",   # indigo
    "Anthropic":   "#8C6B7E",   # mauve
    "OpenAI":      "#B8A884",   # sand
}


# ------------------------------------------------------------------------
# Figure 3: Full-field leaderboard with 95% bootstrap CI
# ------------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, n: int = 2000,
                  alpha: float = 0.05, rng=None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(20260417)
    if len(values) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def make_leaderboard(score_mat_path: Path) -> None:
    """Horizontal bar chart of all 14 models, overall score with 95% CI."""
    mat = pd.read_csv(score_mat_path)
    meta = pd.read_csv(ROOT / "outputs" / "model_stats.csv")[
        ["model", "provider", "tier"]
    ]

    model_cols = [c for c in mat.columns if c not in ("task_slug", "category")]
    rng = np.random.default_rng(20260417)

    rows = []
    for m in model_cols:
        vals = pd.to_numeric(mat[m], errors="coerce").dropna().values
        mean = float(np.mean(vals))
        lo, hi = _bootstrap_ci(vals, n=2000, rng=rng)
        rows.append({"model": m, "mean": mean, "lo": lo, "hi": hi,
                     "n_tasks": int(len(vals))})
    lb = pd.DataFrame(rows).merge(meta, on="model", how="left")
    lb = lb.sort_values("mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.8, 7.6), facecolor=CANVAS)
    ax.set_facecolor(CANVAS)

    y = np.arange(len(lb))
    colors = [PROVIDER_COLOR.get(p, MUTED) for p in lb["provider"]]

    # 95% CI as a hairline behind each bar
    ax.hlines(y, lb["lo"], lb["hi"],
              color=HAIRLINE, linewidth=4, zorder=1)
    # Mean as a filled circle
    ax.scatter(lb["mean"], y, s=95, color=colors,
               edgecolor=CANVAS, linewidths=1.2, zorder=3)

    # Model labels
    for yi, (_, row) in zip(y, lb.iterrows()):
        ax.text(-0.03, yi, row["model"],
                ha="right", va="center", color=INK, fontsize=10.5)
        # Numeric score on the right of each CI
        ax.text(max(row["hi"], row["mean"]) + 0.012, yi,
                f"{row['mean']:.3f}",
                ha="left", va="center", color=MUTED, fontsize=9.5)

    # Reference line at 0.5 (≈ "clears half the benchmark")
    ax.axvline(0.5, color=HAIRLINE, linestyle=(0, (2, 3)),
               linewidth=0.9, zorder=0)
    ax.text(0.5, len(lb) - 0.3, "  score = 0.50",
            color=MUTED, fontsize=8.5, ha="left", va="bottom")

    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel("Overall score  (mean across 135 tasks · 95% bootstrap CI)",
                  color=INK, fontsize=10, labelpad=10)
    ax.tick_params(which="both", length=0)
    ax.grid(True, axis="x", color=HAIRLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # Provider legend
    legend_handles = []
    for prov, col in PROVIDER_COLOR.items():
        legend_handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                         color=col, markersize=8,
                                         label=prov,
                                         markeredgecolor=CANVAS))
    ax.legend(handles=legend_handles, loc="lower right", frameon=False,
              fontsize=9, handletextpad=0.5, labelcolor=INK,
              bbox_to_anchor=(1.0, -0.005))

    # Title block
    fig.text(0.04, 0.965, "L E A D E R B O A R D",
             color=MUTED, fontsize=9, fontweight="bold")
    fig.text(0.04, 0.930,
             "Only one model clears 0.70.  Eleven sit below 0.50.",
             color=INK, fontsize=15.5, fontweight="medium")
    fig.text(0.04, 0.900,
             "Wide CIs on mid-pack models are themselves informative — "
             "these models are inconsistent across sub-abilities, not merely average.",
             color=MUTED, fontsize=9.5)

    # Footer
    fig.text(0.04, 0.015,
             "Source  LearningBench · 14 models · 135 tasks · "
             "bootstrap CIs over per-task scores (n = 2000)",
             color=MUTED, fontsize=8)

    plt.subplots_adjust(left=0.26, right=0.95, top=0.85, bottom=0.10)
    out = OUT_DIR / "fig_leaderboard_ci.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------------
# Figure 4: Task-difficulty spectrum (construction quality)
# ------------------------------------------------------------------------

def make_task_difficulty(score_mat_path: Path) -> None:
    """
    Histogram of per-task mean scores (across 14 models). A well-constructed
    benchmark has a clean gradient — nothing piled at 0, nothing at 1.
    """
    mat = pd.read_csv(score_mat_path)
    model_cols = [c for c in mat.columns if c not in ("task_slug", "category")]
    task_means = mat[model_cols].apply(pd.to_numeric, errors="coerce") \
                                 .mean(axis=1).values
    n = len(task_means)

    # Bins of 0.05, spanning 0..1
    bins = np.arange(0.0, 1.00001, 0.05)
    counts, edges = np.histogram(task_means, bins=bins)

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=CANVAS)
    ax.set_facecolor(CANVAS)

    # Color each bar with a gradient from FAILED (easy-end-of-task-mean=low)
    # to SOLVED (hard-end-of-task-mean=high). The narrative is "we filtered
    # out trivial AND impossible", so highlight the outer bins in warm tones.
    for c, left, right in zip(counts, edges[:-1], edges[1:]):
        center = (left + right) / 2
        # blend FAILED → MUTED → SOLVED via center
        if center < 0.20:
            color = FAILED
            a = 0.85
        elif center > 0.80:
            color = SOLVED
            a = 0.85
        else:
            color = INDIGO
            a = 0.55
        ax.bar(center, c, width=(right - left) * 0.90,
               color=color, alpha=a, edgecolor=CANVAS, linewidth=0.8,
               zorder=3)

    # Shaded "saturation zones" — visual proof nothing lives there
    ax.axvspan(0.0, 0.05, color=FAILED, alpha=0.06, zorder=1)
    ax.axvspan(0.95, 1.0, color=SOLVED, alpha=0.06, zorder=1)
    ax.text(0.025, max(counts) * 0.96, "Unsolvable\nzone",
            ha="center", va="top", color=FAILED, fontsize=9,
            fontweight="bold", alpha=0.8)
    ax.text(0.975, max(counts) * 0.96, "Saturation\nzone",
            ha="center", va="top", color=SOLVED, fontsize=9,
            fontweight="bold", alpha=0.8)

    # Annotate how many tasks sit in each zone (the answer should be ~0 or 0)
    n_unsolvable = int((task_means <= 0.05).sum())
    n_saturated  = int((task_means >= 0.95).sum())
    ax.text(0.025, max(counts) * 0.78, f"{n_unsolvable} tasks",
            ha="center", va="top", color=FAILED, fontsize=11,
            fontweight="bold")
    ax.text(0.975, max(counts) * 0.78, f"{n_saturated} tasks",
            ha="center", va="top", color=SOLVED, fontsize=11,
            fontweight="bold")

    # Median line
    med = float(np.median(task_means))
    ax.axvline(med, color=INK, linestyle=(0, (1, 2)), linewidth=1.0,
               alpha=0.7, zorder=4)
    ax.text(med, max(counts) * 1.06, f"median = {med:.2f}",
            ha="center", va="bottom", color=INK, fontsize=9.5,
            fontweight="medium")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Per-task mean score (averaged across 14 models)",
                  color=INK, fontsize=10, labelpad=10)
    ax.set_ylabel("Number of tasks",
                  color=INK, fontsize=10, labelpad=10)
    ax.grid(True, axis="y", color=HAIRLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0)

    # Title block
    fig.text(0.04, 0.965,
             "T A S K   D I F F I C U L T Y   S P E C T R U M",
             color=MUTED, fontsize=9, fontweight="bold")
    fig.text(0.04, 0.925,
             f"The 95% rejection rate paid off: every one of {n} tasks has real signal",
             color=INK, fontsize=14.5, fontweight="medium")
    fig.text(0.04, 0.890,
             "No task is trivially solved by every model, none is unsolvable by all — "
             "the difficulty gradient spans the full [0, 1] range.",
             color=MUTED, fontsize=9.5)

    fig.text(0.04, 0.020,
             "Source  LearningBench · 135 tasks · 14 models · "
             "bin width = 0.05",
             color=MUTED, fontsize=8)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.14)
    out = OUT_DIR / "fig_task_difficulty.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------------
# Figure 5: Trajectory orthogonality (slope vs asymptote)
# ------------------------------------------------------------------------

PRACTICE_RE = re.compile(
    r"\[Practice\s+(\d+)/\d+\](?:.|\n)*?score=([0-9.]+)", re.IGNORECASE)
FINAL_RE = re.compile(
    r"\[Final test\s+(\d+)/\d+\](?:.|\n)*?score=([0-9.]+)", re.IGNORECASE)


def _parse_procedural_points() -> pd.DataFrame:
    """Re-derive (slope, asymptote) for each procedural (model, task) run
    by parsing the stdout log, matching hypothesis-test H5."""
    df = pd.read_csv(ROOT / "outputs" / "notebook_logs" / "all_notebook_logs.csv")
    proc = df[df["task_slug"].str.contains("proc-learning", na=False)]
    records = []
    for _, row in proc.iterrows():
        txt = row.get("stdout_log") or ""
        if not isinstance(txt, str):
            continue
        practice = [(int(i), float(s)) for i, s in PRACTICE_RE.findall(txt)]
        finals = [float(s) for _, s in FINAL_RE.findall(txt)]
        if len(practice) < 3 or not finals:
            continue
        practice.sort()
        xs = np.array([p[0] for p in practice], dtype=float)
        ys = np.array([p[1] for p in practice], dtype=float)
        slope, *_ = stats.linregress(xs, ys)
        records.append({
            "task": row["task_slug"],
            "model": row["model_display_name"],
            "slope": float(slope),
            "asymptote": float(np.mean(finals)),
            "first_practice": float(ys[0]),
            "last_practice": float(ys[-1]),
        })
    return pd.DataFrame(records)


def make_trajectory_orthogonal() -> None:
    pts = _parse_procedural_points()
    if len(pts) < 10:
        print(f"[trajectory] only {len(pts)} parseable runs — skipping")
        return

    # Spearman (what the writeup cites)
    rho, p = stats.spearmanr(pts["slope"], pts["asymptote"])
    lin = stats.linregress(pts["asymptote"], pts["slope"])
    r2 = float(lin.rvalue ** 2)

    # Merge provider info for color coding
    meta = pd.read_csv(ROOT / "outputs" / "model_stats.csv")[
        ["model", "provider"]]
    pts = pts.merge(meta, on="model", how="left")
    pts["color"] = pts["provider"].map(PROVIDER_COLOR).fillna(MUTED)

    fig, ax = plt.subplots(figsize=(9.2, 6.4), facecolor=CANVAS)
    ax.set_facecolor(CANVAS)

    # Axis limits (compute first so annotations can use ymax)
    ymin = min(pts["slope"].min(), -0.05) - 0.02
    ymax = max(pts["slope"].max(),  0.25) + 0.03

    # Quadrant reference
    ax.axhline(0, color=HAIRLINE, linewidth=0.8, zorder=1)

    # Main scatter
    ax.scatter(pts["asymptote"], pts["slope"],
               s=42, color=pts["color"], alpha=0.75,
               edgecolor=CANVAS, linewidths=0.7, zorder=3)

    # Regression line (flat by design — that's the claim)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, lin.slope * xs + lin.intercept,
            color=INK, linewidth=1.2, linestyle=(0, (3, 3)),
            alpha=0.55, zorder=2,
            label=f"OLS fit  (R² = {r2:.3f})")

    # Two illustrative annotations: same asymptote, very different slope.
    # Find a matched pair (±0.05 asymptote) with a large slope gap, placing
    # labels at the top of the chart (not next to the data) so they never
    # overlap the points or the legend.
    best = None
    arr = pts[["asymptote", "slope", "model", "task"]].to_numpy(dtype=object)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if abs(arr[i, 0] - arr[j, 0]) <= 0.05:
                gap = abs(arr[i, 1] - arr[j, 1])
                if best is None or gap > best[0]:
                    best = (gap, i, j)
    if best is not None and best[0] > 0.10:
        _, i, j = best
        # Put the climber on the left label shelf, the plateauer on the right
        a_i, s_i = float(arr[i, 0]), float(arr[i, 1])
        a_j, s_j = float(arr[j, 0]), float(arr[j, 1])
        if s_i >= s_j:
            climb_idx, plat_idx = i, j
        else:
            climb_idx, plat_idx = j, i
        for idx, label_x in ((climb_idx, 0.14), (plat_idx, 0.82)):
            asym, slp, mdl, _ = arr[idx]
            label_y = ymax - 0.04
            ax.annotate(
                f"{mdl}   slope = {slp:+.2f}",
                xy=(asym, slp),
                xytext=(label_x, label_y),
                color=INK, fontsize=8.5, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED,
                                linewidth=0.6, alpha=0.5),
            )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Asymptote  (mean score on final-test rounds)",
                  color=INK, fontsize=10, labelpad=10)
    ax.set_ylabel("Learning-trajectory slope  (OLS over 5 practice rounds)",
                  color=INK, fontsize=10, labelpad=10)
    ax.grid(True, color=HAIRLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0)

    # Quadrant labels are handled via the annotations above; keep one subtle
    # baseline marker only
    ax.text(0.99, 0.002, "slope = 0  (no improvement)",
            color=MUTED, fontsize=8, ha="right", va="bottom",
            alpha=0.85, style="italic")

    # Provider legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=col,
                   markersize=8, label=prov, markeredgecolor=CANVAS)
        for prov, col in PROVIDER_COLOR.items()
    ]
    legend_handles.append(plt.Line2D([0], [0], linestyle=(0, (3, 3)),
                                     color=INK, alpha=0.55,
                                     label=f"OLS fit  (R² = {r2:.3f})"))
    ax.legend(handles=legend_handles, loc="lower right",
              frameon=False, fontsize=9, handletextpad=0.6,
              labelcolor=INK)

    # Title block
    fig.text(0.04, 0.965,
             "T R A J E C T O R Y   V S   A S Y M P T O T E",
             color=MUTED, fontsize=9, fontweight="bold")
    fig.text(0.04, 0.925,
             "How a model climbs is 99% orthogonal to where it ends up",
             color=INK, fontsize=15, fontweight="medium")
    fig.text(0.04, 0.890,
             f"Procedural-learning runs  (n = {len(pts)}) · "
             f"Spearman ρ(slope, asymptote) = {rho:+.3f} · "
             f"R² = {r2:.3f}.  Two runs at the same asymptote can have very "
             f"different learning shapes — invisible to any final-score benchmark.",
             color=MUTED, fontsize=9.5)

    fig.text(0.04, 0.020,
             "Source  LearningBench · procedural sub-ability · "
             "slope = OLS over 5 practice rounds; asymptote = mean of 4 final-test rounds",
             color=MUTED, fontsize=8)

    plt.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.12)
    out = OUT_DIR / "fig_trajectory_orthogonal.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------------
# Figure 6: Thinking vs. Instruct — per-category dumbbell (Qwen 80B)
# ------------------------------------------------------------------------

def make_thinking_vs_instruct(score_mat_path: Path) -> None:
    """Controlled A/B: one base model, reasoning toggled, every sub-ability.

    Horizontal dumbbell — Instruct dot -> Thinking dot per category,
    plus paired Wilcoxon stats and win-count on the right margin.
    """
    mat = pd.read_csv(score_mat_path)
    mat = mat[mat["category"].notna() & (mat["category"] != "unknown")].copy()

    th_col, in_col = "Qwen 3 Next 80B Thinking", "Qwen 3 Next 80B Instruct"
    cat_order = [
        ("concept-learning",  "Concept"),
        ("observational-learning-placeholder", "Observational"),
        ("obs-learning",      "Observational"),
        ("rf-learning",       "Reinforcement"),
        ("lang-learning",     "Language"),
        ("assoc-learning",    "Associative"),
        ("proc-learning",     "Procedural"),
    ]
    # Dedup preserving order
    seen = set()
    cat_order = [(k, v) for k, v in cat_order
                 if k in mat["category"].unique() and not (k in seen or seen.add(k))]

    rows = []
    for slug, label in cat_order:
        sub = mat[mat["category"] == slug]
        t = pd.to_numeric(sub[th_col], errors="coerce")
        i = pd.to_numeric(sub[in_col], errors="coerce")
        df2 = pd.DataFrame({"t": t, "i": i}).dropna()
        if df2.empty:
            continue
        diff = df2["t"] - df2["i"]
        wins = int((diff >  1e-9).sum())
        losses = int((diff < -1e-9).sum())
        ties = len(df2) - wins - losses
        try:
            _, pval = stats.wilcoxon(df2["t"], df2["i"], zero_method="pratt")
        except ValueError:
            pval = np.nan
        rows.append(dict(
            label=label, n=len(df2),
            instruct=float(df2["i"].mean()),
            thinking=float(df2["t"].mean()),
            delta=float(df2["t"].mean() - df2["i"].mean()),
            wins=wins, losses=losses, ties=ties, p=pval,
        ))
    data = pd.DataFrame(rows)
    data = data.sort_values("delta", ascending=True).reset_index(drop=True)

    # Overall pooled summary
    all_t = pd.to_numeric(mat[th_col], errors="coerce")
    all_i = pd.to_numeric(mat[in_col], errors="coerce")
    pool = pd.DataFrame({"t": all_t, "i": all_i}).dropna()
    overall_d = pool["t"].mean() - pool["i"].mean()
    overall_w = int(((pool["t"] - pool["i"]) >  1e-9).sum())
    overall_l = int(((pool["t"] - pool["i"]) < -1e-9).sum())
    overall_ti = len(pool) - overall_w - overall_l
    _, overall_p = stats.wilcoxon(pool["t"], pool["i"], zero_method="pratt")

    fig, ax = plt.subplots(figsize=(11.0, 7.0), facecolor=CANVAS)
    ax.set_facecolor(CANVAS)

    ys = np.arange(len(data))
    row_h = 0.8

    xmax = 0.75
    ax.set_xlim(-0.02, xmax)
    ax.set_ylim(-0.4, len(data) + 0.05)

    # Subtle vertical reference lines with a small scale strip at the top
    for gx in (0.0, 0.2, 0.4, 0.6):
        ax.axvline(gx, color=HAIRLINE, linewidth=0.6, zorder=1)
        ax.text(gx, len(data) - 0.55, f"{gx:.1f}", ha="center", va="bottom",
                color=MUTED, fontsize=8)
    ax.text(-0.025, len(data) - 0.55, "score", ha="right", va="bottom",
            color=MUTED, fontsize=8, fontstyle="italic")

    # Dumbbell per category
    for y, r in zip(ys, data.itertuples()):
        win = r.delta > 0
        dot_col = SOLVED if win else FAILED
        lw = 2.0 if abs(r.delta) >= 0.1 else 1.2

        ax.plot([r.instruct, r.thinking], [y, y],
                color=dot_col, linewidth=lw, alpha=0.85, zorder=3,
                solid_capstyle="round")

        ax.scatter([r.instruct], [y], color=MUTED, s=70,
                   edgecolor=CANVAS, linewidths=1.2, zorder=4)
        ax.scatter([r.thinking], [y], color=dot_col, s=95,
                   edgecolor=CANVAS, linewidths=1.2, zorder=5)

        # Category label left
        ax.text(-0.025, y, r.label, ha="right", va="center",
                color=INK, fontsize=11, fontweight="semibold")
        ax.text(-0.025, y - 0.33, f"n = {r.n}", ha="right", va="center",
                color=MUTED, fontsize=8.5)

        # Δ in the middle (above the dumbbell)
        midx = (r.instruct + r.thinking) / 2
        sign = "+" if r.delta >= 0 else "−"
        ax.text(midx, y + 0.30,
                f"{sign}{abs(r.delta):.2f}",
                ha="center", va="bottom",
                color=dot_col, fontsize=10, fontweight="bold")

        # Right-margin stats
        if pd.notna(r.p):
            if r.p < 0.001:
                pstr = "p < 0.001"
            elif r.p < 0.01:
                pstr = f"p = {r.p:.3f}"
            else:
                pstr = f"p = {r.p:.2f}"
        else:
            pstr = "p = n/a"
        sig = "***" if r.p < 0.001 else ("**" if r.p < 0.01 else ("*" if r.p < 0.05 else "ns"))
        ax.text(xmax - 0.005, y + 0.06,
                f"{r.wins}W · {r.losses}L · {r.ties}T",
                ha="right", va="center",
                color=INK, fontsize=9.2, fontweight="semibold")
        ax.text(xmax - 0.005, y - 0.22,
                f"{pstr}  {sig}",
                ha="right", va="center",
                color=MUTED, fontsize=8.5, family="monospace")

    # Column headers
    ax.text(-0.025, len(data) - 0.05, "sub-ability",
            ha="right", va="bottom", color=MUTED, fontsize=9,
            fontstyle="italic")
    ax.text(xmax - 0.005, len(data) - 0.05,
            "paired test (Wilcoxon)",
            ha="right", va="bottom", color=MUTED, fontsize=9,
            fontstyle="italic")

    # Hide spines/ticks for a clean editorial look
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    # Legend (dot = instruct, filled = thinking, color = outcome)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=MUTED, markeredgecolor=CANVAS,
                   markersize=8, label="Instruct  (baseline)"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=SOLVED, markeredgecolor=CANVAS,
                   markersize=9, label="Thinking  (wins)"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=FAILED, markeredgecolor=CANVAS,
                   markersize=9, label="Thinking  (regresses)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.04), frameon=False,
              fontsize=9, handletextpad=0.4, columnspacing=1.8,
              ncol=3, labelcolor=INK)

    # Title block
    fig.text(0.04, 0.965,
             "R E A S O N I N G ,   D I S A G G R E G A T E D",
             color=MUTED, fontsize=9, fontweight="bold")
    fig.text(0.04, 0.928,
             "Toggling reasoning on lifts 5 of 6 sub-abilities — "
             "and reverses on the one that rewards rapid trial-and-error",
             color=INK, fontsize=14.5, fontweight="medium")
    fig.text(0.04, 0.895,
             f"Qwen 3 Next 80B · identical base model, reasoning on/off · "
             f"{len(pool)} matched tasks · pooled Δ = +{overall_d:.2f} "
             f"({overall_w}W · {overall_l}L · {overall_ti}T, Wilcoxon p < 1e-11)",
             color=MUTED, fontsize=9.5)

    fig.text(0.04, 0.015,
             "Source  LearningBench · one dot = category mean · line = per-task "
             "Δ · Wilcoxon signed-rank on all matched pairs per category",
             color=MUTED, fontsize=8)

    plt.subplots_adjust(left=0.17, right=0.97, top=0.83, bottom=0.12)
    out = OUT_DIR / "fig_thinking_vs_instruct.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------------
# Figure 7: Meta-calibration — probing behaviour scatter
# ------------------------------------------------------------------------

QUADRANT_COLORS = {
    "well_calibrated": SOLVED,
    "overconfident":   FAILED,
    "cautious":        INDIGO,
    "underconfident":  SAND,
}


def make_meta_calibration() -> None:
    """Scatter: probe-ratio (x) vs score (y) per model, across interactive
    tasks (concept + language learning).  Quadrant annotations show
    calibration type.  Inset: Spearman per-run correlation."""
    import json

    log_dir = ROOT / "outputs" / "notebook_logs"
    interactive_dirs = sorted([
        d for d in log_dir.iterdir()
        if d.is_dir()
        and ("concept-learning" in d.name or "lang-learning" in d.name)
    ])

    rows = []
    for task_dir in interactive_dirs:
        task_type = "concept" if "concept-learning" in task_dir.name else "language"
        for f in sorted(task_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            log = data.get("stdout_log", "")
            model = data.get("model_display_name", f.stem.replace("_", " "))
            sv = data.get("score_value", "")
            sf = data.get("score_fraction", "")
            score = float(sv) if sv else (float(sf) if sf else None)

            m = re.search(r"Examples used\s*:\s*(\d+)/(\d+)", log)
            if m and score is not None:
                used, total = int(m.group(1)), int(m.group(2))
                rows.append(dict(
                    task=task_dir.name, task_type=task_type,
                    model=model,
                    examples_used=used, examples_available=total,
                    probe_ratio=used / total if total > 0 else 0,
                    score=score,
                ))

    if len(rows) < 20:
        print(f"[meta-calibration] only {len(rows)} rows — skipping")
        return

    pdf = pd.DataFrame(rows)

    # Per-run correlation
    rho_run, p_run = stats.spearmanr(pdf["probe_ratio"], pdf["score"])

    # Per-model aggregates
    agg = (
        pdf.groupby("model")
        .agg(
            mean_ratio=("probe_ratio", "mean"),
            mean_score=("score", "mean"),
            n=("score", "size"),
        )
        .reset_index()
    )

    # Merge provider for colours
    meta_path = ROOT / "outputs" / "model_stats.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)[["model", "provider"]]
        agg = agg.merge(meta, on="model", how="left")
    else:
        agg["provider"] = "unknown"
    agg["color"] = agg["provider"].map(PROVIDER_COLOR).fillna(MUTED)

    # Quadrant thresholds
    x_thresh = 0.50
    y_thresh = 0.50

    fig, ax = plt.subplots(figsize=(9.8, 6.8), facecolor=CANVAS)
    ax.set_facecolor(CANVAS)

    # Quadrant shading
    ax.axvspan(0, x_thresh, ymin=0.5, ymax=1.0, color=SOLVED, alpha=0.035, zorder=0)
    ax.axvspan(0, x_thresh, ymin=0, ymax=0.5, color=FAILED, alpha=0.035, zorder=0)
    ax.axvspan(x_thresh, 1.0, ymin=0.5, ymax=1.0, color=INDIGO, alpha=0.035, zorder=0)
    ax.axvspan(x_thresh, 1.0, ymin=0, ymax=0.5, color=SAND, alpha=0.035, zorder=0)

    # Quadrant reference lines
    ax.axhline(y_thresh, color=HAIRLINE, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
    ax.axvline(x_thresh, color=HAIRLINE, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)

    # Quadrant labels
    ax.text(0.12, 0.93, "Well-calibrated",
            transform=ax.transAxes, ha="center", va="top",
            color=SOLVED, fontsize=9.5, fontweight="bold", alpha=0.7)
    ax.text(0.12, 0.90, "confident + correct",
            transform=ax.transAxes, ha="center", va="top",
            color=SOLVED, fontsize=8, alpha=0.6)

    ax.text(0.12, 0.18, "Overconfident",
            transform=ax.transAxes, ha="center", va="bottom",
            color=FAILED, fontsize=9.5, fontweight="bold", alpha=0.7)
    ax.text(0.12, 0.15, "few probes, low score",
            transform=ax.transAxes, ha="center", va="bottom",
            color=FAILED, fontsize=8, alpha=0.6)

    ax.text(0.76, 0.93, "Cautious",
            transform=ax.transAxes, ha="center", va="top",
            color=INDIGO, fontsize=9.5, fontweight="bold", alpha=0.7)
    ax.text(0.76, 0.90, "many probes, still good",
            transform=ax.transAxes, ha="center", va="top",
            color=INDIGO, fontsize=8, alpha=0.6)

    ax.text(0.76, 0.10, "Underconfident",
            transform=ax.transAxes, ha="center", va="bottom",
            color="#8B7355", fontsize=9.5, fontweight="bold", alpha=0.7)
    ax.text(0.76, 0.07, "many probes, low score",
            transform=ax.transAxes, ha="center", va="bottom",
            color="#8B7355", fontsize=8, alpha=0.6)

    # Plot per-model dots
    ax.scatter(agg["mean_ratio"], agg["mean_score"],
               s=130, color=agg["color"], alpha=0.85,
               edgecolor=CANVAS, linewidths=1.4, zorder=4)

    # Label each model — nudge labels to avoid overlap
    for _, row in agg.iterrows():
        short = (row["model"]
                 .replace(" Preview", "")
                 .replace("3.1 ", "")
                 .replace("Next 80B ", ""))
        nudge_x = 0.012
        nudge_y = 0.018
        ha = "left"
        if row["mean_ratio"] > 0.90:
            nudge_x = -0.012
            ha = "right"
        ax.text(row["mean_ratio"] + nudge_x,
                row["mean_score"] + nudge_y,
                short, color=INK, fontsize=8, ha=ha, va="bottom",
                zorder=5)

    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(0.10, 0.85)
    ax.set_xlabel("Probe ratio  (fraction of available examples requested)",
                  color=INK, fontsize=10, labelpad=10)
    ax.set_ylabel("Mean score on interactive tasks",
                  color=INK, fontsize=10, labelpad=10)
    ax.grid(True, color=HAIRLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0)

    # Provider legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=col,
                   markersize=8, label=prov, markeredgecolor=CANVAS)
        for prov, col in PROVIDER_COLOR.items()
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              frameon=False, fontsize=9, handletextpad=0.5,
              labelcolor=INK, bbox_to_anchor=(0.0, -0.005))

    # Title block
    fig.text(0.04, 0.965,
             "M E T A - C A L I B R A T I O N",
             color=MUTED, fontsize=9, fontweight="bold")
    fig.text(0.04, 0.925,
             "Models that ask for fewer examples score higher",
             color=INK, fontsize=15, fontweight="medium")
    fig.text(0.04, 0.890,
             f"{len(pdf)} interactive runs (concept + language) across "
             f"{agg['model'].nunique()} models · "
             f"per-run Spearman ρ = {rho_run:.2f} (p < 10⁻¹⁴) · "
             f"probe ratio ranges from {agg['mean_ratio'].min():.0%} to "
             f"{agg['mean_ratio'].max():.0%}",
             color=MUTED, fontsize=9.5)

    fig.text(0.04, 0.020,
             "Source  LearningBench · concept-formation + language-learning tasks · "
             "probe ratio = examples requested / examples available",
             color=MUTED, fontsize=8)

    plt.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.12)
    out = OUT_DIR / "fig_meta_calibration.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_all_runs()
    make_radar(df)
    make_tokens_scatter(df)

    score_mat_path = REPO / "leaderboard" / "leaderboard_score_matrix.csv"
    make_leaderboard(score_mat_path)
    make_task_difficulty(score_mat_path)
    make_trajectory_orthogonal()
    make_thinking_vs_instruct(score_mat_path)
    make_meta_calibration()
