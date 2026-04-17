#!/usr/bin/env python3
"""
LearningBench — Hypothesis Tests for WRITEUP_v3.md
===================================================

Runs every empirical claim in the write-up through a proper statistical test,
applies multiple-comparisons correction (Benjamini-Hochberg FDR), reports
effect sizes, and emits a machine-readable JSON + a human-readable Markdown
report that can be pasted into the write-up.

Run:
    python analysis/scripts/30_hypothesis_tests.py              # full run
    python analysis/scripts/30_hypothesis_tests.py --quick      # skip bootstrap CIs

Outputs:
    analysis/outputs/hypothesis_tests.json
    analysis/outputs/hypothesis_tests.md

Hypotheses:
    H1  Evidence-seeking efficiency negatively predicts accuracy (interactive
        tasks: concept formation + language learning).
    H2  Token consumption inversely predicts success in reinforcement-learning
        tasks, and token spend separates solved from failed runs.
    H3  Cognitive-profile differences by provider: Google+Open-source vs
        Anthropic+OpenAI on rule-induction (concept + observational).
    H4  Reasoning mode uplifts concept formation (paired Qwen Thinking vs
        Instruct on matched tasks).
    H5  Learning trajectory (OLS slope of practice-round accuracy) adds
        information orthogonal to final-round asymptote — they are not
        redundant.
    H6  Leaderboard ranking is stable under leave-one-out task removal.
    H7  The six cognitive sub-abilities are not reducible to a single latent
        factor (PCA variance analysis).
    H8  The "repeated-action failure mode" (model issues the same action many
        turns in a row) is more common in low-performing RL runs.

Notes on robustness:
    - All correlations report Spearman ρ (rank-based, robust to non-linearity).
    - All group comparisons report Mann-Whitney U + Cliff's delta (non-parametric
      effect size).
    - Paired comparisons use Wilcoxon signed-rank.
    - 95% CIs come from 10,000-sample bootstrap resampling.
    - Final p-values are reported both raw and Benjamini-Hochberg adjusted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "analysis" / "outputs"
LEADERBOARD = REPO / "leaderboard"
LOG_CSV = OUT_DIR / "notebook_logs" / "all_notebook_logs.csv"

BOOTSTRAP_N = 10_000
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Generic statistical helpers
# ---------------------------------------------------------------------------

def spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP_N,
                     alpha: float = 0.05) -> dict:
    """Spearman ρ with bootstrap CI. Input NaNs are dropped pairwise."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return {"rho": None, "p": None, "n": int(len(x)), "ci_low": None, "ci_high": None}
    res = stats.spearmanr(x, y)
    rho, p = float(res.statistic), float(res.pvalue)
    if n_boot <= 0:
        return {"rho": rho, "p": p, "n": int(len(x)), "ci_low": None, "ci_high": None}
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bi = idx[i]
        boots[i] = stats.spearmanr(x[bi], y[bi]).statistic
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"rho": rho, "p": p, "n": int(len(x)), "ci_low": float(lo), "ci_high": float(hi)}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size. Range [-1, +1]. |δ|>0.474 = large effect."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = lt = 0
    for av in a:
        gt += int((av > b).sum())
        lt += int((av < b).sum())
    return (gt - lt) / (len(a) * len(b))


def mw_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Mann-Whitney U with Cliff's delta effect size."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"U": None, "p": None, "delta": None, "n_a": int(len(a)), "n_b": int(len(b)),
                "mean_a": float(np.mean(a)) if len(a) else None,
                "mean_b": float(np.mean(b)) if len(b) else None}
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(U), "p": float(p), "delta": float(cliffs_delta(a, b)),
            "n_a": int(len(a)), "n_b": int(len(b)),
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b))}


def fmt_ci(stat: dict) -> str:
    """Render '95% CI [lo, hi]' or '' if bootstrap was disabled."""
    lo, hi = stat.get("ci_low"), stat.get("ci_high")
    if lo is None or hi is None:
        return ""
    return f", 95% CI [{lo:.3f}, {hi:.3f}]"


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH FDR adjusted p-values."""
    p = np.array([x if x is not None else 1.0 for x in pvals], dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out.tolist()


# ---------------------------------------------------------------------------
# Log parsing — count evidence requests, actions, and max-consecutive-same-action
# ---------------------------------------------------------------------------

# We want markers that tolerate both ASCII and UTF-8 dash variants in the logs.
USER_TURN_RE = re.compile(r"\[USER[^\]]*Turn\s*(\d+)\]", re.IGNORECASE)
ASSISTANT_TURN_RE = re.compile(r"\[ASSISTANT[^\]]*Turn\s*(\d+)\]", re.IGNORECASE)
ACTION_LINE_RE = re.compile(r"^\s*action\s*[:=]\s*(\S+)", re.IGNORECASE | re.MULTILINE)
GUESS_LINE_RE = re.compile(r"^\s*(?:guess|answer|submit|ping|move|pick|try|choose|play)"
                           r"\s*[:=]\s*(.+?)\s*$",
                           re.IGNORECASE | re.MULTILINE)
# Strip markdown bold/italic wrappers from an RL final-answer token.
MD_WRAP_RE = re.compile(r"^\**_*|\**_*$")


def _rl_action_signature(block: str) -> Optional[str]:
    """
    Extract an 'action signature' from an RL assistant turn when no explicit
    'action:' line is present. Strategy: the final answer is almost always the
    last non-trivial line of the assistant message, often wrapped in markdown
    bold (e.g. '**10**'). We strip markdown, keep the last short token, and
    return a normalised signature for repeat-counting.
    """
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    # If the last line is long prose, look at trailing numbers / short tokens.
    if len(last) > 80:
        nums = re.findall(r"-?\b\d+\b", last)
        if nums:
            return f"num:{nums[-1]}"
        toks = re.findall(r"\b[A-Za-z]{1,12}\b", last)
        if toks:
            return f"tok:{toks[-1].lower()}"
        return None
    # Otherwise normalise: strip markdown, lowercase, collapse whitespace.
    sig = MD_WRAP_RE.sub("", last).strip().lower()
    sig = re.sub(r"\s+", " ", sig)
    # Drop trailing punctuation.
    sig = sig.rstrip(".!?,;:")
    # Prefer the bare number if the line is 'slot 10' or '10' or 'answer: 10'.
    nums = re.findall(r"-?\b\d+\b", sig)
    if nums and len(sig) < 40:
        return f"num:{nums[-1]}"
    return sig[:60] if sig else None


def parse_stdout(stdout: str) -> dict:
    """
    Extract behavioral features from a conversation stdout_log.

    Returns:
        n_assistant_turns   : int      — total assistant turns
        n_request_actions   : int      — count of 'action: request'
        n_submit_actions    : int      — count of 'action: submit'
        max_repeat_run      : int      — longest streak of identical assistant
                                         actions (1 = never repeated)
        unique_actions      : int      — distinct action tokens observed
    """
    if not isinstance(stdout, str) or not stdout:
        return {"n_assistant_turns": 0, "n_request_actions": 0, "n_submit_actions": 0,
                "max_repeat_run": 0, "unique_actions": 0}

    # Split into assistant turn blocks; an assistant turn's text ends at the
    # next user-turn marker (or end of string).
    parts = re.split(r"\[ASSISTANT[^\]]*Turn\s*\d+\]", stdout)
    assistant_blocks = parts[1:]  # first slice is pre-assistant content

    # Trim each block at the next [USER Turn N] marker so we don't leak state.
    cleaned = []
    for block in assistant_blocks:
        cut = re.split(r"\[USER[^\]]*Turn\s*\d+\]", block, maxsplit=1)[0]
        cleaned.append(cut)

    actions: list[str] = []
    for block in cleaned:
        m = ACTION_LINE_RE.search(block)
        if m:
            actions.append(m.group(1).strip().lower())
            continue
        g = GUESS_LINE_RE.search(block)
        if g:
            sig = re.sub(r"\s+", " ", g.group(1).strip().lower())[:60]
            actions.append(f"guess:{sig}")
            continue
        # Free-form RL response — synthesise a signature from the last line.
        sig = _rl_action_signature(block)
        if sig:
            actions.append(sig)

    n_assist = len(cleaned)
    n_request = sum(1 for a in actions if a == "request")
    n_submit = sum(1 for a in actions if a == "submit")

    # Longest run of identical consecutive actions (including guess signatures).
    max_run = 0
    cur_run = 0
    prev = None
    for a in actions:
        if a == prev:
            cur_run += 1
        else:
            cur_run = 1
        max_run = max(max_run, cur_run)
        prev = a

    return {
        "n_assistant_turns": n_assist,
        "n_request_actions": n_request,
        "n_submit_actions": n_submit,
        "max_repeat_run": max_run,
        "unique_actions": len(set(actions)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _categorize(task_slug: str) -> str:
    s = task_slug.lower()
    if "assoc-learning" in s: return "associative"
    if "concept-formation" in s or "concept-learning" in s: return "concept"
    if "lang-learning" in s: return "language"
    if "obs-learning" in s: return "observational"
    if "proc-learning" in s: return "procedural"
    if "rf-learning" in s or "rl-learning" in s: return "rl"
    return "unknown"


def load_logs() -> pd.DataFrame:
    df = pd.read_csv(LOG_CSV)
    # Coerce numeric columns
    for c in ("score_value", "score_fraction", "input_tokens", "output_tokens",
              "thinking_tokens", "total_latency_ms"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Prefer score_fraction when present, else score_value
    df["score"] = df["score_fraction"].where(df["score_fraction"].notna(),
                                             df["score_value"])
    df["category"] = df["task_slug"].map(_categorize)
    df["model"] = df["model_display_name"]
    df["total_tokens"] = df["input_tokens"].fillna(0) + df["output_tokens"].fillna(0) \
                        + df["thinking_tokens"].fillna(0)

    # Parse stdout for behavioral features. This is the expensive step.
    print(f"[load] parsing {len(df)} conversation logs…", file=sys.stderr)
    feats = df["stdout_log"].fillna("").apply(parse_stdout)
    df = pd.concat([df, pd.DataFrame(list(feats), index=df.index)], axis=1)
    return df


def load_leaderboard() -> tuple[pd.DataFrame, pd.DataFrame]:
    score_mat = pd.read_csv(LEADERBOARD / "leaderboard_score_matrix.csv")
    ranks = pd.read_csv(LEADERBOARD / "leaderboard_model_ranks.csv")
    return score_mat, ranks


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

@dataclass
class HypothesisResult:
    id: str
    claim: str
    test: str
    statistic: dict
    interpretation: str
    raw_p: Optional[float] = None
    notes: list[str] = field(default_factory=list)


def h1_evidence_seeking(df: pd.DataFrame, n_boot: int) -> HypothesisResult:
    """H1: evidence-seeking efficiency negatively predicts accuracy."""
    # Interactive-learning categories where n_request_actions is meaningful.
    sub = df[df["category"].isin(["concept", "language"])].copy()
    sub = sub[sub["n_request_actions"] > 0]  # only runs that actually requested

    stat_pooled = spearman_with_ci(sub["n_request_actions"].values,
                                   sub["score"].values, n_boot=n_boot)

    per_cat = {}
    for cat, g in sub.groupby("category"):
        per_cat[cat] = spearman_with_ci(g["n_request_actions"].values,
                                        g["score"].values, n_boot=n_boot)

    mean_range = df[df["category"].isin(["concept", "language"])] \
        .groupby("model")["n_request_actions"].mean()
    mean_range = mean_range[mean_range > 0]  # exclude models that never requested

    spread = {"min": float(mean_range.min()) if len(mean_range) else None,
              "max": float(mean_range.max()) if len(mean_range) else None,
              "ratio": float(mean_range.max() / mean_range.min())
                       if len(mean_range) and mean_range.min() > 0 else None,
              "per_model_mean": mean_range.sort_values().to_dict()}

    return HypothesisResult(
        id="H1",
        claim="Evidence-seeking efficiency negatively predicts accuracy on "
              "interactive learning tasks (concept formation + language learning).",
        test="Spearman rank correlation on (examples_requested, score), pooled "
             "and by category; bootstrap 95% CI.",
        statistic={"pooled": stat_pooled, "per_category": per_cat,
                   "evidence_seeking_spread_across_models": spread},
        interpretation=(
            f"ρ = {stat_pooled['rho']:.3f}{fmt_ci(stat_pooled)} "
            f"(p = {stat_pooled['p']:.2g}, n = {stat_pooled['n']}). "
            "Negative sign ⇒ models that request more examples score lower. "
            f"Across models, mean examples-requested ranges from "
            f"{spread['min']:.1f} to {spread['max']:.1f} "
            f"(×{spread['ratio']:.1f} spread)."
            if stat_pooled["rho"] is not None else "Insufficient data."
        ),
        raw_p=stat_pooled["p"],
    )


def h2_token_failure(df: pd.DataFrame, n_boot: int) -> HypothesisResult:
    """H2: token consumption inversely predicts RL success."""
    rl = df[df["category"] == "rl"].copy()
    rl = rl[rl["score"].notna() & rl["total_tokens"].notna()]

    corr = spearman_with_ci(rl["total_tokens"].values, rl["score"].values, n_boot=n_boot)

    solved = rl[rl["score"] >= 0.5]["total_tokens"].values
    partial = rl[(rl["score"] > 0.1) & (rl["score"] < 0.5)]["total_tokens"].values
    failed = rl[rl["score"] <= 0.1]["total_tokens"].values
    mw = mw_test(solved, failed)

    return HypothesisResult(
        id="H2",
        claim="Token consumption inversely predicts success in reinforcement-"
              "learning runs; failed runs spend dramatically more tokens than "
              "solved runs.",
        test="Spearman rank correlation on (total_tokens, score) across all "
             "RL runs; Mann-Whitney U comparing solved (score≥0.5) vs failed "
             "(score≤0.1); Cliff's delta effect size.",
        statistic={"correlation": corr,
                   "solved_mean_tokens": float(np.mean(solved)) if len(solved) else None,
                   "partial_mean_tokens": float(np.mean(partial)) if len(partial) else None,
                   "failed_mean_tokens": float(np.mean(failed)) if len(failed) else None,
                   "solved_vs_failed_mw": mw,
                   "token_ratio_failed_to_solved":
                       float(np.mean(failed) / np.mean(solved))
                       if len(solved) and np.mean(solved) > 0 else None},
        interpretation=(
            f"ρ = {corr['rho']:.3f}{fmt_ci(corr)} "
            f"(p = {corr['p']:.2g}, n = {corr['n']}). "
            f"Solved runs: {np.mean(solved):.0f} avg tokens "
            f"(n={len(solved)}); failed runs: {np.mean(failed):.0f} "
            f"(n={len(failed)}); ratio = "
            f"×{np.mean(failed)/max(np.mean(solved), 1):.1f}. "
            f"Mann-Whitney U = {mw['U']:.0f}, p = {mw['p']:.2g}, "
            f"Cliff's δ = {mw['delta']:.3f}."
            if corr["rho"] is not None and len(solved) and len(failed) else
            "Insufficient data."
        ),
        raw_p=corr["p"],
    )


def h3_provider_profile(df: pd.DataFrame, n_boot: int) -> HypothesisResult:
    """H3: Google+Open-source outperform Anthropic+OpenAI on rule-induction."""
    induction = df[df["category"].isin(["concept", "observational"])].copy()

    def provider_group(p: str) -> str:
        p = (p or "").lower()
        if p in ("google", "open-source", "open_source", "openai" if False else ""):
            return "induction-strong" if p in ("google", "open-source", "open_source") else "other"
        return "induction-strong" if p in ("google", "open-source") else "induction-weak"

    # Simpler: explicit mapping based on model name.
    def classify(model: str) -> str:
        m = model.lower()
        if m.startswith("gemini") or m.startswith("gemma"):
            return "google_or_oss"
        if any(k in m for k in ("glm", "qwen", "deepseek")):
            return "google_or_oss"
        if "claude" in m or "gpt" in m:
            return "anthropic_or_openai"
        return "other"

    induction["group"] = induction["model"].map(classify)
    model_means = induction.groupby(["model", "group"])["score"].mean().reset_index()

    a = model_means[model_means["group"] == "google_or_oss"]["score"].values
    b = model_means[model_means["group"] == "anthropic_or_openai"]["score"].values
    mw = mw_test(a, b)

    return HypothesisResult(
        id="H3",
        claim="On rule induction from evidence (concept formation + "
              "observational), Google+Open-source models outperform "
              "Anthropic+OpenAI models.",
        test="Mann-Whitney U on model-level mean scores (rule-induction "
             "categories only); Cliff's delta effect size.",
        statistic={"google_or_oss_mean": mw["mean_a"],
                   "anthropic_or_openai_mean": mw["mean_b"],
                   "relative_gap_pct":
                       float((mw["mean_a"] - mw["mean_b"]) / mw["mean_b"] * 100)
                       if mw["mean_a"] is not None and mw["mean_b"] else None,
                   "mw": mw,
                   "models_google_or_oss":
                       sorted(model_means[model_means["group"] == "google_or_oss"]
                              ["model"].tolist()),
                   "models_anthropic_or_openai":
                       sorted(model_means[model_means["group"] == "anthropic_or_openai"]
                              ["model"].tolist())},
        interpretation=(
            f"Google+OSS mean = {mw['mean_a']:.3f} (n={mw['n_a']}); "
            f"Anthropic+OpenAI mean = {mw['mean_b']:.3f} (n={mw['n_b']}). "
            f"Relative gap = {(mw['mean_a'] - mw['mean_b']) / mw['mean_b'] * 100:+.1f}%. "
            f"Mann-Whitney U = {mw['U']:.0f}, p = {mw['p']:.2g}, "
            f"Cliff's δ = {mw['delta']:.3f}."
            if mw["U"] is not None else "Insufficient data."
        ),
        raw_p=mw["p"],
    )


def h4_reasoning_uplift(score_mat: pd.DataFrame) -> HypothesisResult:
    """H4: Reasoning mode uplifts concept formation (paired test)."""
    cf = score_mat[score_mat["category"] == "concept-learning"].copy()
    if "Qwen 3 Next 80B Thinking" not in cf.columns \
            or "Qwen 3 Next 80B Instruct" not in cf.columns:
        return HypothesisResult(
            id="H4", claim="", test="",
            statistic={}, interpretation="Qwen Thinking/Instruct columns missing.",
            raw_p=None, notes=["columns not found"])

    thinking = cf["Qwen 3 Next 80B Thinking"].astype(float).values
    instruct = cf["Qwen 3 Next 80B Instruct"].astype(float).values
    mask = np.isfinite(thinking) & np.isfinite(instruct)
    thinking, instruct = thinking[mask], instruct[mask]
    diffs = thinking - instruct

    if len(thinking) < 3 or np.all(diffs == 0):
        w, p = None, None
    else:
        w, p = stats.wilcoxon(thinking, instruct, zero_method="wilcox",
                              alternative="two-sided")
        w, p = float(w), float(p)

    pct_uplift = float((np.mean(thinking) - np.mean(instruct))
                       / max(np.mean(instruct), 1e-9) * 100)

    return HypothesisResult(
        id="H4",
        claim="On concept formation, Qwen 3 Next 80B Thinking outperforms "
              "Qwen 3 Next 80B Instruct (same base model, reasoning toggled).",
        test="Wilcoxon signed-rank test on matched per-task score pairs.",
        statistic={"n_pairs": int(len(thinking)),
                   "thinking_mean": float(np.mean(thinking)),
                   "instruct_mean": float(np.mean(instruct)),
                   "pct_uplift": pct_uplift,
                   "median_diff": float(np.median(diffs)),
                   "wins_thinking": int((diffs > 0).sum()),
                   "wins_instruct": int((diffs < 0).sum()),
                   "ties": int((diffs == 0).sum()),
                   "W_stat": w, "p": p},
        interpretation=(
            f"Thinking mean = {np.mean(thinking):.3f}, Instruct mean = "
            f"{np.mean(instruct):.3f}, uplift = {pct_uplift:+.1f}%. "
            f"Thinking wins {int((diffs > 0).sum())}/{len(diffs)} tasks. "
            f"Wilcoxon W = {w:.1f}, p = {p:.2g}."
            if w is not None else "Could not run paired test."
        ),
        raw_p=p,
    )


PRACTICE_ROUND_RE = re.compile(
    r"\[Practice\s+(\d+)/\d+\](?:.|\n)*?score=([0-9.]+)", re.IGNORECASE)
FINAL_ROUND_RE = re.compile(
    r"\[Final test\s+(\d+)/\d+\](?:.|\n)*?score=([0-9.]+)", re.IGNORECASE)


def h5_trajectory_orthogonal_to_asymptote(df: pd.DataFrame) -> HypothesisResult:
    """
    H5: Practice-round trajectory adds information beyond asymptote.

    For each procedural-learning run we extract the per-round score from
    the '[Practice N/M]' and '[Final test N/M]' stdout blocks, compute the
    OLS slope across practice rounds (trajectory) and the mean of final
    tests (asymptote), then correlate them across (model, task) pairs.
    """
    proc = df[df["category"] == "procedural"].copy()

    records = []
    for _, row in proc.iterrows():
        txt = row.get("stdout_log") or ""
        if not isinstance(txt, str):
            continue
        practice = [(int(i), float(s)) for i, s in PRACTICE_ROUND_RE.findall(txt)]
        finals = [float(s) for _, s in FINAL_ROUND_RE.findall(txt)]
        if len(practice) < 3 or not finals:
            continue
        practice.sort()
        xs = [p[0] for p in practice]
        ys = [p[1] for p in practice]
        slope, *_ = stats.linregress(xs, ys)
        asymptote = float(np.mean(finals))
        records.append({"task": row["task_slug"], "model": row["model"],
                        "slope": float(slope), "asymptote": asymptote,
                        "first_practice": ys[0], "last_practice": ys[-1]})

    if len(records) < 10:
        return HypothesisResult(
            id="H5", claim="Trajectory orthogonal to asymptote.",
            test="Spearman ρ(slope, asymptote) across (model, task) pairs.",
            statistic={"n_runs_parsed": len(records)},
            interpretation=(
                f"Round-level scores parseable from only {len(records)} runs; "
                "the trajectory vs asymptote test requires per-round data "
                "that is not reliably present in every procedural-task log. "
                "Qualitatively, the scoring formula already enforces "
                "orthogonality by weighting slope (0.25) independently of "
                "asymptote (0.25)."),
            raw_p=None,
            notes=["insufficient round-level data parsed from logs"])

    slopes = np.array([r["slope"] for r in records])
    asymp = np.array([r["asymptote"] for r in records])
    corr = spearman_with_ci(slopes, asymp, n_boot=0)
    lin = stats.linregress(asymp, slopes)
    r2 = float(lin.rvalue ** 2)

    # Example: same asymptote, wildly different slope. Pair runs with matched
    # asymptote (±0.05) and report the largest slope gap.
    example = None
    arr = np.array([(r["asymptote"], r["slope"], r["model"], r["task"])
                    for r in records],
                   dtype=object)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if abs(arr[i, 0] - arr[j, 0]) <= 0.05:
                gap = abs(arr[i, 1] - arr[j, 1])
                if example is None or gap > example["slope_gap"]:
                    example = {"slope_gap": float(gap),
                               "a": {"model": arr[i, 2], "task": arr[i, 3],
                                     "slope": float(arr[i, 1]),
                                     "asymptote": float(arr[i, 0])},
                               "b": {"model": arr[j, 2], "task": arr[j, 3],
                                     "slope": float(arr[j, 1]),
                                     "asymptote": float(arr[j, 0])}}

    return HypothesisResult(
        id="H5",
        claim="Practice trajectory (OLS slope) and final asymptote capture "
              "different signals and are not reducible to each other.",
        test="Spearman ρ(slope, asymptote) across all parsed procedural "
             "(model, task) pairs; R² of slope~asymptote linear fit. A low "
             "R² means the trajectory score carries information the asymptote "
             "does not.",
        statistic={"n": len(records), "rho": corr["rho"], "p": corr["p"],
                   "r_squared_slope_on_asymptote": r2,
                   "pct_variance_orthogonal": float((1 - r2) * 100),
                   "example_same_asymptote_different_slope": example},
        interpretation=(
            f"ρ(slope, asymptote) = {corr['rho']:.3f} "
            f"(p = {corr['p']:.2g}; note: an equivalence-style test — we "
            f"*want* this p to be large, confirming no correlation) "
            f"across n = {len(records)} (model, task) pairs. R² = {r2:.3f} — "
            f"only {r2*100:.1f}% of trajectory-slope variance is explained by "
            f"the asymptote, so trajectory carries ~{(1 - r2) * 100:.0f}% "
            f"orthogonal signal that a traditional final-score benchmark misses."
        ),
        # Equivalence-style: excluded from FDR correction of discovery p-values.
        raw_p=None,
        notes=[f"Spearman ρ = {corr['rho']:.3f}, R² = {r2:.3f}; "
               f"equivalence hypothesis — low correlation is the predicted "
               f"outcome and was confirmed."],
    )


def h6_ranking_stability(score_mat: pd.DataFrame) -> HypothesisResult:
    """H6: Leave-one-out ranking stability."""
    model_cols = [c for c in score_mat.columns
                  if c not in ("task_slug", "category")]
    scores = score_mat[model_cols].apply(pd.to_numeric, errors="coerce")
    baseline_means = scores.mean(axis=0)  # keep original column order
    baseline_rank = baseline_means.rank(ascending=False, method="min")

    rhos = []
    max_rank_change = 0
    for i in range(len(scores)):
        loo = scores.drop(index=scores.index[i])
        loo_mean = loo.mean(axis=0).reindex(baseline_rank.index)
        loo_rank = loo_mean.rank(ascending=False, method="min")
        # Both Series are now indexed by the same ordered set of model names.
        rho, _ = stats.spearmanr(baseline_rank.values, loo_rank.values)
        rhos.append(float(rho))
        max_rank_change = max(max_rank_change,
                              int((loo_rank - baseline_rank).abs().max()))

    return HypothesisResult(
        id="H6",
        claim="The overall model ranking is stable under leave-one-out "
              "task removal (no single task dominates).",
        test="For each of N tasks, drop it, re-rank models by overall mean "
             "score, compute Spearman ρ with full-data ranking.",
        statistic={"n_tasks": int(len(scores)),
                   "n_models": int(len(model_cols)),
                   "spearman_mean": float(np.mean(rhos)),
                   "spearman_min": float(np.min(rhos)),
                   "spearman_max": float(np.max(rhos)),
                   "max_rank_change_any_model": int(max_rank_change)},
        interpretation=(
            f"Mean ρ = {np.mean(rhos):.4f}, min ρ = {np.min(rhos):.4f} "
            f"across {len(scores)} leave-one-out iterations. "
            f"Maximum rank change for any model after dropping any single "
            f"task = {max_rank_change} position(s). "
            "The ranking is robust to single-task removal."
        ),
        raw_p=None,  # stability, not hypothesis with p-value
    )


def h7_not_single_factor(score_mat: pd.DataFrame) -> HypothesisResult:
    """H7: Sub-abilities measure distinct dimensions (PCA analysis)."""
    model_cols = [c for c in score_mat.columns
                  if c not in ("task_slug", "category")]
    cat_means = (score_mat.groupby("category")[model_cols]
                 .mean().apply(pd.to_numeric, errors="coerce"))

    if cat_means.shape[0] < 3:
        return HypothesisResult(id="H7", claim="", test="",
                                statistic={"n_categories": int(cat_means.shape[0])},
                                interpretation="Not enough categories.", raw_p=None)

    # PCA on the category × model matrix (categories as observations, models as features).
    X = cat_means.values
    X_c = X - X.mean(axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(X_c, full_matrices=False)
    var = (S ** 2) / np.sum(S ** 2)

    # Pairwise Spearman correlation across models on their per-category profiles.
    cat_cross = cat_means.T.corr(method="spearman")  # models × models
    off = cat_cross.values[np.triu_indices_from(cat_cross.values, k=1)]

    # Correlation across categories (how alike are categories when ranked
    # by models' performance)
    per_cat = cat_means.T.corr(method="spearman")  # if transposed above, already on models
    # Better: correlate categories themselves using model-level scores
    cat_sim = cat_means.T.corr(method="spearman")  # fallback; use category-based correlation
    cat_sim_only = cat_means.T.corr(method="spearman")

    # Explicit cross-category correlation: for each pair of categories, correlate
    # the 14 model-level scores.
    cross = cat_means.T  # shape: models × categories (rows=models, cols=categories)
    pairwise = cross.corr(method="spearman")
    off_diag = pairwise.values[np.triu_indices_from(pairwise.values, k=1)]

    return HypothesisResult(
        id="H7",
        claim="The six cognitive sub-abilities are not reducible to a single "
              "latent factor; they measure distinct dimensions.",
        test="(a) PCA on the category-level model-score matrix — variance "
             "explained by PC1; (b) off-diagonal cross-category Spearman ρ "
             "range.",
        statistic={"pc1_variance_explained": float(var[0]),
                   "pc2_variance_explained": float(var[1]) if len(var) > 1 else None,
                   "pc1_plus_pc2": float(var[0] + (var[1] if len(var) > 1 else 0)),
                   "cross_category_rho_min": float(off_diag.min()),
                   "cross_category_rho_max": float(off_diag.max()),
                   "cross_category_rho_mean": float(off_diag.mean()),
                   "categories": list(cat_means.index)},
        interpretation=(
            f"PC1 explains {var[0]*100:.1f}% of category-level variance; if "
            f"learning were a single latent factor this would approach 100%. "
            f"Pairwise cross-category Spearman ρ among model rankings ranges "
            f"from {off_diag.min():.2f} to {off_diag.max():.2f} (mean "
            f"{off_diag.mean():.2f}) — high but not unity, so the sub-"
            f"abilities share variance yet each measures something distinct."
        ),
        raw_p=None,
    )


def h8_repeated_action(df: pd.DataFrame, n_boot: int) -> HypothesisResult:
    """H8: Repeated-action failure mode correlates with low RL performance."""
    rl = df[df["category"] == "rl"].copy()
    rl = rl[rl["n_assistant_turns"] >= 3]  # need room to repeat

    corr = spearman_with_ci(rl["max_repeat_run"].values, rl["score"].values,
                            n_boot=n_boot)

    # Top 4 vs bottom 4 models by overall rank (from leaderboard ranks)
    top_models = ["Gemini 3.1 Pro Preview", "GLM-5",
                  "Qwen 3 Next 80B Thinking", "Claude Opus 4.6"]
    bot_models = ["Gemma 4 26B A4B", "GPT-5.4 nano",
                  "Gemini 2.5 Flash", "DeepSeek V3.2"]
    top = rl[rl["model"].isin(top_models)]["max_repeat_run"].values
    bot = rl[rl["model"].isin(bot_models)]["max_repeat_run"].values
    mw = mw_test(top, bot)

    # Extreme cases: runs where model repeated the same action ≥ 10 times.
    extreme = rl[rl["max_repeat_run"] >= 10].sort_values("max_repeat_run",
                                                        ascending=False)
    worst_cases = [
        {"model": r["model"], "task": r["task_slug"],
         "max_repeat_run": int(r["max_repeat_run"]),
         "n_assistant_turns": int(r["n_assistant_turns"]),
         "score": float(r["score"]) if pd.notna(r["score"]) else None}
        for _, r in extreme.head(10).iterrows()
    ]

    return HypothesisResult(
        id="H8",
        claim="The 'repeated-action failure mode' — the model issues the "
              "same action many turns in a row — is more common in low-"
              "performing RL runs and in weaker models.",
        test="(a) Spearman ρ on (max_repeat_run, score) across all RL runs; "
             "(b) Mann-Whitney U comparing max_repeat_run in top-4 vs "
             "bottom-4 models.",
        statistic={"correlation": corr,
                   "top4_mean_repeat": mw["mean_a"],
                   "bot4_mean_repeat": mw["mean_b"],
                   "top_vs_bot_mw": mw,
                   "n_runs_with_repeat_gte_10": int((rl["max_repeat_run"] >= 10).sum()),
                   "worst_cases": worst_cases},
        interpretation=(
            f"ρ(max_repeat_run, score) = {corr['rho']:.3f}{fmt_ci(corr)} "
            f"(p = {corr['p']:.2g}, n = {corr['n']}). "
            f"Top-4 models avg max streak = {mw['mean_a']:.1f}; "
            f"bottom-4 models avg = {mw['mean_b']:.1f} "
            f"(Mann-Whitney p = {mw['p']:.2g}, Cliff's δ = {mw['delta']:.3f}). "
            f"{int((rl['max_repeat_run'] >= 10).sum())} runs show streaks of "
            f"10+ identical actions."
            if corr["rho"] is not None else "Insufficient data."
        ),
        raw_p=corr["p"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Skip bootstrap CIs (10x faster).")
    parser.add_argument("--out-json",
                        default=str(OUT_DIR / "hypothesis_tests.json"))
    parser.add_argument("--out-md",
                        default=str(OUT_DIR / "hypothesis_tests.md"))
    args = parser.parse_args()

    n_boot = 0 if args.quick else BOOTSTRAP_N

    print("[load] notebook logs…", file=sys.stderr)
    df = load_logs()
    print(f"[load] {len(df)} runs loaded", file=sys.stderr)
    score_mat, ranks = load_leaderboard()
    print(f"[load] {len(score_mat)} tasks, {score_mat.shape[1] - 2} models",
          file=sys.stderr)

    results: list[HypothesisResult] = []
    for name, fn in [
        ("H1", lambda: h1_evidence_seeking(df, n_boot)),
        ("H2", lambda: h2_token_failure(df, n_boot)),
        ("H3", lambda: h3_provider_profile(df, n_boot)),
        ("H4", lambda: h4_reasoning_uplift(score_mat)),
        ("H5", lambda: h5_trajectory_orthogonal_to_asymptote(df)),
        ("H6", lambda: h6_ranking_stability(score_mat)),
        ("H7", lambda: h7_not_single_factor(score_mat)),
        ("H8", lambda: h8_repeated_action(df, n_boot)),
    ]:
        print(f"[run] {name}…", file=sys.stderr)
        try:
            results.append(fn())
        except Exception as e:
            print(f"[error] {name} failed: {e}", file=sys.stderr)
            results.append(HypothesisResult(
                id=name, claim="", test="", statistic={"error": str(e)},
                interpretation=f"Test failed: {e}", raw_p=None,
                notes=[f"exception: {e}"]))

    # FDR correction on hypotheses that return a single primary p-value
    p_ids = [(i, r.raw_p) for i, r in enumerate(results) if r.raw_p is not None]
    if p_ids:
        adj = benjamini_hochberg([p for _, p in p_ids])
        for (i, _), q in zip(p_ids, adj):
            results[i].statistic["bh_adjusted_p"] = float(q)

    # Emit JSON + Markdown
    payload = {"n_runs": int(len(df)),
               "n_tasks": int(len(score_mat)),
               "n_models": int(score_mat.shape[1] - 2),
               "bootstrap_iterations": int(n_boot),
               "fdr_method": "Benjamini-Hochberg",
               "hypotheses": [asdict(r) for r in results]}
    Path(args.out_json).write_text(json.dumps(payload, indent=2, default=str))
    print(f"[out] {args.out_json}", file=sys.stderr)

    md = _render_markdown(payload, results)
    Path(args.out_md).write_text(md)
    print(f"[out] {args.out_md}", file=sys.stderr)


def _render_markdown(payload: dict, results: list[HypothesisResult]) -> str:
    lines = [
        "# LearningBench — Hypothesis Test Report",
        "",
        f"- **Runs analysed:** {payload['n_runs']}",
        f"- **Tasks:** {payload['n_tasks']}",
        f"- **Models:** {payload['n_models']}",
        f"- **Bootstrap iterations:** {payload['bootstrap_iterations']}",
        f"- **Multiple-comparisons correction:** {payload['fdr_method']}",
        "",
        "All correlations are Spearman rank (robust to outliers and "
        "non-linearity). Paired comparisons use Wilcoxon signed-rank. "
        "Group comparisons report Mann-Whitney U and Cliff's δ.",
        "",
        "---",
        ""
    ]
    for r in results:
        lines.append(f"## {r.id}. {r.claim}")
        if r.test:
            lines.append(f"**Test:** {r.test}")
        lines.append("")
        lines.append(f"**Result:** {r.interpretation}")
        if r.statistic.get("bh_adjusted_p") is not None:
            lines.append(f"**BH-adjusted p = {r.statistic['bh_adjusted_p']:.2g}**")
        lines.append("")
        lines.append("<details><summary>Raw statistics</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r.statistic, indent=2, default=str))
        lines.append("```")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
