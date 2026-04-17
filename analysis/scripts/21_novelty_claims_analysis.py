"""
Novelty Claims Analysis for LearningBench
Produces all statistical evidence for the three headline claims.
Outputs go to analysis/outputs/novelty_claims/
"""

import json
import csv
import os
import math
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parents[2]
LOGS = BASE / "analysis/outputs/notebook_logs/all_notebook_logs.json"
TASK_RUNS = BASE / "analysis/outputs/task_runs/all_task_runs.csv"
FULL_STATS = BASE / "analysis/outputs/full_task_model_stats.csv"
MODEL_STATS = BASE / "analysis/outputs/model_stats.csv"
PROVIDER_STATS = BASE / "analysis/outputs/provider_analysis.csv"
EFF_ABLATION = BASE / "analysis/outputs/efficiency_ablation.csv"
LOO_GLOBAL = BASE / "analysis/outputs/loo_global.csv"
RANDOM_BASELINE = BASE / "analysis/outputs/random_baseline.csv"
OUT = BASE / "analysis/outputs/novelty_claims"
OUT.mkdir(exist_ok=True)


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


# ── Load notebook logs ───────────────────────────────────────────────────────
print("Loading notebook logs...")
with open(LOGS) as f:
    logs = json.load(f)
print(f"  Loaded {len(logs)} entries")


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Active Evidence-Seeking — Requests per Model (CF + Lang tasks)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 1: Active evidence-seeking ===")

cf_lang_rows = []
for d in logs:
    slug = str(d.get("task_slug", ""))
    is_cf = "concept" in slug
    is_ll = "lang" in slug and "lang-learning" in slug or "lang_learning" in slug
    # also catch slugs like "dimval-metathesis-lang-learning"
    is_ll = is_ll or ("lang-learning" in slug)
    if not (is_cf or is_ll):
        continue
    rj = d.get("run_json") or {}
    convs = rj.get("conversations", []) or []
    # Conv 0 = task description, Conv 1 = active interactive session
    active_reqs = 0
    total_reqs = 0
    for conv in convs:
        if conv:
            reqs = conv.get("requests", []) or []
            total_reqs += len(reqs)
    if len(convs) > 1 and convs[1]:
        active_reqs = len(convs[1].get("requests", []) or [])

    score = _safe_float(d.get("score_value") or d.get("score_fraction"))
    cf_lang_rows.append(
        {
            "model": d["model_display_name"],
            "provider": d["provider"],
            "task": slug,
            "category": "concept" if is_cf else "language",
            "total_reqs": total_reqs,
            "active_reqs": active_reqs,
            "score": score,
        }
    )

cf_df = pd.DataFrame(cf_lang_rows)
print(f"  CF/Language runs: {len(cf_df)}")

# Per-model average active requests
per_model_reqs = (
    cf_df.groupby(["model", "provider"])["active_reqs"]
    .agg(mean_requests="mean", std_requests="std", n="count")
    .reset_index()
    .sort_values("mean_requests")
)
per_model_reqs["mean_requests"] = per_model_reqs["mean_requests"].round(2)
per_model_reqs["std_requests"] = per_model_reqs["std_requests"].round(2)
per_model_reqs.to_csv(OUT / "active_querying_per_model.csv", index=False)
print(f"  Request range: {per_model_reqs.mean_requests.min():.1f}–{per_model_reqs.mean_requests.max():.1f}")

# Correlation: active_reqs vs score
rho_ae, p_ae = stats.spearmanr(cf_df["active_reqs"], cf_df["score"])
print(f"  Spearman (active_reqs vs score): rho={rho_ae:.3f}, p={p_ae:.2e}")

# Save summary
ae_summary = {
    "n_runs": len(cf_df),
    "min_mean_requests": per_model_reqs.mean_requests.min(),
    "max_mean_requests": per_model_reqs.mean_requests.max(),
    "min_model": per_model_reqs.iloc[0]["model"],
    "max_model": per_model_reqs.iloc[-1]["model"],
    "spearman_rho": round(rho_ae, 4),
    "spearman_p": round(p_ae, 6),
}
pd.DataFrame([ae_summary]).to_csv(OUT / "active_querying_summary.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Efficiency Ablation — does removing efficiency score change rankings?
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 2: Efficiency ablation ===")

eff_df = pd.read_csv(EFF_ABLATION)
print(f"  Rows: {len(eff_df)}")

# Per-category rank stability
cat_stability = []
for cat, grp in eff_df.groupby("category"):
    rho_cat, p_cat = stats.spearmanr(
        grp["composite_rank"], grp["accuracy_only_rank"]
    )
    max_change = grp["rank_change"].abs().max()
    n_moved = (grp["rank_change"].abs() > 0).sum()
    cat_stability.append(
        {
            "category": cat,
            "n_models": len(grp),
            "spearman_rho": round(rho_cat, 4),
            "max_rank_change": int(max_change),
            "n_moved": int(n_moved),
        }
    )

stab_df = pd.DataFrame(cat_stability)
stab_df.to_csv(OUT / "efficiency_ablation_stability.csv", index=False)
print(stab_df.to_string(index=False))

# Compute raw score gap: composite_score vs simulated_acc_only
eff_df["score_gap"] = eff_df["simulated_acc_only"] - eff_df["composite_score"]
avg_penalty = eff_df["score_gap"].mean()
max_penalty = eff_df["score_gap"].max()
print(f"  Mean accuracy→composite score penalty: {avg_penalty:.4f}")
print(f"  Max penalty (single model): {max_penalty:.4f}")

eff_summary = {
    "categories_tested": len(stab_df),
    "all_rho_1": int((stab_df["spearman_rho"] == 1.0).all()),
    "max_rank_change_ever": int(stab_df["max_rank_change"].max()),
    "mean_score_penalty": round(avg_penalty, 4),
    "max_score_penalty": round(max_penalty, 4),
}
pd.DataFrame([eff_summary]).to_csv(OUT / "efficiency_ablation_summary.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Token-Failure Correlation (RL tasks)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 3: Token-failure correlation (RL) ===")

rl_rows = []
for d in logs:
    slug = str(d.get("task_slug", ""))
    if "rf-learning" not in slug:
        continue
    score = _safe_float(d.get("score_value") or d.get("score_fraction"))
    it = _safe_int(d.get("input_tokens"))
    ot = _safe_int(d.get("output_tokens"))
    tt = _safe_int(d.get("thinking_tokens"))
    rl_rows.append(
        {
            "model": d["model_display_name"],
            "provider": d["provider"],
            "task": slug,
            "input_tokens": it,
            "output_tokens": ot,
            "thinking_tokens": tt,
            "total_tokens": it + ot + tt,
            "score": score,
        }
    )

rl_df = pd.DataFrame(rl_rows)
print(f"  RL runs: {len(rl_df)}, nonzero tokens: {(rl_df.total_tokens > 0).sum()}")

# Overall correlation
rho_rl, p_rl = stats.spearmanr(rl_df["total_tokens"], rl_df["score"])
print(f"  Spearman (tokens vs score, all RL): rho={rho_rl:.3f}, p={p_rl:.2e}")

# Within-task correlation (controls for task difficulty)
within_rhos = []
for task, grp in rl_df.groupby("task"):
    if len(grp) >= 5 and grp.total_tokens.std() > 0:
        r, _ = stats.spearmanr(grp["total_tokens"], grp["score"])
        within_rhos.append(r)
within_mean = float(np.nanmean(within_rhos))
within_median = float(np.nanmedian(within_rhos))
print(
    f"  Within-task rho: mean={within_mean:.3f}, median={within_median:.3f}, n_tasks={len(within_rhos)}"
)

# Solved (>0.5) vs failed (<=0.5)
solved_df = rl_df[rl_df.score > 0.5]
failed_df = rl_df[rl_df.score <= 0.5]
u_stat, u_p = stats.mannwhitneyu(
    solved_df["total_tokens"], failed_df["total_tokens"], alternative="two-sided"
)
solved_mean = solved_df.total_tokens.mean()
failed_mean = failed_df.total_tokens.mean()
token_ratio = failed_mean / max(solved_mean, 1)
print(
    f"  Solved mean tokens: {solved_mean:.0f}, Failed mean tokens: {failed_mean:.0f} "
    f"(ratio={token_ratio:.1f}x)"
)
print(f"  Mann-Whitney: U={u_stat:.0f}, p={u_p:.2e}")

# Score quartile token usage
rl_df["quartile"] = pd.qcut(
    rl_df["score"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop"
)
quartile_tokens = rl_df.groupby("quartile", observed=True)["total_tokens"].agg(
    ["mean", "median", "count"]
).reset_index()
quartile_tokens.to_csv(OUT / "token_failure_by_quartile.csv", index=False)

tf_summary = {
    "n_rl_runs": len(rl_df),
    "n_rl_tasks": rl_df.task.nunique(),
    "overall_spearman_rho": round(rho_rl, 4),
    "overall_spearman_p": round(p_rl, 8),
    "within_task_rho_mean": round(within_mean, 4),
    "within_task_rho_median": round(within_median, 4),
    "n_tasks_within": len(within_rhos),
    "solved_mean_tokens": round(solved_mean, 0),
    "failed_mean_tokens": round(failed_mean, 0),
    "token_ratio_failed_over_solved": round(token_ratio, 2),
    "mannwhitney_u": round(u_stat, 0),
    "mannwhitney_p": round(u_p, 8),
    "n_solved": len(solved_df),
    "n_failed": len(failed_df),
}
pd.DataFrame([tf_summary]).to_csv(OUT / "token_failure_summary.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Provider Rule-Induction Deficit
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 4: Provider rule-induction deficit ===")

model_df = pd.read_csv(MODEL_STATS)
provider_df = pd.read_csv(PROVIDER_STATS)

# Rule-induction = average of concept + observational
model_df["rule_induction"] = (model_df["concept"] + model_df["observational"]) / 2
model_df["rule_induction_rank"] = model_df["rule_induction"].rank(ascending=False)
model_df["rank_delta"] = model_df["rule_induction_rank"] - model_df["rank_overall"]

# Provider-level aggregation for rule-induction vs overall
ri = (
    provider_df[provider_df["category"].isin(["concept", "observational"])]
    .groupby("provider")["mean_score"]
    .mean()
    .reset_index()
    .rename(columns={"mean_score": "rule_induction_mean"})
)
ov = (
    provider_df.groupby("provider")["mean_score"]
    .mean()
    .reset_index()
    .rename(columns={"mean_score": "overall_mean"})
)
prov_ri = ri.merge(ov, on="provider")
prov_ri["deficit"] = prov_ri["overall_mean"] - prov_ri["rule_induction_mean"]
prov_ri["pct_below_overall"] = (prov_ri["deficit"] / prov_ri["overall_mean"] * 100).round(1)
prov_ri.to_csv(OUT / "provider_rule_induction.csv", index=False)
print(prov_ri.round(3).to_string(index=False))

# Mann-Whitney: Anthropic+OpenAI vs Google+Open-source on rule_induction
ac_oi = model_df[model_df["provider"].isin(["Anthropic", "OpenAI"])]["rule_induction"].values
g_os = model_df[model_df["provider"].isin(["Google", "Open-source"])]["rule_induction"].values
u_prov, p_prov = stats.mannwhitneyu(ac_oi, g_os, alternative="less")
print(f"  Anthropic+OpenAI mean RI: {ac_oi.mean():.3f}")
print(f"  Google+Open-source mean RI: {g_os.mean():.3f}")
print(f"  Mann-Whitney (AC+OI < G+OS): U={u_prov:.0f}, p={p_prov:.4f}")

# Per-model rule_induction deficit table (for writeup)
model_df_out = model_df[
    ["model", "provider", "overall_mean", "rank_overall", "rule_induction", "rule_induction_rank", "rank_delta"]
].sort_values("rank_overall")
model_df_out.to_csv(OUT / "model_rule_induction_table.csv", index=False)

# Summary
ri_summary = {
    "anthropic_openai_mean_ri": round(float(ac_oi.mean()), 4),
    "google_opensource_mean_ri": round(float(g_os.mean()), 4),
    "mannwhitney_u": round(u_prov, 0),
    "mannwhitney_p": round(p_prov, 4),
    "anthropic_concept_mean": round(
        float(model_df[model_df.provider == "Anthropic"]["concept"].mean()), 4
    ),
    "openai_concept_mean": round(
        float(model_df[model_df.provider == "OpenAI"]["concept"].mean()), 4
    ),
    "google_concept_mean": round(
        float(model_df[model_df.provider == "Google"]["concept"].mean()), 4
    ),
    "opensource_concept_mean": round(
        float(model_df[model_df.provider == "Open-source"]["concept"].mean()), 4
    ),
    "anthropic_observational_mean": round(
        float(model_df[model_df.provider == "Anthropic"]["observational"].mean()), 4
    ),
    "openai_observational_mean": round(
        float(model_df[model_df.provider == "OpenAI"]["observational"].mean()), 4
    ),
}
pd.DataFrame([ri_summary]).to_csv(OUT / "provider_rule_induction_summary.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Thinking vs Instruct (Qwen pair)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 5: Thinking model gains ===")

qwen_t = model_df[model_df["model"] == "Qwen 3 Next 80B Thinking"].iloc[0]
qwen_i = model_df[model_df["model"] == "Qwen 3 Next 80B Instruct"].iloc[0]

think_rows = []
for cat in ["associative", "concept", "language", "observational", "rl", "overall_mean"]:
    t_val = float(qwen_t[cat])
    i_val = float(qwen_i[cat])
    gain_pct = ((t_val - i_val) / max(i_val, 0.001)) * 100
    think_rows.append(
        {
            "category": cat,
            "thinking": round(t_val, 4),
            "instruct": round(i_val, 4),
            "gain_pct": round(gain_pct, 1),
        }
    )

think_df = pd.DataFrame(think_rows)
think_df.to_csv(OUT / "thinking_vs_instruct.csv", index=False)
print(think_df.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Benchmark Summary Stats (for abstract/results section)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 6: Benchmark summary stats ===")

rb_df = pd.read_csv(RANDOM_BASELINE)
loo_df = pd.read_csv(LOO_GLOBAL)

summary = {
    "total_tasks": 138,
    "total_models": 14,
    "total_runs": len(logs),
    "n_categories": 5,
    "loo_min_spearman": round(float(loo_df["spearman_with_baseline"].min()), 4),
    "loo_mean_spearman": round(float(loo_df["spearman_with_baseline"].mean()), 4),
    "loo_max_rank_change": int(loo_df["max_rank_change"].max()),
    "loo_tasks_causing_2plus_moves": int((loo_df["models_moved_2plus"] > 0).sum()),
    "random_baseline_associative": float(rb_df[rb_df.category == "associative"]["random_baseline"].values[0]),
    "random_baseline_concept": float(rb_df[rb_df.category == "concept"]["random_baseline"].values[0]),
    "random_baseline_rl": float(rb_df[rb_df.category == "rl"]["random_baseline"].values[0]),
    "best_model": model_df.sort_values("rank_overall").iloc[0]["model"],
    "best_model_score": round(float(model_df.sort_values("rank_overall").iloc[0]["overall_mean"]), 4),
}
pd.DataFrame([summary]).to_csv(OUT / "benchmark_summary.csv", index=False)
print(pd.DataFrame([summary]).T.to_string())


# ════════════════════════════════════════════════════════════════════════════
# MASTER NUMBERS FILE — for copy-paste into writeup
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Writing master numbers file ===")

leaderboard = model_df[
    ["model", "provider", "tier", "associative", "concept", "language", "observational", "rl", "overall_mean", "rank_overall"]
].sort_values("rank_overall")
leaderboard.columns = [
    "Model", "Provider", "Tier",
    "Assoc.", "Concept", "Language", "Observ.", "RL", "Overall", "Rank"
]
leaderboard = leaderboard.round(3)
leaderboard.to_csv(OUT / "leaderboard.csv", index=False)

print("\n✓ All outputs written to analysis/outputs/novelty_claims/")
print("  - active_querying_per_model.csv")
print("  - active_querying_summary.csv")
print("  - efficiency_ablation_stability.csv")
print("  - efficiency_ablation_summary.csv")
print("  - token_failure_by_quartile.csv")
print("  - token_failure_summary.csv")
print("  - provider_rule_induction.csv")
print("  - provider_rule_induction_summary.csv")
print("  - model_rule_induction_table.csv")
print("  - thinking_vs_instruct.csv")
print("  - benchmark_summary.csv")
print("  - leaderboard.csv")
