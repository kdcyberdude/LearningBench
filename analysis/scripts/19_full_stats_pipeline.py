"""
19_full_stats_pipeline.py

Pipeline:
  1. Build scores matrix from leaderboard JSONs (134 tasks x 14 models)
  2. Download all 134 kernel outputs to get token/timing for each task
     (each kernel stores data for the most-recently-run model)
  3. Merge into a single CSV: task, category, model, score, avg_input_tokens,
     avg_output_tokens, wall_sec, n_requests
  4. Compute aggregated stats: per-model x per-category and overall
  5. Save analysis/outputs/full_task_model_stats.csv
     and analysis/outputs/aggregate_stats.csv

Usage:
  python analysis/scripts/19_full_stats_pipeline.py
  python analysis/scripts/19_full_stats_pipeline.py --skip-download  # use cached
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

KAGGLE_USER = "kdcyberdude"
PROJECT_ROOT = Path(__file__).parent.parent.parent
LB_DIR       = PROJECT_ROOT / "leaderboards2"
OUT_DIR      = PROJECT_ROOT / "analysis" / "outputs"
KERNEL_DIR   = OUT_DIR / "kernel_logs_all"
KERNEL_DIR.mkdir(parents=True, exist_ok=True)

# Category derived from which leaderboard file the task appears in
LB_TO_CATEGORY = {
    "kdcyberdude_associativelearning_leaderboard.json":  "Associative",
    "kdcyberdude_conceptlearning_leaderboard.json":      "Concept Formation",
    "kdcyberdude_languagelearning_leaderboard.json":     "Language Learning",
    "kdcyberdude_observationallearning_leaderboard.json": "Observational",
    "kdcyberdude_rlbench_leaderboard.json":              "Reinforcement Learning",
}

MODEL_DISPLAY = {
    "gemini-3-1-pro-preview":            "Gemini 3.1 Pro Preview",
    "glm-5":                             "GLM-5",
    "qwen3-next-80b-a3b-thinking":       "Qwen 3 Next 80B Thinking",
    "gpt-5-4-2026-03-05":                "GPT-5.4",
    "claude-opus-4-6default":            "Claude Opus 4.6",
    "gemini-2-5-flash":                  "Gemini 2.5 Flash",
    "claude-sonnet-4-6default":          "Claude Sonnet 4.6",
    "gemini-3-1-flash-lite-preview":     "Gemini 3.1 Flash-Lite Preview",
    "deepseek-v3-2":                     "DeepSeek V3.2",
    "claude-haiku-4-520251001":          "Claude Haiku 4.5",
    "gpt-5-4-mini-2026-03-17":           "GPT-5.4 mini",
    "gemma-4-26b-a4b":                   "Gemma 4 26B A4B",
    "qwen3-next-80b-a3b-instruct":       "Qwen 3 Next 80B Instruct",
    "gpt-5-4-nano-2026-03-17":           "GPT-5.4 nano",
}

# model_part strings that appear inside run.json filenames
MODEL_PARTS = [
    "anthropic_claude-haiku-4-520251001",
    "anthropic_claude-opus-4-6default",
    "anthropic_claude-sonnet-4-6default",
    "deepseek-ai_deepseek-v3.2",
    "google_gemini-2.5-flash",
    "google_gemini-3.1-flash-lite-preview",
    "google_gemini-3.1-pro-preview",
    "google_gemma-4-26b-a4b",
    "openai_gpt-5.4-2026-03-05",
    "openai_gpt-5.4-mini-2026-03-17",
    "openai_gpt-5.4-nano-2026-03-17",
    "qwen_qwen3-next-80b-a3b-instruct",
    "qwen_qwen3-next-80b-a3b-thinking",
    "zai_glm-5",
]

MP_TO_DISPLAY = {
    "anthropic_claude-haiku-4-520251001":    "Claude Haiku 4.5",
    "anthropic_claude-opus-4-6default":      "Claude Opus 4.6",
    "anthropic_claude-sonnet-4-6default":    "Claude Sonnet 4.6",
    "deepseek-ai_deepseek-v3.2":             "DeepSeek V3.2",
    "google_gemini-2.5-flash":               "Gemini 2.5 Flash",
    "google_gemini-3.1-flash-lite-preview":  "Gemini 3.1 Flash-Lite Preview",
    "google_gemini-3.1-pro-preview":         "Gemini 3.1 Pro Preview",
    "google_gemma-4-26b-a4b":                "Gemma 4 26B A4B",
    "openai_gpt-5.4-2026-03-05":             "GPT-5.4",
    "openai_gpt-5.4-mini-2026-03-17":        "GPT-5.4 mini",
    "openai_gpt-5.4-nano-2026-03-17":        "GPT-5.4 nano",
    "qwen_qwen3-next-80b-a3b-instruct":      "Qwen 3 Next 80B Instruct",
    "qwen_qwen3-next-80b-a3b-thinking":      "Qwen 3 Next 80B Thinking",
    "zai_glm-5":                             "GLM-5",
}


# ---------------------------------------------------------------------------
# Step 1: Build scores from leaderboard JSONs
# ---------------------------------------------------------------------------
def build_scores() -> dict[tuple[str,str], dict]:
    """Returns {(task_slug, model_display): {score, category}} """
    records = {}
    task_category = {}

    for lb_file in sorted(LB_DIR.glob("*.json")):
        category = LB_TO_CATEGORY.get(lb_file.name, "Unknown")
        lb = json.loads(lb_file.read_text())
        for model_row in lb.get("rows", []):
            model_name = model_row.get("modelVersionName", "")
            for tr in model_row.get("taskResults", []):
                slug = tr.get("benchmarkTaskSlug", "").split("/")[-1]
                task_name = tr.get("benchmarkTaskName", "") or slug
                if not slug or len(slug) < 4:
                    continue
                task_category[slug] = (task_name, category)
                score = None
                r = tr.get("result", {})
                if r.get("hasNumericResult"):
                    score = r.get("numericResult", {}).get("value")
                if score is not None:
                    records[(slug, model_name)] = {"score": score, "category": category, "task_name": task_name}
    return records, task_category


# ---------------------------------------------------------------------------
# Step 2: Download kernel outputs for all tasks
# ---------------------------------------------------------------------------
def parse_run_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        start = datetime.fromisoformat(data["startTime"].replace("Z", "+00:00"))
        end   = datetime.fromisoformat(data["endTime"].replace("Z", "+00:00"))
        wall  = round((end - start).total_seconds(), 1)

        convs = data.get("conversations", [])
        lats, inp, out, cost = [], 0, 0, 0
        for conv in convs:
            for req in conv.get("requests", []):
                m = req.get("metrics", {})
                lat = m.get("totalBackendLatencyMs")
                if lat is not None:
                    lats.append(int(lat))
                inp  += int(m.get("inputTokens",  0) or 0)
                out  += int(m.get("outputTokens", 0) or 0)
                cost += int(m.get("inputTokensCostNanodollars",  0) or 0)
                cost += int(m.get("outputTokensCostNanodollars", 0) or 0)

        n = len(lats)
        model_part = None
        model_slug = data.get("modelVersion", {}).get("slug", "")
        for mp in MODEL_PARTS:
            if mp.split("_", 1)[1] in model_slug:
                model_part = mp
                break

        return {
            "model_part":        model_part,
            "model_display":     MP_TO_DISPLAY.get(model_part, model_slug),
            "task_name":         data["taskVersion"]["name"],
            "wall_sec":          wall,
            "n_requests":        n,
            "total_input_tokens":  inp,
            "total_output_tokens": out,
            "avg_input_tokens":  round(inp / n, 1) if n else 0,
            "avg_output_tokens": round(out / n, 1) if n else 0,
            "avg_latency_ms":    round(sum(lats) / n, 0) if n else 0,
        }
    except Exception as e:
        return {"_error": str(e)}


def download_kernel(slug: str, delay: float = 2.0) -> dict | None:
    dest = KERNEL_DIR / slug
    # Check cache first
    existing = list(dest.glob("*.run.json")) if dest.exists() else []
    if existing:
        for f in existing:
            m = parse_run_json(f)
            if "model_display" in m:
                m["_cached"] = True
                return m

    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    time.sleep(delay)
    r = subprocess.run(
        ["kaggle", "kernels", "output",
         f"{KAGGLE_USER}/{slug}", "-p", str(dest), "-q"],
        capture_output=True, text=True, timeout=90,
    )
    run_files = [dest / f for f in os.listdir(dest) if f.endswith(".run.json")] if dest.exists() else []
    if not run_files:
        shutil.rmtree(dest, ignore_errors=True)
        return None
    return parse_run_json(run_files[0])


def download_all_kernels(task_slugs: list[str]) -> dict[str, dict]:
    """Returns task_slug -> metrics dict"""
    results = {}
    total = len(task_slugs)
    for i, slug in enumerate(task_slugs):
        m = download_kernel(slug, delay=2.5)
        cached = m.pop("_cached", False) if m else False
        tag = "[cached]" if cached else ""
        if m and "model_display" in m:
            results[slug] = m
            n = m["n_requests"]
            ai = m["avg_input_tokens"]
            ao = m["avg_output_tokens"]
            print(f"  [{i+1:3}/{total}] {slug[:50]:<52} {m['model_display'][:28]:<30} "
                  f"reqs={n:<4} avgIn={ai:<8} avgOut={ao} {tag}")
        else:
            err = m.get("_error","?") if m else "download failed"
            print(f"  [{i+1:3}/{total}] {slug[:50]:<52} ERROR: {err}")
    return results


# ---------------------------------------------------------------------------
# Step 3: Merge and write full CSV
# ---------------------------------------------------------------------------
def build_full_csv(scores: dict, token_data: dict[str,dict]) -> list[dict]:
    """Merges scores with token data. token_data is keyed by task_slug."""
    rows = []
    for (task_slug, model_display), info in sorted(scores.items()):
        td = token_data.get(task_slug, {})
        # Only attach token data if it matches this model
        td_model = td.get("model_display", "")
        has_token = (td_model == model_display)

        rows.append({
            "task_slug":          task_slug,
            "task_name":          info["task_name"],
            "category":           info["category"],
            "model":              model_display,
            "score":              round(info["score"], 4),
            "avg_input_tokens":   td.get("avg_input_tokens", "") if has_token else "",
            "avg_output_tokens":  td.get("avg_output_tokens", "") if has_token else "",
            "wall_sec":           td.get("wall_sec", "") if has_token else "",
            "n_requests":         td.get("n_requests", "") if has_token else "",
            "total_input_tokens": td.get("total_input_tokens", "") if has_token else "",
            "total_output_tokens":td.get("total_output_tokens", "") if has_token else "",
            "avg_latency_ms":     td.get("avg_latency_ms", "") if has_token else "",
        })
    return rows


# ---------------------------------------------------------------------------
# Step 4: Aggregate stats
# ---------------------------------------------------------------------------
def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    import statistics

    # Group scores and token data by (model, category) and (model, "Overall")
    def safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    groups: dict[tuple, dict] = defaultdict(lambda: {
        "scores": [], "avg_in": [], "avg_out": [], "wall": [], "n_req": []
    })

    for row in rows:
        for grp_cat in [row["category"], "Overall"]:
            key = (row["model"], grp_cat)
            g = groups[key]
            s = safe_float(row["score"])
            if s is not None:
                g["scores"].append(s)
            if row["avg_input_tokens"] != "":
                g["avg_in"].append(safe_float(row["avg_input_tokens"]))
                g["avg_out"].append(safe_float(row["avg_output_tokens"]))
                g["wall"].append(safe_float(row["wall_sec"]))
                g["n_req"].append(safe_float(row["n_requests"]))

    def mean_r(lst, decimals=1):
        lst = [x for x in lst if x is not None]
        return round(statistics.mean(lst), decimals) if lst else ""

    agg_rows = []
    for (model, category), g in sorted(groups.items()):
        agg_rows.append({
            "model":                model,
            "category":             category,
            "n_tasks":              len(g["scores"]),
            "mean_score":           mean_r(g["scores"], 4),
            "avg_input_tokens":     mean_r(g["avg_in"], 1),
            "avg_output_tokens":    mean_r(g["avg_out"], 1),
            "avg_wall_sec":         mean_r(g["wall"], 1),
            "avg_n_requests":       mean_r(g["n_req"], 1),
            "n_token_samples":      len(g["avg_in"]),
        })
    return agg_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip kernel downloads; use only cached files")
    args = parser.parse_args()

    print("=" * 70)
    print("Step 1: Building scores from leaderboard JSONs...")
    scores, task_category = build_scores()
    all_task_slugs = sorted({slug for slug, _ in scores.keys()})
    print(f"  {len(scores)} (task, model) score pairs across {len(all_task_slugs)} tasks")

    if not args.skip_download:
        print(f"\nStep 2: Downloading {len(all_task_slugs)} kernel outputs...")
        print("  (rate-limited to ~1 request per 2.5 s; expect ~6 min)")
        token_data = download_all_kernels(all_task_slugs)
    else:
        print("\nStep 2: Loading cached kernel data (--skip-download)...")
        token_data = {}
        for slug in all_task_slugs:
            dest = KERNEL_DIR / slug
            existing = list(dest.glob("*.run.json")) if dest.exists() else []
            if existing:
                m = parse_run_json(existing[0])
                if "model_display" in m:
                    token_data[slug] = m

    coverage = len(token_data)
    print(f"\n  Token data obtained for {coverage}/{len(all_task_slugs)} tasks")

    print("\nStep 3: Merging scores + token data...")
    full_rows = build_full_csv(scores, token_data)
    full_csv  = OUT_DIR / "full_task_model_stats.csv"
    fieldnames = ["task_slug","task_name","category","model","score",
                  "avg_input_tokens","avg_output_tokens","wall_sec",
                  "n_requests","total_input_tokens","total_output_tokens","avg_latency_ms"]
    with open(full_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(full_rows)
    print(f"  Saved → {full_csv}  ({len(full_rows)} rows)")

    print("\nStep 4: Computing aggregate stats...")
    agg_rows = aggregate(full_rows)
    agg_csv  = OUT_DIR / "aggregate_stats.csv"
    agg_fields = ["model","category","n_tasks","mean_score",
                  "avg_input_tokens","avg_output_tokens","avg_wall_sec",
                  "avg_n_requests","n_token_samples"]
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"  Saved → {agg_csv}  ({len(agg_rows)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
