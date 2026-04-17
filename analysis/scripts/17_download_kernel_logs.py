"""
17_download_kernel_logs.py — Parallel download of all model run.json files across all tasks.

Architecture reality (discovered during investigation):
  - Each task is a Kaggle kernel; each model run = a new kernel version.
  - Kaggle only retains output files from the LATEST version per kernel.
  - Therefore, only ONE model's run.json is directly accessible per task kernel.
  - To get ALL 14 models' run.json files, we must find 14 different task kernels
    where each model is the CURRENT (latest) version.
  - Strategy: scan all ~154 task kernels, note the current model for each, then
    use kaggle CLI to download that model's run.json from it.
  - Result: per-model timing/token data sampled from ONE representative task per model.
    For cross-task analysis, we need to run this once per new benchmark round.

Output:
  analysis/outputs/kernel_logs/
    {model_part}/{task_prefix}-run_id_Run_1_{model_part}.run.json
  analysis/outputs/kernel_logs/manifest.json   — model → {task, kernel, scriptVersionId}
  analysis/outputs/kernel_logs_parsed.csv      — flat table: model, task, metrics

Usage:
  python analysis/scripts/17_download_kernel_logs.py
  python analysis/scripts/17_download_kernel_logs.py --task-only inhibitory-summation-assoc-learning
  python analysis/scripts/17_download_kernel_logs.py --workers 8
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE_USER = "kdcyberdude"
KAGGLE_KEY  = "bd24a8e62cb84624fb000acb4f47c8d3"
API_BASE    = "https://api.kaggle.com/v1/kernels.KernelsApiService"

PROJECT_ROOT = Path(__file__).parent.parent.parent
LEADERBOARD_DIR = PROJECT_ROOT / "leaderboards"
OUT_DIR = Path(__file__).parent.parent / "outputs" / "kernel_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All 14 model filename parts as they appear in run.json filenames
# Format: provider_model-slug  (/ → _)
# Some models have naming quirks noted below
ALL_MODEL_PARTS = [
    "anthropic_claude-haiku-4-520251001",   # NB: no hyphen before "20251001"
    "anthropic_claude-opus-4-6default",     # NB: no hyphen before "default"
    "anthropic_claude-sonnet-4-6default",
    "deepseek-ai_deepseek-v3.2",            # NB: provider is "deepseek-ai" not "deepseek"
    "google_gemini-2.5-flash",
    "google_gemini-3.1-flash-lite-preview",
    "google_gemini-3.1-pro-preview",
    "google_gemma-4-26b-a4b",               # NB: truncated — missing "-it" suffix in filenames
    "openai_gpt-5.4-2026-03-05",
    "openai_gpt-5.4-mini-2026-03-17",
    "openai_gpt-5.4-nano-2026-03-17",
    "qwen_qwen3-next-80b-a3b-instruct",
    "qwen_qwen3-next-80b-a3b-thinking",
    "zai_glm-5",
]

# Normalised display names for each model filename part
MODEL_DISPLAY = {
    "anthropic_claude-haiku-4-520251001":    "Claude Haiku 4.5",
    "anthropic_claude-opus-4-6default":      "Claude Opus 4.6",
    "anthropic_claude-sonnet-4-6default":    "Claude Sonnet 4.6",
    "deepseek-ai_deepseek-v3.2":             "DeepSeek V3.2",
    "google_gemini-2.5-flash":               "Gemini 2.5 Flash",
    "google_gemini-3.1-flash-lite-preview":  "Gemini 3.1 Flash-Lite",
    "google_gemini-3.1-pro-preview":         "Gemini 3.1 Pro",
    "google_gemma-4-26b-a4b":                "Gemma 4 26B",
    "openai_gpt-5.4-2026-03-05":             "GPT-5.4",
    "openai_gpt-5.4-mini-2026-03-17":        "GPT-5.4 mini",
    "openai_gpt-5.4-nano-2026-03-17":        "GPT-5.4 nano",
    "qwen_qwen3-next-80b-a3b-instruct":      "Qwen3 80B Instruct",
    "qwen_qwen3-next-80b-a3b-thinking":      "Qwen3 80B Thinking",
    "zai_glm-5":                             "GLM-5",
}

LEADERBOARD_FILES = [
    "kdcyberdude_associativelearning_leaderboard.json",
    "kdcyberdude_conceptlearning_leaderboard.json",
    "kdcyberdude_observationallearning_leaderboard.json",
    "kdcyberdude_rlbench_leaderboard.json",
    "kdcyberdude_languagelearning_leaderboard.json",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _curl_post_silent(endpoint: str, payload: dict) -> dict:
    r = subprocess.run(
        ["curl", "-s", "-X", "POST",
         "-H", "Content-Type: application/json",
         "-u", f"{KAGGLE_USER}:{KAGGLE_KEY}",
         "-d", json.dumps(payload),
         endpoint],
        capture_output=True, text=True, timeout=20,
    )
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def _curl_post_location(endpoint: str, payload: dict) -> str | None:
    """POST and return the Location redirect header value, or None."""
    r = subprocess.run(
        ["curl", "-v", "-s", "-X", "POST",
         "-H", "Content-Type: application/json",
         "-u", f"{KAGGLE_USER}:{KAGGLE_KEY}",
         "-d", json.dumps(payload),
         endpoint],
        capture_output=True, text=True, timeout=20,
    )
    m = re.search(r"location: (https://[^\r\n]+)", r.stderr, re.IGNORECASE)
    return m.group(1).strip() if m else None


def collect_all_kernels() -> list[str]:
    """Return sorted list of unique task kernel slugs from all leaderboards."""
    kernels: set[str] = set()
    for lb_name in LEADERBOARD_FILES:
        lb_path = LEADERBOARD_DIR / lb_name
        if not lb_path.exists():
            print(f"  [warn] missing leaderboard: {lb_path}")
            continue
        lb = json.loads(lb_path.read_text())
        for row in lb.get("rows", []):
            for tr in row.get("taskResults", []):
                slug = tr.get("benchmarkTaskSlug", "").split("/")[-1]
                if slug and not slug.isdigit() and len(slug) > 4:
                    kernels.add(slug)
    return sorted(kernels)


def get_current_model_for_kernel(kernel_slug: str) -> tuple[str, str] | None:
    """
    Return (model_part, task_prefix) for the CURRENT (latest) version's run.json,
    or None if not found / not a benchmark task kernel.
    """
    d = _curl_post_silent(f"{API_BASE}/ListKernelFiles",
                          {"userName": KAGGLE_USER, "kernelSlug": kernel_slug})
    for f in d.get("files", []):
        name = f["name"]
        if name.endswith(".run.json") and "run_id_Run_1_" in name:
            model_part = name.split("run_id_Run_1_")[1].replace(".run.json", "")
            task_prefix = name.split("-run_id_Run_1_")[0]
            return model_part, task_prefix
    return None


def get_script_version_id(kernel_slug: str) -> str | None:
    """Get scriptVersionId for the current version of a kernel via task.json redirect."""
    d = _curl_post_silent(f"{API_BASE}/GetKernel",
                          {"userName": KAGGLE_USER, "kernelSlug": kernel_slug})
    v_num = d.get("metadata", {}).get("currentVersionNumber", 0)
    if not v_num:
        return None

    task_prefix = kernel_slug.replace("-", "_")
    url = _curl_post_location(f"{API_BASE}/DownloadKernelOutput", {
        "ownerSlug": KAGGLE_USER,
        "kernelSlug": kernel_slug,
        "filePath": f"{task_prefix}.task.json",
        "versionNumber": v_num,
    })
    if not url:
        return None
    m = re.search(r"/kf/(\d+)/", url)
    return m.group(1) if m else None


def download_kernel_output(kernel_slug: str, dest_dir: Path) -> list[str]:
    """
    Use kaggle CLI to download all output files for the current version.
    Returns list of downloaded filenames.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["kaggle", "kernels", "output",
         f"{KAGGLE_USER}/{kernel_slug}", "-p", str(dest_dir), "-q"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return []
    return [f for f in os.listdir(dest_dir) if f.endswith(".run.json")]


def parse_run_json(path: Path) -> dict:
    """Extract metrics from a run.json file. Returns {} on failure."""
    try:
        data = json.loads(path.read_text())
        start = datetime.fromisoformat(data["startTime"].replace("Z", "+00:00"))
        end   = datetime.fromisoformat(data["endTime"].replace("Z", "+00:00"))
        total_sec = (end - start).total_seconds()

        convs = data.get("conversations", [])
        latencies, input_toks, output_toks, cost_nano = [], 0, 0, 0
        for conv in convs:
            for req in conv.get("requests", []):
                m = req.get("metrics", {})
                lat = m.get("totalBackendLatencyMs")
                if lat is not None:
                    latencies.append(int(lat))
                input_toks  += int(m.get("inputTokens",  0) or 0)
                output_toks += int(m.get("outputTokens", 0) or 0)
                cost_nano   += int(m.get("inputTokensCostNanodollars", 0) or 0)

        results = data.get("results", [])
        score = None
        for res in results:
            if res.get("type") == "AGGREGATED":
                nr = res.get("numericResult", {})
                if "value" in nr:
                    score = nr["value"]
                    break

        return {
            "model_slug":       data["modelVersion"]["slug"],
            "task_name":        data["taskVersion"]["name"],
            "total_wall_sec":   total_sec,
            "n_api_requests":   len(latencies),
            "median_lat_ms":    sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "mean_lat_ms":      int(sum(latencies) / len(latencies)) if latencies else 0,
            "total_lat_ms":     sum(latencies),
            "input_tokens":     input_toks,
            "output_tokens":    output_toks,
            "cost_nanodollars": cost_nano,
            "conversations":    len(convs),
            "score":            score,
            "start_time":       data["startTime"],
        }
    except Exception as e:
        return {"_parse_error": str(e)}


# ---------------------------------------------------------------------------
# Phase 1: Scan all kernels to build model → kernel mapping
# ---------------------------------------------------------------------------
def scan_kernels_for_model_map(
    all_kernels: list[str],
    workers: int = 8,
) -> dict[str, dict]:
    """
    Returns model_part → {kernel_slug, task_prefix} for each model that is
    the CURRENT version in at least one kernel.

    Scans in parallel; stops once all 14 models are found.
    """
    remaining = set(ALL_MODEL_PARTS)
    model_map: dict[str, dict] = {}
    lock_done = False

    print(f"\n[Phase 1] Scanning {len(all_kernels)} kernels to find one per model...")

    def scan_one(kernel_slug: str):
        nonlocal lock_done
        if lock_done:
            return None
        result = get_current_model_for_kernel(kernel_slug)
        if result:
            model_part, task_prefix = result
            return kernel_slug, model_part, task_prefix
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_one, ks): ks for ks in all_kernels}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            kernel_slug, model_part, task_prefix = res
            if model_part in remaining and model_part not in model_map:
                model_map[model_part] = {
                    "kernel_slug": kernel_slug,
                    "task_prefix": task_prefix,
                }
                remaining.discard(model_part)
                display = MODEL_DISPLAY.get(model_part, model_part)
                print(f"  ✓ {display:<28} ← {kernel_slug}")
                if not remaining:
                    lock_done = True
                    # Cancel remaining futures quickly
                    for f in futures:
                        f.cancel()
                    break
            time.sleep(0.0)   # yield

    if remaining:
        print(f"  [warn] {len(remaining)} models not found as current in any kernel:")
        for mp in sorted(remaining):
            print(f"    – {MODEL_DISPLAY.get(mp, mp)} ({mp})")
    return model_map


# ---------------------------------------------------------------------------
# Phase 2: Download run.json for each found model
# ---------------------------------------------------------------------------
def download_model_logs(
    model_map: dict[str, dict],
    workers: int = 6,
) -> dict[str, dict]:
    """
    Downloads run.json for each model from its source kernel.
    Returns model_part → {path, script_version_id, metrics}
    """
    print(f"\n[Phase 2] Downloading {len(model_map)} run.json files...")

    def download_one(model_part: str, info: dict) -> tuple[str, dict]:
        kernel_slug = info["kernel_slug"]
        task_prefix = info["task_prefix"]

        dest_dir = OUT_DIR / model_part
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{task_prefix}-run_id_Run_1_{model_part}.run.json"
        local_path = dest_dir / filename

        # Skip if already downloaded and non-empty
        if local_path.exists() and local_path.stat().st_size > 200:
            metrics = parse_run_json(local_path)
            vid = get_script_version_id(kernel_slug)
            return model_part, {
                "path": str(local_path),
                "script_version_id": vid,
                "kaggle_url": f"https://www.kaggle.com/code/{KAGGLE_USER}/{kernel_slug}?scriptVersionId={vid}" if vid else "",
                "metrics": metrics,
                "cached": True,
            }

        # Download via kaggle CLI
        tmp_dir = OUT_DIR / f"_tmp_{model_part[:20]}"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        r = subprocess.run(
            ["kaggle", "kernels", "output",
             f"{KAGGLE_USER}/{kernel_slug}", "-p", str(tmp_dir), "-q"],
            capture_output=True, text=True, timeout=60,
        )

        # Find the run.json for our model in the downloaded files
        if not tmp_dir.exists():
            return model_part, {"error": "CLI failed"}

        run_file = None
        for f in tmp_dir.iterdir():
            if f.name.endswith(".run.json") and f"run_id_Run_1_{model_part}" in f.name:
                run_file = f
                break

        if run_file is None or run_file.stat().st_size < 100:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return model_part, {"error": f"run.json not found in {kernel_slug}"}

        # Move to permanent location
        shutil.copy2(run_file, local_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        vid = get_script_version_id(kernel_slug)
        metrics = parse_run_json(local_path)

        return model_part, {
            "path": str(local_path),
            "script_version_id": vid,
            "kaggle_url": f"https://www.kaggle.com/code/{KAGGLE_USER}/{kernel_slug}?scriptVersionId={vid}" if vid else "",
            "metrics": metrics,
            "cached": False,
        }

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, mp, info): mp
            for mp, info in model_map.items()
        }
        for fut in concurrent.futures.as_completed(futures):
            model_part = futures[fut]
            mp, result = fut.result()
            results[mp] = result
            display = MODEL_DISPLAY.get(mp, mp)
            if "error" in result:
                print(f"  ✗ {display:<28} ERROR: {result['error']}")
            else:
                m = result.get("metrics", {})
                cached = " (cached)" if result.get("cached") else ""
                score_str = f"{m.get('score', 'N/A')}"
                print(
                    f"  ✓ {display:<28} score={score_str:<7} "
                    f"wall={m.get('total_wall_sec', 0):.0f}s  "
                    f"backend={m.get('total_lat_ms', 0):,}ms  "
                    f"in={m.get('input_tokens', 0):,} out={m.get('output_tokens', 0):,}"
                    f"{cached}"
                )

    return results


# ---------------------------------------------------------------------------
# Phase 3: Save manifest + flat CSV
# ---------------------------------------------------------------------------
def save_outputs(
    model_map: dict[str, dict],
    download_results: dict[str, dict],
) -> None:
    import csv

    # Manifest JSON
    manifest = {}
    for mp, info in model_map.items():
        dl = download_results.get(mp, {})
        manifest[mp] = {
            "display_name":     MODEL_DISPLAY.get(mp, mp),
            "source_kernel":    info["kernel_slug"],
            "task_prefix":      info["task_prefix"],
            "script_version_id": dl.get("script_version_id"),
            "kaggle_url":       dl.get("kaggle_url", ""),
            "local_path":       dl.get("path", ""),
            "metrics":          dl.get("metrics", {}),
        }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Saved manifest → {manifest_path}")

    # Flat CSV
    csv_path = OUT_DIR.parent / "kernel_logs_parsed.csv"
    fieldnames = [
        "model_part", "display_name", "source_kernel", "script_version_id", "kaggle_url",
        "score", "total_wall_sec", "n_api_requests", "median_lat_ms", "mean_lat_ms",
        "total_lat_ms", "input_tokens", "output_tokens", "cost_nanodollars",
        "conversations", "task_name", "start_time",
    ]
    rows = []
    for mp, info in sorted(manifest.items()):
        m = info.get("metrics", {})
        rows.append({
            "model_part":         mp,
            "display_name":       info["display_name"],
            "source_kernel":      info["source_kernel"],
            "script_version_id":  info.get("script_version_id", ""),
            "kaggle_url":         info.get("kaggle_url", ""),
            "score":              m.get("score", ""),
            "total_wall_sec":     m.get("total_wall_sec", ""),
            "n_api_requests":     m.get("n_api_requests", ""),
            "median_lat_ms":      m.get("median_lat_ms", ""),
            "mean_lat_ms":        m.get("mean_lat_ms", ""),
            "total_lat_ms":       m.get("total_lat_ms", ""),
            "input_tokens":       m.get("input_tokens", ""),
            "output_tokens":      m.get("output_tokens", ""),
            "cost_nanodollars":   m.get("cost_nanodollars", ""),
            "conversations":      m.get("conversations", ""),
            "task_name":          m.get("task_name", ""),
            "start_time":         m.get("start_time", ""),
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV      → {csv_path}")

    # Human-readable summary table
    print(f"\n{'='*110}")
    print(f"{'Model':<28} {'Src Task':<40} {'Score':<8} {'Wall(s)':<9} {'Backend(ms)':<13} {'InTok':<9} {'OutTok'}")
    print("-" * 110)
    for row in sorted(rows, key=lambda r: -(float(r["score"]) if r["score"] != "" else 0)):
        print(
            f"{row['display_name']:<28} {row['source_kernel'][:39]:<40} "
            f"{str(row['score']):<8} {str(row['total_wall_sec'])[:7]:<9} "
            f"{str(row['total_lat_ms']):<13} {str(row['input_tokens']):<9} {row['output_tokens']}"
        )

    # Kaggle URLs
    print(f"\n{'='*110}")
    print("Kaggle URLs (one per model — links to the exact notebook version):")
    print(f"{'='*110}")
    for row in sorted(rows, key=lambda r: r["display_name"]):
        print(f"  {row['display_name']:<28}  {row['kaggle_url']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download Kaggle kernel run.json files for all models")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers (default: 6)")
    parser.add_argument("--scan-workers", type=int, default=10, help="Workers for kernel scanning (default: 10)")
    parser.add_argument("--task-only", type=str, default=None,
                        help="Only scan a single specific kernel slug (for testing)")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Skip Phase 1 scan; load manifest.json instead")
    args = parser.parse_args()

    print(f"Output directory: {OUT_DIR}")
    all_kernels = collect_all_kernels()
    print(f"Total task kernels: {len(all_kernels)}")

    if args.task_only:
        all_kernels = [args.task_only]

    # Phase 1: find which kernel has each model as current version
    manifest_path = OUT_DIR / "manifest.json"
    if args.skip_scan and manifest_path.exists():
        print("\n[Phase 1] Loading existing manifest (--skip-scan)...")
        existing = json.loads(manifest_path.read_text())
        model_map = {
            mp: {"kernel_slug": v["source_kernel"], "task_prefix": v["task_prefix"]}
            for mp, v in existing.items()
            if v.get("source_kernel")
        }
        print(f"  Loaded {len(model_map)} models from manifest")
    else:
        model_map = scan_kernels_for_model_map(all_kernels, workers=args.scan_workers)

    if not model_map:
        print("No models found. Exiting.")
        sys.exit(1)

    # Phase 2: download run.json files
    download_results = download_model_logs(model_map, workers=args.workers)

    # Phase 3: save outputs
    save_outputs(model_map, download_results)

    n_ok = sum(1 for v in download_results.values() if "error" not in v)
    print(f"\nDone: {n_ok}/{len(model_map)} models downloaded successfully.")


if __name__ == "__main__":
    main()
