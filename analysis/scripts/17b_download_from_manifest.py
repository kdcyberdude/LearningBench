"""
17b_download_from_manifest.py — Fast download using pre-built manifest.

The manifest from script 17 already found which kernel slug hosts each model.
This script just downloads those kernels and extracts metrics — but uses the
correct underscore-based filename matching (the CLI downloads files with
underscored task prefixes even when the kernel slug uses hyphens).

Usage:
  python analysis/scripts/17b_download_from_manifest.py
"""
from __future__ import annotations

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
OUT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "kernel_logs"
MANIFEST_PATH = OUT_DIR / "manifest.json"
CSV_OUT = PROJECT_ROOT / "analysis" / "outputs" / "kernel_logs_parsed.csv"

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

# Models not in manifest (Opus and 3 Google models) — need to find their kernels
EXTRA_MODELS_TO_SCAN = {
    "anthropic_claude-opus-4-6default",
    "google_gemini-3.1-flash-lite-preview",
    "google_gemini-3.1-pro-preview",
}


def parse_run_json(path: Path) -> dict:
    """Extract aggregate metrics from run.json."""
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
                cost_nano   += int(m.get("inputTokensCostNanodollars",  0) or 0)
                cost_nano   += int(m.get("outputTokensCostNanodollars", 0) or 0)

        results = data.get("results", [])
        score = None
        for res in results:
            if res.get("type") == "AGGREGATED":
                nr = res.get("numericResult", {})
                if "value" in nr:
                    score = nr["value"]
                    break

        n_req = len(latencies)
        return {
            "model_slug":       data["modelVersion"]["slug"],
            "task_name":        data["taskVersion"]["name"],
            "total_wall_sec":   round(total_sec, 1),
            "n_api_requests":   n_req,
            "median_lat_ms":    sorted(latencies)[n_req // 2] if latencies else 0,
            "mean_lat_ms":      int(sum(latencies) / n_req) if latencies else 0,
            "total_lat_ms":     sum(latencies),
            "input_tokens":     input_toks,
            "output_tokens":    output_toks,
            "cost_nanodollars": cost_nano,
            "avg_input_tokens": round(input_toks / n_req, 1) if n_req else 0,
            "avg_output_tokens": round(output_toks / n_req, 1) if n_req else 0,
            "conversations":    len(convs),
            "score":            score,
            "start_time":       data["startTime"],
        }
    except Exception as e:
        return {"_parse_error": str(e)}


def download_kernel(kernel_slug: str, dest_dir: Path, delay: float = 2.0) -> list[Path]:
    """Download all output files for a kernel. Returns list of run.json paths."""
    shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    time.sleep(delay)  # be nice to the API
    r = subprocess.run(
        ["kaggle", "kernels", "output",
         f"{KAGGLE_USER}/{kernel_slug}", "-p", str(dest_dir), "-q"],
        capture_output=True, text=True, timeout=90,
    )
    if r.returncode != 0:
        print(f"    [ERROR] kaggle CLI failed: {r.stderr.strip()[:200]}")
        return []
    return [dest_dir / f for f in os.listdir(dest_dir) if f.endswith(".run.json")]


def main():
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        print("Run 17_download_kernel_logs.py first to build the manifest.")
        return

    manifest = json.loads(MANIFEST_PATH.read_text())
    print(f"Loaded manifest with {len(manifest)} models.")

    # Also try to find missing models by scanning some kernels
    missing = [mp for mp in EXTRA_MODELS_TO_SCAN if mp not in manifest or not manifest[mp].get("source_kernel")]
    if missing:
        print(f"\nMissing from manifest: {[MODEL_DISPLAY.get(m,m) for m in missing]}")
        print("Scanning leaderboard JSONs for their kernels...")
        # Load all kernels from leaderboard files
        leaderboard_dir = PROJECT_ROOT / "leaderboards2"
        all_kernels = set()
        for lb_file in leaderboard_dir.glob("*.json"):
            lb = json.loads(lb_file.read_text())
            for row in lb.get("rows", []):
                for tr in row.get("taskResults", []):
                    slug = tr.get("benchmarkTaskSlug", "").split("/")[-1]
                    if slug and len(slug) > 4:
                        all_kernels.add(slug)

        # Quick scan to find missing models
        import concurrent.futures
        import re
        API_BASE = "https://api.kaggle.com/v1/kernels.KernelsApiService"
        found_extra = {}

        def scan_for_missing(kernel_slug):
            r = subprocess.run(
                ["curl", "-s", "-X", "POST",
                 "-H", "Content-Type: application/json",
                 "-u", f"{KAGGLE_USER}:bd24a8e62cb84624fb000acb4f47c8d3",
                 "-d", json.dumps({"userName": KAGGLE_USER, "kernelSlug": kernel_slug}),
                 f"{API_BASE}/ListKernelFiles"],
                capture_output=True, text=True, timeout=15,
            )
            try:
                d = json.loads(r.stdout)
            except Exception:
                return None
            for f in d.get("files", []):
                name = f.get("name", "")
                if name.endswith(".run.json") and "run_id_Run_1_" in name:
                    mp = name.split("run_id_Run_1_")[1].replace(".run.json", "")
                    if mp in missing:
                        task_prefix = name.split("-run_id_Run_1_")[0] if "-run_id_Run_1_" in name else name.split("_run_id_Run_1_")[0]
                        return kernel_slug, mp, task_prefix
            return None

        scanned = 0
        for slug in sorted(all_kernels):
            if not missing:
                break
            time.sleep(0.3)
            result = scan_for_missing(slug)
            scanned += 1
            if result:
                kernel_slug, mp, task_prefix = result
                found_extra[mp] = {"source_kernel": kernel_slug, "task_prefix": task_prefix}
                missing.remove(mp)
                print(f"  ✓ {MODEL_DISPLAY.get(mp,mp)} ← {kernel_slug}")
            if scanned % 20 == 0:
                print(f"  Scanned {scanned} kernels... {len(missing)} still missing")

        # Merge into manifest
        for mp, info in found_extra.items():
            manifest[mp] = {
                "display_name": MODEL_DISPLAY.get(mp, mp),
                "source_kernel": info["source_kernel"],
                "task_prefix": info["task_prefix"],
                "metrics": {}
            }

    # Download each model's run.json
    results = {}
    for i, (model_part, info) in enumerate(manifest.items()):
        kernel_slug = info.get("source_kernel")
        if not kernel_slug:
            print(f"  [SKIP] {MODEL_DISPLAY.get(model_part, model_part)} — no source kernel")
            continue

        display = MODEL_DISPLAY.get(model_part, model_part)
        print(f"\n[{i+1}/{len(manifest)}] {display} ← {kernel_slug}")

        dest_dir = OUT_DIR / model_part
        # Check if we already have a good run.json
        existing = list(dest_dir.glob("*.run.json")) if dest_dir.exists() else []
        if existing:
            run_path = existing[0]
            metrics = parse_run_json(run_path)
            if "input_tokens" in metrics and metrics["input_tokens"] > 0:
                print(f"  [cached] {run_path.name}")
                results[model_part] = metrics
                continue

        run_files = download_kernel(kernel_slug, dest_dir, delay=3.0)
        if not run_files:
            print(f"  [ERROR] No run.json downloaded for {display}")
            continue

        # Find the matching run.json for this model
        model_file = None
        for rf in run_files:
            if f"run_id_Run_1_{model_part}" in rf.name:
                model_file = rf
                break
        if model_file is None and run_files:
            # Take first run.json (single-model kernels)
            model_file = run_files[0]
            print(f"  [warn] Using {model_file.name} (model part not in name)")

        if model_file:
            metrics = parse_run_json(model_file)
            results[model_part] = metrics
            n = metrics.get("n_api_requests", "?")
            inp = metrics.get("input_tokens", "?")
            out = metrics.get("output_tokens", "?")
            wall = metrics.get("total_wall_sec", "?")
            print(f"  ✓ reqs={n}  in={inp}  out={out}  wall={wall}s")
        else:
            print(f"  [ERROR] Could not find run.json for {display}")

    # Update manifest with metrics
    for mp, metrics in results.items():
        if mp in manifest:
            manifest[mp]["metrics"] = metrics
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved manifest → {MANIFEST_PATH}")

    # Write CSV
    fieldnames = [
        "model_part", "display_name", "source_kernel",
        "score", "total_wall_sec", "n_api_requests",
        "median_lat_ms", "mean_lat_ms", "total_lat_ms",
        "input_tokens", "output_tokens",
        "avg_input_tokens", "avg_output_tokens",
        "cost_nanodollars", "conversations", "task_name", "start_time",
    ]
    rows = []
    for mp in sorted(manifest.keys()):
        info = manifest[mp]
        m = info.get("metrics", {})
        rows.append({
            "model_part":         mp,
            "display_name":       MODEL_DISPLAY.get(mp, mp),
            "source_kernel":      info.get("source_kernel", ""),
            "score":              m.get("score", ""),
            "total_wall_sec":     m.get("total_wall_sec", ""),
            "n_api_requests":     m.get("n_api_requests", ""),
            "median_lat_ms":      m.get("median_lat_ms", ""),
            "mean_lat_ms":        m.get("mean_lat_ms", ""),
            "total_lat_ms":       m.get("total_lat_ms", ""),
            "input_tokens":       m.get("input_tokens", ""),
            "output_tokens":      m.get("output_tokens", ""),
            "avg_input_tokens":   m.get("avg_input_tokens", ""),
            "avg_output_tokens":  m.get("avg_output_tokens", ""),
            "cost_nanodollars":   m.get("cost_nanodollars", ""),
            "conversations":      m.get("conversations", ""),
            "task_name":          m.get("task_name", ""),
            "start_time":         m.get("start_time", ""),
        })
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV → {CSV_OUT}")

    # Summary
    print(f"\n{'='*100}")
    print(f"{'Model':<28} {'Task':<35} {'Wall(s)':<9} {'Reqs':<6} {'InTok':<8} {'OutTok':<8} {'AvgIn':<8} {'AvgOut'}")
    print(f"{'-'*100}")
    for row in sorted(rows, key=lambda r: r["display_name"]):
        if row["input_tokens"]:
            print(f"{row['display_name']:<28} {str(row['task_name'])[:34]:<35} "
                  f"{str(row['total_wall_sec']):<9} {str(row['n_api_requests']):<6} "
                  f"{str(row['input_tokens']):<8} {str(row['output_tokens']):<8} "
                  f"{str(row['avg_input_tokens']):<8} {row['avg_output_tokens']}")
        else:
            print(f"{row['display_name']:<28} {'(no data)'}")


if __name__ == "__main__":
    main()
