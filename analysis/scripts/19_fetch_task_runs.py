#!/usr/bin/env python3
"""
19_fetch_task_runs.py — Fetch all model runs for every benchmark task
using the Kaggle internal API (ListBenchmarkTaskRuns).

Unlike the cross-kernel sampling approach in script 17, this method:
  - Uses the browser session to call /api/i/benchmarks.BenchmarkTaskRunService/ListBenchmarkTaskRuns
  - Gets ALL 14 model runs for a given task in a single API call
  - Returns rich metadata: model name, score, latency, token counts, cost, session ID
  - Does NOT require downloading kernel outputs or kernel output files

Authentication:
  Cookies are extracted from your Brave (or Chrome) browser via browser-cookie3
  and cached in .kaggle_session.json at the project root (gitignored).
  Run with --refresh-cookies to re-extract them from the browser.

Usage:
  python3 19_fetch_task_runs.py                         # fetch all tasks
  python3 19_fetch_task_runs.py --task spurious-hue-true-edge-assoc-learning
  python3 19_fetch_task_runs.py --refresh-cookies       # refresh from browser
  python3 19_fetch_task_runs.py --workers 8             # parallel workers
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SESSION_FILE = ROOT / ".kaggle_session.json"
OUTPUTS_DIR = ROOT / "analysis" / "outputs" / "task_runs"
TASKS_DIR = ROOT / "analysis" / "outputs" / "kernel_logs_all"

OWNER_SLUG = "kdcyberdude"
API_URL = "https://www.kaggle.com/api/i/benchmarks.BenchmarkTaskRunService/ListBenchmarkTaskRuns"

# ---------------------------------------------------------------------------
# Cookie management
# ---------------------------------------------------------------------------

def load_session() -> dict:
    """Load saved session from .kaggle_session.json."""
    if not SESSION_FILE.exists():
        sys.exit(
            f"No session file found at {SESSION_FILE}.\n"
            "Run with --refresh-cookies to extract cookies from your Brave browser."
        )
    return json.loads(SESSION_FILE.read_text())


def refresh_cookies(browser: str = "brave") -> dict:
    """Extract fresh cookies from the browser and save to .kaggle_session.json."""
    try:
        import browser_cookie3
    except ImportError:
        sys.exit("Install browser-cookie3: pip install browser-cookie3")

    print(f"Extracting cookies from {browser}...")
    if browser == "brave":
        raw = browser_cookie3.brave(domain_name=".kaggle.com")
    elif browser == "chrome":
        raw = browser_cookie3.chrome(domain_name=".kaggle.com")
    else:
        sys.exit(f"Unsupported browser: {browser}. Use 'brave' or 'chrome'.")

    cookie_dict = {c.name: c.value for c in raw}
    if not cookie_dict:
        sys.exit("No cookies found. Make sure you're logged into Kaggle in that browser.")

    session_data = {
        "_comment": "Kaggle browser session credentials — DO NOT COMMIT (gitignored)",
        "_instructions": "Refresh: python3 analysis/scripts/19_fetch_task_runs.py --refresh-cookies",
        "cookies": cookie_dict,
        "xsrf_token": cookie_dict.get("XSRF-TOKEN", ""),
        "build_version": cookie_dict.get("build-hash", ""),
        "owner_slug": OWNER_SLUG,
    }
    SESSION_FILE.write_text(json.dumps(session_data, indent=2))
    print(f"Saved {len(cookie_dict)} cookies to {SESSION_FILE}")
    return session_data


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def fetch_task_runs(task_slug: str, session: dict, version_number: int | None = None) -> list[dict]:
    """Call ListBenchmarkTaskRuns for a single task slug. Returns list of run dicts."""
    cookies = session["cookies"]
    xsrf = session["xsrf_token"]
    build = session["build_version"]
    owner = session.get("owner_slug", OWNER_SLUG)

    slug_selector: dict = {"taskSlug": task_slug, "ownerSlug": owner}
    if version_number is not None:
        slug_selector["versionNumber"] = version_number

    body = {
        "filter": {
            "taskIdentifier": {"slugSelector": slug_selector},
            "taskRunIds": [],
            "modelVersionIds": [],
        },
        "pageSize": 0,
        "pageToken": "",
        "skip": 0,
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "origin": "https://www.kaggle.com",
        "referer": f"https://www.kaggle.com/benchmarks/tasks/{owner}/{task_slug}",
        "x-xsrf-token": xsrf,
        "x-kaggle-build-version": build,
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        ),
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=headers, cookies=cookies, json=body, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("benchmarkTaskRuns", [])
            elif resp.status_code in (401, 403):
                print(f"  [AUTH ERROR {resp.status_code}] {task_slug} — re-run with --refresh-cookies")
                return []
            else:
                print(f"  [HTTP {resp.status_code}] {task_slug} attempt {attempt+1}")
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            print(f"  [NETWORK ERROR] {task_slug}: {e}")
            time.sleep(2 ** attempt)
    return []


# ---------------------------------------------------------------------------
# Parse a run into a flat dict
# ---------------------------------------------------------------------------

def parse_run(run: dict) -> dict:
    """Flatten a benchmark task run into a single-level dict for CSV."""
    tv = run.get("taskVersion", {})
    mv = run.get("modelVersion", {})
    org = mv.get("organization", {})
    results = run.get("results", [{}])
    r0 = results[0] if results else {}
    metrics = r0.get("totalMetrics", {})
    numeric = r0.get("numericResult", {})
    assertions = r0.get("assertionStatuses", [])

    passed = sum(1 for a in assertions if "PASSED" in a)
    total_a = len(assertions)
    score_frac = passed / total_a if total_a else None
    score_val = numeric.get("value") if isinstance(numeric, dict) else None

    cost_in = metrics.get("inputTokensCostNanodollars", 0) or 0
    cost_out = metrics.get("outputTokensCostNanodollars", 0) or 0
    cost_usd = (cost_in + cost_out) / 1e9

    return {
        "run_id": run.get("id"),
        "task_slug": tv.get("slug", ""),
        "task_name": tv.get("name", ""),
        "task_version_id": tv.get("id"),
        "task_version_number": tv.get("versionNumber"),
        "model_version_id": mv.get("id"),
        "model_slug": mv.get("slug", ""),
        "model_display_name": mv.get("displayName", ""),
        "model_proxy_slug": mv.get("modelProxySlug", ""),
        "provider": org.get("slug", ""),
        "state": run.get("state", ""),
        "start_time": run.get("startTime", ""),
        "end_time": run.get("endTime", ""),
        "score_fraction": score_frac,
        "score_value": score_val,
        "assertions_passed": passed,
        "assertions_total": total_a,
        "input_tokens": metrics.get("inputTokens", 0),
        "output_tokens": metrics.get("outputTokens", 0),
        "thinking_tokens": metrics.get("thinkingTokens", 0),
        "total_latency_ms": metrics.get("totalBackendLatencyMs", 0),
        "input_cost_nanodollars": cost_in,
        "output_cost_nanodollars": cost_out,
        "cost_usd": cost_usd,
        "output_kernel_session_id": run.get("outputKernelSessionId"),
    }


# ---------------------------------------------------------------------------
# Discover tasks
# ---------------------------------------------------------------------------

def discover_tasks() -> list[str]:
    """Find all task slugs from kernel_logs_all output directory."""
    if not TASKS_DIR.exists():
        print(f"Warning: {TASKS_DIR} not found. Run script 17 first or specify --task.")
        return []
    slugs = sorted(
        d.name
        for d in TASKS_DIR.iterdir()
        if d.is_dir() and not d.name.isdigit()
    )
    return slugs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", help="Fetch a single task slug only")
    parser.add_argument("--refresh-cookies", action="store_true", help="Re-extract cookies from browser")
    parser.add_argument("--browser", default="brave", choices=["brave", "chrome"], help="Browser to extract cookies from")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers (default: 6)")
    parser.add_argument("--output", default=str(OUTPUTS_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Session ---
    if args.refresh_cookies:
        session = refresh_cookies(args.browser)
    else:
        session = load_session()
    print(f"Session loaded: {len(session['cookies'])} cookies, XSRF={session['xsrf_token'][:20]}...")

    # --- Task list ---
    if args.task:
        task_slugs = [args.task]
    else:
        task_slugs = discover_tasks()
        if not task_slugs:
            sys.exit("No tasks found. Use --task <slug> or run script 17 first.")
        print(f"Discovered {len(task_slugs)} tasks to fetch.")

    # --- Fetch ---
    all_rows: list[dict] = []
    errors: list[str] = []

    def fetch_and_parse(slug: str) -> tuple[str, list[dict], str | None]:
        runs = fetch_task_runs(slug, session)
        rows = [parse_run(r) for r in runs]
        err = None if rows else f"No runs returned for {slug}"
        return slug, rows, err

    print(f"\nFetching {len(task_slugs)} tasks with {args.workers} workers...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_and_parse, s): s for s in task_slugs}
        for i, fut in enumerate(as_completed(futures), 1):
            slug, rows, err = fut.result()
            if err:
                errors.append(err)
                print(f"  [{i:3}/{len(task_slugs)}] ERROR  {slug}")
            else:
                all_rows.extend(rows)
                # Save per-task JSON
                task_file = out_dir / f"{slug}.json"
                task_file.write_text(json.dumps(rows, indent=2))
                print(f"  [{i:3}/{len(task_slugs)}] OK     {slug} ({len(rows)} runs)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {len(all_rows)} total runs, {len(errors)} errors.")

    # --- Save combined CSV ---
    if all_rows:
        import csv
        csv_file = out_dir / "all_task_runs.csv"
        fieldnames = list(all_rows[0].keys())
        with csv_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved combined CSV → {csv_file}  ({len(all_rows)} rows)")

    # --- Print summary table for single-task mode ---
    if args.task and all_rows:
        print(f"\n{'Model':<32} {'Score':>6} {'LatMs':>8} {'InTok':>7} {'OutTok':>7} {'Think':>7} {'Cost$':>8} {'SessionID':>12}")
        print("-" * 95)
        for row in all_rows:
            score = f"{row['assertions_passed']}/{row['assertions_total']}" if row['assertions_total'] else str(row['score_value'] or "N/A")
            print(
                f"{row['model_display_name']:<32} {score:>6} "
                f"{row['total_latency_ms']:>8} {row['input_tokens']:>7} {row['output_tokens']:>7} "
                f"{row['thinking_tokens']:>7} {row['cost_usd']:>8.5f} {str(row['output_kernel_session_id']):>12}"
            )

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
