#!/usr/bin/env python3
"""
20_fetch_notebook_logs.py — Download per-model notebook stdout logs and run.json files
for all benchmark tasks.

For every (task, model) pair in all_task_runs.csv this script:
  1. Calls GetKernelViewModel (internal Kaggle API) with the output_kernel_session_id
     to obtain a fresh signed renderedOutputUrl and signed run.json downloadUrl.
  2. Downloads the rendered HTML notebook and extracts the stdout trace from the
     second <pre> block (task name header, prompt, per-item responses, score).
  3. Downloads the .run.json sidecar file (full conversation / turn-level data).
  4. Saves everything as a structured JSON file per (task, model) pair.
  5. Produces a combined all_notebook_logs.json (array) and all_notebook_logs.csv.

Resume: already-saved files are skipped, so the script can be interrupted and re-run.

Usage:
  # Refresh cookies first (if needed):
  python3 analysis/scripts/20_fetch_notebook_logs.py --refresh-cookies

  # Fetch all tasks, 6 workers:
  python3 analysis/scripts/20_fetch_notebook_logs.py --workers 6

  # Single task for validation:
  python3 analysis/scripts/20_fetch_notebook_logs.py --task spurious-hue-true-edge-assoc-learning

  # Single task + single model:
  python3 analysis/scripts/20_fetch_notebook_logs.py \
      --task spurious-hue-true-edge-assoc-learning \
      --model "GPT-5.4"

  # Force re-download even if already saved:
  python3 analysis/scripts/20_fetch_notebook_logs.py --force --workers 6
"""

import argparse
import csv
import html as htmllib
import json
import random
import re
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
ALL_RUNS_CSV = ROOT / "analysis" / "outputs" / "task_runs" / "all_task_runs.csv"
OUT_DIR = ROOT / "analysis" / "outputs" / "notebook_logs"

OWNER_SLUG = "kdcyberdude"
VIEWMODEL_URL = "https://www.kaggle.com/api/i/kernels.LegacyKernelsService/GetKernelViewModel"

# Some tasks use numeric IDs as their benchmark task slug, but the actual Kaggle
# kernel slug (used for GetKernelViewModel) is the hyphenated human-readable name.
TASK_SLUG_TO_KERNEL_SLUG: dict[str, str] = {
    # Numeric benchmark task IDs → human-readable kernel slugs
    "10000": "odd-letter-score-pair-assoc-learning",
    "10001": "contextual-flip-assoc-learning",
    "10002": "latent-inhibition-assoc-learning",
    # Benchmark task slug differs from the actual Kaggle kernel slug
    "adaptive-sort-rule-proc-learning":       "adaptive-sort-proc-learning",
    "agglutinative-morphology-obs-learning":  "agglutinative-morphology1-obs-learning",
    "lattice-meet-join-obs-learning":          "lattice-meet-join-divisibility-obs-learning",
    "syntax-tree-rewrite-obs-learning":        "syntax-tree-adverb-rewrite-obs-learning",
    "hidden-modal-logic-kripke-obs-learning":  "new-benchmark-task-616bf",
}

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def load_session() -> dict:
    if not SESSION_FILE.exists():
        sys.exit(
            f"[ERROR] No session file at {SESSION_FILE}.\n"
            "Run with --refresh-cookies first."
        )
    return json.loads(SESSION_FILE.read_text())


def refresh_cookies(browser: str = "brave") -> dict:
    try:
        import browser_cookie3
    except ImportError:
        sys.exit("pip install browser-cookie3 first.")

    print(f"[cookies] Extracting from {browser}…")
    fn = getattr(browser_cookie3, browser.lower())
    jar = fn(domain_name=".kaggle.com")

    cookies: dict[str, str] = {}
    for c in jar:
        cookies[c.name] = c.value

    want = {"XSRF-TOKEN", "ka_sessionid", "__Host-KAGGLEID", "CSRF-TOKEN",
            "ka_db", "CLIENT-TOKEN", "build-hash", "GCLB", "ACCEPTED_COOKIES"}
    found = {k: v for k, v in cookies.items() if k in want}
    xsrf = cookies.get("XSRF-TOKEN", "")
    build = cookies.get("build-hash", "")

    if not xsrf:
        sys.exit("[ERROR] XSRF-TOKEN not found — are you logged into Kaggle in that browser?")

    data = {
        "_comment": "Kaggle browser session — DO NOT COMMIT (gitignored)",
        "_instructions": "python3 analysis/scripts/20_fetch_notebook_logs.py --refresh-cookies",
        "cookies": found,
        "xsrf_token": xsrf,
        "build_version": build,
        "owner_slug": OWNER_SLUG,
    }
    SESSION_FILE.write_text(json.dumps(data, indent=2))
    print(f"[cookies] Saved {len(found)} cookies to {SESSION_FILE.name}")
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_headers(session: dict) -> dict:
    # Full browser headers are required to avoid reCAPTCHA bot detection.
    # Kaggle's internal /api/i/ endpoints validate request fingerprints.
    return {
        "Content-Type": "application/json",
        "x-xsrf-token": session.get("xsrf_token", ""),
        "x-kaggle-build-version": session.get("build_version", ""),
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/135.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.kaggle.com/",
        "Origin": "https://www.kaggle.com",
        "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    }


def _safe_filename(model_display_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_display_name.replace(" ", "_"))


def _extract_stdout(html_text: str) -> str:
    """Return the notebook stdout trace from the rendered HTML.

    The rendered notebook has (at minimum) two <pre> blocks:
      pre[0] = Python source code of the task cell
      pre[1] = stdout output (the _log_trace print with PROMPT/RESPONSES/SCORE)

    We identify the real stdout block by looking for 'SCORE: <decimal>'
    which is always present in the benchmark trace but never in source code.
    Fallback: return the last pre block if the primary heuristic fails.
    """
    pre_blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", html_text, re.DOTALL)
    for block in pre_blocks:
        clean = htmllib.unescape(re.sub(r"<[^>]+>", "", block)).strip()
        # Real stdout always contains "SCORE: 0.xxxx" or "SCORE: 1.0000"
        if re.search(r"SCORE:\s+[\d.]+", clean):
            return clean
    # Fallback: last pre block (usually stdout for notebooks with one output cell)
    if pre_blocks:
        return htmllib.unescape(re.sub(r"<[^>]+>", "", pre_blocks[-1])).strip()
    return ""


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_one(
    row: dict,
    session: dict,
    out_dir: Path,
    force: bool = False,
    max_retries: int = 4,
) -> dict:
    """Fetch notebook log for one (task, model) row from all_task_runs.csv.

    Returns a result dict suitable for aggregation.
    """
    task_slug: str = row["task_slug"]
    model_name: str = row["model_display_name"]
    session_id_str: str = row.get("output_kernel_session_id", "")
    model_safe = _safe_filename(model_name)

    result = {
        "task_slug": task_slug,
        "model_display_name": model_name,
        "model_proxy_slug": row.get("model_proxy_slug", ""),
        "provider": row.get("provider", ""),
        "output_kernel_session_id": session_id_str,
        "score_fraction": row.get("score_fraction", ""),
        "score_value": row.get("score_value", ""),
        "total_latency_ms": row.get("total_latency_ms", ""),
        "input_tokens": row.get("input_tokens", ""),
        "output_tokens": row.get("output_tokens", ""),
        "thinking_tokens": row.get("thinking_tokens", ""),
        "cost_usd": row.get("cost_usd", ""),
        # populated below
        "stdout_log": "",
        "run_json": None,
        "notebook_url": "",
        "status": "pending",
        "error": "",
    }

    if not session_id_str:
        result["status"] = "no_session_id"
        result["error"] = "output_kernel_session_id missing in CSV"
        return result

    session_id = int(session_id_str)
    kernel_slug = TASK_SLUG_TO_KERNEL_SLUG.get(task_slug, task_slug)
    result["notebook_url"] = (
        f"https://www.kaggle.com/code/{OWNER_SLUG}/{kernel_slug}"
        f"?scriptVersionId={session_id}"
    )

    # Where to save
    task_dir = out_dir / task_slug
    task_dir.mkdir(parents=True, exist_ok=True)
    out_file = task_dir / f"{model_safe}.json"

    if out_file.exists() and not force:
        cached = json.loads(out_file.read_text())
        cached["status"] = "cached"
        return cached

    headers = _make_headers(session)
    cookies = session.get("cookies", {})

    # Small random jitter to stagger parallel requests and reduce reCAPTCHA triggers
    time.sleep(random.uniform(0.2, 1.0))

    # -----------------------------------------------------------------------
    # Step 1: GetKernelViewModel → get fresh signed URLs
    # -----------------------------------------------------------------------
    rendered_url = ""
    run_json_download_url = ""

    for attempt in range(1, max_retries + 1):
        try:
            viewmodel_body: dict = {
                "authorUserName": OWNER_SLUG,
                "kernelSlug": kernel_slug,
                "kernelVersionId": session_id,
            }
            resp = requests.post(
                VIEWMODEL_URL,
                json=viewmodel_body,
                headers=headers,
                cookies=cookies,
                timeout=30,
            )
            # On 404, retry once without the versionId — some kernels only serve
            # via their current (latest) run, not specific historical version IDs.
            if resp.status_code == 404 and "kernelVersionId" in viewmodel_body:
                viewmodel_body_no_ver = {k: v for k, v in viewmodel_body.items() if k != "kernelVersionId"}
                resp_no_ver = requests.post(
                    VIEWMODEL_URL,
                    json=viewmodel_body_no_ver,
                    headers=headers,
                    cookies=cookies,
                    timeout=30,
                )
                if resp_no_ver.status_code == 200:
                    resp = resp_no_ver
            if resp.status_code in (401, 403):
                result["status"] = "auth_error"
                result["error"] = (
                    f"HTTP {resp.status_code} — session expired, run --refresh-cookies"
                )
                return result
            if resp.status_code == 429:
                # Rate limited — back off longer
                wait = 5 * attempt
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                result["status"] = "viewmodel_error"
                result["error"] = f"GetKernelViewModel HTTP {resp.status_code}: {resp.text[:200]}"
                return result

            body = resp.text.strip()
            if not body:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                result["status"] = "empty_response"
                result["error"] = "GetKernelViewModel returned empty body"
                return result

            try:
                vm = json.loads(body)
            except json.JSONDecodeError as exc:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                result["status"] = "json_decode_error"
                result["error"] = f"GetKernelViewModel bad JSON: {exc} | body[:100]={body[:100]}"
                return result

            kernel_run = vm.get("kernelRun", {})
            rendered_url = kernel_run.get("renderedOutputUrl", "")

            # Find the .run.json download URL among outputFiles
            for f in vm.get("outputFiles", []):
                if f.get("name", "").endswith(".run.json"):
                    run_json_download_url = f.get("downloadUrl", "")
                    break
            break

        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                result["status"] = "request_error"
                result["error"] = f"GetKernelViewModel exception: {exc}"
                return result

    # -----------------------------------------------------------------------
    # Step 2: Download rendered HTML → extract stdout
    # -----------------------------------------------------------------------
    if rendered_url:
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(rendered_url, timeout=60)
                if r.status_code == 200:
                    result["stdout_log"] = _extract_stdout(r.text)
                    break
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
            except requests.RequestException:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

    # -----------------------------------------------------------------------
    # Step 3: Download run.json (full conversation / turn data)
    # -----------------------------------------------------------------------
    if run_json_download_url:
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(run_json_download_url, timeout=60)
                if r.status_code == 200:
                    result["run_json"] = r.json()
                    break
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
            except (requests.RequestException, json.JSONDecodeError):
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

    result["status"] = "ok"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def load_all_runs(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download notebook stdout logs + run.json for every task×model pair."
    )
    parser.add_argument("--task", help="Fetch a single task slug only")
    parser.add_argument("--model", help='Fetch a single model only (e.g. "GPT-5.4")')
    parser.add_argument("--refresh-cookies", action="store_true",
                        help="Re-extract cookies from the browser first")
    parser.add_argument("--browser", default="brave", choices=["brave", "chrome"],
                        help="Browser to extract cookies from (default: brave)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel download workers (default: 2; Kaggle reCAPTCHA triggers at >3 concurrency)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output file already exists")
    parser.add_argument("--output", default=str(OUT_DIR),
                        help="Output directory (default: analysis/outputs/notebook_logs)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Session
    if args.refresh_cookies:
        session = refresh_cookies(args.browser)
    else:
        session = load_session()

    # Load CSV
    if not ALL_RUNS_CSV.exists():
        sys.exit(
            f"[ERROR] {ALL_RUNS_CSV} not found.\n"
            "Run python3 analysis/scripts/19_fetch_task_runs.py first."
        )
    all_rows = load_all_runs(ALL_RUNS_CSV)
    print(f"[info] Loaded {len(all_rows)} rows from all_task_runs.csv")

    # Filter
    if args.task:
        all_rows = [r for r in all_rows if r["task_slug"] == args.task]
        if not all_rows:
            sys.exit(f"[ERROR] No rows found for task '{args.task}'")
    if args.model:
        all_rows = [r for r in all_rows if r["model_display_name"] == args.model]
        if not all_rows:
            sys.exit(f"[ERROR] No rows found for model '{args.model}'")

    print(f"[info] Will process {len(all_rows)} (task, model) pairs")
    print(f"[info] Output → {out_dir}")
    print(f"[info] Workers: {args.workers} | Force: {args.force}\n")

    results: list[dict] = []
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, row, session, out_dir, args.force): row
            for row in all_rows
        }
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            row = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                errors += 1
                res = {
                    "task_slug": row.get("task_slug", "?"),
                    "model_display_name": row.get("model_display_name", "?"),
                    "status": "exception",
                    "error": str(exc),
                }

            results.append(res)
            status = res.get("status", "?")
            if status not in ("ok", "cached"):
                errors += 1
                print(
                    f"  [{done}/{total}] ✗ {res['task_slug']} / "
                    f"{res['model_display_name']} — {status}: {res.get('error','')}"
                )
            else:
                has_stdout = bool(res.get("stdout_log"))
                has_run_json = res.get("run_json") is not None
                marker = "✓" if status == "ok" else "~"
                flag = f"stdout={'✓' if has_stdout else '✗'} run_json={'✓' if has_run_json else '✗'}"
                print(
                    f"  [{done}/{total}] {marker} {res['task_slug']} / "
                    f"{res['model_display_name']} — {flag}"
                )

    # ------------------------------------------------------------------
    # Save aggregated outputs
    # ------------------------------------------------------------------
    # 1. Combined JSON array (full data)
    combined_json = out_dir / "all_notebook_logs.json"
    combined_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # 2. Combined CSV (without the large run_json blob)
    csv_columns = [
        "task_slug", "model_display_name", "model_proxy_slug", "provider",
        "output_kernel_session_id", "notebook_url",
        "score_fraction", "score_value",
        "total_latency_ms", "input_tokens", "output_tokens", "thinking_tokens", "cost_usd",
        "stdout_log", "status", "error",
    ]
    combined_csv = out_dir / "all_notebook_logs.csv"
    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    cached_count = sum(1 for r in results if r.get("status") == "cached")
    stdout_count = sum(1 for r in results if r.get("stdout_log"))
    runjson_count = sum(1 for r in results if r.get("run_json") is not None)
    err_count = sum(1 for r in results if r.get("status") not in ("ok", "cached"))

    print(f"\n{'='*60}")
    print(f"  Completed : {ok_count} downloaded, {cached_count} from cache")
    print(f"  Errors    : {err_count}")
    print(f"  Stdout log: {stdout_count} / {len(results)} pairs have stdout")
    print(f"  Run JSON  : {runjson_count} / {len(results)} pairs have run.json")
    print(f"  Output    : {out_dir}")
    print(f"  Combined  : all_notebook_logs.json ({len(results)} records)")
    print(f"  CSV       : all_notebook_logs.csv")
    print(f"{'='*60}\n")

    if args.task and len(all_rows) <= 20:
        # Print a small table for single-task runs
        print(f"\n{'Model':<35} {'Stdout':^8} {'RunJSON':^8} {'Score':>8}  Status")
        print("-" * 75)
        for r in sorted(results, key=lambda x: x.get("model_display_name", "")):
            print(
                f"  {r.get('model_display_name','?'):<33} "
                f"{'✓' if r.get('stdout_log') else '✗':^8} "
                f"{'✓' if r.get('run_json') else '✗':^8} "
                f"{str(r.get('score_fraction',''))[:6]:>8}  "
                f"{r.get('status','?')}"
            )


if __name__ == "__main__":
    main()
