#!/usr/bin/env python3
"""
Download Kaggle kernel .ipynb files in parallel, convert to .py, and delete .ipynb.
Organizes output into category subdirectories.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
LEADERBOARD_DIR = Path(__file__).parent.parent / "leaderboards"
OUTPUT_BASE = Path(__file__).parent.parent / "downloaded_tasks"
KAGGLE_USER = "kdcyberdude"
MAX_WORKERS = 2          # conservative to stay below Kaggle rate limit
REQUEST_DELAY = 2.0      # minimum seconds between each download start
MAX_RETRIES = 5
RETRY_DELAY = 30         # seconds between retries on 429

LEADERBOARD_FILES = {
    "kdcyberdude_associativelearning_leaderboard.json": "associative_learning",
    "kdcyberdude_conceptlearning_leaderboard.json": "concept_learning",
    "kdcyberdude_languagelearning_leaderboard.json": "language_learning",
    "kdcyberdude_observationallearning_leaderboard.json": "observational_learning",
    "kdcyberdude_rlbench_leaderboard.json": "reinforcement_learning",
}

KAGGLE_BIN = shutil.which("kaggle") or "kaggle"
JUPYTER_BIN = shutil.which("jupyter") or "jupyter"

# Some tasks have a numeric benchmarkTaskSlug (e.g. "10001") in the leaderboard JSON
# that does NOT match the actual Kaggle kernel slug.  This table maps those numeric
# slugs to their real kernel slugs so the script is fully reproducible on any machine.
SLUG_OVERRIDES: dict[str, str] = {
    "10000": "odd-letter-score-pair-assoc-learning",
    "10001": "contextual-flip-assoc-learning",
    "10002": "latent-inhibition-assoc-learning",
}

# ── Throttle state ───────────────────────────────────────────────────────────
_throttle_lock = threading.Lock()
_last_request_time: list[float] = [0.0]


# ── Helpers ──────────────────────────────────────────────────────────────────

def collect_tasks() -> list[tuple[str, str, str]]:
    """Return list of (category, task_name, true_slug) from leaderboard JSON."""
    seen: dict[str, tuple[str, str, str]] = {}  # slug -> (cat, name, slug)
    for filename, category in LEADERBOARD_FILES.items():
        path = LEADERBOARD_DIR / filename
        with open(path) as fh:
            data = json.load(fh)
        for row in data.get("rows", []):
            for tr in row.get("taskResults", []):
                name = tr.get("benchmarkTaskName", "")
                slug_path = tr.get("benchmarkTaskSlug", "")
                if name and slug_path:
                    true_slug = slug_path.split("/")[-1]
                    key = f"{category}/{true_slug}"
                    if key not in seen:
                        seen[key] = (category, name, true_slug)
    return sorted(seen.values(), key=lambda x: (x[0], x[2]))


def run(cmd: list, cwd=None, timeout: int = 120):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def _throttled_download(kernel_ref: str, tmp: str):
    """Rate-limited download with exponential back-off on 429."""
    rc, out, err = 1, "", "not started"
    for attempt in range(1, MAX_RETRIES + 1):
        # enforce minimum gap between any two download starts
        with _throttle_lock:
            now = time.monotonic()
            wait = REQUEST_DELAY - (now - _last_request_time[0])
            if wait > 0:
                time.sleep(wait)
            _last_request_time[0] = time.monotonic()

        rc, out, err = run([KAGGLE_BIN, "kernels", "pull", kernel_ref], cwd=tmp)
        if rc == 0:
            return rc, out, err
        if "429" in err or "Too Many Requests" in err:
            backoff = RETRY_DELAY * attempt
            print(f"  [rate-limit] {kernel_ref}, backoff {backoff}s (attempt {attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(backoff)
        elif attempt < MAX_RETRIES:
            time.sleep(5)
    return rc, out, err


def download_and_convert(category: str, task_name: str, slug: str) -> dict:
    # Some leaderboard slugs are numeric IDs; use the real Kaggle kernel slug for
    # the actual download, but keep the leaderboard slug as the output filename.
    kernel_slug = SLUG_OVERRIDES.get(slug, slug)
    kernel_ref = f"{KAGGLE_USER}/{kernel_slug}"
    out_dir = OUTPUT_BASE / category
    out_dir.mkdir(parents=True, exist_ok=True)

    py_path = out_dir / f"{slug}.py"
    if py_path.exists():
        return {"task": task_name, "slug": slug, "category": category, "status": "skipped", "msg": "already exists"}

    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err = _throttled_download(kernel_ref, tmp)
        if rc != 0:
            return {
                "task": task_name,
                "slug": slug,
                "category": category,
                "status": "error",
                "msg": f"download failed: {err.strip()[:200]}",
            }

        ipynb_files = list(Path(tmp).glob("*.ipynb"))
        if not ipynb_files:
            return {
                "task": task_name,
                "slug": slug,
                "category": category,
                "status": "error",
                "msg": "no .ipynb found after download",
            }
        ipynb_path = ipynb_files[0]

        rc, out, err = run(
            [JUPYTER_BIN, "nbconvert", "--to", "script", "--no-prompt", str(ipynb_path)],
            cwd=tmp,
            timeout=60,
        )
        if rc != 0:
            return {
                "task": task_name,
                "slug": slug,
                "category": category,
                "status": "error",
                "msg": f"nbconvert failed: {err.strip()[:200]}",
            }

        py_files = list(Path(tmp).glob("*.py"))
        if not py_files:
            return {
                "task": task_name,
                "slug": slug,
                "category": category,
                "status": "error",
                "msg": "no .py found after nbconvert",
            }

        shutil.move(str(py_files[0]), str(py_path))

    return {"task": task_name, "slug": slug, "category": category, "status": "ok", "msg": ""}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_tasks = collect_tasks()  # list of (category, task_name, slug)
    total = len(all_tasks)
    cats = {}
    for cat, name, slug in all_tasks:
        cats.setdefault(cat, 0)
        cats[cat] += 1
    print(f"Found {total} tasks across {len(cats)} categories")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"\nUsing {MAX_WORKERS} workers, {REQUEST_DELAY}s inter-request delay")
    est_min = (total * REQUEST_DELAY / MAX_WORKERS) / 60
    print(f"Estimated time: ~{est_min:.0f} min (excluding retries)\n")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    completed = 0
    errors = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_and_convert, cat, name, slug): (cat, name, slug)
            for cat, name, slug in all_tasks
        }
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            if res["status"] == "ok":
                print(f"[{completed:3d}/{total}] ✓  {res['category']}/{res['slug']}", flush=True)
            elif res["status"] == "skipped":
                skipped += 1
                print(f"[{completed:3d}/{total}] →  skipped {res['category']}/{res['slug']}", flush=True)
            else:
                errors.append(res)
                print(f"[{completed:3d}/{total}] ✗  {res['category']}/{res['slug']}: {res['msg']}", flush=True)

    print()
    print(f"Done. {completed - len(errors) - skipped} downloaded, {skipped} skipped, {len(errors)} errors.")
    if errors:
        print("\nFailed tasks:")
        for e in errors:
            print(f"  {e['category']}/{e['slug']} ({e['task']}): {e['msg']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
