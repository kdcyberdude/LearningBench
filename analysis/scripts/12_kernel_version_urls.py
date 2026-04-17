"""
Generate a mapping of all task kernels → their Kaggle URLs per version.

Uses two Kaggle API calls per kernel:
  1. POST GetKernel           → currentVersionNumber
  2. POST DownloadKernelOutput (per version) → 302 redirect containing scriptVersionId

Output: analysis/outputs/kernel_version_urls.json
        analysis/outputs/kernel_version_urls.csv
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

KAGGLE_USER = "kdcyberdude"
KAGGLE_KEY  = "bd24a8e62cb84624fb000acb4f47c8d3"
BASE_URL    = "https://www.kaggle.com/code"
API_BASE    = "https://api.kaggle.com/v1/kernels.KernelsApiService"

LEADERBOARD_FILES = [
    "leaderboards/kdcyberdude_associativelearning_leaderboard.json",
    "leaderboards/kdcyberdude_conceptlearning_leaderboard.json",
    "leaderboards/kdcyberdude_observationallearning_leaderboard.json",
    "leaderboards/kdcyberdude_rlbench_leaderboard.json",
    "leaderboards/kdcyberdude_languagelearning_leaderboard.json",
]

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR   = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def curl_post(endpoint, payload):
    """POST to Kaggle API with basic auth, return parsed JSON."""
    result = subprocess.run(
        ["curl", "-s", "-X", "POST",
         "-H", "Content-Type: application/json",
         "-u", f"{KAGGLE_USER}:{KAGGLE_KEY}",
         "-d", json.dumps(payload),
         endpoint],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return json.loads(result.stdout)
    except Exception:
        return {}


def curl_post_redirect_url(endpoint, payload):
    """POST to Kaggle API, return the Location header URL (follows redirect)."""
    result = subprocess.run(
        ["curl", "-v", "-s", "-X", "POST",
         "-H", "Content-Type: application/json",
         "-u", f"{KAGGLE_USER}:{KAGGLE_KEY}",
         "-d", json.dumps(payload),
         endpoint],
        capture_output=True, text=True, timeout=30,
    )
    m = re.search(r"location: (https://[^\r\n]+)", result.stderr, re.IGNORECASE)
    return m.group(1).strip() if m else None


def get_version_count(kernel_slug):
    """Return currentVersionNumber for the kernel (0 on error)."""
    d = curl_post(f"{API_BASE}/GetKernel",
                  {"userName": KAGGLE_USER, "kernelSlug": kernel_slug})
    return d.get("metadata", {}).get("currentVersionNumber", 0)


def get_script_version_id(kernel_slug, version_number, filename):
    """Return scriptVersionId (str) for a specific version, or None."""
    redirect = curl_post_redirect_url(
        f"{API_BASE}/DownloadKernelOutput",
        {
            "ownerSlug": KAGGLE_USER,
            "kernelSlug": kernel_slug,
            "filePath": filename,
            "versionNumber": version_number,
        },
    )
    if redirect:
        m = re.search(r"/kf/(\d+)/", redirect)
        return m.group(1) if m else None
    return None


def collect_task_kernels():
    """Return {kernel_slug: task_filename} from all leaderboard JSONs."""
    kernels = {}
    for lb_rel in LEADERBOARD_FILES:
        lb_path = PROJECT_ROOT / lb_rel
        if not lb_path.exists():
            print(f"  [warn] missing: {lb_path}")
            continue
        lb = json.loads(lb_path.read_text())
        for row in lb.get("rows", []):
            for tr in row.get("taskResults", []):
                slug = tr.get("benchmarkTaskSlug", "")
                kernel_slug = slug.split("/")[-1]
                if not kernel_slug or len(kernel_slug) < 4 or kernel_slug.isdigit():
                    continue
                if kernel_slug not in kernels:
                    # Guess the task.json filename from the kernel slug
                    # Pattern: hyphens → underscores, e.g. "temporal-pairing-kmp-assoc-learning"
                    # → "temporal_pairing_kmp_assoc_learning.task.json"
                    task_filename = kernel_slug.replace("-", "_") + ".task.json"
                    kernels[kernel_slug] = task_filename
    return kernels


def main():
    task_kernels = collect_task_kernels()
    print(f"Found {len(task_kernels)} unique task kernels")

    results = {}  # kernel_slug → {"n_versions": N, "versions": {1: {url, script_version_id}}}

    for i, (kernel_slug, task_filename) in enumerate(sorted(task_kernels.items()), 1):
        print(f"\n[{i}/{len(task_kernels)}] {kernel_slug}")

        n_versions = get_version_count(kernel_slug)
        if n_versions == 0:
            print(f"  [skip] could not get version count")
            results[kernel_slug] = {"n_versions": 0, "versions": {}}
            time.sleep(0.3)
            continue

        print(f"  versions: {n_versions}")
        version_map = {}

        for v in range(1, n_versions + 1):
            vid = get_script_version_id(kernel_slug, v, task_filename)
            if vid:
                kaggle_url = f"{BASE_URL}/{KAGGLE_USER}/{kernel_slug}?scriptVersionId={vid}"
                version_map[v] = {"script_version_id": vid, "url": kaggle_url}
                print(f"    v{v}: {vid} → {kaggle_url}")
            else:
                print(f"    v{v}: [no redirect found]")
                version_map[v] = {"script_version_id": None, "url": None}
            time.sleep(0.15)  # polite rate limit

        results[kernel_slug] = {"n_versions": n_versions, "versions": version_map}
        time.sleep(0.2)

    # Save JSON
    json_path = OUTPUT_DIR / "kernel_version_urls.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved JSON: {json_path}")

    # Save CSV
    import csv
    csv_path = OUTPUT_DIR / "kernel_version_urls.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel_slug", "version_number", "script_version_id", "url"])
        for kernel_slug, info in sorted(results.items()):
            for v, vdata in sorted(info["versions"].items()):
                writer.writerow([
                    kernel_slug, v,
                    vdata.get("script_version_id", ""),
                    vdata.get("url", ""),
                ])
    print(f"✓ Saved CSV:  {csv_path}")

    # Summary stats
    total_versions = sum(len(v["versions"]) for v in results.values())
    total_urls     = sum(
        1 for v in results.values()
        for vd in v["versions"].values()
        if vd.get("url")
    )
    print(f"\n=== Summary ===")
    print(f"  Kernels processed : {len(results)}")
    print(f"  Total versions    : {total_versions}")
    print(f"  Valid URLs found  : {total_urls}")


if __name__ == "__main__":
    main()
