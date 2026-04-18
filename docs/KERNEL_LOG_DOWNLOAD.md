# Kaggle Benchmark Run Data — Architecture & How-To Guide

**File:** `analysis/docs/KERNEL_LOG_DOWNLOAD.md`  
**Last updated:** 2026-04-15  
**Scripts:** `analysis/scripts/19_fetch_task_runs.py` (metrics) · `analysis/scripts/20_fetch_notebook_logs.py` (stdout + run.json)

---

## Overview

Two scripts together give complete coverage of all benchmark evaluation data:

| Script | What it fetches | Auth | Output |
|---|---|---|---|
| `19_fetch_task_runs.py` | Metrics: scores, tokens, costs, latency for all 14 models × all tasks | Browser session | `all_task_runs.csv` |
| `20_fetch_notebook_logs.py` | Notebook stdout trace + full conversation run.json per model × task | Browser session | `notebook_logs/` |

Run script 19 first, then script 20. Script 20 reads the `output_kernel_session_id` column produced by script 19.

---

## What Data Is Available

### Script 19 — Aggregated metrics (per task × model)

| Field | Description |
|---|---|
| `task_slug` | Task URL slug |
| `model_display_name` | Human-readable model name |
| `score_fraction` | Fraction correct (0.0–1.0) |
| `score_value` | Raw numeric score when available |
| `assertions_passed / assertions_total` | e.g. `3/4` |
| `total_latency_ms` | Pure model inference time (no Kaggle overhead) |
| `input_tokens` | Prompt tokens |
| `output_tokens` | Response tokens |
| `thinking_tokens` | Extended reasoning tokens (thinking models only) |
| `cost_usd` | `(input + output cost nanodollars) / 1e9` |
| `output_kernel_session_id` | Unique Kaggle run ID — links notebook URL and run.json |

### Script 20 — Full per-model run data (per task × model)

| Field | Description |
|---|---|
| `stdout_log` | The full printed execution trace: task name, prompt, per-item responses (✓/✗), final SCORE |
| `run_json` | Complete multi-turn LLM conversation: every `request` and `response` turn with role, content, and assertion results |
| `notebook_url` | Direct URL to the Kaggle notebook at that exact version |
| All fields from script 19 | Score, latency, tokens, cost — duplicated for convenience |

---

## Architecture: How It Works

### Step 1 — Script 19: Get metrics for all models

**API endpoint:**
```
POST https://www.kaggle.com/api/i/benchmarks.BenchmarkTaskRunService/ListBenchmarkTaskRuns
```

A single call per task returns all 14 model run records for the **latest version** of that task. No kernel file downloads needed.

**Request body:**
```json
{
  "filter": {
    "taskIdentifier": {
      "slugSelector": {
        "taskSlug": "spurious-hue-true-edge-assoc-learning",
        "ownerSlug": "kdcyberdude"
      }
    },
    "taskRunIds": [],
    "modelVersionIds": []
  },
  "pageSize": 0,
  "pageToken": "",
  "skip": 0
}
```

**Key response fields:**
```json
{
  "benchmarkTaskRuns": [
    {
      "id": 141794,
      "modelVersion": { "displayName": "GPT-5.4", "organization": { "slug": "openai" } },
      "state": "BENCHMARK_TASK_RUN_STATE_COMPLETED",
      "results": [
        {
          "type": "AGGREGATED",
          "totalMetrics": {
            "inputTokens": 458,
            "outputTokens": 38,
            "thinkingTokens": 0,
            "totalBackendLatencyMs": 988,
            "inputTokensCostNanodollars": 1145000,
            "outputTokensCostNanodollars": 570000
          },
          "assertionStatuses": ["FAILED", "FAILED", "PASSED", "FAILED"]
        }
      ],
      "outputKernelSessionId": 310210849
    }
  ]
}
```

The `outputKernelSessionId` is the key that links to the Kaggle notebook version and enables script 20.

---

### Step 2 — Script 20: Get stdout trace + run.json

For each `(task_slug, output_kernel_session_id)` pair from script 19's CSV:

**Sub-step A — Get signed download URLs via GetKernelViewModel:**
```
POST https://www.kaggle.com/api/i/kernels.LegacyKernelsService/GetKernelViewModel
```

Request body:
```json
{
  "authorUserName": "kdcyberdude",
  "kernelSlug": "spurious-hue-true-edge-assoc-learning",
  "kernelVersionId": 310210849
}
```

Note the exact field names: `authorUserName` (not `authorSlug`), `kernelVersionId` (not `kernelSlug`/`versionNumber`).

Response fields of interest:
```json
{
  "kernelRun": {
    "id": 310210849,
    "renderedOutputUrl": "https://www.kaggleusercontent.com/kf/310210849/eyJ..."
  },
  "outputFiles": [
    {
      "name": "spurious_hue_true_edge_assoc_learning-run_id_Run_1_openai_gpt-5.4-2026-03-05.run.json",
      "downloadUrl": "https://www.kaggleusercontent.com/kf/310210849/eyJ..."
    },
    {
      "name": "spurious_hue_true_edge_assoc_learning.task.json",
      "downloadUrl": "https://www.kaggleusercontent.com/kf/310210849/eyJ..."
    }
  ]
}
```

Both URLs are **short-lived signed CDN URLs** — they must be fetched immediately after this call. Do not cache them.

**Sub-step B — Download `renderedOutputUrl` → extract stdout:**

The rendered URL returns an HTML page of the notebook. The page always contains two `<pre>` blocks:
- `pre[0]` — Python source code of the task cell
- `pre[1]` — Execution stdout (the `_log_trace()` output)

The stdout block is identified by the presence of `SCORE: <decimal>` (which is never in source code). It looks like:

```
============================================================
  spurious_hue_true_edge
============================================================

  TASK: Tests resistance to spurious correlation...

  PROMPT:
Items are classified into GROUP_X or GROUP_Y.
Training examples:
  item_1: tint=veld facet=riven grain=pellate → GROUP_X
  ...

  RESPONSES:
    item_9: got='GROUP_X'  expected='GROUP_X'  ✓
    item_10: got='GROUP_Y'  expected='GROUP_Y'  ✓
    item_11: got='GROUP_X'  expected='GROUP_X'  ✓
    item_12: got='GROUP_Y'  expected='GROUP_Y'  ✓

  SCORE: 1.0000
============================================================
```

**Sub-step C — Download `.run.json` → full conversation:**

The `.run.json` sidecar contains:
```json
{
  "taskVersion": "...",
  "modelVersion": "...",
  "state": "COMPLETED",
  "startTime": "...",
  "endTime": "...",
  "conversations": [
    {
      "requests": [
        {
          "contents": [
            { "role": "CONTENT_ROLE_USER", "parts": [{ "text": "Items are classified..." }] },
            { "role": "CONTENT_ROLE_ASSISTANT", "parts": [{ "text": "GROUP_X\nGROUP_Y..." }] }
          ]
        }
      ]
    }
  ],
  "results": { ... },
  "assertions": [ ... ]
}
```

---

## Authentication

Both scripts use the same browser session cookies — **not** the Kaggle API key.

Kaggle's `/api/i/` internal endpoints require browser session cookies. The Kaggle CLI / public `/api/v1/` key does not work for these endpoints.

**Required cookies:**
| Cookie | Purpose |
|---|---|
| `XSRF-TOKEN` | Anti-CSRF token (also sent as `x-xsrf-token` header) |
| `ka_sessionid` | Kaggle session identity |
| `__Host-KAGGLEID` | Kaggle user identity |
| `CSRF-TOKEN`, `ka_db`, `CLIENT-TOKEN` | Additional auth cookies |
| `build-hash` | Sent as `x-kaggle-build-version` header |

**Required headers (in addition to cookies):**
```python
{
    "Content-Type": "application/json",
    "x-xsrf-token": "<XSRF-TOKEN value>",
    "x-kaggle-build-version": "<build-hash value>",
    "User-Agent": "Mozilla/5.0 (Macintosh; ...) Chrome/135.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.kaggle.com/",
    "Origin": "https://www.kaggle.com",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}
```

**Important:** The full `User-Agent` and `sec-fetch-*` headers must match a real browser. Kaggle triggers a Google reCAPTCHA page when the request fingerprint looks like a bot. This manifests as an HTML page response instead of JSON (the script detects and retries this).

Cookies are extracted from the live browser using `browser-cookie3` and cached in `.kaggle_session.json` (gitignored). Session typically lasts several days.

---

## Setup (First Time)

### 1. Install dependencies

```bash
pip install browser-cookie3 requests
```

### 2. Extract cookies from your browser

Make sure you are **logged into Kaggle** in Brave or Chrome, then run:

```bash
cd /path/to/learning_eval
python3 analysis/scripts/19_fetch_task_runs.py --refresh-cookies
# or for Chrome:
python3 analysis/scripts/19_fetch_task_runs.py --refresh-cookies --browser chrome
```

This creates `.kaggle_session.json` at the project root. It is gitignored. **Never commit this file.**

Re-run this command whenever your session expires (typically after a few days). Signs of expiry: HTTP 401/403 responses, or the script returning `auth_error` status.

---

## Running the Scripts

### Step 1: Fetch all metrics (script 19)

```bash
# All tasks, 6 workers (~60 seconds):
python3 analysis/scripts/19_fetch_task_runs.py --workers 6

# Single task for validation:
python3 analysis/scripts/19_fetch_task_runs.py --task spurious-hue-true-edge-assoc-learning

# Refresh cookies and fetch all:
python3 analysis/scripts/19_fetch_task_runs.py --refresh-cookies --workers 6
```

Produces:
- `analysis/outputs/task_runs/all_task_runs.csv` — 1,835+ rows (131 tasks × 14-15 models)
- `analysis/outputs/task_runs/<task-slug>.json` — one JSON per task

### Step 2: Fetch stdout logs + run.json files (script 20)

```bash
# All tasks, 2 workers (recommended — higher concurrency triggers reCAPTCHA):
python3 analysis/scripts/20_fetch_notebook_logs.py --workers 2

# Single task (for validation/spot-check):
python3 analysis/scripts/20_fetch_notebook_logs.py \
    --task spurious-hue-true-edge-assoc-learning

# Single task + single model:
python3 analysis/scripts/20_fetch_notebook_logs.py \
    --task spurious-hue-true-edge-assoc-learning \
    --model "GPT-5.4"

# Force re-download of already-cached files:
python3 analysis/scripts/20_fetch_notebook_logs.py --force --workers 2

# Refresh cookies first if session expired:
python3 analysis/scripts/20_fetch_notebook_logs.py --refresh-cookies --workers 2
```

**Resume capability:** Script 20 skips any `(task, model)` pairs that already have a saved output file. If the run is interrupted, re-run the same command — it will continue from where it left off.

Expected runtime for all tasks: ~30 minutes at 2 workers (131 tasks × 14 models = 1,834 pairs, ~1 second each).

---

## Output Files

### Script 19 outputs

```
analysis/outputs/task_runs/
├── all_task_runs.csv                                    ← Combined flat table
├── spurious-hue-true-edge-assoc-learning.json           ← Per-task JSON (14 runs)
└── ... (one JSON per task, ~131 files)
```

### Script 20 outputs

```
analysis/outputs/notebook_logs/
├── all_notebook_logs.json                               ← Combined array (all pairs)
├── all_notebook_logs.csv                                ← Combined CSV (no run_json blob)
├── spurious-hue-true-edge-assoc-learning/
│   ├── GPT-5_4.json                                     ← stdout_log + run_json + metrics
│   ├── Claude_Opus_4_6.json
│   ├── Gemini_3_1_Pro_Preview.json
│   └── ... (one JSON per model, 14 files)
└── ... (one directory per task, ~131 directories)
```

### Per-(task, model) JSON schema (script 20 output)

```json
{
  "task_slug": "spurious-hue-true-edge-assoc-learning",
  "model_display_name": "Gemini 3.1 Pro Preview",
  "model_proxy_slug": "google/gemini-3.1-pro-preview",
  "provider": "google",
  "output_kernel_session_id": "310210849",
  "score_fraction": "1.0",
  "score_value": "",
  "total_latency_ms": "30679",
  "input_tokens": "517",
  "output_tokens": "330",
  "thinking_tokens": "0",
  "cost_usd": "0.022454",
  "stdout_log": "===...=== spurious_hue_true_edge ...SCORE: 1.0000 ===...===",
  "run_json": {
    "taskVersion": "...",
    "modelVersion": "...",
    "conversations": [ { "requests": [ { "contents": [...] } ] } ],
    "results": { ... },
    "assertions": [ ... ]
  },
  "notebook_url": "https://www.kaggle.com/code/kdcyberdude/spurious-hue-true-edge-assoc-learning?scriptVersionId=310210849",
  "status": "ok",
  "error": ""
}
```

### `all_task_runs.csv` columns (script 19)

| Column | Type | Description |
|---|---|---|
| `run_id` | int | Kaggle internal run ID |
| `task_slug` | str | Task URL slug |
| `task_name` | str | Task Python function name |
| `task_version_id` | int | Task version DB ID |
| `task_version_number` | int | Version number (latest = highest) |
| `model_version_id` | int | Model version DB ID |
| `model_slug` | str | Short model slug |
| `model_display_name` | str | Human-readable name |
| `model_proxy_slug` | str | Full provider/model/version slug |
| `provider` | str | Provider org slug |
| `state` | str | `BENCHMARK_TASK_RUN_STATE_COMPLETED` etc. |
| `start_time` | ISO datetime | Run start |
| `end_time` | ISO datetime | Run end |
| `score_fraction` | float | `assertions_passed / assertions_total` |
| `score_value` | float | Raw numeric result (when available) |
| `assertions_passed` | int | Number of assertions that passed |
| `assertions_total` | int | Total assertions in the task |
| `input_tokens` | int | Prompt tokens |
| `output_tokens` | int | Response tokens |
| `thinking_tokens` | int | Reasoning tokens (thinking models) |
| `total_latency_ms` | int | Pure model inference time (ms) |
| `input_cost_nanodollars` | int | Input cost × 1e9 |
| `output_cost_nanodollars` | int | Output cost × 1e9 |
| `cost_usd` | float | `(input + output cost) / 1e9` |
| `output_kernel_session_id` | int | Links to Kaggle notebook URL and run.json |

---

## Notebook URL Format

```
https://www.kaggle.com/code/kdcyberdude/{task-slug}?scriptVersionId={output_kernel_session_id}
```

Example:
```
https://www.kaggle.com/code/kdcyberdude/spurious-hue-true-edge-assoc-learning?scriptVersionId=310210849
```

---

## Concurrency Notes

**Script 19:** Safe at 6 workers. The `ListBenchmarkTaskRuns` endpoint is tolerant of parallel requests because it is a read-only listing API without reCAPTCHA.

**Script 20:** Maximum 2 workers. The `GetKernelViewModel` endpoint triggers Google reCAPTCHA when more than ~3 simultaneous requests come from the same session. Symptoms: the response body is an HTML reCAPTCHA challenge page instead of JSON. The script detects this and retries with exponential backoff, but keeping workers ≤ 2 avoids it entirely. At 2 workers, 1,835 pairs complete in ~30 minutes.

---

## The Older Scripts (17 / 18) — Context

### `17_download_kernel_logs.py` — Cross-kernel sampling (legacy)

**Problem:** The public Kaggle API only exposes output files from the *latest* kernel version. Since each benchmark task kernel gets overwritten with a new model version for each model run, only the last model's `*.run.json` is accessible per kernel. For full 14-model coverage, one must scan all ~154 task kernels to find one kernel where each model happens to be the current version.

**How it worked:**
1. Calls `ListKernelFiles` (public API, key-authenticated) for each kernel to check which model's `run.json` is current
2. Downloads via `kaggle kernels output` CLI
3. Parses each `*.run.json` for timing and token data
4. Saves `analysis/outputs/kernel_logs/manifest.json` and `kernel_logs_parsed.csv`

**Limitation:** Gives one sample *per model* (not per task-model pair), from whichever task that model happened to run last. Does not provide per-task scores.

### `18_timing_hypotheses.py` — Hypothesis tests on kernel log data (legacy)

Reads `kernel_logs_parsed.csv` (output of script 17) and runs statistical tests for H14–H18. Now superseded by running the same tests on `all_task_runs.csv` from script 19, which has full task × model coverage.

### Why scripts 19 + 20 are better

| Dimension | Scripts 17+18 | Scripts 19+20 |
|---|---|---|
| Coverage | 1 sample per model (14 total) | All 14 models × all 131 tasks (~1,835 rows) |
| Stdout logs | Not available | Full `_log_trace` output per model per task |
| Conversation data | Partial (one task's run.json per model) | Complete per task × model pair |
| Auth required | Public API key | Browser session cookies |
| Speed | 2–5 min (scan + download) | 19: <60s; 20: ~30 min |
| Score data | From source task only | From the actual task being analyzed |

---

## Known Kernel Slug Overrides

Some benchmark task slugs do not match the actual Kaggle kernel slug. Script 20 maintains a
`TASK_SLUG_TO_KERNEL_SLUG` mapping to handle these. The current overrides are:

| Benchmark task slug | Actual kernel slug | Reason |
|---|---|---|
| `10000` | `odd-letter-score-pair-assoc-learning` | Numeric task ID — no hyphenated slug |
| `10001` | `contextual-flip-assoc-learning` | Numeric task ID |
| `10002` | `latent-inhibition-assoc-learning` | Numeric task ID; also requires fallback (no versionId) |
| `adaptive-sort-rule-proc-learning` | `adaptive-sort-proc-learning` | Kernel was created under shorter name |
| `agglutinative-morphology-obs-learning` | `agglutinative-morphology1-obs-learning` | Kernel has `1` suffix |
| `lattice-meet-join-obs-learning` | `lattice-meet-join-divisibility-obs-learning` | Kernel has extra specificity in name |
| `syntax-tree-rewrite-obs-learning` | `syntax-tree-adverb-rewrite-obs-learning` | Same — kernel name is more specific |
| `hidden-modal-logic-kripke-obs-learning` | `new-benchmark-task-616bf` | Kernel was never renamed from autogenerated slug |
| `counterfactual-sequence-rewrite-assoc-learning` | `cumulative-state-rewrite-assoc-learning` | Kernel created under different name |
| `divisor-count-rf-learning` | `new-benchmark-task-436e8` | Autogenerated kernel slug |
| `finite-state-transducer-obs-learning` | `new-benchmark-task-f463f` | Autogenerated kernel slug |
| `graph-shortest-path-rf-learning` | `new-benchmark-task-7f806` | Autogenerated kernel slug |
| `gray-hamming-rf-learning` | `new-benchmark-task-3c19f` | Autogenerated kernel slug |
| `hidden-token-filter-obs-learning` | `new-benchmark-task-d4254` | Autogenerated kernel slug |
| `kelstran-tone-lang-learning` | `new-benchmark-task-e7cce` | Autogenerated kernel slug |
| `latent-set-variant-assoc-learning` | `new-benchmark-task-faf75` | Autogenerated kernel slug |
| `manhattan-point-rf-learning` | `new-benchmark-task-94f38` | Autogenerated kernel slug |
| `mealy-machine-output-obs-learning` | `new-benchmark-task-9f8e4` | Autogenerated kernel slug |
| `mixed-radix-number-lang-learning` | `new-benchmark-task-6fbe4` | Autogenerated kernel slug |
| `prentova-allomorphy-wugtest-lang-learning` | `new-benchmark-task-8453a` | Autogenerated kernel slug |
| `regex-intersection-membership-obs-learning` | `new-benchmark-task-63b13` | Autogenerated kernel slug |
| `rule90-step-rf-learning` | `new-benchmark-task-3731c` | Autogenerated kernel slug |
| `sensory-preconditioning-assoc-learning` | `inference-dyad-operators-assoc-learning` | Kernel created under different name |
| `serial-chain-reconstruction-assoc-learning` | `glyph-bind-assoc-learning` | Kernel created under different name |
| `skelth-allomorph-lang-learning` | `new-benchmark-task-b3b35` | Autogenerated kernel slug |
| `skovar-deletion-lang-learning` | `new-benchmark-task-39f3a` | Autogenerated kernel slug |
| `telvari-evidentiality-lang-learning` | `new-benchmark-task-878f2` | Autogenerated kernel slug |
| `titration-curve-diprotic-obs-learning` | `new-benchmark-task-81929` | Autogenerated kernel slug |
| `xor-subset-hamming-rf-learning` | `new-benchmark-task-442d1` | Autogenerated kernel slug |

**Permanently inaccessible (run_json not available):**

| Task slug | Reason |
|---|---|
| `hidden-modal-logic-kripke2-obs-learning` | Kernel exists (`sourceKernelId=115383318`) but `GetKernelViewModel` returns 404. `GetBenchmarkTaskRun` also returns no `conversations` data. Metrics (tokens, latency, cost) are available via `GetBenchmarkTaskRun`. |
| `hidden-priority-order-obs-learning` | Same — `sourceKernelId=115378304`. Metrics available but `run_json` / `stdout_log` are not retrievable. |

These two tasks have scores in `all_task_runs.csv` but have empty `run_json` and `stdout_log` in `all_notebook_logs.csv`.

### Fallback: no-version-ID requests

Script 20 automatically retries `GetKernelViewModel` without `kernelVersionId` when the versioned
request returns 404. This handles tasks like `10002` where the specific historical run versions
are no longer accessible but the current kernel run is still servable.

---

## Troubleshooting

### `No session file found`
```bash
python3 analysis/scripts/19_fetch_task_runs.py --refresh-cookies
```

### `auth_error` (HTTP 401/403)
Your session has expired. Re-run with `--refresh-cookies`.

### `json_decode_error` with `<!doctype html>` in body
Kaggle returned a reCAPTCHA page. Reduce `--workers` to 1 or 2, or add a longer delay. The script will auto-retry with exponential backoff.

### `No cookies found`
Make sure you are logged into Kaggle in Brave/Chrome before running. Try closing and reopening the browser once.

### `No tasks found` (script 19)
Script 19 discovers tasks from `analysis/outputs/kernel_logs_all/`. If that directory is empty, either run script 17 first, or pass `--task <slug>` directly.

### Script 20: `output_kernel_session_id missing`
Run script 19 first to populate `all_task_runs.csv`, then run script 20.

### Rate limiting / slow responses
Reduce `--workers` (try `--workers 1`). For script 19, the API is generally tolerant of 6 parallel requests. For script 20, keep it at 2.

---

## Session File Format

`.kaggle_session.json` (gitignored — never commit):

```json
{
  "_comment": "Kaggle browser session credentials — DO NOT COMMIT (gitignored)",
  "_instructions": "Refresh: python3 analysis/scripts/19_fetch_task_runs.py --refresh-cookies",
  "cookies": {
    "XSRF-TOKEN": "CfDJ8...",
    "ka_sessionid": "5a008a...",
    "__Host-KAGGLEID": "CfDJ8...",
    "CSRF-TOKEN": "CfDJ8...",
    "ka_db": "CfDJ8...",
    "CLIENT-TOKEN": "eyJhbGci...",
    "build-hash": "94ff71b9...",
    "GCLB": "CJv5...",
    "ACCEPTED_COOKIES": "true"
  },
  "xsrf_token": "CfDJ8...",
  "build_version": "94ff71b9...",
  "owner_slug": "kdcyberdude"
}
```

---

## CLI Reference

### Script 19

```
usage: 19_fetch_task_runs.py [-h] [--task TASK] [--refresh-cookies]
                              [--browser {brave,chrome}] [--workers N]
                              [--output DIR]

  --task TASK           Fetch a single task slug only
  --refresh-cookies     Re-extract cookies from the browser
  --browser             brave (default) or chrome
  --workers N           Parallel workers (default: 6)
  --output DIR          Output directory (default: analysis/outputs/task_runs)
```

### Script 20

```
usage: 20_fetch_notebook_logs.py [-h] [--task TASK] [--model MODEL]
                                  [--refresh-cookies] [--browser {brave,chrome}]
                                  [--workers N] [--force] [--output DIR]

  --task TASK           Fetch a single task slug only
  --model MODEL         Fetch a single model only (e.g. "GPT-5.4")
  --refresh-cookies     Re-extract cookies from the browser
  --browser             brave (default) or chrome
  --workers N           Parallel workers (default: 2; max safe = 3)
  --force               Re-download even if output file already exists
  --output DIR          Output directory (default: analysis/outputs/notebook_logs)
```
