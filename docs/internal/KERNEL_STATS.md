# Kernel Execution Statistics — Token Usage & Timing Analysis

**Generated from:** Kaggle kernel execution logs (`run.json` files)  
**Coverage:** 126 / 134 task kernels downloaded; token data available for 82 (task, model) pairs across 10 models  
**Full data:** `analysis/outputs/full_task_model_stats.csv` (1,875 rows — 134 tasks × 14 models with scores + token data where available)  
**Aggregate data:** `analysis/outputs/aggregate_stats.csv` (84 rows — 14 models × 6 groups)

---

## 1. What the Data Contains

Each row in `full_task_model_stats.csv` captures:

| Column | Description |
|---|---|
| `task_slug` | Unique URL-safe task identifier |
| `task_name` | Human-readable task name |
| `category` | Learning sub-ability (Associative / Concept Formation / Language Learning / Observational / Reinforcement Learning) |
| `model` | Model display name |
| `score` | Task score ∈ [0, 1] for that model |
| `avg_input_tokens` | Average input tokens **per API request** for this task run |
| `avg_output_tokens` | Average output tokens **per API request** |
| `wall_sec` | Total wall-clock execution time (seconds) for the whole task |
| `n_requests` | Total number of API requests made during the task |
| `total_input_tokens` | Sum of all input tokens across all requests |
| `total_output_tokens` | Sum of all output tokens across all requests |
| `avg_latency_ms` | Average backend latency per request (milliseconds) |

> **Note on coverage:** Each Kaggle task kernel stores output only from the *most recently run* model version. With 14 models rotating through 134 task kernels, token data is available for the model that most recently ran each kernel — roughly one model per task, producing ~82 complete (task, model) pairs out of a possible 1,875. The 4 models with zero samples (Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5, Gemini 3.1 Flash-Lite Preview) had their kernels overwritten by subsequent model runs before this download.

---

## 2. Overall Averages per Model

Averages computed across all token-sampled task runs for each model. `#Samples` = number of tasks for which token data was available.

| Model | Tier | Mean Score | Avg Input Tok/Req | Avg Output Tok/Req | Avg Wall Time (s) | Avg Requests | # Token Samples |
|---|---|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | Frontier | 0.843 | 787 | 45 | 365 | 11.7 | 3 |
| GLM-5 | Frontier | 0.672 | 489 | 3,894 | 217 | 1.0 | 5 |
| Qwen 3 Next 80B Thinking | Standard | 0.603 | 2,370 | 14,572 | 262 | 7.4 | 7 |
| GPT-5.4 | Frontier | 0.486 | 790 | 36 | 11 | 6.1 | 7 |
| Claude Opus 4.6 | Frontier | 0.477 | — | — | — | — | 0 |
| Gemini 2.5 Flash | Standard | 0.462 | 504 | 27 | 215 | 6.3 | 6 |
| Claude Sonnet 4.6 | Standard | 0.450 | — | — | — | — | 0 |
| Gemini 3.1 Flash-Lite Preview | Efficient | 0.436 | — | — | — | — | 0 |
| DeepSeek V3.2 | Standard | 0.434 | 1,273 | 1,913 | 78 | 0.8 | 5 |
| Claude Haiku 4.5 | Efficient | 0.367 | — | — | — | — | 0 |
| GPT-5.4 mini | Standard | 0.349 | 840 | 32 | 8 | 10.8 | 4 |
| Gemma 4 26B A4B | Efficient | 0.347 | 19,732 | 51 | 459 | 2.8 | 4 |
| Qwen 3 Next 80B Instruct | Standard | 0.337 | 8,994 | 770 | 72 | 11.7 | 22 |
| GPT-5.4 nano | Efficient | 0.244 | 1,387 | 63 | 24 | 19.8 | 19 |

**Key observations:**
- **GLM-5 and Qwen Thinking** generate far more output tokens than other models — they produce verbose chain-of-thought reasoning in their responses
- **GPT-5.4** is the most token-efficient frontier model: 790 input / 36 output per request with only 11s average wall time
- **Gemma 4 26B** shows anomalously high input tokens (19,732) — explained by the `temporal-pairing-tnr` task which accumulated a 75,729-token context in its single request
- **Qwen Instruct** has high input tokens (8,994) because RL tasks accumulate long in-context conversation histories across 10–30 turns
- **GPT-5.4 nano** makes the most requests per task (19.8 avg) — this is the Efficient-tier model given the most RL-style tasks in its sample

---

## 3. Per-Category Averages

### 3.1 Associative Learning (20 tasks)

| Model | Score | Avg Input Tok | Avg Output Tok | Avg Wall (s) | # Samples |
|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | 0.935 | 1,150 | 104 | 42 | 1 |
| GLM-5 | 0.772 | 1,401 | 13,008 | 226 | 1 |
| Claude Opus 4.6 | 0.655 | — | — | — | 0 |
| GPT-5.4 | 0.647 | 458 | 38 | 2 | 1 |
| Claude Sonnet 4.6 | 0.637 | — | — | — | 0 |
| Qwen 3 Next 80B Thinking | 0.608 | 8,184 | 51,306 | 268 | 1 |
| Gemini 2.5 Flash | 0.602 | — | — | — | 0 |
| Gemini 3.1 Flash-Lite Preview | 0.562 | — | — | — | 0 |
| Claude Haiku 4.5 | 0.507 | — | — | — | 0 |
| DeepSeek V3.2 | 0.497 | — | — | — | 0 |
| Gemma 4 26B A4B | 0.495 | 75,729 | 33 | 6 | 1 |
| GPT-5.4 mini | 0.472 | — | — | — | 0 |
| Qwen 3 Next 80B Instruct | 0.456 | 745 | 73 | 3 | 7 |
| GPT-5.4 nano | 0.426 | — | — | — | 0 |

*Qwen Thinking's 51K output tokens on one associative task is unusual — this was the `temporal-pairing-kmp` task, where the model produced extensive chain-of-thought before each answer.*

### 3.2 Concept Formation (18 tasks)

| Model | Score | Avg Input Tok | Avg Output Tok | Avg Wall (s) | # Samples |
|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | 0.799 | 456 | 22 | 71 | 1 |
| GLM-5 | 0.568 | 301 | 4,167 | 195 | 1 |
| Qwen 3 Next 80B Thinking | 0.563 | 459 | 1,995 | 83 | 1 |
| Gemini 2.5 Flash | 0.533 | 503 | 33 | 203 | 3 |
| Claude Sonnet 4.6 | 0.332 | — | — | — | 0 |
| Gemini 3.1 Flash-Lite Preview | 0.313 | — | — | — | 0 |
| GPT-5.4 | 0.285 | 1,268 | 34 | 19 | 3 |
| Claude Opus 4.6 | 0.259 | — | — | — | 0 |
| Gemma 4 26B A4B | 0.255 | 853 | 62 | 1,825 | 1 |
| GPT-5.4 mini | 0.224 | 549 | 30 | 5 | 1 |
| DeepSeek V3.2 | 0.194 | — | — | — | 0 |
| Claude Haiku 4.5 | 0.193 | — | — | — | 0 |
| Qwen 3 Next 80B Instruct | 0.186 | — | — | — | 0 |
| GPT-5.4 nano | 0.149 | 1,099 | 32 | 22 | 2 |

*Gemma 4 26B took 1,825s on the `disjunctive-noise` concept task — the slowest single-task execution in the benchmark.*

### 3.3 Language Learning (26 tasks)

| Model | Score | Avg Input Tok | Avg Output Tok | Avg Wall (s) | # Samples |
|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | 0.756 | — | — | — | 0 |
| GLM-5 | 0.693 | 742 | 2,297 | 58 | 1 |
| Qwen 3 Next 80B Thinking | 0.623 | 723 | 6,046 | 62 | 1 |
| GPT-5.4 | 0.600 | — | — | — | 0 |
| Claude Opus 4.6 | 0.510 | — | — | — | 0 |
| Gemini 3.1 Flash-Lite Preview | 0.492 | — | — | — | 0 |
| DeepSeek V3.2 | 0.473 | — | — | — | 0 |
| Gemini 2.5 Flash | 0.470 | 1,050 | 23 | 341 | 1 |
| Claude Sonnet 4.6 | 0.467 | — | — | — | 0 |
| GPT-5.4 mini | 0.463 | — | — | — | 0 |
| Qwen 3 Next 80B Instruct | 0.415 | 825 | 25 | 5 | 1 |
| Claude Haiku 4.5 | 0.349 | — | — | — | 0 |
| GPT-5.4 nano | 0.315 | — | — | — | 0 |
| Gemma 4 26B A4B | 0.271 | — | — | — | 0 |

### 3.4 Observational Learning (40 tasks)

| Model | Score | Avg Input Tok | Avg Output Tok | Avg Wall (s) | # Samples |
|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | 0.851 | — | — | — | 0 |
| Qwen 3 Next 80B Thinking | 0.584 | 354 | 17,354 | 252 | 2 |
| GLM-5 | 0.573 | 0* | 0* | 302 | 2 |
| DeepSeek V3.2 | 0.426 | 1,273 | 1,913 | 78 | 5 |
| Gemini 2.5 Flash | 0.354 | 322 | 28 | 66 | 1 |
| Claude Opus 4.6 | 0.340 | — | — | — | 0 |
| Qwen 3 Next 80B Instruct | 0.309 | 1,178 | 57 | 24 | 3 |
| GPT-5.4 | 0.296 | 489 | 50 | 3 | 2 |
| Gemini 3.1 Flash-Lite Preview | 0.288 | — | — | — | 0 |
| Claude Sonnet 4.6 | 0.268 | — | — | — | 0 |
| Claude Haiku 4.5 | 0.244 | — | — | — | 0 |
| GPT-5.4 mini | 0.218 | 1,599 | 56 | 1 | 1 |
| Gemma 4 26B A4B | 0.218 | 1,173 | 55 | 3 | 2 |
| GPT-5.4 nano | 0.143 | 968 | 41 | 2 | 4 |

*\* GLM-5 shows 0 tokens on 2 observational tasks where the run.json contained 0 API requests — the model process timed out before issuing any API calls, resulting in a task failure (score was near 0 for those tasks).*

### 3.5 Reinforcement Learning (34 tasks)

| Model | Score | Avg Input Tok | Avg Output Tok | Avg Wall (s) | # Samples |
|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | 0.871 | 754 | 10 | 982 | 1 |
| GLM-5 | 0.771 | — | — | — | 0 |
| GPT-5.4 | 0.632 | 291 | 15 | 9 | 1 |
| Qwen 3 Next 80B Thinking | 0.630 | 3,260 | 3,974 | 459 | 2 |
| Claude Opus 4.6 | 0.624 | — | — | — | 0 |
| Claude Sonnet 4.6 | 0.603 | — | — | — | 0 |
| Gemini 3.1 Flash-Lite Preview | 0.555 | — | — | — | 0 |
| Claude Haiku 4.5 | 0.537 | — | — | — | 0 |
| Gemma 4 26B A4B | 0.522 | — | — | — | 0 |
| DeepSeek V3.2 | 0.510 | — | — | — | 0 |
| Gemini 2.5 Flash | 0.455 | 142 | 14 | 276 | 1 |
| GPT-5.4 mini | 0.407 | 606 | 21 | 13 | 2 |
| Qwen 3 Next 80B Instruct | 0.316 | 17,117 | 1,476 | 135 | 11 |
| GPT-5.4 nano | 0.244 | 1,561 | 75 | 31 | 13 |

*RL tasks show the highest wall times due to multi-turn exploration (20–60 steps). Gemini 3.1 Pro's 982s on `mastermind-aggregate` is 30 requests × ~33s each. Qwen Instruct's 17K avg input tokens reflect accumulating full conversation history in the context window across many turns.*

---

## 4. Notable Patterns

### Token Verbosity vs. Performance
Models with highest output tokens per request — GLM-5 (3,894) and Qwen Thinking (14,572) — are ranked #2 and #3 overall. Both models produce extensive chain-of-thought reasoning in their outputs. This suggests verbose internal monologue correlates with stronger learning performance, at least on these task types.

### Fastest vs. Slowest Execution
- **Fastest:** GPT-5.4 (avg 11s per task), GPT-5.4 mini (8s) — rapid, low-latency inference with terse outputs
- **Slowest:** Gemma 4 26B (459s avg), Gemini 3.1 Pro RL tasks (up to 982s) — slow due to long multi-turn RL episodes or high API latency

### Request Count and RL Tasks
RL tasks drive request counts up to 20–60 per task. GPT-5.4 nano averaged 19.8 requests per task (heavily sampled on RL tasks), while GLM-5 averaged 1.0 (mostly sampled on single-turn observational tasks). Comparing raw token counts across models is only meaningful within the same task type.

### Input Token Inflation in Long-Context Tasks
Some tasks accumulate very long contexts:
- `temporal-pairing-tnr` (Gemma 4 26B): **75,729 tokens** input — single-turn task with a large pre-filled observation log
- `bitstring-hamming-rf-learning` (Qwen Instruct): ~84K avg input over 30 turns — each turn appends the full conversation history
- `temporal-pairing-kmp` (Qwen Thinking): **51,306 output tokens** — the model generated extensive reasoning traces

---

## 5. Data Files

| File | Description |
|---|---|
| `analysis/outputs/full_task_model_stats.csv` | Full 1,875-row table: every (task, model) pair with score + token/timing data where available |
| `analysis/outputs/aggregate_stats.csv` | 84-row summary: per-model × per-category (5 categories + Overall = 6 rows × 14 models) |
| `analysis/outputs/kernel_logs_all/` | Raw downloaded `run.json` files, one sub-folder per task (126 folders) |

---

## 6. How to Reproduce This Data

### What the data is

Each Kaggle benchmark task runs as a **kernel** (notebook) on Kaggle's infrastructure. When a kernel executes, it produces output files including a `run.json` file. This file contains the complete record of the task execution: every API request made to the model, with the full conversation contents, token counts, latency, and cost metadata.

### Architecture constraint

Kaggle only retains output files from the **latest version** of each kernel. Since each task kernel is shared across all 14 models (each model run creates a new kernel version), only the most recently run model's `run.json` is accessible at any given time. This means token/timing data for any specific (task, model) pair is available only until another model runs that task.

### Step-by-step reproduction

**Prerequisites:**
```bash
pip install kaggle
# Set up ~/.kaggle/kaggle.json with your credentials, or:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**1. List all task kernel slugs**

The kernel slug for each task matches its task slug (e.g., `base7-decode-rf-learning`). You can extract all slugs from the leaderboard JSON files:
```python
import json
from pathlib import Path

slugs = set()
for lb_file in Path("leaderboards2").glob("*.json"):
    lb = json.loads(lb_file.read_text())
    for row in lb.get("rows", []):
        for tr in row.get("taskResults", []):
            slug = tr.get("benchmarkTaskSlug", "").split("/")[-1]
            if slug and len(slug) > 4:
                slugs.add(slug)
print(sorted(slugs))
```

**2. Download a kernel's output files**
```bash
kaggle kernels output kdcyberdude/<task-slug> -p ./output_dir -q
```

This downloads all output files for the current (latest) version of that kernel. The `run.json` file will be named:
```
<task_name_with_underscores>-run_id_Run_1_<provider>_<model-slug>.run.json
```
Example: `base7_decode_rf_learning-run_id_Run_1_openai_gpt-5.4-nano-2026-03-17.run.json`

**3. Parse the run.json**

```python
import json
from datetime import datetime

def parse_run_json(path):
    data = json.loads(open(path).read())
    
    start = datetime.fromisoformat(data["startTime"].replace("Z", "+00:00"))
    end   = datetime.fromisoformat(data["endTime"].replace("Z", "+00:00"))
    wall_sec = (end - start).total_seconds()
    
    model_slug = data["modelVersion"]["slug"]
    task_name  = data["taskVersion"]["name"]
    
    input_tokens = output_tokens = n_requests = 0
    total_latency_ms = 0
    
    for conv in data.get("conversations", []):
        for req in conv.get("requests", []):
            metrics = req.get("metrics", {})
            input_tokens  += int(metrics.get("inputTokens", 0) or 0)
            output_tokens += int(metrics.get("outputTokens", 0) or 0)
            lat = metrics.get("totalBackendLatencyMs")
            if lat:
                total_latency_ms += int(lat)
            n_requests += 1
    
    return {
        "model":              model_slug,
        "task":               task_name,
        "wall_sec":           wall_sec,
        "n_requests":         n_requests,
        "total_input_tokens": input_tokens,
        "total_output_tokens":output_tokens,
        "avg_input_tokens":   input_tokens / n_requests if n_requests else 0,
        "avg_output_tokens":  output_tokens / n_requests if n_requests else 0,
        "avg_latency_ms":     total_latency_ms / n_requests if n_requests else 0,
    }
```

**4. Run the full pipeline**

The script `analysis/scripts/19_full_stats_pipeline.py` automates all steps:
```bash
python analysis/scripts/19_full_stats_pipeline.py
# Use --skip-download to reuse cached files:
python analysis/scripts/19_full_stats_pipeline.py --skip-download
```

### Re-running for fresh data

Because Kaggle only stores the latest version, the model coverage will change as new model runs overwrite old ones. To get comprehensive coverage across all 14 models, you would need to:
1. Run each model on a dedicated task that no other model will overwrite, **or**
2. Download kernel outputs immediately after each model evaluation run before the next model runs, **or**
3. Use the Kaggle API to retrieve historical kernel versions (if your account has access to the benchmark's version history)

For the purposes of this benchmark analysis, the 82 (task, model) pairs captured represent the state of Kaggle's kernel storage at the time of download (April 15, 2026).
