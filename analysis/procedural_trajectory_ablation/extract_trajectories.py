"""
Procedural Learning Trajectory Ablation Study
==============================================
Extracts per-round scores from stdout_log conversation histories for all
procedural learning tasks × models, then computes learning trajectory
statistics (OLS slope, classification, component estimates).

Outputs:
  trajectories.csv        — one row per (task, model, round), with round score
  trajectory_summary.csv  — one row per (task, model), aggregated stats
  trajectory_report.md    — human-readable ablation study report
"""

import json
import re
import os
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOGS_DIR = Path("/Users/kd/Desktop/proj/learning_eval/analysis/outputs/notebook_logs")
OUT_DIR = Path("/Users/kd/Desktop/proj/learning_eval/analysis/procedural_trajectory_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROC_TASKS = [
    "adaptive-sort-rule-proc-learning",
    "boolean-circuit-proc-learning",
    "dialect-morphology-proc-learning",
    "grammar-induction-proc-learning",
    "lights-out-variant-proc-learning",
    "nim-variant-proc-learning",
    "opponent-strategy-proc-learning",
    "packet-filter-proc-learning",
    "sql-reverse-engineering-proc-learning",
    "state-machine-password-proc-learning",
    "voting-protocol-proc-learning",
]

# Short display names
TASK_SHORT = {
    "adaptive-sort-rule-proc-learning": "adaptive-sort-rule",
    "boolean-circuit-proc-learning": "boolean-circuit",
    "dialect-morphology-proc-learning": "dialect-morphology",
    "grammar-induction-proc-learning": "grammar-induction",
    "lights-out-variant-proc-learning": "lights-out-variant",
    "nim-variant-proc-learning": "nim-variant",
    "opponent-strategy-proc-learning": "opponent-strategy",
    "packet-filter-proc-learning": "packet-filter",
    "sql-reverse-engineering-proc-learning": "sql-reverse-engineering",
    "state-machine-password-proc-learning": "state-machine-password",
    "voting-protocol-proc-learning": "voting-protocol",
}

# Model display name normalization (filename stem → readable name)
MODEL_NAME_MAP = {
    "Claude_Haiku_4_5": "Claude Haiku",
    "Claude_Opus_4_6": "Claude Opus",
    "Claude_Sonnet_4_6": "Claude Sonnet",
    "DeepSeek_V3_2": "DeepSeek",
    "GLM-5": "GLM-5",
    "GPT-5_4": "GPT-5.4",
    "GPT-5_4_mini": "GPT-mini",
    "GPT-5_4_nano": "GPT-nano",
    "Gemini_2_5_Flash": "G-Flash",
    "Gemini_3_1_Flash-Lite_Preview": "G-Lite",
    "Gemini_3_1_Pro_Preview": "G-Pro",
    "Gemini_3_Flash_Preview": "G-Flash-3",
    "Gemma_4_26B_A4B": "Gemma",
    "Qwen_3_Next_80B_Instruct": "Qwen-I",
    "Qwen_3_Next_80B_Thinking": "Qwen-T",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
PRACTICE_SCORE_RE = re.compile(
    r"\[Practice\s+(\d+)/\d+\].*?(?:PASS|FAIL).*?score=([\d.]+)", re.DOTALL
)
FINAL_SCORE_RE = re.compile(
    r"\[Final test\s+(\d+)/\d+\].*?(?:PASS|FAIL).*?score=([\d.]+)", re.DOTALL
)
OVERALL_SCORE_RE = re.compile(r"Final score\s*:\s*([\d.]+)")


def parse_stdout(stdout: str):
    """Return (practice_scores, transfer_scores, overall_score).
    practice_scores: list of (round_num, score) for rounds 1..5
    transfer_scores: list of (test_num, score) for final tests
    """
    practice_scores = []
    transfer_scores = []

    # Split into per-round blocks by the Practice / Final test headers
    # We'll use line-by-line scanning to be robust to varying formats
    lines = stdout.split("\n")
    current_round = None
    current_type = None  # "practice" or "transfer"
    buffer = []

    def flush_buffer(buf, rtype, rnum):
        text = "\n".join(buf)
        m = re.search(r"score=([\d.]+)", text)
        if m:
            score = float(m.group(1))
            return (rnum, score)
        return None

    for line in lines:
        # Detect Practice header
        m = re.match(r"\s*\[Practice\s+(\d+)/\d+\]", line)
        if m:
            if buffer and current_round is not None:
                result = flush_buffer(buffer, current_type, current_round)
                if result:
                    if current_type == "practice":
                        practice_scores.append(result)
                    else:
                        transfer_scores.append(result)
            current_round = int(m.group(1))
            current_type = "practice"
            buffer = [line]
            continue

        # Detect Final test header
        m = re.match(r"\s*\[Final test\s+(\d+)/\d+\]", line)
        if m:
            if buffer and current_round is not None:
                result = flush_buffer(buffer, current_type, current_round)
                if result:
                    if current_type == "practice":
                        practice_scores.append(result)
                    else:
                        transfer_scores.append(result)
            current_round = int(m.group(1))
            current_type = "transfer"
            buffer = [line]
            continue

        if buffer is not None:
            buffer.append(line)

    # Flush last buffer
    if buffer and current_round is not None:
        result = flush_buffer(buffer, current_type, current_round)
        if result:
            if current_type == "practice":
                practice_scores.append(result)
            else:
                transfer_scores.append(result)

    # Overall score
    m = OVERALL_SCORE_RE.search(stdout)
    overall = float(m.group(1)) if m else None

    return practice_scores, transfer_scores, overall


def ols_slope(scores):
    """Compute OLS slope of scores vs round index (1-based).
    Returns None if fewer than 2 points.
    """
    if len(scores) < 2:
        return None
    n = len(scores)
    xs = list(range(1, n + 1))
    ys = scores
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def classify_trajectory(slope, threshold=0.03):
    """Classify a learning trajectory based on OLS slope."""
    if slope is None:
        return "insufficient_data"
    if slope > threshold:
        return "improving"
    elif slope < -threshold:
        return "deteriorating"
    else:
        return "flat"


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------
all_round_rows = []    # for trajectories.csv
summary_rows = []      # for trajectory_summary.csv

for task_folder in PROC_TASKS:
    task_dir = LOGS_DIR / task_folder
    if not task_dir.exists():
        print(f"WARNING: {task_folder} not found in logs")
        continue

    task_short = TASK_SHORT[task_folder]

    for json_file in sorted(task_dir.glob("*.json")):
        model_stem = json_file.stem
        model_display = MODEL_NAME_MAP.get(model_stem, model_stem)

        with open(json_file) as f:
            data = json.load(f)

        stdout = data.get("stdout_log", "")
        if not stdout:
            print(f"  WARNING: No stdout_log for {task_short} / {model_display}")
            continue

        practice_scores, transfer_scores, overall = parse_stdout(stdout)

        # Extract just the numeric scores in order
        p_scores = [s for _, s in sorted(practice_scores)]
        t_scores = [s for _, s in sorted(transfer_scores)]

        # Per-round rows
        for i, s in enumerate(p_scores, 1):
            all_round_rows.append({
                "task": task_short,
                "model": model_display,
                "phase": "practice",
                "round": i,
                "score": s,
            })
        for i, s in enumerate(t_scores, 1):
            all_round_rows.append({
                "task": task_short,
                "model": model_display,
                "phase": "transfer",
                "round": i,
                "score": s,
            })

        # Summary stats
        slope = ols_slope(p_scores)
        classification = classify_trajectory(slope)
        first_score = p_scores[0] if p_scores else None
        last_score = p_scores[-1] if p_scores else None
        mean_practice = sum(p_scores) / len(p_scores) if p_scores else None
        mean_transfer = sum(t_scores) / len(t_scores) if t_scores else None
        delta = (last_score - first_score) if (first_score is not None and last_score is not None) else None

        summary_rows.append({
            "task": task_short,
            "model": model_display,
            "n_practice_rounds": len(p_scores),
            "n_transfer_tests": len(t_scores),
            "practice_round_1": round(first_score, 4) if first_score is not None else "",
            "practice_round_last": round(last_score, 4) if last_score is not None else "",
            "delta_first_to_last": round(delta, 4) if delta is not None else "",
            "ols_slope": round(slope, 5) if slope is not None else "",
            "trajectory_class": classification,
            "mean_practice_score": round(mean_practice, 4) if mean_practice is not None else "",
            "mean_transfer_score": round(mean_transfer, 4) if mean_transfer is not None else "",
            "composite_score": data.get("score_value", overall),
        })

        slope_str = f"{slope:.4f}" if slope is not None else "N/A"
        print(f"  {task_short:35s} {model_display:15s}  practice={p_scores}  slope={slope_str}  → {classification}")

# ---------------------------------------------------------------------------
# Write trajectories.csv
# ---------------------------------------------------------------------------
traj_path = OUT_DIR / "trajectories.csv"
with open(traj_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["task", "model", "phase", "round", "score"])
    writer.writeheader()
    writer.writerows(all_round_rows)
print(f"\nWrote {len(all_round_rows)} rows to {traj_path}")

# ---------------------------------------------------------------------------
# Write trajectory_summary.csv
# ---------------------------------------------------------------------------
summary_path = OUT_DIR / "trajectory_summary.csv"
with open(summary_path, "w", newline="") as f:
    fields = ["task", "model", "n_practice_rounds", "n_transfer_tests",
              "practice_round_1", "practice_round_last", "delta_first_to_last",
              "ols_slope", "trajectory_class", "mean_practice_score",
              "mean_transfer_score", "composite_score"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"Wrote {len(summary_rows)} rows to {summary_path}")
