#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


METHOD_PATTERN = re.compile(r"\n\s*method:\s*([a-zA-Z0-9_]+)\n")
TOK_PATTERN = re.compile(r"\n\s*tok_len:\s*(\d+)\n")
CFG_PATTERN = re.compile(r"--cfg',\s*'([^']+)'")


def detect_kind(summary):
    if "eval/checkpoint" in summary:
        return "eval"
    if "client_summarized_weighted_avg/train_avg_loss" in summary:
        return "train"
    if "val_acc" in summary or "train_avg_loss" in summary:
        return "train"
    return "other"


def extract_method(config_text):
    methods = METHOD_PATTERN.findall(config_text)
    for method in methods:
        if method != "none":
            return method
    return "unknown"


def extract_tok_len(config_text):
    match = TOK_PATTERN.search(config_text)
    if match:
        return int(match.group(1))
    return None


def extract_cfg_name(debug_text):
    match = CFG_PATTERN.search(debug_text)
    if match:
        return Path(match.group(1)).name
    return "unknown"


def load_run_row(run_dir):
    files_dir = run_dir / "files"
    logs_dir = run_dir / "logs"

    summary_path = files_dir / "wandb-summary.json"
    config_path = files_dir / "config.yaml"
    debug_path = logs_dir / "debug.log"

    if not summary_path.exists() or not config_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return None

    config_text = config_path.read_text(errors="ignore")
    debug_text = debug_path.read_text(errors="ignore") if debug_path.exists() else ""

    runtime_s = summary.get("_runtime")
    if runtime_s is None:
        runtime_s = summary.get("_wandb", {}).get("runtime")

    row = {
        "run_id": run_dir.name,
        "kind": detect_kind(summary),
        "method": extract_method(config_text),
        "tok_len": extract_tok_len(config_text),
        "task": summary.get("eval/task", ""),
        "runtime_s": runtime_s,
        "gsm8k_accuracy": summary.get("eval/gsm8k_accuracy"),
        "eval_checkpoint": summary.get("eval/checkpoint", ""),
        "cfg_name": extract_cfg_name(debug_text),
    }
    return row


def write_rows_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "kind",
        "method",
        "tok_len",
        "task",
        "runtime_s",
        "gsm8k_accuracy",
        "cfg_name",
        "eval_checkpoint",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_agg_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buckets = defaultdict(list)
    for row in rows:
        runtime_s = row.get("runtime_s")
        if runtime_s is None:
            continue
        try:
            runtime_s = float(runtime_s)
        except Exception:
            continue
        key = (row["kind"], row["method"], row["tok_len"], row["task"])
        buckets[key].append(runtime_s)

    fields = [
        "kind",
        "method",
        "tok_len",
        "task",
        "n_runs",
        "mean_runtime_s",
        "min_runtime_s",
        "max_runtime_s",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (kind, method, tok_len, task), vals in sorted(buckets.items()):
            writer.writerow(
                {
                    "kind": kind,
                    "method": method,
                    "tok_len": tok_len,
                    "task": task,
                    "n_runs": len(vals),
                    "mean_runtime_s": round(sum(vals) / len(vals), 3),
                    "min_runtime_s": round(min(vals), 3),
                    "max_runtime_s": round(max(vals), 3),
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Profile local W&B runs and export timing CSV files."
    )
    parser.add_argument(
        "--wandb-dir",
        default="wandb",
        help="Path to local W&B directory (default: wandb)",
    )
    parser.add_argument(
        "--out-rows",
        default="eval_result/wandb_run_profile_rows.csv",
        help="Output CSV path for per-run rows",
    )
    parser.add_argument(
        "--out-agg",
        default="eval_result/wandb_run_profile_agg.csv",
        help="Output CSV path for grouped aggregates",
    )
    parser.add_argument(
        "--run-prefix",
        default="",
        help="Optional run dir prefix filter, e.g. run-20260403_",
    )
    args = parser.parse_args()

    wandb_dir = Path(args.wandb_dir)
    run_dirs = sorted(wandb_dir.glob("run-*"))

    rows = []
    for run_dir in run_dirs:
        if args.run_prefix and not run_dir.name.startswith(args.run_prefix):
            continue
        row = load_run_row(run_dir)
        if row is not None:
            rows.append(row)

    write_rows_csv(Path(args.out_rows), rows)
    write_agg_csv(Path(args.out_agg), rows)

    print(f"Profiled runs: {len(rows)}")
    print(f"Per-run CSV: {args.out_rows}")
    print(f"Aggregate CSV: {args.out_agg}")


if __name__ == "__main__":
    main()
