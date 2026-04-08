#!/usr/bin/env python3
"""
Run the LLM (Qwen 2.5 3B / GSM8K) wall-clock timing matrix.

Usage:
    python scripts/run_llm_wallclock_matrix.py --gpu 0
    python scripts/run_llm_wallclock_matrix.py --gpu 0 --methods svd,factor_fedavg --scenarios shared_b
    python scripts/run_llm_wallclock_matrix.py --gpu 0 --dry-run
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WALLCLOCK_DIR = REPO_ROOT / "federatedscope/llm/yamls/wallclock"
RESULTS_FILE = WALLCLOCK_DIR / "RESULTS.md"
OUTDIR_ROOT = REPO_ROOT / "exp/llm_wallclock"
CKPT_ROOT = REPO_ROOT / "exp/llm_wallclock_ckpts"

METHOD_TO_CFG = {
    "factor_fedavg": WALLCLOCK_DIR / "qwen25_3b_factor_fedavg.yaml",
    "factor_fedprox": WALLCLOCK_DIR / "qwen25_3b_factor_fedprox.yaml",
    "fa_lora": WALLCLOCK_DIR / "qwen25_3b_fa_lora.yaml",
    "svd": WALLCLOCK_DIR / "qwen25_3b_svd.yaml",
    "svd_no_gram": WALLCLOCK_DIR / "qwen25_3b_svd_no_gram.yaml",
    "fedma": WALLCLOCK_DIR / "qwen25_3b_fedma.yaml",
    "full_rank": WALLCLOCK_DIR / "qwen25_3b_full_rank.yaml",
}

METHOD_ORDER = list(METHOD_TO_CFG.keys())
SCENARIO_ORDER = ["shared_b", "personalized_b"]

SCENARIO_LOCAL_PARAMS = {
    "shared_b": [],
    "personalized_b": ["lora_B"],
}

INVALID_COMBOS = {
    ("fa_lora", "personalized_b"): "freeze_A baseline requires shared lora_B",
    ("full_rank", "personalized_b"):
        "full-rank aggregation requires shared lora_B",
}


def parse_csv_arg(value, allowed):
    if value is None:
        return allowed
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"Unknown values: {invalid}")
    return [item for item in allowed if item in selected]


def ensure_results_header(results_path: Path, reset=False):
    header = (
        "# LLM Wall-Clock Results (Qwen 2.5 3B / GSM8K)\n\n"
        "This file is appended automatically by "
        "`scripts/run_llm_wallclock_matrix.py`.\n\n"
        "| Method | Scenario | Status | Server Round Avg (s) | "
        "Client Round Avg Mean (s) | Client Round Avg Std (s) | "
        "Server Timed Rounds | Client Timed Rounds Mean | Notes |\n"
        "| --- | --- | --- | ---: | ---: | ---: | "
        "---: | ---: | --- |\n")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        is_empty = f.tell() == 0
        if reset:
            f.seek(0)
            f.truncate()
            f.write(header)
        elif is_empty:
            f.write(header)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def format_float(value):
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def append_markdown_row(results_path: Path, row):
    row_line = (
        f"| {row['method']} | {row['scenario']} | "
        f"{row['status']} | {row['server_round_avg']} | "
        f"{row['client_round_avg_mean']} | {row['client_round_avg_std']} | "
        f"{row['server_timed_rounds']} | {row['client_timed_rounds_mean']} | "
        f"{row['notes']} |\n")
    row_prefix = f"| {row['method']} | {row['scenario']} |"
    with results_path.open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        lines = f.readlines()
        lines = [line for line in lines if not line.startswith(row_prefix)]
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(row_line)
        f.seek(0)
        f.truncate()
        f.writelines(lines)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def parse_system_metrics(outdir: Path):
    candidates = sorted(outdir.rglob("system_metrics.log"))
    if not candidates:
        raise FileNotFoundError(f"Missing system_metrics.log under {outdir}")
    sys_path = candidates[-1]

    server = None
    clients = []
    with sys_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            res = json.loads(line)
            worker_id = res.get("id")
            if worker_id in ["sys_avg", "sys_std"]:
                continue
            if worker_id == 0:
                server = res
            elif isinstance(worker_id, int) and worker_id > 0:
                clients.append(res)

    if server is None:
        raise ValueError(f"No server metrics found in {sys_path}")
    if not clients:
        raise ValueError(f"No client metrics found in {sys_path}")
    return server, clients, sys_path.parent


def mean(values):
    return sum(values) / len(values) if values else None


def std(values):
    if not values:
        return None
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def summarize_metrics(method, scenario, server, clients, notes=""):
    client_round_values = [
        client.get("client_round_time_avg_seconds_excluding_first_round")
        for client in clients
        if client.get(
            "client_round_time_avg_seconds_excluding_first_round") is not None
    ]
    client_timed_rounds = [
        client.get("client_round_time_timed_rounds")
        for client in clients
        if client.get("client_round_time_timed_rounds") is not None
    ]

    return {
        "method": method,
        "scenario": scenario,
        "status": "ok",
        "server_round_avg": format_float(
            server.get(
                "server_round_time_avg_seconds_excluding_first_round")),
        "client_round_avg_mean": format_float(mean(client_round_values)),
        "client_round_avg_std": format_float(std(client_round_values)),
        "server_timed_rounds": str(
            server.get("server_round_time_timed_rounds", "N/A")),
        "client_timed_rounds_mean": format_float(mean(client_timed_rounds)),
        "notes": notes or "`ok`",
    }


def append_nonrun_row(results_path: Path, method, scenario, status, notes):
    append_markdown_row(
        results_path,
        {
            "method": method,
            "scenario": scenario,
            "status": status,
            "server_round_avg": "N/A",
            "client_round_avg_mean": "N/A",
            "client_round_avg_std": "N/A",
            "server_timed_rounds": "N/A",
            "client_timed_rounds_mean": "N/A",
            "notes": notes,
        },
    )


def build_case_command(gpu, method, scenario, outdir, save_to):
    cfg_path = METHOD_TO_CFG[method]
    cmd = [
        sys.executable,
        "federatedscope/main.py",
        "--cfg",
        str(cfg_path),
        "device",
        "0",
        "eval_device",
        "0",
        "outdir",
        str(outdir),
        "federate.save_to",
        str(save_to),
        "wandb.use",
        "False",
    ]
    local_param = SCENARIO_LOCAL_PARAMS[scenario]
    if local_param:
        cmd += ["personalization.local_param", str(local_param)]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, help="Physical GPU id")
    parser.add_argument("--methods", help="Comma-separated method list")
    parser.add_argument("--scenarios", help="Comma-separated scenario list")
    parser.add_argument(
        "--results-file",
        default=str(RESULTS_FILE),
        help="Markdown table output path",
    )
    parser.add_argument(
        "--root-outdir",
        default=str(OUTDIR_ROOT),
        help="Root directory for experiment outputs",
    )
    parser.add_argument(
        "--root-save-dir",
        default=str(CKPT_ROOT),
        help="Root directory for checkpoints",
    )
    parser.add_argument("--reset-results", action="store_true")
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Stream each case log to stdout while saving it",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = parse_csv_arg(args.methods, METHOD_ORDER)
    scenarios = parse_csv_arg(args.scenarios, SCENARIO_ORDER)

    results_path = Path(args.results_file)
    outdir_root = Path(args.root_outdir)
    save_root = Path(args.root_save_dir)
    ensure_results_header(results_path, reset=args.reset_results)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for method in methods:
        for scenario in scenarios:
            invalid_reason = INVALID_COMBOS.get((method, scenario))
            if invalid_reason is not None:
                append_nonrun_row(
                    results_path, method, scenario, "N/A", invalid_reason
                )
                continue

            case_name = f"gsm8k__{method}__{scenario}"
            outdir = outdir_root / scenario / method
            save_to = save_root / f"{case_name}.ckpt"
            log_path = outdir / "command.log"

            shutil.rmtree(outdir, ignore_errors=True)
            outdir.mkdir(parents=True, exist_ok=True)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            for stale_ckpt in save_to.parent.glob(f"{case_name}*.ckpt*"):
                if stale_ckpt.exists():
                    stale_ckpt.unlink()

            cmd = build_case_command(
                args.gpu, method, scenario, outdir, save_to
            )
            if args.dry_run:
                print(" ".join(cmd))
                continue

            print(
                f"[start] method={method} scenario={scenario} "
                f"gpu={args.gpu} log={log_path}",
                flush=True,
            )
            with log_path.open("w") as log_f:
                log_f.write("Command:\n")
                log_f.write(" ".join(cmd) + "\n\n")
                log_f.flush()
                if args.show_logs:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        log_f.write(line)
                        log_f.flush()
                        print(line, end="")
                    proc.wait()
                else:
                    proc = subprocess.run(
                        cmd,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                    )

            if proc.returncode != 0:
                print(
                    f"[failed] method={method} scenario={scenario} "
                    f"exit={proc.returncode} log={log_path}",
                    flush=True,
                )
                append_nonrun_row(
                    results_path,
                    method,
                    scenario,
                    "failed",
                    f"`exit={proc.returncode}`",
                )
                continue

            try:
                server, clients, metrics_dir = parse_system_metrics(outdir)
                row = summarize_metrics(
                    method,
                    scenario,
                    server,
                    clients,
                    notes=f"`{metrics_dir}`",
                )
                append_markdown_row(results_path, row)
                print(
                    f"[done] method={method} scenario={scenario} "
                    f"server_round_avg={row['server_round_avg']}s "
                    f"client_round_avg_mean={row['client_round_avg_mean']}s "
                    f"metrics={metrics_dir}",
                    flush=True,
                )
            except Exception as error:
                print(
                    f"[parse_failed] method={method} scenario={scenario} "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
                append_nonrun_row(
                    results_path,
                    method,
                    scenario,
                    "parse_failed",
                    f"`{type(error).__name__}: {error}`",
                )


if __name__ == "__main__":
    main()
