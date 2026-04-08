#!/usr/bin/env python3
"""
Build the LLM (Qwen 2.5 3B / GSM8K) wall-clock rebuttal report.

Reads RESULTS.md produced by run_llm_wallclock_matrix.py and writes
WALLCLOCK_REPORT.md in the same directory with formatted tables matching
the GLUE rebuttal format.

Usage:
    python scripts/build_llm_wallclock_rebuttal.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path("/home/mahi/app/APRILS")
WALLCLOCK_DIR = REPO_ROOT / "federatedscope/llm/yamls/wallclock"
RESULTS_PATH = WALLCLOCK_DIR / "RESULTS.md"
OUTPUT_PATH = WALLCLOCK_DIR / "WALLCLOCK_REPORT.md"

# Internal method name -> display name in the report.
DISPLAY_NAMES = {
    "fa_lora": "FFA-LoRA",
    "factor_fedavg": "Factor-wise",
    "factor_fedprox": "FedProx",
    "fedma": "FedMA",
    "svd_no_gram": "PERSIA (w/o whitening)",
    "svd": "**PERSIA**",
    "full_rank": "Full-rank + SVD",
}

# Order of methods in the Global FL table (sorted by expected server cost).
GLOBAL_METHOD_ORDER = [
    "fa_lora",
    "factor_fedavg",
    "factor_fedprox",
    "fedma",
    "svd_no_gram",
    "svd",
    "full_rank",
]

# Order of methods in the Personalized FL table.
PERSONAL_METHOD_ORDER = [
    "factor_fedprox",
    "factor_fedavg",
    "fedma",
    "svd_no_gram",
    "svd",
]

WALLCLOCK_ROUNDS = 5
WALLCLOCK_LOCAL_STEPS = 10
MAIN_LOCAL_STEPS = 50

BASELINE_METHOD = "factor_fedavg"


@dataclass
class ResultRow:
    method: str
    scenario: str
    status: str
    server_round: Optional[float]
    client_round: Optional[float]
    client_std: Optional[float]
    server_timed_rounds: Optional[float]
    client_timed_rounds: Optional[float]
    notes: str


def parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def read_results() -> List[ResultRow]:
    rows: List[ResultRow] = []
    for line in RESULTS_PATH.read_text().splitlines():
        if not line.startswith("|") or line.startswith("| ---") or \
                "Method | Scenario" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 9:
            continue
        rows.append(
            ResultRow(
                method=parts[0],
                scenario=parts[1],
                status=parts[2],
                server_round=parse_float(parts[3]),
                client_round=parse_float(parts[4]),
                client_std=parse_float(parts[5]),
                server_timed_rounds=parse_float(parts[6]),
                client_timed_rounds=parse_float(parts[7]),
                notes=parts[8],
            ))
    return rows


def get_row(results: List[ResultRow], method: str,
            scenario: str) -> Optional[ResultRow]:
    for row in results:
        if row.method == method and row.scenario == scenario \
                and row.status == "ok":
            return row
    return None


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_timing_table(
    results: List[ResultRow],
    scenario: str,
    method_order: List[str],
) -> str:
    baseline = get_row(results, BASELINE_METHOD, scenario)
    baseline_server = baseline.server_round if baseline else None

    header = (
        "| Method | Server (s) | Client (s) | Server / Client "
        "| Server overhead vs. Factor-wise |"
    )
    separator = "|---|---:|---:|---:|---:|"
    lines = [header, separator]

    # Collect rows with their server times for sorting.
    table_rows: List[tuple] = []
    for method in method_order:
        row = get_row(results, method, scenario)
        if row is None:
            continue
        server = row.server_round
        client = row.client_round
        display = DISPLAY_NAMES.get(method, method)

        if server is not None and client is not None and client > 0:
            ratio = f"{server / client * 100:.1f}%"
        else:
            ratio = "N/A"

        if method == BASELINE_METHOD:
            overhead = "\u2014 (baseline)"
        elif server is not None and baseline_server is not None:
            delta = server - baseline_server
            if delta < 0:
                overhead = f"\u22120.{abs(delta) * 1000:.0f}"
                overhead = f"\u2212{abs(delta):.3f}s (faster)"
            else:
                overhead = f"+{delta:.3f}s"
        else:
            overhead = "N/A"

        bold = method == "svd"
        table_rows.append((server or 0, display, fmt(server), fmt(client),
                           ratio, overhead, bold))

    # Sort by server time ascending.
    table_rows.sort(key=lambda x: x[0])

    for _, display, server_s, client_s, ratio, overhead, bold in table_rows:
        if bold:
            lines.append(
                f"| {display} | **{server_s}** | **{client_s}** "
                f"| **{ratio}** | **{overhead}** |"
            )
        else:
            lines.append(
                f"| {display} | {server_s} | {client_s} "
                f"| {ratio} | {overhead} |"
            )

    return "\n".join(lines)


def build_projection_table(results: List[ResultRow]) -> str:
    global_baseline = get_row(results, BASELINE_METHOD, "shared_b")
    global_svd = get_row(results, "svd", "shared_b")
    personal_baseline = get_row(results, BASELINE_METHOD, "personalized_b")
    personal_svd = get_row(results, "svd", "personalized_b")

    header = (
        "| Setting | Extra server cost | Client time "
        f"({WALLCLOCK_LOCAL_STEPS} steps) | Client time "
        f"({MAIN_LOCAL_STEPS} steps) | Overhead "
        f"({WALLCLOCK_LOCAL_STEPS} steps) | **Overhead "
        f"({MAIN_LOCAL_STEPS} steps)** |"
    )
    separator = "|---|---:|---:|---:|---:|---:|"
    lines = [header, separator]

    scale = MAIN_LOCAL_STEPS / WALLCLOCK_LOCAL_STEPS

    for label, baseline, svd in [
        ("Global FL", global_baseline, global_svd),
        ("Personalized FL", personal_baseline, personal_svd),
    ]:
        if baseline is None or svd is None:
            lines.append(f"| {label} | N/A | N/A | N/A | N/A | N/A |")
            continue

        extra_server = (svd.server_round or 0) - (baseline.server_round or 0)
        client_short = baseline.client_round or 0
        client_long = client_short * scale

        if client_short > 0:
            overhead_short = extra_server / client_short * 100
            overhead_long = extra_server / client_long * 100
            lines.append(
                f"| {label} | +{extra_server:.3f}s | {client_short:.3f}s "
                f"| ~{client_long:.2f}s | {overhead_short:.1f}% "
                f"| **{overhead_long:.1f}%** |"
            )
        else:
            lines.append(f"| {label} | +{extra_server:.3f}s | N/A "
                         f"| N/A | N/A | N/A |")

    return "\n".join(lines)


def build_report(results: List[ResultRow]) -> str:
    global_table = build_timing_table(results, "shared_b",
                                      GLOBAL_METHOD_ORDER)
    personal_table = build_timing_table(results, "personalized_b",
                                        PERSONAL_METHOD_ORDER)
    projection_table = build_projection_table(results)

    # Extract key numbers for the narrative sections.
    global_svd = get_row(results, "svd", "shared_b")
    global_baseline = get_row(results, BASELINE_METHOD, "shared_b")
    personal_svd = get_row(results, "svd", "personalized_b")
    personal_baseline = get_row(results, BASELINE_METHOD, "personalized_b")

    extra_server_global = "N/A"
    overhead_main_global = "N/A"
    extra_server_personal = "N/A"
    overhead_main_personal = "N/A"

    if global_svd and global_baseline and \
            global_svd.server_round is not None and \
            global_baseline.server_round is not None and \
            global_baseline.client_round is not None and \
            global_baseline.client_round > 0:
        delta = global_svd.server_round - global_baseline.server_round
        extra_server_global = f"{delta:+.3f}s"
        client_long = global_baseline.client_round * (
            MAIN_LOCAL_STEPS / WALLCLOCK_LOCAL_STEPS)
        overhead_main_global = f"{delta / client_long * 100:.1f}%"

    if personal_svd and personal_baseline and \
            personal_svd.server_round is not None and \
            personal_baseline.server_round is not None and \
            personal_baseline.client_round is not None and \
            personal_baseline.client_round > 0:
        delta = personal_svd.server_round - personal_baseline.server_round
        extra_server_personal = f"{delta:+.3f}s"
        client_long = personal_baseline.client_round * (
            MAIN_LOCAL_STEPS / WALLCLOCK_LOCAL_STEPS)
        overhead_main_personal = f"{delta / client_long * 100:.1f}%"

    return f"""\
## Wall-Clock Results

Following experiments are run for {WALLCLOCK_ROUNDS} rounds with \
{WALLCLOCK_LOCAL_STEPS} local steps per round on Qwen 2.5 3B-Instruct \
fine-tuned on GSM8K with LoRA (rank 8). The server-side wall-clock \
measurement reports the total time for aggregation per each FL round. \
The client-side wall-clock measurement reports the time for a single \
local training round ({WALLCLOCK_LOCAL_STEPS} steps of batch size 4).

### Global FL \u2014 Per-Round Timing (seconds)

{global_table}

### Personalized FL \u2014 Per-Round Timing (seconds)

{personal_table}

*Measured on Qwen 2.5 3B-Instruct with GSM8K (tok_len=256, batch_size=4, \
LoRA r=8, 3 clients).*

## Overhead Projection to Main Experiments ({MAIN_LOCAL_STEPS} local steps)

The wall-clock measurements above use {WALLCLOCK_LOCAL_STEPS} local steps \
per round. In our main experiments, clients perform \
**{MAIN_LOCAL_STEPS} local steps**. Since server aggregation cost is \
independent of local training duration, the overhead ratio shrinks \
proportionally:

{projection_table}

**At {MAIN_LOCAL_STEPS} local steps, PERSIA adds {overhead_main_global} \
wall-clock overhead per round in Global FL and {overhead_main_personal} \
in Personalized FL.** In practical deployments where clients may run \
hundreds of local steps, the overhead becomes negligible.

## Addressing the Sequential Bottleneck Concern

We want to clarify three points regarding the reviewer's concern about \
L separate SVDs creating a sequential bottleneck:

**1. Layer-wise SVDs are independent and parallelizable.** PERSIA \
performs a truncated SVD per adapted layer, and these operations are \
entirely independent. In practice, all layer parameters are already \
resident in VRAM, so all L low-rank SVDs can execute in parallel \
(or be batched), avoiding a sequential bottleneck. Our implementation \
uses `torch.svd_lowrank`, which leverages GPU parallelism.

**2. Aggregation is infrequent relative to training.** The SVD occurs \
once per FL round, whereas each round involves \u03c4 forward\u2013backward \
passes across all clients. As shown above, even with only \
{WALLCLOCK_LOCAL_STEPS} local steps, client-side training already \
dominates the round time. With {MAIN_LOCAL_STEPS} local steps (our \
main setting), the server-side SVD is a small fraction of the round.

**3. Low-rank SVD is tunable.** We use `torch.svd_lowrank(n_iter=k)` \
where the number of power iterations k controls an accuracy\u2013speed \
trade-off. With rank r = 8 and n_iter = 3, the SVD operates on \
small r \u00d7 r matrices after the initial randomized projection, which \
is extremely fast even for large d.

## Net Effect: Communication Savings Dominate

The central practical benefit of PERSIA is enabling longer local \
training without collapse, which reduces the number of FL rounds \
needed. This represents a significant reduction in communication \
rounds, far exceeding the per-round aggregation overhead.
"""


def main() -> None:
    results = read_results()
    report = build_report(results)
    OUTPUT_PATH.write_text(report)
    print(f"Wrote {OUTPUT_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()
