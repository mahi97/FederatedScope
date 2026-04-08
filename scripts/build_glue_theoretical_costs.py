#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from federatedscope.core.configs.config import global_cfg
from federatedscope.glue.model.model_builder import get_llm


REPO_ROOT = Path("/home/mahi/app/APRILS")
CFG_DIR = REPO_ROOT / "federatedscope/glue/yamls/wallclock"
OUTPUT_PATH = CFG_DIR / "THEORETICAL_COSTS.md"

TASK_TO_NUM_LABELS = OrderedDict([
    ("cola", 2),
    ("mnli", 3),
    ("mrpc", 2),
    ("qnli", 2),
    ("qqp", 2),
    ("rte", 2),
    ("sst2", 2),
    ("stsb", 1),
    ("wnli", 2),
])

GROUP_TO_TASKS = OrderedDict([
    ("binary", ["cola", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]),
    ("three_class", ["mnli"]),
    ("regression", ["stsb"]),
])

METHOD_SCENARIOS = [
    ("factor_fedavg", "shared_b"),
    ("factor_fedavg", "personalized_b"),
    ("factor_fedprox", "shared_b"),
    ("factor_fedprox", "personalized_b"),
    ("fa_lora", "shared_b"),
    ("fa_lora", "personalized_b"),
    ("svd", "shared_b"),
    ("svd", "personalized_b"),
    ("svd_no_gram", "shared_b"),
    ("svd_no_gram", "personalized_b"),
    ("fedma", "shared_b"),
    ("fedma", "personalized_b"),
    ("full_rank", "shared_b"),
    ("full_rank", "personalized_b"),
]


def matmul_flops(m: int, n: int, p: int) -> int:
    return 2 * m * n * p


def weighted_avg_flops(num_params: int, client_num: int) -> int:
    return num_params * client_num + num_params * (client_num - 1)


def randomized_lowrank_svd_flops(m: int,
                                 n: int,
                                 q: int,
                                 niter: int = 3) -> int:
    # Proxy for torch.svd_lowrank with q=r and niter=3.
    return int((4 * niter + 4) * m * n * q + 4 * q * q * n +
               (8.0 / 3.0) * (q**3))


def fmt_count(value: int) -> str:
    return f"{value:,}"


def fmt_flops(value: int) -> str:
    abs_value = float(value)
    if abs_value >= 1e12:
        return f"{abs_value / 1e12:.3f}T"
    if abs_value >= 1e9:
        return f"{abs_value / 1e9:.3f}G"
    if abs_value >= 1e6:
        return f"{abs_value / 1e6:.3f}M"
    if abs_value >= 1e3:
        return f"{abs_value / 1e3:.3f}K"
    return str(int(value))


def fmt_params_bytes(num_params: int, bytes_per_param: int = 2) -> str:
    total_bytes = num_params * bytes_per_param
    mib = total_bytes / (1024**2)
    return f"{mib:.2f} MiB"


def load_common_cfg() -> Dict:
    with (CFG_DIR / "roberta_large_svd.yaml").open() as f:
        return yaml.safe_load(f)


def inspect_roberta_glue(num_labels: int) -> Dict:
    cfg = global_cfg.clone()
    cfg.defrost()
    cfg.model.type = "FacebookAI/roberta-large@huggingface_llm"
    cfg.data.type = "sst2@glue"
    cfg.data.num_labels = num_labels
    cfg.data.splitter = "iid"
    cfg.federate.client_num = 3
    cfg.llm.adapter.use = True
    cfg.llm.adapter.args = [{
        "adapter_package": "peft",
        "adapter_method": "lora",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05
    }]
    cfg.freeze()

    model = get_llm(cfg)
    named_params = list(model.named_parameters())
    all_param_count = sum(param.numel() for _, param in named_params)

    lora_a = [(name, tuple(param.shape), param.numel())
              for name, param in named_params if "lora_A" in name]
    lora_b = [(name, tuple(param.shape), param.numel())
              for name, param in named_params if "lora_B" in name]
    classifier = [(name, tuple(param.shape), param.numel())
                  for name, param in named_params
                  if "classifier" in name and param.requires_grad]

    if not lora_a or not lora_b:
        raise RuntimeError("Failed to detect LoRA A/B shapes")

    num_lora_modules = len(lora_a)
    rank = lora_a[0][1][0]
    hidden_dim = lora_a[0][1][1]
    lora_a_params = sum(numel for _, _, numel in lora_a)
    lora_b_params = sum(numel for _, _, numel in lora_b)
    classifier_params = sum(numel for _, _, numel in classifier)

    base_model = model.model.base_model.model
    input_dict = {
        "input_ids": torch.zeros((16, 128), dtype=torch.long),
        "attention_mask": torch.ones((16, 128), dtype=torch.long),
    }
    train_step_flops = int(base_model.floating_point_ops(input_dict))

    return {
        "all_params": all_param_count,
        "num_lora_modules": num_lora_modules,
        "rank": rank,
        "hidden_dim": hidden_dim,
        "lora_a_params": lora_a_params,
        "lora_b_params": lora_b_params,
        "classifier_params": classifier_params,
        "trainable_params_default": lora_a_params + lora_b_params +
        classifier_params,
        "trainable_params_fa_lora": lora_b_params + classifier_params,
        "train_step_flops": train_step_flops,
    }


def build_method_rows(common_cfg: Dict, model_stats_by_labels: Dict[int,
                                                                    Dict]):
    client_num = common_cfg["federate"]["client_num"]
    local_steps = common_cfg["train"]["local_update_steps"]
    total_rounds = common_cfg["federate"]["total_round_num"]

    shared_a = model_stats_by_labels[2]["lora_a_params"]
    shared_b = model_stats_by_labels[2]["lora_b_params"]
    num_lora_modules = model_stats_by_labels[2]["num_lora_modules"]
    rank = model_stats_by_labels[2]["rank"]
    hidden_dim = model_stats_by_labels[2]["hidden_dim"]

    gram_upload_round = num_lora_modules * (
        matmul_flops(rank, hidden_dim, rank) +
        matmul_flops(rank, rank, hidden_dim) + 10 * (rank**3))
    receive_reconstruct_round = num_lora_modules * (
        matmul_flops(hidden_dim, rank, hidden_dim) +
        matmul_flops(hidden_dim, hidden_dim, rank) +
        (4 * rank * rank * hidden_dim) + int((8.0 / 3.0) * (rank**3)))
    svd_a_only_round = num_lora_modules * randomized_lowrank_svd_flops(
        client_num * rank, hidden_dim, rank)
    svd_b_shared_round = num_lora_modules * (
        client_num *
        (matmul_flops(hidden_dim, rank, hidden_dim) +
         matmul_flops(hidden_dim, hidden_dim, rank)) +
        weighted_avg_flops(hidden_dim * rank, client_num))
    fedma_align_round = num_lora_modules * (
        (client_num - 1) * (rank * rank * (3 * hidden_dim - 1) + rank**3) +
        weighted_avg_flops(rank * hidden_dim, client_num))
    full_rank_round = num_lora_modules * (
        client_num * matmul_flops(hidden_dim, rank, hidden_dim) +
        weighted_avg_flops(hidden_dim * hidden_dim, client_num) +
        randomized_lowrank_svd_flops(hidden_dim, hidden_dim, rank) +
        4 * hidden_dim * rank * rank)

    factor_shared_server = weighted_avg_flops(shared_a + shared_b, client_num)
    factor_personal_server = weighted_avg_flops(shared_a, client_num)
    fa_lora_server = weighted_avg_flops(shared_b, client_num)

    rows = []
    for method, scenario in METHOD_SCENARIOS:
        valid = True
        note = ""

        if method == "fa_lora" and scenario == "personalized_b":
            valid = False
            note = "Invalid: freeze_A leaves no shared trainable parameter once lora_B is also personalized."
        elif method == "full_rank" and scenario == "personalized_b":
            valid = False
            note = "Invalid: full-rank server aggregation needs both A and B to form BA."

        if not valid:
            rows.append({
                "method": method,
                "scenario": scenario,
                "valid": False,
                "trainable_default": None,
                "shared_params": None,
                "client_extra_round": None,
                "server_round": None,
                "note": note,
            })
            continue

        if method in {"factor_fedavg", "factor_fedprox", "svd", "svd_no_gram", "fedma", "full_rank"}:
            trainable_kind = "default"
        else:
            trainable_kind = "fa_lora"

        if method == "factor_fedavg":
            shared_params = shared_a + shared_b if scenario == "shared_b" else shared_a
            client_extra_round = 0
            server_round = factor_shared_server if scenario == "shared_b" else factor_personal_server
            note = "Plain factor-wise averaging on transmitted trainable tensors."
        elif method == "factor_fedprox":
            shared_params = shared_a + shared_b if scenario == "shared_b" else shared_a
            label_2_stats = model_stats_by_labels[2]
            prox_extra = 12 * label_2_stats["all_params"]
            client_extra_round = prox_extra
            server_round = factor_shared_server if scenario == "shared_b" else factor_personal_server
            note = "Same server path as factor_fedavg, plus FedProx adds a full-model proximal scan every batch."
        elif method == "fa_lora":
            shared_params = shared_b
            client_extra_round = 0
            server_round = fa_lora_server
            note = "Only LoRA-B is trainable/shared; LoRA-A is frozen and excluded from upload."
        elif method == "svd":
            shared_params = shared_a + shared_b if scenario == "shared_b" else shared_a
            client_extra_round = gram_upload_round if scenario == "personalized_b" else 0
            server_round = svd_a_only_round + (
                svd_b_shared_round if scenario == "shared_b" else 0)
            note = "Client-side SVD logic is active only for personalized-B svd; shared-B svd uses plain client upload/update."
        elif method == "svd_no_gram":
            shared_params = shared_a + shared_b if scenario == "shared_b" else shared_a
            client_extra_round = 0
            server_round = svd_a_only_round + (
                svd_b_shared_round if scenario == "shared_b" else 0)
            note = "Same server path as svd, but the client always uses plain upload/update."
        elif method == "fedma":
            shared_params = shared_a + shared_b if scenario == "shared_b" else shared_a
            client_extra_round = 0
            server_round = fedma_align_round + (
                svd_b_shared_round if scenario == "shared_b" else 0)
            note = "Server aligns LoRA-A rows first; clients use plain upload/update."
        elif method == "full_rank":
            shared_params = shared_a + shared_b
            client_extra_round = 0
            server_round = full_rank_round
            note = "Server averages BA in full rank, then factorizes back to rank-r LoRA; clients use plain upload/update."
        else:
            raise ValueError(method)

        rows.append({
            "method": method,
            "scenario": scenario,
            "valid": True,
            "trainable_kind": trainable_kind,
            "shared_params": shared_params,
            "client_extra_round": client_extra_round,
            "server_round": int(server_round),
            "note": note,
            "total_rounds": total_rounds,
            "local_steps": local_steps,
            "gram_upload_round": gram_upload_round,
            "receive_reconstruct_round": receive_reconstruct_round,
        })
    return rows


def render_markdown(common_cfg: Dict, model_stats_by_labels: Dict[int, Dict],
                    method_rows: List[Dict]) -> str:
    batch_size = common_cfg["dataloader"]["batch_size"]
    local_steps = common_cfg["train"]["local_update_steps"]
    tok_len = common_cfg["llm"]["tok_len"]
    client_num = common_cfg["federate"]["client_num"]
    total_rounds = common_cfg["federate"]["total_round_num"]

    lines: List[str] = []
    lines.append("# GLUE Theoretical Cost Model")
    lines.append("")
    lines.append("This file estimates the computation carried by each GLUE wall-clock baseline in the current rebuttal setup.")
    lines.append("")
    lines.append("Scope:")
    lines.append(f"- Config family: `{CFG_DIR}`")
    lines.append(
        f"- Backbone: `FacebookAI/roberta-large@huggingface_llm`, LoRA rank `8`, query/value adapters on all `24` layers")
    lines.append(
        f"- FL setup: `client_num={client_num}`, `local_update_steps={local_steps}`, `batch_size={batch_size}`, `tok_len={tok_len}`, `total_round_num={total_rounds}`"
    )
    lines.append(
        "- Common client training FLOPs are taken from the model class's own `floating_point_ops(...)` estimator for a `16 x 128` batch."
    )
    lines.append(
        "- Method-specific deltas are derived directly from the code paths in `svd_trainer.py`, `svd_aggregator.py`, `clients_avg_aggregator.py`, and `trainer_fedprox.py`."
    )
    lines.append(
        "- FLOP counts below are analytical proxies for multiply-add style work. They do not include Python scheduling overhead, queue overhead, or GPU kernel launch overhead."
    )
    lines.append("")

    lines.append("## Shared Constants")
    lines.append("")
    shared = model_stats_by_labels[2]
    lines.append(
        f"- Total model params (binary-task classifier): `{fmt_count(shared['all_params'])}`")
    lines.append(
        f"- LoRA-A params: `{fmt_count(shared['lora_a_params'])}` across `{shared['num_lora_modules']}` matrices of shape `({shared['rank']}, {shared['hidden_dim']})`"
    )
    lines.append(
        f"- LoRA-B params: `{fmt_count(shared['lora_b_params'])}` across `{shared['num_lora_modules']}` matrices of shape `({shared['hidden_dim']}, {shared['rank']})`"
    )
    lines.append(
        f"- Shared A+B payload: `{fmt_count(shared['lora_a_params'] + shared['lora_b_params'])}` params, about `{fmt_params_bytes(shared['lora_a_params'] + shared['lora_b_params'])}` at fp16"
    )
    lines.append(
        f"- Shared A-only or B-only payload: `{fmt_count(shared['lora_a_params'])}` params, about `{fmt_params_bytes(shared['lora_a_params'])}` at fp16"
    )
    lines.append(
        f"- Each client processes `{batch_size * local_steps}` examples and `{batch_size * local_steps * tok_len}` tokens per round"
    )
    lines.append("")

    lines.append("## Task Groups")
    lines.append("")
    lines.append(
        "| Group | Tasks | Num Labels | Classifier Params | Common Client Train FLOPs / Step | Common Client Train FLOPs / Round | Common Client Train FLOPs / 3 Rounds |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for group, tasks in GROUP_TO_TASKS.items():
        num_labels = TASK_TO_NUM_LABELS[tasks[0]]
        stats = model_stats_by_labels[num_labels]
        round_flops = stats["train_step_flops"] * local_steps
        total_flops = round_flops * total_rounds
        lines.append(
            f"| {group} | {', '.join(tasks)} | {num_labels} | {fmt_count(stats['classifier_params'])} | {fmt_flops(stats['train_step_flops'])} | {fmt_flops(round_flops)} | {fmt_flops(total_flops)} |"
        )
    lines.append("")

    lines.append("## Method Delta Formulas")
    lines.append("")
    lines.append("Notation:")
    lines.append(f"- `C = {client_num}` clients")
    lines.append(f"- `M = {shared['num_lora_modules']}` LoRA modules")
    lines.append(f"- `d = {shared['hidden_dim']}` hidden width")
    lines.append(f"- `r = {shared['rank']}` LoRA rank")
    lines.append(f"- `P_A = {fmt_count(shared['lora_a_params'])}`")
    lines.append(f"- `P_B = {fmt_count(shared['lora_b_params'])}`")
    lines.append("")
    lines.append("Exact proxy terms used in the tables:")
    lines.append("- Weighted averaging of `P` shared scalars across `C` clients: `(2C - 1)P`")
    lines.append(
        "- Client Gram upload transform in `svdtrainer.get_model_para()`: `M * [2rdr + 2rrd + 10r^3]`"
    )
    lines.append(
        "- Client shared-B reconstruction in `svdtrainer.update()`: `M * [2d^2r + 2d^2r + 4r^2d + (8/3)r^3]`"
    )
    lines.append(
        "- `svd` A-factor server path: `M * randomized_lowrank_svd(Cr, d, r, niter=3)`"
    )
    lines.append(
        "- `fedma` A-factor server path: `M * [(C-1)(r^2(3d-1) + r^3) + (2C-1)rd]`"
    )
    lines.append(
        "- `full_rank` server path: `M * [C * 2d^2r + (2C-1)d^2 + randomized_lowrank_svd(d, d, r, niter=3) + 4dr^2]`"
    )
    lines.append(
        "- `fedprox` extra client term per round: `12 * P_all` because the proximal regularizer scans the full model every batch for `4` local steps"
    )
    lines.append("")

    lines.append("## Method / Scenario Deltas")
    lines.append("")
    lines.append(
        "| Method | Scenario | Valid | Shared Params / Round | Shared Payload @ fp16 | Client Extra FLOPs / Round | Client Extra FLOPs / 3 Rounds | Server Aggregation FLOPs / Round | Server Aggregation FLOPs / 3 Rounds | Notes |"
    )
    lines.append(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in method_rows:
        if not row["valid"]:
            lines.append(
                f"| {row['method']} | {row['scenario']} | no | N/A | N/A | N/A | N/A | N/A | N/A | {row['note']} |"
            )
            continue
        lines.append(
            f"| {row['method']} | {row['scenario']} | yes | {fmt_count(row['shared_params'])} | {fmt_params_bytes(row['shared_params'])} | {fmt_flops(row['client_extra_round'])} | {fmt_flops(row['client_extra_round'] * total_rounds)} | {fmt_flops(row['server_round'])} | {fmt_flops(row['server_round'] * total_rounds)} | {row['note']} |"
        )
    lines.append("")

    lines.append("## Combined Client Totals By Task Group")
    lines.append("")
    lines.append(
        "| Group | Method | Scenario | Valid | Local Trainable Params | Common Client FLOPs / Round | Client Extra FLOPs / Round | Client Total FLOPs / Round | Client Total FLOPs / 3 Rounds | Server Aggregation FLOPs / Round |"
    )
    lines.append(
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group, tasks in GROUP_TO_TASKS.items():
        num_labels = TASK_TO_NUM_LABELS[tasks[0]]
        stats = model_stats_by_labels[num_labels]
        common_round = stats["train_step_flops"] * local_steps
        for row in method_rows:
            if not row["valid"]:
                lines.append(
                    f"| {group} | {row['method']} | {row['scenario']} | no | N/A | {fmt_flops(common_round)} | N/A | N/A | N/A | N/A |"
                )
                continue
            trainable_params = stats[
                "trainable_params_default"] if row["trainable_kind"] == "default" else stats[
                    "trainable_params_fa_lora"]
            total_round = common_round + row["client_extra_round"]
            lines.append(
                f"| {group} | {row['method']} | {row['scenario']} | yes | {fmt_count(trainable_params)} | {fmt_flops(common_round)} | {fmt_flops(row['client_extra_round'])} | {fmt_flops(total_round)} | {fmt_flops(total_round * total_rounds)} | {fmt_flops(row['server_round'])} |"
            )
    lines.append("")

    lines.append("## Main Takeaways")
    lines.append("")
    lines.append(
        "- Client rounds are dominated by the same RoBERTa-large local training term: about `15.001T` FLOPs per round for the binary GLUE tasks, `15.001T` for MNLI, and `15.001T` for STS-B."
    )
    lines.append(
        "- `factor_fedavg`, `fa_lora`, `svd`, `svd_no_gram`, `fedma`, and `full_rank` are therefore very close on client-side arithmetic unless they add an explicit extra path. The largest shared-B client extra in this codebase is the `svdtrainer.update()` reconstruction, about `1.623G` FLOPs per round."
    )
    lines.append(
        "- `fedprox` is special: its extra arithmetic term is only `4.286G` FLOPs per round, but it also scans the full `357M`-parameter model every batch. That memory traffic is likely a bigger wall-clock driver than the raw extra FLOPs."
    )
    lines.append(
        "- Server-side separation is much larger than client-side separation. In the current code, the expected aggregation ordering is roughly: `factor_fedavg / factor_fedprox / fa_lora` < `fedma (personalized_B)` < `svd / svd_no_gram (personalized_B)` < `fedma (shared_B)` ≈ `svd / svd_no_gram (shared_B)` < `full_rank (shared_B)`."
    )
    lines.append(
        "- Personalized-B scenarios halve the shared payload from `786,432` to `393,216` params for all valid methods that still transmit A, and they remove the shared-B receive-time `B` reconstruction cost from the client."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    common_cfg = load_common_cfg()
    model_stats_by_labels = {
        num_labels: inspect_roberta_glue(num_labels)
        for num_labels in sorted(set(TASK_TO_NUM_LABELS.values()))
    }
    method_rows = build_method_rows(common_cfg, model_stats_by_labels)
    markdown = render_markdown(common_cfg, model_stats_by_labels, method_rows)
    OUTPUT_PATH.write_text(markdown)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
