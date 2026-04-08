#!/usr/bin/env python3
"""Analyze LoRA pkl dumps and compare aggregation methods.

Reads lora_values_per_round*.pkl produced by ClientsAvgAggregator and computes
aggregated W for each layer/round with four methods:
1) full-rank: mean_i (B_i @ A_i)
2) factor-wise: mean_i(B_i) @ mean_i(A_i)
3) full-rank + SVD: rank-r SVD of full-rank W, then W = B A
4) mahi: G_i = sqrt(B_i^T B_i), Z_i = G_i A_i, SVD(stack(Z_i)) -> A = V^T,
   B_i' = B_i (A_i V), B = mean_i(B_i'), W = B A

Plots per-layer and average-across-layers distances to full-rank W.
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


METHODS = ["full", "factor", "svd", "mahi"]
METRICS = [
    "fro",        # Relative Frobenius norm of difference
    "rel_fro",    # Relative Frobenius error
    "mae",        # Mean absolute error
    "max_abs",    # Max absolute error
    "cos_sim",    # Cosine similarity of flattened matrices
    "cos_dist",   # 1 - cosine similarity
]


def safe_matrix_sqrt(G: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safely compute sqrt of a symmetric PSD matrix (from SVDTrainer)."""
    t = torch.from_numpy(G).double()
    t = 0.5 * (t + t.T)
    e, V = torch.linalg.eigh(t)
    e_pos = torch.clamp(e, min=0.0)
    return (V * torch.sqrt(e_pos + eps)).matmul(V.T).cpu().numpy()


def truncated_svd(W: np.ndarray, rank: int, niter: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rank-r truncated SVD using torch.svd_lowrank."""
    if W.size == 0:
        raise ValueError("Empty matrix passed to SVD")
    m, n = W.shape
    r = min(rank, m, n)
    if r < 1:
        raise ValueError(f"Invalid rank {rank} for shape {W.shape}")
    t = torch.from_numpy(W).float()
    U, S, V = torch.svd_lowrank(t, q=r, niter=niter)
    Vt = V.T
    return U.cpu().numpy(), S.cpu().numpy(), Vt.cpu().numpy()


def compute_W_full(A_list: List[np.ndarray], B_list: List[np.ndarray]) -> np.ndarray:
    n = min(len(A_list), len(B_list))
    if n == 0:
        raise ValueError("No client pairs to compute W_full")
    out_dim, r = B_list[0].shape
    r2, in_dim = A_list[0].shape
    if r != r2:
        raise ValueError(f"Incompatible shapes B {B_list[0].shape} and A {A_list[0].shape}")
    W = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(n):
        W += B_list[i] @ A_list[i]
    return W / float(n)


def compute_W_factor(A_list: List[np.ndarray], B_list: List[np.ndarray]) -> np.ndarray:
    n = min(len(A_list), len(B_list))
    if n == 0:
        raise ValueError("No client pairs to compute W_factor")
    A_mean = np.mean(np.stack(A_list[:n], axis=0), axis=0)
    B_mean = np.mean(np.stack(B_list[:n], axis=0), axis=0)
    return B_mean @ A_mean


def compute_W_svd(W_full: np.ndarray, rank: int, niter: int = 3) -> np.ndarray:
    U, S, Vt = truncated_svd(W_full, rank=rank, niter=niter)
    sqrtS = np.sqrt(S)
    B = U * sqrtS[None, :]
    A = sqrtS[:, None] * Vt
    return B @ A


def compute_W_mahi(
    A_list: List[np.ndarray],
    B_list: List[np.ndarray],
    rank: int,
    niter: int = 3,
    eps: float = 1e-12,
) -> np.ndarray:
    n = min(len(A_list), len(B_list))
    if n == 0:
        raise ValueError("No client pairs to compute W_mahi")

    # G_i and Z_i
    Zs = []
    for i in range(n):
        B = B_list[i]
        A = A_list[i]
        G = safe_matrix_sqrt(B.T @ B, eps=eps)
        Zs.append(G @ A)

    stacked_Z = np.vstack(Zs)
    U, S, Vt = truncated_svd(stacked_Z, rank=rank, niter=niter)
    V = Vt.T

    # Update B_i and average
    Bs = []
    for i in range(n):
        B = B_list[i]
        A = A_list[i]
        Bs.append(B @ (A @ V))

    B_mean = np.mean(np.stack(Bs, axis=0), axis=0)
    return B_mean @ Vt


def compute_metrics(W: np.ndarray, W_full: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    diff = W - W_full
    fro = float(np.linalg.norm(diff, ord="fro"))
    base = float(np.linalg.norm(W_full, ord="fro"))
    rel_fro = fro / (base + eps)
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    flat_w = W.reshape(-1)
    flat_f = W_full.reshape(-1)
    denom = (np.linalg.norm(flat_w) * np.linalg.norm(flat_f) + eps)
    cos_sim = float(np.dot(flat_w, flat_f) / denom)
    cos_dist = 1.0 - cos_sim
    return {
        "fro": fro,
        "rel_fro": rel_fro,
        "mae": mae,
        "max_abs": max_abs,
        "cos_sim": cos_sim,
        "cos_dist": cos_dist,
    }


def base_layer_key(a_key: str) -> str:
    return a_key.replace("lora_A", "lora")


def build_round_layer_map(round_dict: Dict[str, Tuple[List, List]]) -> Dict[str, Tuple[List, List, str, str]]:
    layer_map = {}
    for key in round_dict:
        if "lora_A" not in key:
            continue
        bkey = key.replace("lora_A", "lora_B")
        if bkey not in round_dict:
            continue
        A_list, _ = round_dict[key]
        _, B_list = round_dict[bkey]
        layer_map[base_layer_key(key)] = (A_list, B_list, key, bkey)
    return layer_map


def safe_filename(name: str, max_len: int = 160) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if len(safe) <= max_len:
        return safe
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return safe[: max_len - 9] + "_" + digest


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_metric_per_layer(
    rounds: List[int],
    metrics: Dict[str, Dict[str, Dict[str, List[float]]]],
    out_dir: str,
    include_full: bool = False,
) -> None:
    methods = METHODS if include_full else [m for m in METHODS if m != "full"]
    per_layer_dir = os.path.join(out_dir, "per_layer")
    ensure_dir(per_layer_dir)

    for metric_name, layer_dict in metrics.items():
        metric_dir = os.path.join(per_layer_dir, metric_name)
        ensure_dir(metric_dir)
        for layer, method_dict in layer_dict.items():
            plt.figure(figsize=(10, 4))
            for method in methods:
                vals = method_dict.get(method, [])
                if not vals:
                    continue
                plt.plot(rounds, vals, label=method)
            title = layer if len(layer) < 100 else (layer[:97] + "...")
            plt.title(f"{metric_name} vs round | {title}")
            plt.xlabel("round")
            plt.ylabel(metric_name)
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            fname = safe_filename(layer) + ".png"
            plt.tight_layout()
            plt.savefig(os.path.join(metric_dir, fname), dpi=150)
            plt.close()



def plot_metric_avg(
    rounds: List[int],
    metrics: Dict[str, Dict[str, Dict[str, List[float]]]],
    out_dir: str,
    include_full: bool = False,
) -> None:
    methods = METHODS if include_full else [m for m in METHODS if m != "full"]
    avg_dir = os.path.join(out_dir, "avg")
    ensure_dir(avg_dir)

    for metric_name, layer_dict in metrics.items():
        plt.figure(figsize=(10, 4))
        for method in methods:
            series = []
            for r_idx in range(len(rounds)):
                vals = []
                for layer in layer_dict:
                    v = layer_dict[layer][method][r_idx]
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    vals.append(v)
                series.append(float(np.mean(vals)) if vals else np.nan)
            plt.plot(rounds, series, label=method)
        plt.title(f"{metric_name} vs round | average across layers")
        plt.xlabel("round")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(avg_dir, f"avg_{metric_name}.png"), dpi=150)
        plt.close()


def save_metrics_json(rounds, metrics, out_dir) -> None:
    def to_serializable(val):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val

    serial = {
        "rounds": rounds,
        "methods": METHODS,
        "metrics": list(metrics.keys()),
        "layers": list(next(iter(metrics.values())).keys()) if metrics else [],
        "data": {},
    }
    for metric_name, layer_dict in metrics.items():
        serial["data"][metric_name] = {}
        for layer, method_dict in layer_dict.items():
            serial["data"][metric_name][layer] = {}
            for method, series in method_dict.items():
                serial["data"][metric_name][layer][method] = [to_serializable(v) for v in series]

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(serial, f, indent=2)


def analyze_pkl(
    pkl_path: str,
    out_dir: str,
    rank: int = 8,
    niter: int = 3,
    eps: float = 1e-12,
    include_full: bool = False,
) -> None:
    print(f"[INFO] Analyzing {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    rounds = sorted(data.keys())
    round_maps = {}
    layers = set()
    for r in rounds:
        layer_map = build_round_layer_map(data[r])
        round_maps[r] = layer_map
        layers.update(layer_map.keys())

    layers = sorted(layers)
    if not layers:
        raise ValueError(f"No LoRA layer pairs found in {pkl_path}")

    metrics = {
        metric: {
            layer: {method: [np.nan] * len(rounds) for method in METHODS}
            for layer in layers
        }
        for metric in METRICS
    }

    print(f"[INFO] {len(rounds)} rounds, {len(layers)} layers found.")
    for ridx, r in enumerate(tqdm(rounds, desc=f"Rounds ({os.path.basename(pkl_path)})")):
        layer_map = round_maps[r]
        for layer in tqdm(layers, desc=f"  Layers (round {r})", leave=False):
            if layer not in layer_map:
                continue
            A_list, B_list, _, _ = layer_map[layer]
            n = min(len(A_list), len(B_list))
            if n == 0:
                continue
            # Convert to float32 for compute
            A = [np.array(A_list[i], dtype=np.float32) for i in range(n)]
            B = [np.array(B_list[i], dtype=np.float32) for i in range(n)]

            try:
                W_full = compute_W_full(A, B)
                W_factor = compute_W_factor(A, B)
                W_svd = compute_W_svd(W_full, rank=rank, niter=niter)
                W_mahi = compute_W_mahi(A, B, rank=rank, niter=niter, eps=eps)
            except Exception as e:
                print(f"[WARN] round {r} layer {layer}: {e}")
                continue

            W_by_method = {
                "full": W_full,
                "factor": W_factor,
                "svd": W_svd,
                "mahi": W_mahi,
            }
            for method, W in W_by_method.items():
                mvals = compute_metrics(W, W_full, eps=eps)
                for metric_name, val in mvals.items():
                    metrics[metric_name][layer][method][ridx] = float(val)

    ensure_dir(out_dir)
    save_metrics_json(rounds, metrics, out_dir)
    plot_metric_per_layer(rounds, metrics, out_dir, include_full=include_full)
    plot_metric_avg(rounds, metrics, out_dir, include_full=include_full)

    print(f"[INFO] Saved metrics and plots to: {out_dir}\n")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LoRA aggregation methods from pkl dumps")
    parser.add_argument(
        "--pkl",
        action="append",
        help="Path to lora_values_per_round*.pkl (can be repeated)",
    )
    parser.add_argument(
        "--pkl-glob",
        help="Glob for pkl files, e.g. federatedscope/core/aggregators/lora_values_per_round*.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default="exp/lora_analysis",
        help="Output directory for metrics and plots",
    )
    parser.add_argument("--rank", type=int, default=8, help="Rank for truncated SVD")
    parser.add_argument("--niter", type=int, default=3, help="Iterations for svd_lowrank")
    parser.add_argument("--eps", type=float, default=1e-12, help="Epsilon for safe sqrt and metrics")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for svd_lowrank reproducibility")
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="Include full-rank baseline in plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    pkl_paths = []
    if args.pkl:
        pkl_paths.extend(args.pkl)
    if args.pkl_glob:
        pkl_paths.extend(glob.glob(args.pkl_glob))

    if not pkl_paths:
        raise SystemExit("No pkl files provided. Use --pkl or --pkl-glob.")

    for pkl_path in pkl_paths:
        if not os.path.isfile(pkl_path):
            print(f"[WARN] Skipping missing file: {pkl_path}")
            continue
        base = os.path.splitext(os.path.basename(pkl_path))[0]
        out_dir = os.path.join(args.out_dir, base)
        analyze_pkl(
            pkl_path=pkl_path,
            out_dir=out_dir,
            rank=args.rank,
            niter=args.niter,
            eps=args.eps,
            include_full=args.include_full,
        )


if __name__ == "__main__":
    main()
