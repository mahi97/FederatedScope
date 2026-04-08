"""
Script to analyze LoRA per-round values and compare aggregation techniques.

- Reads lora_values_per_round*.pkl files.
- Computes for each round and each LoRA layer:
    1. Full-rank W: 1/n sum(B_i A_i)
    2. Factor-wise W: (1/n sum(A_i)) (1/n sum(B_i))
    3. Full-rank SVD W: rank-8 SVD of full-rank W, reconstruct as BA
    4. Mahi technique (see below)
- Computes distance metrics between each method and full-rank.
- Plots per-layer and averaged errors vs round.

Requirements:
    - torch
    - numpy
    - matplotlib
    - scipy

Usage:
    python analyze_lora_aggregation.py
"""
import os
import glob
import pickle
import torch
import matplotlib.pyplot as plt

def safesqrt(mat, eps=1e-8):
    # Stable sqrt of symmetric positive semi-definite matrix using torch
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(torch.sqrt(eigvals + eps)) @ eigvecs.T

def l2_error(A, B):
    return torch.norm(A - B).item()

def cosine_sim(A, B):
    return (torch.sum(A * B) / (torch.norm(A) * torch.norm(B) + 1e-8)).item()

def frobenius_norm(A, B):
    return torch.norm(A - B, p='fro').item()

def analyze_lora_pkl_files(pkl_pattern):
    files = sorted(glob.glob(pkl_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pkl_pattern}")
    # Load all rounds from all files
    all_rounds = {}
    for f in files:
        with open(f, 'rb') as pf:
            d = pickle.load(pf)
            all_rounds.update(d)
    all_rounds = dict(sorted(all_rounds.items()))
    # Structure: all_rounds[round][layer] = (list_of_lora_A, list_of_lora_B)
    return all_rounds

def compute_aggregations_per_round(round_data, rank=8):
    # round_data: dict[layer] = (list_of_lora_A, list_of_lora_B)
    results = {}
    for layer, (A_list, B_list) in round_data.items():
        A = torch.stack([torch.tensor(a) for a in A_list])  # (n, r, d)
        B = torch.stack([torch.tensor(b) for b in B_list])  # (n, d, r)
        n_clients = A.shape[0]
        # 1. Full-rank W
        W_full = torch.mean(torch.stack([B[i] @ A[i] for i in range(n_clients)]), dim=0)
        # 2. Factor-wise W
        A_mean = torch.mean(A, dim=0)
        B_mean = torch.mean(B, dim=0)
        W_factor = B_mean @ A_mean
        # 3. Full-rank SVD W
        U, S, Vh = torch.linalg.svd(W_full, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
        B_svd = U_r @ torch.diag(torch.sqrt(S_r))
        A_svd = torch.diag(torch.sqrt(S_r)) @ Vh_r
        W_svd = B_svd @ A_svd
        # 4. Mahi technique
        G_list = [safesqrt(B[i].T @ B[i]) for i in range(n_clients)]
        Z_list = [G_list[i] @ A[i] for i in range(n_clients)]
        Z_stack = torch.cat(Z_list, dim=0)
        U_m, S_m, Vh_m = torch.linalg.svd(Z_stack, full_matrices=False)
        Vh_m_r = Vh_m[:rank, :]
        A_mahi = Vh_m_r
        B_mahi_list = [B[i] @ (A[i] @ Vh_m_r.T) for i in range(n_clients)]
        B_mahi = torch.mean(torch.stack(B_mahi_list), dim=0)
        W_mahi = B_mahi @ A_mahi
        results[layer] = {
            'full': W_full,
            'factor': W_factor,
            'svd': W_svd,
            'mahi': W_mahi
        }
    return results

def compute_errors_vectorized(all_agg_results):
    # all_agg_results: dict[round][layer][method] = W (torch.Tensor)
    # Returns: dict[round][layer][method][metric]
    rounds = list(all_agg_results.keys())
    layers = list(next(iter(all_agg_results.values())).keys())
    methods = ['factor', 'svd', 'mahi']
    errors_per_round = {}
    for r in rounds:
        errors_per_round[r] = {}
        for layer in layers:
            res = all_agg_results[r][layer]
            W_full = res['full']
            # Stack all method Ws for this layer/round
            Ws = torch.stack([res[m] for m in methods])  # (3, ...)
            W_full_rep = W_full.unsqueeze(0).expand_as(Ws)
            diff = Ws - W_full_rep
            l2 = torch.norm(diff.view(diff.shape[0], -1), dim=1)
            fro = torch.norm(diff, p='fro', dim=(1,2)) if diff.ndim == 3 else l2
            cos = torch.sum(Ws.view(Ws.shape[0], -1) * W_full.view(-1), dim=1) / (
                torch.norm(Ws.view(Ws.shape[0], -1), dim=1) * torch.norm(W_full.view(-1)) + 1e-8)
            errors_per_round[r][layer] = {
                m: {'l2': l2[i].item(), 'fro': fro[i].item(), 'cosine': cos[i].item()} for i, m in enumerate(methods)
            }
    return errors_per_round

def plot_errors(errors_per_round, save_dir=None):
    import matplotlib.pyplot as plt
    methods = ['factor', 'svd', 'mahi']
    layers = list(next(iter(errors_per_round.values())).keys())
    rounds = list(errors_per_round.keys())
    for metric in ['l2', 'fro', 'cosine']:
        plt.figure(figsize=(12, 6))
        for layer in layers:
            vals = {m: [errors_per_round[r][layer][m][metric] for r in rounds] for m in methods}
            for m in methods:
                plt.plot(rounds, vals[m], label=f'{layer}-{m}')
        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.title(f'{metric} error per layer/method')
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{metric}_per_layer.png'))
        plt.show()
    # Average over layers
    for metric in ['l2', 'fro', 'cosine']:
        plt.figure(figsize=(12, 6))
        for m in methods:
            vals = [torch.mean(torch.tensor([errors_per_round[r][layer][m][metric] for layer in layers])) for r in rounds]
            plt.plot(rounds, vals, label=m)
        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.title(f'{metric} error averaged over layers')
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{metric}_avg.png'))
        plt.show()

def main():
    pkl_pattern = os.path.join(os.path.dirname(__file__), 'lora_values_per_round*.pkl')
    all_rounds = analyze_lora_pkl_files(pkl_pattern)
    all_agg_results = {}
    for round_num, round_data in all_rounds.items():
        agg_results = compute_aggregations_per_round(round_data, rank=8)
        all_agg_results[round_num] = agg_results
    errors_per_round = compute_errors_vectorized(all_agg_results)
    plot_errors(errors_per_round)

if __name__ == '__main__':
    main()
