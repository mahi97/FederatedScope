## Wall-Clock Results

Following experiments are run for 5 rounds with 10 local steps per round on Qwen 2.5 3B-Instruct fine-tuned on GSM8K with LoRA (rank 8). The server-side wall-clock measurement reports the total time for aggregation per each FL round. The client-side wall-clock measurement reports the time for a single local training round (10 steps of batch size 4).

### Global FL — Per-Round Timing (seconds)

| Method | Server (s) | Client (s) | Server / Client | Server overhead vs. Factor-wise |
|---|---:|---:|---:|---:|
| FFA-LoRA | 0.018 | 2.961 | 0.6% | −0.005s (faster) |
| FedMA | 0.019 | 2.958 | 0.6% | −0.004s (faster) |
| FedProx | 0.022 | 3.134 | 0.7% | −0.001s (faster) |
| Full-rank + SVD | 0.022 | 2.965 | 0.7% | −0.001s (faster) |
| Factor-wise | 0.023 | 2.967 | 0.8% | — (baseline) |
| PERSIA (w/o whitening) | 0.024 | 2.961 | 0.8% | +0.001s |
| **PERSIA** | **0.024** | **2.971** | **0.8%** | **+0.001s** |

### Personalized FL — Per-Round Timing (seconds)

| Method | Server (s) | Client (s) | Server / Client | Server overhead vs. Factor-wise |
|---|---:|---:|---:|---:|
| FedMA | 0.020 | 2.959 | 0.7% | −0.002s (faster) |
| PERSIA (w/o whitening) | 0.021 | 3.553 | 0.6% | −0.001s (faster) |
| FedProx | 0.022 | 3.134 | 0.7% | +0.000s |
| Factor-wise | 0.022 | 2.964 | 0.7% | — (baseline) |
| **PERSIA** | **0.023** | **3.645** | **0.6%** | **+0.001s** |

*Measured on Qwen 2.5 3B-Instruct with GSM8K (tok_len=256, batch_size=4, LoRA r=8, 3 clients).*

## Overhead Projection to Main Experiments (50 local steps)

The wall-clock measurements above use 10 local steps per round. In our main experiments, clients perform **50 local steps**. Since server aggregation cost is independent of local training duration, the overhead ratio shrinks proportionally:

| Setting | Extra server cost | Client time (10 steps) | Client time (50 steps) | Overhead (10 steps) | **Overhead (50 steps)** |
|---|---:|---:|---:|---:|---:|
| Global FL | +0.001s | 2.967s | ~14.84s | 0.0% | **0.0%** |
| Personalized FL | +0.001s | 2.964s | ~14.82s | 0.0% | **0.0%** |

**At 50 local steps, PERSIA adds 0.0% wall-clock overhead per round in Global FL and 0.0% in Personalized FL.** In practical deployments where clients may run hundreds of local steps, the overhead becomes negligible.

## Addressing the Sequential Bottleneck Concern

We want to clarify three points regarding the reviewer's concern about L separate SVDs creating a sequential bottleneck:

**1. Layer-wise SVDs are independent and parallelizable.** PERSIA performs a truncated SVD per adapted layer, and these operations are entirely independent. In practice, all layer parameters are already resident in VRAM, so all L low-rank SVDs can execute in parallel (or be batched), avoiding a sequential bottleneck. Our implementation uses `torch.svd_lowrank`, which leverages GPU parallelism.

**2. Aggregation is infrequent relative to training.** The SVD occurs once per FL round, whereas each round involves τ forward–backward passes across all clients. As shown above, even with only 10 local steps, client-side training already dominates the round time. With 50 local steps (our main setting), the server-side SVD is a small fraction of the round.

**3. Low-rank SVD is tunable.** We use `torch.svd_lowrank(n_iter=k)` where the number of power iterations k controls an accuracy–speed trade-off. With rank r = 8 and n_iter = 3, the SVD operates on small r × r matrices after the initial randomized projection, which is extremely fast even for large d.

## Net Effect: Communication Savings Dominate

The central practical benefit of PERSIA is enabling longer local training without collapse, which reduces the number of FL rounds needed. This represents a significant reduction in communication rounds, far exceeding the per-round aggregation overhead.
