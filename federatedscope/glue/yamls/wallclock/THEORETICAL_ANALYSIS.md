# Theoretical Computational Cost Analysis (from code)

**Model**: RoBERTa-large, d=1024 hidden dim, r=8 LoRA rank, M=48 LoRA modules (query+value across 24 layers).
**FL**: N clients, tau local SGD steps per round, batch size b, sequence length s.

All costs below are per communication round, across all M adapted layers.

---

## 1. Common Client Training Cost

Every method shares the same local training loop (RoBERTa-large forward + backward):

```
FLOPs_train = tau * b * (6 * P_model * s)
```

Where `P_model = 355M` (RoBERTa-large parameters), factor 6 accounts for forward (2x) + backward (4x) multiply-adds.

| tau | Batch | Seq | Training FLOPs/round | Note |
|---:|---:|---:|---:|:---|
| 4 | 16 | 128 | 14.0 T | Wall-clock benchmark |
| 40 | 16 | 128 | 140.0 T | Paper's main setup |
| 640 | 16 | 128 | 2,240 T | Paper's max local steps |

---

## 2. PERSIA Server (Global FL, `svd_aggregator._para_weighted_avg`)

### Step 1: SVD of stacked right factors

Code (`svd_aggregator.py:46-51`):
```python
A = torch.stack([A_i for each client], dim=0).reshape(-1, d)  # (Nr) x d
U, S, V = torch.svd_lowrank(A, q=r, niter=3)
A_avg = V.t()  # r x d
```

`torch.svd_lowrank` implements randomized SVD (Halko et al., 2011):
- Draw random Gaussian Omega: d x (r + p), where p ~ r (oversampling)
- Y = A @ Omega: (Nr x d)(d x 2r) -> Nr x 2r, cost O(Nr * d * 2r)
- Power iterations (niter=3): each refines Y via Y = A @ (A^T @ Y)
  - A^T @ Y: (d x Nr)(Nr x 2r) -> d x 2r, cost O(d * Nr * 2r)
  - A @ result: (Nr x d)(d x 2r) -> Nr x 2r, cost O(Nr * d * 2r)
  - Per iteration: O(4Ndr^2), three iterations: O(12Ndr^2)
- QR of Y: O(Nr * (2r)^2) = O(Nr^3) -- negligible
- Small SVD: O(Nr * r^2) -- negligible

**Total SVD per module**: O(2Ndr^2 + 12Ndr^2) = O(14Ndr^2)

With N=3, r=8, d=1024: 14 * 3 * 1024 * 64 = **2,752,512 FLOPs**
Across M=48 modules: **132.1M FLOPs**

### Step 2: B-factor reconstruction and averaging

Code (`svd_aggregator.py:60-68`):
```python
B_avg = mean([B_i @ A_i @ V for each client])
# Evaluated left-to-right: (B_i @ A_i) @ V
```

**As implemented** (left-to-right evaluation):
- B_i @ A_i: (d x r)(r x d) -> d x d, cost O(d^2 * r) per client
- result @ V: (d x d)(d x r) -> d x r, cost O(d^2 * r) per client
- Per client: O(2d^2r), total: O(2Nd^2r)

With N=3, r=8, d=1024: 2 * 3 * 1024^2 * 8 = **50,331,648 FLOPs** per module
Across M=48: **2.416G FLOPs**

**As described in paper** (Algorithm 1, lines 16-17: Q_i = A_i @ V, then B'_i = B_i @ Q_i):
- Q_i = A_i @ V: (r x d)(d x r) -> r x r, cost O(dr^2) per client
- B'_i = B_i @ Q_i: (d x r)(r x r) -> d x r, cost O(dr^2) per client
- Per client: O(2dr^2), total: O(2Ndr^2)

With N=3, r=8, d=1024: 2 * 3 * 1024 * 64 = **393,216 FLOPs** per module
Across M=48: **18.9M FLOPs**

### PERSIA Global FL Server Total

| Component | As Implemented | As in Paper (Algorithm 1) |
|:---|---:|---:|
| SVD (M=48 modules) | 132.1M | 132.1M |
| B reconstruction (M=48) | 2,416M | 18.9M |
| Averaging B' (M=48) | 1.2M | 1.2M |
| **Total** | **~2.55G** | **~152M** |

The implementation uses naive matrix ordering that creates a d x d intermediate. The paper's claimed O(Ndr^2) complexity is achieved by computing Q_i = A_i @ V_r first (a one-line reordering). The difference is 128x at d=1024, r=8.

**Note**: Even the unoptimized implementation takes only ~55ms on GPU, because 2.55G FLOPs at modern GPU throughput (~10 TFLOPS for small ops) is ~0.25ms of pure compute — the rest is kernel launch and memory overhead.

---

## 3. PERSIA Client (Personalized FL, `svd_trainer.py`)

### Step 3a: Gram whitening in `get_model_para()` (lines 114-121)

Only active when `personalized_b=True` AND `use_gram=True`:

```python
G = B.T @ B           # (r x d)(d x r) -> r x r, cost O(dr^2)
G2 = sqrt(G + eps*I)  # eigendecomposition of r x r, cost O(r^3)
Z = G2 @ A            # (r x r)(r x d) -> r x d, cost O(r^2 * d)
```

Per module: O(dr^2 + r^3 + r^2d) = O(2dr^2 + r^3)
With r=8, d=1024: 2*1024*64 + 512 = **131,584 FLOPs**
Across M=48: **6.3M FLOPs**

### Step 3b: B reconstruction in `update()` (lines 60-63)

Only active when `personalized_b=True`:

```python
pinv_A = solve(A @ A^T + eps*I, A).T   # O(r^2*d + r^3) per module
new_B = old_B @ old_A @ pinv_A          # See below
```

**As implemented** (left-to-right):
- old_B @ old_A: (d x r)(r x d) -> d x d, cost O(d^2r)
- result @ pinv_A: (d x d)(d x r) -> d x r, cost O(d^2r)
- Total: O(2d^2r) per module

**As in paper** (right-to-left via Q_i = A_i @ V_r):
- Q_i = A_i @ V_r: (r x d)(d x r) -> r x r, cost O(dr^2)
- B'_i = B_i @ Q_i: (d x r)(r x r) -> d x r, cost O(dr^2)
- Total: O(2dr^2) per module

| | As Implemented | As in Paper |
|:---|---:|---:|
| Per module | 2 * 1024^2 * 8 = 16.8M | 2 * 1024 * 64 = 131K |
| M=48 modules | 805M | 6.3M |

### PERSIA PFL Client Extra Total

| Component | As Implemented | As in Paper |
|:---|---:|---:|
| Gram whitening (M=48) | 6.3M | 6.3M |
| B reconstruction (M=48) | 805M | 6.3M |
| **Total client extra** | **~811M** | **~12.6M** |

---

## 4. FedAvg Server (`clients_avg_aggregator._para_weighted_avg`)

Simple weighted average of all LoRA parameters:

```python
avg_A = sum(w_i * A_i)   # N weighted sums of r x d matrices
avg_B = sum(w_i * B_i)   # N weighted sums of d x r matrices
```

Per module: O(N * r * d) + O(N * d * r) = O(2Ndr)
With N=3, r=8, d=1024: 2 * 3 * 1024 * 8 = **49,152 FLOPs**
Across M=48: **2.36M FLOPs**

---

## 5. FedProx Client Extra

FedProx adds a proximal term `(mu/2)||theta - theta_global||^2` to the loss. This requires:
- One subtraction of the full parameter vector per batch: O(P_total) = O(357M)
- One norm computation per batch: O(P_total)
- Across tau * b batches (actually tau steps): O(tau * P_total)

With tau=4: 4 * 357M = **1.43G FLOPs** extra per round (but mainly memory-traffic bound)

Server: identical to FedAvg = **2.36M FLOPs**

---

## 6. Full-Rank Server (`svd_aggregator._full_rank_avg`)

```python
w_avg = mean(B_i @ A_i)                    # N dense products d x d
U, S, V = torch.svd_lowrank(w_avg, q=r, niter=3)  # SVD of d x d
A_new = sqrt(S) @ V^T                      # r x d
B_new = U @ sqrt(S)                         # d x r
```

| Step | FLOPs per module | Note |
|:---|---:|:---|
| N x (B_i @ A_i): d x d products | N * 2dr * d = O(Nd^2r) | Forms dense d x d matrix |
| SVD of d x d matrix | O(d^2 * r * niter) | Randomized, but on d x d input |
| Reconstruct factors | O(dr) | Negligible |

With N=3, r=8, d=1024, niter=3:
- Dense products: 3 * 2 * 1024^2 * 8 = 50.3M per module
- SVD: ~6 * 1024^2 * 8 * 3 = 150.9M per module (estimate)
- Total: ~201M per module
- Across M=48: **~9.66G FLOPs**

**Memory**: requires O(d^2) = 1024^2 * 2 bytes = 2 MB per module for the dense matrix. Across 48 modules: ~96 MB.

---

## 7. FedMA Server (`svd_aggregator._para_aligned_avg`)

```python
# For each module, align LoRA-A rows via Hungarian algorithm:
cost = ||a_i^row - a_ref^row||^2    # r x r cost matrix, O(r^2 * d)
perm = linear_sum_assignment(cost)  # O(r^3)
# Then B reconstruction same as PERSIA
```

Per module:
- Alignment: (N-1) * [O(r^2d) + O(r^3)] = O(Nr^2d)
- B reconstruction (code): O(2Nd^2r) — same left-to-right issue as PERSIA
- B reconstruction (paper): O(2Ndr^2)

| | As Implemented | Optimal |
|:---|---:|---:|
| Alignment (M=48) | 48 * 2 * 64 * 1024 = 6.3M | 6.3M |
| B reconstruction (M=48) | 2.42G | 18.9M |
| **Total** | **~2.42G** | **~25.2M** |

---

## 8. FFA-LoRA Server

Only LoRA-B is trainable and shared (LoRA-A is frozen). Aggregation is simple averaging of B:

Per module: O(N * d * r) = O(Ndr)
With N=3, r=8, d=1024: 3 * 1024 * 8 = 24,576
Across M=48: **1.18M FLOPs**

---

## 9. Summary Table

### Server aggregation cost (M=48 modules, N=3 clients, d=1024, r=8)

| Method | FLOPs (Paper's Algorithm) | FLOPs (Current Code) | Complexity Class |
|:---|---:|---:|:---|
| FedAvg | 2.36M | 2.36M | O(NMdr) |
| FFA-LoRA | 1.18M | 1.18M | O(NMdr) |
| FedProx | 2.36M | 2.36M | O(NMdr) |
| FedMA | 25.2M | 2.42G | O(NMdr^2) / O(NMd^2r) |
| **PERSIA** | **152M** | **2.55G** | **O(NMdr^2)** / O(NMd^2r) |
| Full-Rank | 9.66G | 9.66G | O(NMd^2r) |

### Client extra cost (M=48 modules, per client per round)

| Method | Extra FLOPs (Paper) | Extra FLOPs (Code) | Complexity Class |
|:---|---:|---:|:---|
| FedAvg | 0 | 0 | 0 |
| FFA-LoRA | 0 | 0 | 0 |
| FedProx | 1.43G | 1.43G | O(tau * P_model) |
| FedMA | 0 | 0 | 0 |
| PERSIA (Global FL) | 0 | 0 | 0 |
| PERSIA (PFL, whitening) | 12.6M | 811M | O(Mdr^2) / O(Md^2r) |
| Full-Rank | 0 | 0 | 0 |

### Communication per client per round

| Method | Upload | Download |
|:---|:---|:---|
| FedAvg (shared_b) | A + B = 2 * r * d = 1.50 MiB | Same |
| FedAvg (personalized_b) | A only = r * d = 0.75 MiB | Same |
| FFA-LoRA | B only = d * r = 0.75 MiB | Same |
| PERSIA (Global FL) | A + B = 1.50 MiB | A + B = 1.50 MiB |
| PERSIA (PFL) | Z (= r * d) = 0.75 MiB | V (= d * r) = 0.75 MiB |
| Full-Rank | A + B = 1.50 MiB | A + B = 1.50 MiB |

### Overhead relative to training (14T FLOPs at tau=4, or 140T at tau=40)

| Method | Server FLOPs (Paper) | Server / Train (tau=4) | Server / Train (tau=40) |
|:---|---:|---:|---:|
| FedAvg | 2.36M | 0.000017% | 0.0000017% |
| PERSIA | 152M | 0.0011% | 0.00011% |
| Full-Rank | 9.66G | 0.069% | 0.0069% |

**All server aggregation costs are negligible compared to training.** Even Full-Rank at 9.66G is only 0.069% of a single client's training round at tau=4.

---

## 10. Scaling Laws

### With model width d (fixing r=8, N=10, M proportional to d)

| Model | d | M (approx) | PERSIA Server | Full-Rank Server | Ratio |
|:---|---:|---:|---:|---:|---:|
| RoBERTa-base | 768 | 24 | O(24 * 10 * 768 * 64) = 11.8M | O(24 * 10 * 768^2 * 8) = 1.13G | 96x |
| RoBERTa-large | 1024 | 48 | O(48 * 10 * 1024 * 64) = 31.5M | O(48 * 10 * 1024^2 * 8) = 4.03G | 128x |
| LLaMA-7B | 4096 | 64 | O(64 * 10 * 4096 * 64) = 167.8M | O(64 * 10 * 4096^2 * 8) = 85.9G | 512x |
| LLaMA-13B | 5120 | 80 | O(80 * 10 * 5120 * 64) = 262.1M | O(80 * 10 * 5120^2 * 8) = 167.8G | 640x |
| LLaMA-70B | 8192 | 160 | O(160 * 10 * 8192 * 64) = 838.9M | O(160 * 10 * 8192^2 * 8) = 858.9G | 1024x |

**Full-Rank memory** at d=8192: 8192^2 * 2 bytes = 128 MB per layer, 160 layers = **20.5 GB** just for aggregation buffers.
**PERSIA memory** at d=8192: 10 * 8192 * 8 * 2 bytes = 1.3 MB per layer, 160 layers = **208 MB**.

### With number of clients N (fixing d=1024, r=8, M=48)

All methods scale linearly in N. PERSIA server overhead per additional client:
- Paper: 48 * (14 * 1024 * 64 + 2 * 1024 * 64) = 48 * 1,048,576 = ~50.3M FLOPs
- FedAvg: 48 * 2 * 1024 * 8 = ~786K FLOPs
- Ratio: ~64x (= r^2/1 = 64), constant regardless of N

### With LoRA rank r (fixing d=1024, N=10, M=48)

| r | PERSIA Server | FedAvg Server | PERSIA/FedAvg | Full-Rank Server | PERSIA/Full-Rank |
|---:|---:|---:|---:|---:|---:|
| 4 | 37.7M | 3.93M | 9.6x | 4.03G | 107x cheaper |
| 8 | 152M | 7.86M | 19.3x | 4.03G | 26.5x cheaper |
| 16 | 608M | 15.7M | 38.7x | 4.03G | 6.6x cheaper |
| 32 | 2.43G | 31.5M | 77.3x | 4.03G | 1.7x cheaper |
| 64 | 9.73G | 62.9M | 154.7x | 4.03G | 0.4x (more expensive) |

PERSIA's O(r^2) scaling means it becomes more expensive than Full-Rank when r > sqrt(d) = 32 at d=1024. In practice, LoRA ranks are typically 4-16, well within PERSIA's efficient regime.

---

## 11. Aggregation vs Training: Relative Significance

The most important context for aggregation cost is how it compares to training.

### Measured wall-clock ratios (from experiments)

| Scale (tau) | Client Train (s) | PERSIA Server (s) | Server / Round (%) | Full 200-round Server Total |
|---:|---:|---:|---:|:---|
| 4 (benchmark) | 0.635 | 0.055 | **8.0%** | 11.0s out of ~4 min |
| 40 (paper main) | 6.35 | 0.055 | **0.86%** | 11.0s out of ~40 min |
| 640 (paper max) | 101.5 | 0.055 | **0.054%** | 11.0s out of ~640 min |

**Over the full 200-round main experiment (~200 minutes), PERSIA's total server aggregation time is 11 seconds = 0.09% of total wall-clock.**

For comparison, even if we used FedAvg (5.2 seconds total), the savings would be 5.8 seconds over a 200-minute experiment — less than the time to drink a sip of coffee.

### FLOP ratios

| Method | Server FLOPs (all M layers) | Client Train FLOPs (tau=40) | Server / Train |
|:---|---:|---:|---:|
| FedAvg | 2.36M | 140T | 1.7 x 10^-8 |
| PERSIA | 152M | 140T | 1.1 x 10^-6 |
| Full-Rank | 9.66G | 140T | 6.9 x 10^-5 |

Even Full-Rank aggregation is **0.000069%** of a single client's training round. The aggregation is fundamentally a rounding error compared to the forward+backward passes through a 355M-parameter transformer.

---

## 12. Layer Parallelism: Does It Change Anything?

**Short answer: No, it does not change any conclusion.**

### Current implementation: sequential over M=48 modules

All methods (FedAvg, FedMA, PERSIA, Full-Rank) iterate over LoRA modules in a Python for-loop:

```python
for key in avg_model:
    if 'lora_A' in key:
        # process one module
```

Each module is independent — there are no cross-layer dependencies in any aggregation method. So all M=48 modules *could* be parallelized.

### What parallelism would save

| Method | Sequential (ms) | Per-module (ms) | Parallel (ms) | Savings |
|:---|---:|---:|---:|:---|
| FedAvg | 26 | 0.54 | ~0.54 | 25 ms |
| FedMA | 33 | 0.69 | ~0.69 | 32 ms |
| PERSIA | 55 | 1.15 | ~1.15 | 54 ms |
| Full-Rank | 59 | 1.23 | ~1.23 | 58 ms |

### Why it doesn't matter

1. **All methods benefit equally.** Parallelism divides all methods' times by M=48. The *relative* overhead between methods stays exactly the same. PERSIA goes from 55ms to 1.15ms; FedAvg goes from 26ms to 0.54ms. The difference drops from 29ms to 0.6ms, but it was already negligible.

2. **The sequential time is already negligible.** At the paper's main setting (tau=40), PERSIA's 55ms server time is 0.86% of the 6.4s round. Parallelizing it to 1.15ms would save 54ms — reducing the round from 6.4s to 6.346s, a 0.84% improvement. This is well within measurement noise.

3. **The bottleneck is always client training.** Even with perfect server parallelism (0ms aggregation), the round time would go from 6.4s to 6.35s — because the client's local SGD dominates at >99%.

4. **Complexity class is unchanged.** The paper reports per-layer complexity O(Ndr^2). With M layers, the total is O(MNdr^2) sequential or O(Ndr^2) parallel. Either way, the scaling with respect to d, N, and r is the same, and the absolute time is sub-second.

### Bottom line

Layer parallelism is an orthogonal implementation optimization that applies equally to all methods. It would reduce PERSIA's already-negligible 55ms to ~1ms. Since aggregation is <1% of the round time at the paper's operating scale regardless, parallelism does not change any scalability conclusion or relative comparison.
