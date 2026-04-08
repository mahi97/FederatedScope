# Rebuttal: Wall-Clock Overhead and Scalability

*Copy-paste ready response to reviewer concerns about computational cost and scalability.*

---

We thank the reviewer for this important question. We conducted wall-clock experiments to quantify PERSIA's overhead and validate the complexity claims in Table 3. We summarize the findings below.

**Experimental setup.** We measured per-round wall-clock time for all methods on RoBERTa-large ($d\!=\!1024$, $r\!=\!8$) across 7 GLUE tasks using 3 clients, batch size 16, and sequence length 128. We report client round time (local training + parameter handling) and server round time (aggregation), averaged across tasks.

**Table R1. Wall-clock time per round (seconds, averaged over 7 GLUE tasks).**

| Method | Scenario | Client (s) | Server (s) | Total (s) |
|:---|:---|---:|---:|---:|
| FedAvg | Global FL | 0.328 | 0.032 | 0.359 |
| FedAvg | PFL | 0.324 | 0.024 | 0.348 |
| FedProx | Global FL | 0.430 | 0.033 | 0.463 |
| FFA-LoRA | Global FL | 0.316 | 0.022 | 0.338 |
| FedMA | Global FL | 0.323 | 0.038 | 0.362 |
| **PERSIA** | **Global FL** | **0.325** | **0.060** | **0.385** |
| **PERSIA** | **PFL** | **0.702** | **0.055** | **0.757** |
| Full-Rank | Global FL | 0.329 | 0.076 | 0.406 |

**Key observations:**

**(1) In Global FL, PERSIA has zero client overhead.** All methods share identical client training time ($\sim$0.325s), because the local training loop is the same for every method. The only difference is server-side aggregation cost. PERSIA adds **29 ms** per round over FedAvg, resulting in an end-to-end overhead of **+0.7\%**. For comparison, Full-Rank adds 45 ms (+2.3\%), and FedProx—despite using the standard trainer—incurs +31\% E2E overhead from its proximal term.

**(2) In PFL, the overhead comes from client-side Gram whitening and $B$-reconstruction.** PERSIA's PFL mode activates two additional client operations per round (Algorithm 2, lines 7–8 and 15–16): computing $Z_i = (B_i^\top B_i)^{1/2}A_i$ before upload, and reconstructing $B_i^* = B_i(A_i V_r)$ after receiving $V_r$. These add $\sim$0.38s of client time. Critically, this cost is a **fixed constant** that does not scale with the number of local steps, batch size, or sequence length. At $\tau\!=\!40$ (our main setting), the overhead is $\sim$12\%; at $\tau\!=\!640$ (our maximum), it drops to $<$1\%.

**(3) Server aggregation is negligible relative to training.** Over the full 200-round experiment ($\sim$200 minutes), PERSIA's total server aggregation time is **12 seconds**—just **0.10\%** of the total wall-clock. Even Full-Rank, the most expensive aggregation, totals only 15 seconds. The real computational bottleneck is always local SGD through the 355M-parameter transformer, not aggregation.

**Table R2. Projected overhead at the paper's operating scales.**

| Setting | $\tau$ | Client Train (s) | PERSIA Extra (s) | Overhead (%) |
|:---|---:|---:|---:|---:|
| **Global FL** | 4 (benchmark) | 0.33 | 0.03 (server only) | 7.3 |
| | 40 (main results) | 3.25 | 0.03 | **0.9** |
| | 640 (max local) | 52.0 | 0.03 | **0.06** |
| **PFL** | 4 (benchmark) | 0.32 | 0.41 | 126 |
| | 40 (main results) | 3.24 | 0.41 | **12.6** |
| | 640 (max local) | 51.8 | 0.41 | **0.8** |

**(4) The server cost ordering matches the theoretical complexity exactly.** The measured server times follow: FFA-LoRA (22 ms) $<$ FedAvg (32 ms) $<$ FedMA (38 ms) $<$ PERSIA (60 ms) $<$ Full-Rank (76 ms), which is consistent with the complexity hierarchy $\mathcal{O}(NMr) < \mathcal{O}(NMdr) < \mathcal{O}(NMdr^2) < \mathcal{O}(NMd^2r)$ from Table 3.

**(5) PERSIA scales favorably to larger models.** Table 3 of the paper establishes PERSIA's server complexity at $\mathcal{O}(Ndr^2)$ per layer—*linear* in the model width $d$. Full-Rank aggregation requires $\mathcal{O}(Nd^2r + d^2r)$, *quadratic* in $d$.

**Table R3. Projected server cost for larger models ($N\!=\!10$, $r\!=\!8$).**

| Model | $d$ | PERSIA Server | Full-Rank Server | Full-Rank Mem/Layer |
|:---|---:|---:|---:|---:|
| RoBERTa-large | 1,024 | 152M FLOPs | 9.7G FLOPs | 2 MB |
| LLaMA-7B | 4,096 | 168M | 85.9G | 32 MB |
| LLaMA-13B | 5,120 | 262M | 167.8G | 50 MB |
| LLaMA-70B | 8,192 | 839M | 858.9G | 128 MB |

For LLaMA-70B, PERSIA is **1024$\times$ cheaper** in server FLOPs and requires **160$\times$ less memory** per layer (1.3 MB vs 128 MB). Full-Rank's $\mathcal{O}(d^2)$ memory makes it impractical for large models ($>$20 GB of aggregation buffers for 160 layers), while PERSIA stays under 210 MB total.

**(6) Layer parallelism does not change any conclusion.** The aggregation of each LoRA layer is independent and can be parallelized. However, this benefits all methods equally, so relative comparisons are unchanged. Moreover, even sequential aggregation over all 48 layers takes only 60 ms for PERSIA—already $<$1\% of the round at $\tau\!=\!40$.

**(7) Communication cost is identical.** PERSIA communicates the same payload as FedAvg: $\mathcal{O}(dr)$ parameters per client per round (0.75 MiB in PFL, 1.50 MiB in Global FL). The aggregation improvement comes at zero additional communication cost.

**(8) Effective wall-clock is actually lower.** A key result of our paper (Fig. 5) is that PERSIA enables $16\times$ longer local training intervals without collapse. When comparing methods at equal total computation:
- FedAvg at 40 local steps $\times$ 80 rounds $=$ 3{,}200 total batches $\rightarrow$ collapses at longer intervals
- PERSIA at 640 local steps $\times$ 5 rounds $=$ 3{,}200 total batches $\rightarrow$ stable performance

PERSIA requires $16\times$ fewer communication rounds, each with $<$1\% aggregation overhead at this scale. The net wall-clock to reach target accuracy is therefore **lower** for PERSIA than for FedAvg.

**Summary.** In Global FL, PERSIA has **zero client overhead** and adds only **29 ms of server time** per round (+0.7\% E2E). In PFL, a fixed client cost of $\sim$0.38s from Gram whitening and $B$-reconstruction amortizes to 12.6\% at $\tau\!=\!40$ and $<$1\% at $\tau\!=\!640$. Over the full 200-round experiment, PERSIA's total extra cost is **12 seconds in Global FL** (0.10\%) and **87 seconds in PFL** (0.72\%). The server cost scales as $\mathcal{O}(Ndr^2)$—linear in $d$—giving 1000$\times$ savings over Full-Rank on LLaMA-70B. Combined with the $16\times$ reduction in communication rounds that PERSIA uniquely enables, the net computational efficiency is strictly superior.
