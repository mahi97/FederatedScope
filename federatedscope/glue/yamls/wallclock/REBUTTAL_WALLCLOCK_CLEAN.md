# GLUE Rebuttal Wall-Clock Summary (Clean Version)

This version keeps only claims that are jointly supported by the measured timings and the theoretical cost model.

## Stable Task Set

- The raw sheet contains clear timing outliers on `cola` and `mnli` even for the baselines themselves.
- To avoid making a rebuttal claim from unstable rows, this version uses the automatically detected stable subset: `mrpc`, `qnli`, `qqp`, `rte`, `sst2`, `wnli`.
- `STS-B` is still excluded because every method failed there.

## Numerical Summary On Stable Tasks

### shared_b

| Method | Stable-task coverage | Median client round (s) | Median server round (s) | Median client FL end (min) | Client / factor_fedavg | Server / factor_fedavg | FL end / factor_fedavg | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factor_fedavg | 6/6 | 0.312 | 0.024 | 0.118 | 1.00x | 1.00x | 1.00x | Reference baseline |
| factor_fedprox | 6/6 | 0.398 | 0.025 | 0.132 | 1.27x | 1.00x | 1.12x | FedAvg server path, extra proximal pass on client |
| fa_lora | 6/6 | 0.304 | 0.018 | 0.113 | 0.97x | 0.75x | 0.96x | Only LoRA-B is shared |
| svd | 6/6 | 0.635 | 0.054 | 0.193 | 2.03x | 2.25x | 1.64x | Our method |
| svd_no_gram | 6/6 | 0.611 | 0.053 | 0.187 | 1.95x | 2.21x | 1.60x | Matched SVD-family ablation without Gram upload |
| fedma | 6/6 | 0.629 | 0.033 | 0.189 | 1.97x | 1.35x | 1.60x | Alignment-based SVD-family baseline |
| full_rank | 3/6 | 0.623 | 0.060 | 0.190 | 1.94x | 2.50x | 1.63x | Full-rank server reconstruction baseline |

### personalized_b

| Method | Stable-task coverage | Median client round (s) | Median server round (s) | Median client FL end (min) | Client / factor_fedavg | Server / factor_fedavg | FL end / factor_fedavg | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factor_fedavg | 6/6 | 0.311 | 0.019 | 0.117 | 1.00x | 1.00x | 1.00x | Reference baseline |
| factor_fedprox | 6/6 | 0.396 | 0.020 | 0.130 | 1.28x | 1.05x | 1.13x | FedAvg server path, extra proximal pass on client |
| fa_lora | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Only LoRA-B is shared |
| svd | 6/6 | 0.635 | 0.046 | 0.194 | 2.05x | 2.45x | 1.65x | Our method |
| svd_no_gram | 6/6 | 0.609 | 0.047 | 0.188 | 1.95x | 2.47x | 1.61x | Matched SVD-family ablation without Gram upload |
| fedma | 6/6 | 0.612 | 0.026 | 0.188 | 1.95x | 1.37x | 1.59x | Alignment-based SVD-family baseline |
| full_rank | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Full-rank server reconstruction baseline |

## Theory Summary

### shared_b

| Method | Valid | Payload / round | Client extra FLOPs / round | Server agg FLOPs / round | Theoretical client / factor_fedavg | Scalability note |
| --- | --- | --- | --- | --- | --- | --- |
| factor_fedavg | yes | 1.50 MiB | 0 | 3.932M | 1.00x | Linear in C, FedAvg-style averaging of A+B |
| factor_fedprox | yes | 1.50 MiB | 4.286G | 3.932M | 1.00x | Linear in C, FedAvg-style averaging of A+B |
| fa_lora | yes | 0.75 MiB | 0 | 1.966M | 1.00x | Linear in C, LoRA-B-only sharing |
| svd | yes | 1.50 MiB | 0 | 4.997G | 1.00x | Linear in C, SVD on A plus shared-B reconstruction |
| svd_no_gram | yes | 1.50 MiB | 0 | 4.997G | 1.00x | Linear in C, same server path as svd |
| fedma | yes | 1.50 MiB | 0 | 4.855G | 1.00x | Linear in C, row alignment on A plus shared-B reconstruction |
| full_rank | yes | 1.50 MiB | 0 | 9.135G | 1.00x | Linear in C, largest server constant due full-rank factorization |

### personalized_b

| Method | Valid | Payload / round | Client extra FLOPs / round | Server agg FLOPs / round | Theoretical client / factor_fedavg | Scalability note |
| --- | --- | --- | --- | --- | --- | --- |
| factor_fedavg | yes | 0.75 MiB | 0 | 1.966M | 1.00x | Linear in C, FedAvg-style averaging of A-only |
| factor_fedprox | yes | 0.75 MiB | 4.286G | 1.966M | 1.00x | Linear in C, FedAvg-style averaging of A-only |
| fa_lora | no | N/A | N/A | N/A | N/A | Invalid: freeze_A leaves no shared trainable parameter once lora_B is also personalized. |
| svd | yes | 0.75 MiB | 12.829M | 163.643M | 1.00x | Linear in C, low-rank SVD on stacked A only |
| svd_no_gram | yes | 0.75 MiB | 0 | 163.643M | 1.00x | Linear in C, same server path as svd without Gram upload |
| fedma | yes | 0.75 MiB | 0 | 20.883M | 1.00x | Linear in C, row alignment on A |
| full_rank | no | N/A | N/A | N/A | N/A | Invalid: full-rank server aggregation needs both A and B to form BA. |

## What We Can Defend Cleanly

| Scenario | svd / svd_no_gram client round | svd / svd_no_gram server round | svd / svd_no_gram client FL end | Supported claim |
| --- | --- | --- | --- | --- |
| shared_b | 1.02x | 1.02x | 1.03x | Our method and the no-Gram ablation are effectively tied |
| personalized_b | 1.03x | 0.98x | 1.02x | The personalized-B Gram path adds only a very small constant |

## Rebuttal-Ready Conclusions

- Against the plain factor-FedAvg baseline, the current `svd` implementation costs about 2.03x client-round time and 2.25x server-round time in `shared_b`, and about 2.04x client-round time and 2.45x server-round time in `personalized_b`. End-to-end client FL time is about 1.63x to 1.65x of factor-FedAvg.
- This does **not** imply worse asymptotic scalability. The analytical model still keeps `svd` linear in the number of clients and low-rank in communication payload: `1.50 MiB` per round in `shared_b` and `0.75 MiB` in `personalized_b`.
- The incremental overhead specific to **our method** is best isolated by comparing `svd` to `svd_no_gram`, not by comparing the whole SVD-family pipeline to `factor_fedavg`. On the stable tasks, `svd` and `svd_no_gram` are essentially tied, which matches the theory because the extra Gram transform is tiny compared with the `15.001T` client training FLOPs per round.
- Therefore the defensible claim is: **our method is scalable in the same asymptotic sense as the baseline, but the current implementation has a constant wall-clock premium relative to factor-FedAvg; the part unique to our method contributes only a very small additional overhead beyond the shared SVD-family implementation cost.**

## Suggested Rebuttal Text

In the rebuttal wall-clock benchmark at `3` clients, our `svd` method is slower than plain factor-FedAvg in absolute wall-clock, with about 2.03x client-round and 2.25x server-round time in `shared_b`, and about 2.04x client-round and 2.45x server-round time in `personalized_b` on the stable six-task subset. However, this should not be interpreted as a poor scaling property of the algorithm itself. The theoretical model shows that our method remains linear in the number of clients and keeps the same low-rank communication payload (`0.75` to `1.50 MiB` per round). More importantly, when we compare against the matched ablation `svd_no_gram`, which shares the same SVD pipeline but removes the method-specific Gram step, the measured overhead of our method is only about `1.02x` to `1.03x`, exactly consistent with the fact that the extra client arithmetic is negligible relative to the `15.001T` FLOPs already spent in local RoBERTa-large training. Thus, the main practical premium comes from the current SVD-family implementation path, not from the method-specific computation, and the method remains scalable with a moderate constant-factor overhead.
