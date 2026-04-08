# GLUE Rebuttal Wall-Clock Analysis

This file is generated from `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/RESULTS.md` and `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/THEORETICAL_COSTS.md` by `scripts/build_glue_wallclock_rebuttal.py`.

## Scope

- Numerical summaries use the measured wall-clock values in `RESULTS.md`.
- Theoretical summaries use the analytical FLOP and payload model in `THEORETICAL_COSTS.md`.
- `STS-B` is excluded from every numerical summary because every run failed.
- Invalid method/scenario pairs remain visible as `N/A`.
- A stable-task subset is detected automatically from the factor-FedAvg baseline; in the current data this subset is `mrpc`, `qnli`, `qqp`, `rte`, `sst2`, `wnli`.

## Coverage

| Method | shared_b coverage | personalized_b coverage | Notes |
| --- | --- | --- | --- |
| factor_fedavg | 8 ok, 1 failed | 8 ok, 1 failed | STS-B failed for every method |
| factor_fedprox | 8 ok, 1 failed | 8 ok, 1 failed | STS-B failed for every method |
| fa_lora | 8 ok, 1 failed | 9 invalid | STS-B failed for every method; freeze_A baseline requires shared lora_B |
| svd | 8 ok, 1 failed | 8 ok, 1 failed | STS-B failed for every method |
| svd_no_gram | 8 ok, 1 failed | 8 ok, 1 failed | STS-B failed for every method |
| fedma | 8 ok, 1 failed | 8 ok, 1 failed | STS-B failed for every method |
| full_rank | 5 ok, 4 failed | 9 invalid | STS-B failed for every method; full-rank aggregation requires shared lora_B; shared_b also failed on rte, sst2, wnli |

## Numerical Results

### shared_b

| Method | OK tasks | Median client round (s) | Median server round (s) | Median client FL end (min) | Median client / factor_fedavg | Median server / factor_fedavg | Median FL end / factor_fedavg | Coverage note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factor_fedavg | 8 | 0.314 | 0.024 | 0.121 | 1.00x | 1.00x | 1.00x | 1 failed, 8/9 ok |
| factor_fedprox | 8 | 0.401 | 0.025 | 0.133 | 1.27x | 1.00x | 1.11x | 1 failed, 8/9 ok |
| fa_lora | 8 | 0.304 | 0.018 | 0.114 | 0.97x | 0.75x | 0.96x | 1 failed, 8/9 ok |
| svd | 8 | 0.640 | 0.054 | 0.194 | 1.99x | 2.23x | 1.62x | 1 failed, 8/9 ok |
| svd_no_gram | 8 | 0.617 | 0.054 | 0.191 | 1.92x | 2.21x | 1.48x | 1 failed, 8/9 ok |
| fedma | 8 | 0.627 | 0.033 | 0.189 | 1.94x | 1.33x | 1.49x | 1 failed, 8/9 ok |
| full_rank | 5 | 0.641 | 0.060 | 0.190 | 1.90x | 2.50x | 1.56x | 4 failed, 5/9 ok |

### personalized_b

| Method | OK tasks | Median client round (s) | Median server round (s) | Median client FL end (min) | Median client / factor_fedavg | Median server / factor_fedavg | Median FL end / factor_fedavg | Coverage note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factor_fedavg | 8 | 0.312 | 0.019 | 0.118 | 1.00x | 1.00x | 1.00x | 1 failed, 8/9 ok |
| factor_fedprox | 8 | 0.400 | 0.021 | 0.131 | 1.27x | 1.05x | 1.12x | 1 failed, 8/9 ok |
| fa_lora | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| svd | 8 | 0.645 | 0.048 | 0.200 | 2.02x | 2.45x | 1.65x | 1 failed, 8/9 ok |
| svd_no_gram | 8 | 0.633 | 0.048 | 0.193 | 1.94x | 2.47x | 1.61x | 1 failed, 8/9 ok |
| fedma | 8 | 0.615 | 0.026 | 0.188 | 1.93x | 1.37x | 1.58x | 1 failed, 8/9 ok |
| full_rank | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |

## Theoretical Results

### shared_b

| Method | Valid | Payload / round | Client extra FLOPs / round | Server agg FLOPs / round | Theoretical client / factor_fedavg | Theoretical server / factor_fedavg | Scalability note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factor_fedavg | yes | 1.50 MiB | 0 | 3.932M | 1.00x | 1.00x | Linear in C, FedAvg-style averaging of A+B |
| factor_fedprox | yes | 1.50 MiB | 4.286G | 3.932M | 1.00x | 1.00x | Linear in C, FedAvg-style averaging of A+B |
| fa_lora | yes | 0.75 MiB | 0 | 1.966M | 1.00x | 0.50x | Linear in C, LoRA-B-only sharing |
| svd | yes | 1.50 MiB | 0 | 4.997G | 1.00x | 1270.85x | Linear in C, SVD on A plus shared-B reconstruction |
| svd_no_gram | yes | 1.50 MiB | 0 | 4.997G | 1.00x | 1270.85x | Linear in C, same server path as svd |
| fedma | yes | 1.50 MiB | 0 | 4.855G | 1.00x | 1234.74x | Linear in C, row alignment on A plus shared-B reconstruction |
| full_rank | yes | 1.50 MiB | 0 | 9.135G | 1.00x | 2323.25x | Linear in C, largest server constant due full-rank factorization |

### personalized_b

| Method | Valid | Payload / round | Client extra FLOPs / round | Server agg FLOPs / round | Theoretical client / factor_fedavg | Theoretical server / factor_fedavg | Scalability note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factor_fedavg | yes | 0.75 MiB | 0 | 1.966M | 1.00x | 1.00x | Linear in C, FedAvg-style averaging of A-only |
| factor_fedprox | yes | 0.75 MiB | 4.286G | 1.966M | 1.00x | 1.00x | Linear in C, FedAvg-style averaging of A-only |
| fa_lora | no | N/A | N/A | N/A | N/A | N/A | Invalid: freeze_A leaves no shared trainable parameter once lora_B is also personalized. |
| svd | yes | 0.75 MiB | 12.829M | 163.643M | 1.00x | 83.24x | Linear in C, low-rank SVD on stacked A only |
| svd_no_gram | yes | 0.75 MiB | 0 | 163.643M | 1.00x | 83.24x | Linear in C, same server path as svd without Gram upload |
| fedma | yes | 0.75 MiB | 0 | 20.883M | 1.00x | 10.62x | Linear in C, row alignment on A |
| full_rank | no | N/A | N/A | N/A | N/A | N/A | Invalid: full-rank server aggregation needs both A and B to form BA. |

## Match / Mismatch Analysis

| Question | Theory says | Numbers show | Verdict | Interpretation |
| --- | --- | --- | --- | --- |
| Client cost for shared_b svd vs factor_fedavg | Nearly identical: both have 15.001T client train FLOPs/round and svd has 0 extra client FLOPs | Median client round is 2.03x on stable tasks | Mismatch | The wall-clock gap is implementation or runtime overhead, not arithmetic from the analytical model |
| Incremental cost of svd vs svd_no_gram | Almost identical in shared_b and personalized_b; client-side Gram logic only matters for personalized_b and is tiny relative to 15.001T local training | Stable-task shared_b svd / svd_no_gram client median = 1.04x | Match | This is the cleanest way to isolate the overhead specific to our method |
| Server ordering inside the SVD family | fedma < svd ≈ svd_no_gram < full_rank in shared_b; fedma < svd ≈ svd_no_gram in personalized_b | Stable-task medians follow the same ordering | Match | Theory is useful for ordering and asymptotics, but not for converting FLOPs into exact time factors |
| Magnitude of server slowdown for svd vs factor_fedavg | About 1270.85x more aggregation FLOPs in shared_b | Only about 2.25x in measured server round time | Compressed in practice | All server rounds are below 0.1 s, so fixed Python and orchestration costs dominate |
| FedProx overhead | Only 4.286G extra arithmetic/round, but repeated full-model scans should hurt memory traffic | Stable-task client median is 1.27x vs factor_fedavg | Match | FedProx behaves like a memory-bound slowdown, exactly as the theory note suggests |
| cola / mnli raw timings | Should look similar to the other tasks because local steps, batch size, and token length are fixed | They are clear outliers even for the baselines themselves | Mismatch | These rows should be shown transparently but excluded from the final rebuttal claim set |

## Main Reading

- The raw numbers do show that the current `svd` implementation is slower than `factor_fedavg` in wall-clock at `C=3`.
- The cleanest theory/measurement match is not `svd` vs `factor_fedavg`, but `svd` vs `svd_no_gram`: those two are nearly tied, which is exactly what the analytical model predicts.
- Theoretical server FLOP ratios are much larger than measured server time ratios because the absolute server times are tiny, so fixed runtime overhead compresses the wall-clock differences.
- `cola` and `mnli` are visibly unstable even for the baselines, so they should be shown transparently but not used as the main rebuttal evidence.
