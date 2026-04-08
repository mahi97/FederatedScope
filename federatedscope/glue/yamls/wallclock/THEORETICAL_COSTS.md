# GLUE Theoretical Cost Model

This file estimates the computation carried by each GLUE wall-clock baseline in the current rebuttal setup.

Scope:
- Config family: `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock`
- Backbone: `FacebookAI/roberta-large@huggingface_llm`, LoRA rank `8`, query/value adapters on all `24` layers
- FL setup: `client_num=3`, `local_update_steps=4`, `batch_size=16`, `tok_len=128`, `total_round_num=3`
- Common client training FLOPs are taken from the model class's own `floating_point_ops(...)` estimator for a `16 x 128` batch.
- Method-specific deltas are derived directly from the code paths in `svd_trainer.py`, `svd_aggregator.py`, `clients_avg_aggregator.py`, and `trainer_fedprox.py`.
- FLOP counts below are analytical proxies for multiply-add style work. They do not include Python scheduling overhead, queue overhead, or GPU kernel launch overhead.

## Shared Constants

- Total model params (binary-task classifier): `357,199,876`
- LoRA-A params: `393,216` across `48` matrices of shape `(8, 1024)`
- LoRA-B params: `393,216` across `48` matrices of shape `(1024, 8)`
- Shared A+B payload: `786,432` params, about `1.50 MiB` at fp16
- Shared A-only or B-only payload: `393,216` params, about `0.75 MiB` at fp16
- Each client processes `64` examples and `8192` tokens per round

## Task Groups

| Group | Tasks | Num Labels | Classifier Params | Common Client Train FLOPs / Step | Common Client Train FLOPs / Round | Common Client Train FLOPs / 3 Rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| binary | cola, mrpc, qnli, qqp, rte, sst2, wnli | 2 | 1,051,650 | 3.750T | 15.001T | 45.004T |
| three_class | mnli | 3 | 1,052,675 | 3.750T | 15.001T | 45.004T |
| regression | stsb | 1 | 1,050,625 | 3.750T | 15.001T | 45.003T |

## Method Delta Formulas

Notation:
- `C = 3` clients
- `M = 48` LoRA modules
- `d = 1024` hidden width
- `r = 8` LoRA rank
- `P_A = 393,216`
- `P_B = 393,216`

Exact proxy terms used in the tables:
- Weighted averaging of `P` shared scalars across `C` clients: `(2C - 1)P`
- Client Gram upload transform in `svdtrainer.get_model_para()`: `M * [2rdr + 2rrd + 10r^3]`
- Client shared-B reconstruction in `svdtrainer.update()`: `M * [2d^2r + 2d^2r + 4r^2d + (8/3)r^3]`
- `svd` A-factor server path: `M * randomized_lowrank_svd(Cr, d, r, niter=3)`
- `fedma` A-factor server path: `M * [(C-1)(r^2(3d-1) + r^3) + (2C-1)rd]`
- `full_rank` server path: `M * [C * 2d^2r + (2C-1)d^2 + randomized_lowrank_svd(d, d, r, niter=3) + 4dr^2]`
- `fedprox` extra client term per round: `12 * P_all` because the proximal regularizer scans the full model every batch for `4` local steps

## Method / Scenario Deltas

| Method | Scenario | Valid | Shared Params / Round | Shared Payload @ fp16 | Client Extra FLOPs / Round | Client Extra FLOPs / 3 Rounds | Server Aggregation FLOPs / Round | Server Aggregation FLOPs / 3 Rounds | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| factor_fedavg | shared_b | yes | 786,432 | 1.50 MiB | 0 | 0 | 3.932M | 11.796M | Plain factor-wise averaging on transmitted trainable tensors. |
| factor_fedavg | personalized_b | yes | 393,216 | 0.75 MiB | 0 | 0 | 1.966M | 5.898M | Plain factor-wise averaging on transmitted trainable tensors. |
| factor_fedprox | shared_b | yes | 786,432 | 1.50 MiB | 4.286G | 12.859G | 3.932M | 11.796M | Same server path as factor_fedavg, plus FedProx adds a full-model proximal scan every batch. |
| factor_fedprox | personalized_b | yes | 393,216 | 0.75 MiB | 4.286G | 12.859G | 1.966M | 5.898M | Same server path as factor_fedavg, plus FedProx adds a full-model proximal scan every batch. |
| fa_lora | shared_b | yes | 393,216 | 0.75 MiB | 0 | 0 | 1.966M | 5.898M | Only LoRA-B is trainable/shared; LoRA-A is frozen and excluded from upload. |
| fa_lora | personalized_b | no | N/A | N/A | N/A | N/A | N/A | N/A | Invalid: freeze_A leaves no shared trainable parameter once lora_B is also personalized. |
| svd | shared_b | yes | 786,432 | 1.50 MiB | 0 | 0 | 4.997G | 14.992G | Client-side SVD logic is active only for personalized-B svd; shared-B svd uses plain client upload/update. |
| svd | personalized_b | yes | 393,216 | 0.75 MiB | 12.829M | 38.486M | 163.643M | 490.930M | Client-side SVD logic is active only for personalized-B svd; shared-B svd uses plain client upload/update. |
| svd_no_gram | shared_b | yes | 786,432 | 1.50 MiB | 0 | 0 | 4.997G | 14.992G | Same server path as svd, but the client always uses plain upload/update. |
| svd_no_gram | personalized_b | yes | 393,216 | 0.75 MiB | 0 | 0 | 163.643M | 490.930M | Same server path as svd, but the client always uses plain upload/update. |
| fedma | shared_b | yes | 786,432 | 1.50 MiB | 0 | 0 | 4.855G | 14.564G | Server aligns LoRA-A rows first; clients use plain upload/update. |
| fedma | personalized_b | yes | 393,216 | 0.75 MiB | 0 | 0 | 20.883M | 62.650M | Server aligns LoRA-A rows first; clients use plain upload/update. |
| full_rank | shared_b | yes | 786,432 | 1.50 MiB | 0 | 0 | 9.135G | 27.406G | Server averages BA in full rank, then factorizes back to rank-r LoRA; clients use plain upload/update. |
| full_rank | personalized_b | no | N/A | N/A | N/A | N/A | N/A | N/A | Invalid: full-rank server aggregation needs both A and B to form BA. |

## Combined Client Totals By Task Group

| Group | Method | Scenario | Valid | Local Trainable Params | Common Client FLOPs / Round | Client Extra FLOPs / Round | Client Total FLOPs / Round | Client Total FLOPs / 3 Rounds | Server Aggregation FLOPs / Round |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| binary | factor_fedavg | shared_b | yes | 1,838,082 | 15.001T | 0 | 15.001T | 45.004T | 3.932M |
| binary | factor_fedavg | personalized_b | yes | 1,838,082 | 15.001T | 0 | 15.001T | 45.004T | 1.966M |
| binary | factor_fedprox | shared_b | yes | 1,838,082 | 15.001T | 4.286G | 15.006T | 45.017T | 3.932M |
| binary | factor_fedprox | personalized_b | yes | 1,838,082 | 15.001T | 4.286G | 15.006T | 45.017T | 1.966M |
| binary | fa_lora | shared_b | yes | 1,444,866 | 15.001T | 0 | 15.001T | 45.004T | 1.966M |
| binary | fa_lora | personalized_b | no | N/A | 15.001T | N/A | N/A | N/A | N/A |
| binary | svd | shared_b | yes | 1,838,082 | 15.001T | 0 | 15.001T | 45.004T | 4.997G |
| binary | svd | personalized_b | yes | 1,838,082 | 15.001T | 12.829M | 15.001T | 45.004T | 163.643M |
| binary | svd_no_gram | shared_b | yes | 1,838,082 | 15.001T | 0 | 15.001T | 45.004T | 4.997G |
| binary | svd_no_gram | personalized_b | yes | 1,838,082 | 15.001T | 0 | 15.001T | 45.004T | 163.643M |
| binary | fedma | shared_b | yes | 1,838,082 | 15.001T | 0 | 15.001T | 45.004T | 4.855G |
| binary | fedma | personalized_b | yes | 1,838,082 | 15.001T | 0 | 15.001T | 45.004T | 20.883M |
| binary | full_rank | shared_b | yes | 1,838,082 | 15.001T | 0 | 15.001T | 45.004T | 9.135G |
| binary | full_rank | personalized_b | no | N/A | 15.001T | N/A | N/A | N/A | N/A |
| three_class | factor_fedavg | shared_b | yes | 1,839,107 | 15.001T | 0 | 15.001T | 45.004T | 3.932M |
| three_class | factor_fedavg | personalized_b | yes | 1,839,107 | 15.001T | 0 | 15.001T | 45.004T | 1.966M |
| three_class | factor_fedprox | shared_b | yes | 1,839,107 | 15.001T | 4.286G | 15.006T | 45.017T | 3.932M |
| three_class | factor_fedprox | personalized_b | yes | 1,839,107 | 15.001T | 4.286G | 15.006T | 45.017T | 1.966M |
| three_class | fa_lora | shared_b | yes | 1,445,891 | 15.001T | 0 | 15.001T | 45.004T | 1.966M |
| three_class | fa_lora | personalized_b | no | N/A | 15.001T | N/A | N/A | N/A | N/A |
| three_class | svd | shared_b | yes | 1,839,107 | 15.001T | 0 | 15.001T | 45.004T | 4.997G |
| three_class | svd | personalized_b | yes | 1,839,107 | 15.001T | 12.829M | 15.001T | 45.004T | 163.643M |
| three_class | svd_no_gram | shared_b | yes | 1,839,107 | 15.001T | 0 | 15.001T | 45.004T | 4.997G |
| three_class | svd_no_gram | personalized_b | yes | 1,839,107 | 15.001T | 0 | 15.001T | 45.004T | 163.643M |
| three_class | fedma | shared_b | yes | 1,839,107 | 15.001T | 0 | 15.001T | 45.004T | 4.855G |
| three_class | fedma | personalized_b | yes | 1,839,107 | 15.001T | 0 | 15.001T | 45.004T | 20.883M |
| three_class | full_rank | shared_b | yes | 1,839,107 | 15.001T | 0 | 15.001T | 45.004T | 9.135G |
| three_class | full_rank | personalized_b | no | N/A | 15.001T | N/A | N/A | N/A | N/A |
| regression | factor_fedavg | shared_b | yes | 1,837,057 | 15.001T | 0 | 15.001T | 45.003T | 3.932M |
| regression | factor_fedavg | personalized_b | yes | 1,837,057 | 15.001T | 0 | 15.001T | 45.003T | 1.966M |
| regression | factor_fedprox | shared_b | yes | 1,837,057 | 15.001T | 4.286G | 15.005T | 45.016T | 3.932M |
| regression | factor_fedprox | personalized_b | yes | 1,837,057 | 15.001T | 4.286G | 15.005T | 45.016T | 1.966M |
| regression | fa_lora | shared_b | yes | 1,443,841 | 15.001T | 0 | 15.001T | 45.003T | 1.966M |
| regression | fa_lora | personalized_b | no | N/A | 15.001T | N/A | N/A | N/A | N/A |
| regression | svd | shared_b | yes | 1,837,057 | 15.001T | 0 | 15.001T | 45.003T | 4.997G |
| regression | svd | personalized_b | yes | 1,837,057 | 15.001T | 12.829M | 15.001T | 45.003T | 163.643M |
| regression | svd_no_gram | shared_b | yes | 1,837,057 | 15.001T | 0 | 15.001T | 45.003T | 4.997G |
| regression | svd_no_gram | personalized_b | yes | 1,837,057 | 15.001T | 0 | 15.001T | 45.003T | 163.643M |
| regression | fedma | shared_b | yes | 1,837,057 | 15.001T | 0 | 15.001T | 45.003T | 4.855G |
| regression | fedma | personalized_b | yes | 1,837,057 | 15.001T | 0 | 15.001T | 45.003T | 20.883M |
| regression | full_rank | shared_b | yes | 1,837,057 | 15.001T | 0 | 15.001T | 45.003T | 9.135G |
| regression | full_rank | personalized_b | no | N/A | 15.001T | N/A | N/A | N/A | N/A |

## Main Takeaways

- Client rounds are dominated by the same RoBERTa-large local training term: about `15.001T` FLOPs per round for the binary GLUE tasks, `15.001T` for MNLI, and `15.001T` for STS-B.
- `factor_fedavg`, `fa_lora`, `svd`, `svd_no_gram`, `fedma`, and `full_rank` are therefore very close on client-side arithmetic unless they add an explicit extra path. The largest shared-B client extra in this codebase is the `svdtrainer.update()` reconstruction, about `1.623G` FLOPs per round.
- `fedprox` is special: its extra arithmetic term is only `4.286G` FLOPs per round, but it also scans the full `357M`-parameter model every batch. That memory traffic is likely a bigger wall-clock driver than the raw extra FLOPs.
- Server-side separation is much larger than client-side separation. In the current code, the expected aggregation ordering is roughly: `factor_fedavg / factor_fedprox / fa_lora` < `fedma (personalized_B)` < `svd / svd_no_gram (personalized_B)` < `fedma (shared_B)` ≈ `svd / svd_no_gram (shared_B)` < `full_rank (shared_B)`.
- Personalized-B scenarios halve the shared payload from `786,432` to `393,216` params for all valid methods that still transmit A, and they remove the shared-B receive-time `B` reconstruction cost from the client.

