# GLUE wall-clock benchmark

This folder contains fast GLUE timing configs for rebuttal tables.

Goal:
- compare server and client wall-clock across baselines
- finish quickly with only a few FL rounds
- append a markdown table directly into `RESULTS.md`

Common setup:
- backbone: `FacebookAI/roberta-large@huggingface_llm`
- LoRA rank: `8`
- clients: `3`
- rounds: `3`
- local steps: `4` batches
- train subset per task: up to `384`
- validation subset per task: up to `96`
- classifier is always personalized for GLUE

Scenarios:
- `shared_b`: `personalization.local_param=['classifier']`
- `personalized_b`: `personalization.local_param=['classifier', 'lora_B']`

Methods:
- `factor_fedavg`
- `factor_fedprox`
- `fa_lora`
- `svd`
- `svd_no_gram`
- `fedma`
- `full_rank`

Method constraints:
- `fa_lora` is only valid for `shared_b`
- `full_rank` is only valid for `shared_b`

The matrix runner records markdown rows in:
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/RESULTS.md`
- appends are file-locked, so multiple GPU shards can write into the same table safely

The analytical cost report lives in:
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/THEORETICAL_COSTS.md`

It reads:
- `system_metrics.log` for worker summaries
- `round_timing.log` indirectly through the summary fields written into `system_metrics.log`

Reported timing columns come from the recently added timers:
- `server_round_time_avg_seconds_excluding_first_round`
- `client_round_time_avg_seconds_excluding_first_round`

The first timed round is excluded from the reported round average.

Main scripts:
- `/home/mahi/app/APRILS/scripts/run_glue_wallclock_matrix.sh`
- `/home/mahi/app/APRILS/scripts/run_glue_wallclock_one.sh`
- `/home/mahi/app/APRILS/scripts/run_glue_wallclock_matrix.py`
- `/home/mahi/app/APRILS/scripts/build_glue_theoretical_costs.py`

Run them from the `aprils` environment, or set `PYTHON=/home/mahi/miniconda3/envs/aprils/bin/python`.
Add `--show-logs` if you want the per-case log streamed to the terminal while it is also saved to `command.log`.
