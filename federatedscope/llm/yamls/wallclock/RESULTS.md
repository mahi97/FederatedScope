# LLM Wall-Clock Results (Qwen 2.5 3B / GSM8K)

This file is appended automatically by `scripts/run_llm_wallclock_matrix.py`.

| Method | Scenario | Status | Server Round Avg (s) | Client Round Avg Mean (s) | Client Round Avg Std (s) | Server Timed Rounds | Client Timed Rounds Mean | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| factor_fedavg | shared_b | ok | 0.023 | 2.967 | 0.015 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/shared_b/factor_fedavg/fedavg_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| factor_fedavg | personalized_b | ok | 0.022 | 2.964 | 0.015 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/personalized_b/factor_fedavg/fedavg_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| factor_fedprox | shared_b | ok | 0.022 | 3.134 | 0.018 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/shared_b/factor_fedprox/fedavg_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| factor_fedprox | personalized_b | ok | 0.022 | 3.134 | 0.017 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/personalized_b/factor_fedprox/fedavg_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| fa_lora | shared_b | ok | 0.018 | 2.961 | 0.017 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/shared_b/fa_lora/fedavg_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| svd | shared_b | ok | 0.024 | 2.971 | 0.017 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/shared_b/svd/svd_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| svd | personalized_b | ok | 0.023 | 3.645 | 0.041 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/personalized_b/svd/svd_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| svd_no_gram | shared_b | ok | 0.024 | 2.961 | 0.017 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/shared_b/svd_no_gram/svd_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| svd_no_gram | personalized_b | ok | 0.021 | 3.553 | 0.034 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/personalized_b/svd_no_gram/svd_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| fedma | shared_b | ok | 0.019 | 2.958 | 0.016 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/shared_b/fedma/svd_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| fedma | personalized_b | ok | 0.020 | 2.959 | 0.016 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/personalized_b/fedma/svd_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| full_rank | shared_b | ok | 0.022 | 2.965 | 0.016 | 4 | 5.000 | `/home/mahi/app/APRILS/exp/llm_wallclock/shared_b/full_rank/svd_Qwen/Qwen2.5-3B-Instruct@huggingface_llm_on_gsm8k@llm_lr0.005_lstep10` |
| full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
