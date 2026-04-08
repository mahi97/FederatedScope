# GLUE Wall-Clock Results

This file is appended automatically by `scripts/run_glue_wallclock_matrix.py`.

| Task | Method | Scenario | Status | Server Round Avg (s) | Client Round Avg Mean (s) | Client Round Avg Std (s) | Server FL End (min) | Client FL End Mean (min) | Server Timed Rounds | Client Timed Rounds Mean | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cola | factor_fedavg | shared_b | ok | 0.026 | 1.452 | 0.041 | 0.358 | 0.330 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/shared_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | factor_fedavg | personalized_b | ok | 0.019 | 1.729 | 0.178 | 0.377 | 0.349 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/personalized_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | factor_fedprox | shared_b | ok | 0.026 | 1.754 | 0.170 | 0.390 | 0.361 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/shared_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | factor_fedprox | personalized_b | ok | 0.022 | 1.689 | 0.244 | 0.394 | 0.365 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/personalized_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | fa_lora | shared_b | ok | 0.026 | 1.495 | 0.044 | 0.360 | 0.330 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/shared_b/fa_lora/fedavg_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| cola | svd | shared_b | ok | 0.056 | 1.928 | 0.210 | 0.443 | 0.426 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/shared_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | svd | personalized_b | ok | 0.052 | 2.258 | 0.346 | 0.472 | 0.452 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/personalized_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | svd_no_gram | shared_b | ok | 0.081 | 1.989 | 0.101 | 0.456 | 0.443 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/shared_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | svd_no_gram | personalized_b | ok | 0.054 | 1.785 | 0.086 | 0.434 | 0.416 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/personalized_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| mnli | factor_fedavg | shared_b | ok | 0.030 | 1.464 | 0.020 | 0.359 | 0.330 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/shared_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | factor_fedavg | personalized_b | ok | 0.023 | 1.617 | 0.077 | 0.372 | 0.344 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/personalized_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | factor_fedprox | shared_b | ok | 0.031 | 1.606 | 0.115 | 0.396 | 0.367 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/shared_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | factor_fedprox | personalized_b | ok | 0.021 | 0.402 | 0.002 | 0.160 | 0.131 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/personalized_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | fa_lora | shared_b | ok | 0.018 | 0.304 | 0.000 | 0.144 | 0.114 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/shared_b/fa_lora/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| mnli | svd | shared_b | ok | 0.060 | 0.641 | 0.033 | 0.228 | 0.209 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/shared_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | svd | personalized_b | ok | 0.048 | 0.683 | 0.038 | 0.230 | 0.211 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/personalized_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | svd_no_gram | shared_b | ok | 0.054 | 0.660 | 0.047 | 0.216 | 0.197 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/shared_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | svd_no_gram | personalized_b | ok | 0.052 | 0.715 | 0.059 | 0.234 | 0.213 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/personalized_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mrpc | factor_fedavg | shared_b | ok | 0.033 | 0.328 | 0.000 | 0.153 | 0.122 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/shared_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | factor_fedavg | personalized_b | ok | 0.022 | 0.317 | 0.001 | 0.152 | 0.118 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/personalized_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | factor_fedprox | shared_b | ok | 0.033 | 0.406 | 0.004 | 0.169 | 0.134 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/shared_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | factor_fedprox | personalized_b | ok | 0.021 | 0.421 | 0.004 | 0.168 | 0.135 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/personalized_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | fa_lora | shared_b | ok | 0.020 | 0.309 | 0.001 | 0.150 | 0.117 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/shared_b/fa_lora/fedavg_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| mrpc | svd | shared_b | ok | 0.061 | 0.680 | 0.039 | 0.223 | 0.203 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/shared_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | svd | personalized_b | ok | 0.061 | 0.784 | 0.032 | 0.241 | 0.218 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/personalized_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | svd_no_gram | shared_b | ok | 0.059 | 0.680 | 0.034 | 0.227 | 0.203 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/shared_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | svd_no_gram | personalized_b | ok | 0.048 | 0.655 | 0.038 | 0.223 | 0.196 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/personalized_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| qnli | factor_fedavg | shared_b | ok | 0.024 | 0.314 | 0.000 | 0.150 | 0.120 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/shared_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | factor_fedavg | personalized_b | ok | 0.019 | 0.313 | 0.001 | 0.148 | 0.118 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/personalized_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | factor_fedprox | shared_b | ok | 0.024 | 0.397 | 0.000 | 0.162 | 0.131 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/shared_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | factor_fedprox | personalized_b | ok | 0.032 | 0.456 | 0.002 | 0.176 | 0.146 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/personalized_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | fa_lora | shared_b | ok | 0.018 | 0.304 | 0.001 | 0.144 | 0.115 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/shared_b/fa_lora/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| qnli | svd | shared_b | ok | 0.054 | 0.613 | 0.001 | 0.213 | 0.193 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/shared_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | svd | personalized_b | ok | 0.046 | 0.622 | 0.001 | 0.216 | 0.194 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/personalized_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | svd_no_gram | shared_b | ok | 0.053 | 0.607 | 0.001 | 0.210 | 0.190 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/shared_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | svd_no_gram | personalized_b | ok | 0.047 | 0.606 | 0.001 | 0.210 | 0.190 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/personalized_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qqp | factor_fedavg | shared_b | ok | 0.024 | 0.311 | 0.001 | 0.144 | 0.116 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/shared_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qqp | factor_fedavg | personalized_b | ok | 0.019 | 0.308 | 0.004 | 0.146 | 0.117 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/personalized_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qqp | factor_fedprox | shared_b | ok | 0.025 | 0.396 | 0.001 | 0.161 | 0.132 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/shared_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qqp | factor_fedprox | personalized_b | ok | 0.020 | 0.393 | 0.000 | 0.160 | 0.130 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/personalized_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qqp | fa_lora | shared_b | ok | 0.017 | 0.303 | 0.001 | 0.142 | 0.113 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/shared_b/fa_lora/fedavg_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qqp | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| qqp | svd | shared_b | ok | 0.054 | 0.607 | 0.001 | 0.208 | 0.189 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/shared_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| cola | fedma | shared_b | ok | 0.034 | 0.606 | 0.001 | 0.210 | 0.191 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/shared_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | fedma | personalized_b | ok | 0.028 | 0.607 | 0.001 | 0.207 | 0.188 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/personalized_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | full_rank | shared_b | ok | 0.072 | 0.737 | 0.002 | 0.241 | 0.214 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/cola/shared_b/full_rank/svd_FacebookAI/roberta-large@huggingface_llm_on_cola@glue_lr0.02_lstep4` |
| cola | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
| mnli | fedma | shared_b | ok | 0.033 | 0.631 | 0.037 | 0.206 | 0.187 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/shared_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| qqp | svd | personalized_b | ok | 0.048 | 0.633 | 0.007 | 0.213 | 0.194 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/personalized_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| mnli | fedma | personalized_b | ok | 0.026 | 0.630 | 0.035 | 0.208 | 0.188 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/personalized_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | full_rank | shared_b | ok | 0.062 | 0.641 | 0.037 | 0.210 | 0.190 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mnli/shared_b/full_rank/svd_FacebookAI/roberta-large@huggingface_llm_on_mnli@glue_lr0.02_lstep4` |
| mnli | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
| mrpc | fedma | shared_b | ok | 0.035 | 0.640 | 0.040 | 0.215 | 0.194 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/shared_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | fedma | personalized_b | ok | 0.027 | 0.623 | 0.030 | 0.208 | 0.188 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/personalized_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| qqp | svd_no_gram | shared_b | ok | 0.053 | 0.596 | 0.001 | 0.211 | 0.192 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/shared_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| mrpc | full_rank | shared_b | ok | 0.058 | 0.623 | 0.034 | 0.209 | 0.190 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/mrpc/shared_b/full_rank/svd_FacebookAI/roberta-large@huggingface_llm_on_mrpc@glue_lr0.02_lstep4` |
| mrpc | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
| qnli | fedma | shared_b | ok | 0.034 | 0.624 | 0.005 | 0.211 | 0.192 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/shared_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | fedma | personalized_b | ok | 0.026 | 0.601 | 0.002 | 0.207 | 0.188 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/personalized_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qqp | svd_no_gram | personalized_b | ok | 0.047 | 0.597 | 0.004 | 0.205 | 0.186 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/personalized_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qnli | full_rank | shared_b | ok | 0.060 | 0.653 | 0.025 | 0.220 | 0.202 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qnli/shared_b/full_rank/svd_FacebookAI/roberta-large@huggingface_llm_on_qnli@glue_lr0.02_lstep4` |
| qnli | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
| qqp | fedma | shared_b | ok | 0.033 | 0.602 | 0.002 | 0.205 | 0.186 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/shared_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qqp | fedma | personalized_b | ok | 0.026 | 0.598 | 0.001 | 0.202 | 0.183 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/personalized_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qqp | full_rank | shared_b | ok | 0.060 | 0.604 | 0.001 | 0.208 | 0.189 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/qqp/shared_b/full_rank/svd_FacebookAI/roberta-large@huggingface_llm_on_qqp@glue_lr0.02_lstep4` |
| qqp | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
| rte | factor_fedavg | shared_b | ok | 0.024 | 0.311 | 0.000 | 0.163 | 0.134 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/shared_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | factor_fedavg | personalized_b | ok | 0.019 | 0.311 | 0.001 | 0.147 | 0.118 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/personalized_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | factor_fedprox | shared_b | ok | 0.024 | 0.398 | 0.001 | 0.158 | 0.129 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/shared_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | factor_fedprox | personalized_b | ok | 0.020 | 0.397 | 0.002 | 0.159 | 0.129 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/personalized_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | fa_lora | shared_b | ok | 0.018 | 0.306 | 0.000 | 0.142 | 0.113 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/shared_b/fa_lora/fedavg_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| rte | svd | shared_b | ok | 0.054 | 0.642 | 0.034 | 0.214 | 0.192 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/shared_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | svd | personalized_b | ok | 0.047 | 0.653 | 0.037 | 0.225 | 0.205 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/personalized_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | svd_no_gram | shared_b | ok | 0.055 | 0.619 | 0.036 | 0.205 | 0.184 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/shared_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | svd_no_gram | personalized_b | ok | 0.061 | 0.754 | 0.036 | 0.225 | 0.204 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/personalized_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | fedma | shared_b | ok | 0.032 | 0.633 | 0.028 | 0.206 | 0.185 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/shared_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | fedma | personalized_b | ok | 0.026 | 0.634 | 0.039 | 0.208 | 0.188 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/rte/personalized_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_rte@glue_lr0.02_lstep4` |
| rte | full_rank | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| rte | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
| sst2 | factor_fedavg | shared_b | ok | 0.024 | 0.312 | 0.000 | 0.142 | 0.114 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/shared_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | factor_fedavg | personalized_b | ok | 0.019 | 0.311 | 0.001 | 0.142 | 0.114 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/personalized_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | factor_fedprox | shared_b | ok | 0.024 | 0.395 | 0.001 | 0.161 | 0.131 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/shared_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | factor_fedprox | personalized_b | ok | 0.020 | 0.393 | 0.000 | 0.158 | 0.129 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/personalized_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | fa_lora | shared_b | ok | 0.018 | 0.303 | 0.001 | 0.143 | 0.113 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/shared_b/fa_lora/fedavg_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| sst2 | svd | shared_b | ok | 0.053 | 0.631 | 0.039 | 0.214 | 0.194 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/shared_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | svd | personalized_b | ok | 0.046 | 0.637 | 0.039 | 0.207 | 0.188 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/personalized_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | svd_no_gram | shared_b | ok | 0.054 | 0.616 | 0.035 | 0.203 | 0.184 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/shared_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | svd_no_gram | personalized_b | ok | 0.047 | 0.612 | 0.032 | 0.202 | 0.183 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/personalized_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | fedma | shared_b | ok | 0.033 | 0.633 | 0.037 | 0.216 | 0.193 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/shared_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | fedma | personalized_b | ok | 0.026 | 0.623 | 0.032 | 0.207 | 0.187 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/sst2/personalized_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_sst2@glue_lr0.02_lstep4` |
| sst2 | full_rank | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| sst2 | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
| stsb | factor_fedavg | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | factor_fedavg | personalized_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | factor_fedprox | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | factor_fedprox | personalized_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | fa_lora | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| stsb | svd | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | svd | personalized_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | svd_no_gram | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | svd_no_gram | personalized_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | fedma | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | fedma | personalized_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | full_rank | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| stsb | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
| wnli | factor_fedavg | shared_b | ok | 0.024 | 0.313 | 0.001 | 0.145 | 0.116 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/shared_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | factor_fedavg | personalized_b | ok | 0.019 | 0.311 | 0.000 | 0.142 | 0.114 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/personalized_b/factor_fedavg/fedavg_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | factor_fedprox | shared_b | ok | 0.025 | 0.403 | 0.001 | 0.167 | 0.135 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/shared_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | factor_fedprox | personalized_b | ok | 0.019 | 0.395 | 0.001 | 0.158 | 0.129 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/personalized_b/factor_fedprox/fedavg_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | fa_lora | shared_b | ok | 0.018 | 0.305 | 0.004 | 0.139 | 0.111 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/shared_b/fa_lora/fedavg_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | fa_lora | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | freeze_A baseline requires shared lora_B |
| wnli | svd | shared_b | ok | 0.055 | 0.639 | 0.021 | 0.211 | 0.191 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/shared_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | svd | personalized_b | ok | 0.046 | 0.600 | 0.002 | 0.200 | 0.181 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/personalized_b/svd/svd_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | svd_no_gram | shared_b | ok | 0.053 | 0.316 | 0.001 | 0.145 | 0.117 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/shared_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | svd_no_gram | personalized_b | ok | 0.047 | 0.601 | 0.005 | 0.203 | 0.183 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/personalized_b/svd_no_gram/svd_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | fedma | shared_b | ok | 0.032 | 0.314 | 0.000 | 0.143 | 0.115 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/shared_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | fedma | personalized_b | ok | 0.026 | 0.310 | 0.001 | 0.141 | 0.113 | 2 | 3.000 | `/home/mahi/app/APRILS/exp/glue_wallclock/wnli/personalized_b/fedma/svd_FacebookAI/roberta-large@huggingface_llm_on_wnli@glue_lr0.02_lstep4` |
| wnli | full_rank | shared_b | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `exit=1` |
| wnli | full_rank | personalized_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | full-rank aggregation requires shared lora_B |
