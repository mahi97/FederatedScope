# GLUE Wall-Clock Run Order

## Files

Configs:
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/roberta_large_factor_fedavg.yaml`
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/roberta_large_factor_fedprox.yaml`
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/roberta_large_fa_lora.yaml`
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/roberta_large_svd.yaml`
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/roberta_large_svd_no_gram.yaml`
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/roberta_large_fedma.yaml`
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/roberta_large_full_rank.yaml`

Scripts:
- `/home/mahi/app/APRILS/scripts/run_glue_wallclock_matrix.sh`
- `/home/mahi/app/APRILS/scripts/run_glue_wallclock_one.sh`
- `/home/mahi/app/APRILS/scripts/run_glue_wallclock_matrix.py`
- `/home/mahi/app/APRILS/scripts/build_glue_theoretical_costs.py`

Results table:
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/RESULTS.md`
- `/home/mahi/app/APRILS/federatedscope/glue/yamls/wallclock/THEORETICAL_COSTS.md`

## Default order in the matrix script

Tasks:
1. `cola`
2. `mnli`
3. `mrpc`
4. `qnli`
5. `qqp`
6. `rte`
7. `sst2`
8. `stsb`
9. `wnli`

Methods:
1. `factor_fedavg`
2. `factor_fedprox`
3. `fa_lora`
4. `svd`
5. `svd_no_gram`
6. `fedma`
7. `full_rank`

Scenarios:
1. `shared_b`
2. `personalized_b`

Invalid combinations are appended as `N/A` rows automatically:
- `fa_lora` + `personalized_b`
- `full_rank` + `personalized_b`

## Run everything on one GPU

Use the `aprils` environment, or prefix the commands with `PYTHON=/home/mahi/miniconda3/envs/aprils/bin/python`.

```bash
./scripts/run_glue_wallclock_matrix.sh --gpu 4 --reset-results
```

If you want live terminal logs as well:

```bash
./scripts/run_glue_wallclock_matrix.sh --gpu 4 --reset-results --show-logs
```

## Recommended 4-GPU sharding

GPU 4:
```bash
./scripts/run_glue_wallclock_matrix.sh --gpu 4 --reset-results --tasks cola,mnli,mrpc
```

GPU 5:
```bash
./scripts/run_glue_wallclock_matrix.sh --gpu 5 --tasks qnli,qqp
```

GPU 6:
```bash
./scripts/run_glue_wallclock_matrix.sh --gpu 6 --tasks rte,sst2
```

GPU 7:
```bash
./scripts/run_glue_wallclock_matrix.sh --gpu 7 --tasks stsb,wnli
```

Only the first shard should use `--reset-results`. The results file appends are locked, so the other shards can write into the same markdown table safely.

## Rerun one case

```bash
./scripts/run_glue_wallclock_one.sh 4 sst2 svd shared_b
./scripts/run_glue_wallclock_one.sh 5 qqp factor_fedprox personalized_b
```
