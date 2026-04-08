#!/bin/sh

conda activate aprils

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml federate.method 'svd' trainer.type 'svdtrainer' data.type 'qnli@glue' data.splitter 'iid' device 0 >/dev/null 2>&1 & 

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml federate.method 'svd' trainer.type 'svdtrainer' data.type 'rte@glue' data.splitter 'iid' device 1 >/dev/null 2>&1 &

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml federate.method 'svd' trainer.type 'svdtrainer' data.type 'mnli@glue' data.splitter 'lda' device 2 >/dev/null 2>&1 &

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml federate.method 'svd' trainer.type 'svdtrainer' data.type 'qqp@glue' data.splitter 'lda' device 3 >/dev/null 2>&1 &

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml federate.method 'svd' trainer.type 'svdtrainer' data.type 'sst2@glue' data.splitter 'lda' device 4 >/dev/null 2>&1 &

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml personalization.local_param "['classifier']" federate.method 'svd' data.type 'qqp@glue' data.splitter 'iid' device 5 >/dev/null 2>&1 & 

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml personalization.local_param "['classifier']" federate.method 'svd' data.type 'qqp@glue' data.splitter 'lda' device 6 >/dev/null 2>&1 &

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml personalization.local_param "['classifier']" federate.method 'svd' data.type 'rte@glue' data.splitter 'lda' device 7 >/dev/null 2>&1 &

wait

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml personalization.local_param "['classifier']" federate.method 'svd' trainer.type 'svdtrainer' data.type 'rte@glue' data.splitter 'iid' device 0 >/dev/null 2>&1 &

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml personalization.local_param "['classifier']" federate.method 'svd' trainer.type 'svdtrainer' data.type 'mnli@glue' data.splitter 'lda' device 1 >/dev/null 2>&1 &

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml personalization.local_param "['classifier']" federate.method 'svd' trainer.type 'svdtrainer' data.type 'sst2@glue' data.splitter 'lda' device 2 >/dev/null 2>&1 &

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml personalization.local_param "['classifier']" federate.method 'svd' trainer.type 'svdtrainer' data.type 'rte@glue' data.splitter 'lda' device 5 >/dev/null 2>&1 &

wait

