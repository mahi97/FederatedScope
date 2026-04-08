#!/bin/sh

conda activate aprils

python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/llm/yamls/fedsa-lora.yaml device 0 >/dev/null 2>&1 &