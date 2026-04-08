# FedSA-LoRA

The implementation of [Selective Aggregation for Low-Rank Adaptation in Federated Learning](https://openreview.net/forum?id=iX3uESGdsO) [ICLR 2025]. \
[Pengxin Guo](https://pengxin-guo.github.io), [Shuang Zeng](https://scholar.google.com/citations?user=yTP1oqkAAAAJ&hl=en), Yanran Wang, Huijie Fan, Feifei Wang, and [Liangqiong Qu](https://liangqiong.github.io).

<img src="./figs/FedSA-LoRA.png" alt="framework" width="700" /> 

##### Figure 1. The illustration of (a) LoRA, (b) FFA-LoRA, and (c) FedSA-LoRA. In LoRA, both $A$ and $B$ matrices are trainable and shared with the server for aggregation. In FFA-LoRA, only $B$ matrices are trainable and shared with the server for aggregation, while $A$ matrices are fixed after initialization. In FedSA-LoRA, both $A$ and $B$ matrices are trainable, but only $A$ matrices are shared with the server for aggregation while $B$ matrices are kept locally.


## Installation

Our code is based on Python version 3.10. The project uses `uv` and `pyproject.toml` for dependency management. CUDA-enabled PyTorch wheels are configured through uv's `pytorch-cu126` index, so `torch` and `torchvision` are installed with GPU support on Linux/Windows.

Install the LLM dependencies with:
```shell
uv python install 3.10
uv sync --locked --extra llm
```

Verify that PyTorch can see the GPU:
```shell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Training

Now, we can fine-tune a LLM with FedSA-LoRA:

```shell
uv run python federatedscope/main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml

# or 

uv run python main.py --cfg ~/app/APRILS/federatedscope/glue/yamls/svd-lora.yaml

```

## Acknowledgement


We would like to thank the authors for releasing the public repository: [FederatedScope-LLM](https://github.com/alibaba/FederatedScope/tree/llm).
