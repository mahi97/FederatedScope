import os
import sys
import torch

print(os.getcwd())
os.chdir('..')
print(os.getcwd())

# Respect the caller's Hugging Face online/offline settings.
# Offline mode can still be enabled explicitly through the standard env vars.

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, get_ds_rank
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # Allow project-local env files to override cache/data roots even when
    # the loaded yaml sets explicit defaults.
    env_hf_home = os.environ.get('HF_HOME')
    env_data_root = os.environ.get('HF_DATASETS_CACHE') or \
        os.environ.get('DATA_DIR')
    if env_hf_home:
        init_cfg.llm.cache.model = env_hf_home
    if env_data_root:
        init_cfg.data.root = env_data_root

    # Redirect experiment outputs to APRILS_OUTPUT_DIR / APRILS_EXP_DIR so
    # nothing is written under the project tree or ~/.cache.
    env_output_dir = os.environ.get('APRILS_OUTPUT_DIR')
    env_exp_dir = os.environ.get('APRILS_EXP_DIR')
    if env_output_dir:
        init_cfg.outdir = env_output_dir
    if env_exp_dir:
        init_cfg.outdir = env_exp_dir

    # Point wandb at the storage root so logs stay off the project dir.
    env_wandb_dir = os.environ.get('WANDB_DIR')
    if env_wandb_dir:
        os.makedirs(env_wandb_dir, exist_ok=True)

    if init_cfg.llm.deepspeed.use:
        import deepspeed
        deepspeed.init_distributed()

    update_logger(init_cfg, clear_before_add=True, rank=get_ds_rank())
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)

    if init_cfg.federate.client_idx_for_local_train != 0:
        init_cfg.federate.client_num = 1
        new_data = {0: data[0]} if 0 in data.keys() else dict()
        new_data[1] = data[init_cfg.federate.client_idx_for_local_train]
        data = new_data
    
    init_cfg.freeze()
    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=client_cfgs)
    _ = runner.run()