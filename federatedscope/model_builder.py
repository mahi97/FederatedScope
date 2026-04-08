import os
import torch
from federatedscope.glue.model.adapter_builder import AdapterModel

# Enable offline mode if environment variable is set
OFFLINE_MODE = os.environ.get('HF_HUB_OFFLINE', '0') == '1' or \
               os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'

# Also set environment variables to prevent any network calls
if OFFLINE_MODE:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


def get_model_from_huggingface(model_name, config):
    from transformers import AutoModelForSequenceClassification

    kwargs = {}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model
    
    # added by me, for GLUE
    kwargs['num_labels'] = config.data.num_labels
    
    # Always try local_files_only first if cache is specified
    try:
        kwargs['local_files_only'] = True
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    except OSError:
        # Fall back to downloading if local files not found and not in offline mode
        if OFFLINE_MODE:
            raise OSError(
                f"Cannot load model '{model_name}' in offline mode. "
                f"Please run make_offline.py first to download required models."
            )
        kwargs['local_files_only'] = False
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)


def get_llm(config):    
    model_name, model_hub = config.model.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)
    
    # for FFA-LoRA & FFA-VeRA
    if config.federate.freeze_A:
        for name, param in model.named_parameters():
            if "lora_A" in name or "vera_lambda_d" in name:
                param.requires_grad = False
    
    # save initial lora parameters, for local training
    if config.federate.method == "local":
        if config.llm.adapter.args[0].get('adapter_method', '') == "vera":
            initial_lora_params = {name: param.clone() for name, param in model.named_parameters() if 'vera' in name}
        else:
            initial_lora_params = {name: param.clone() for name, param in model.named_parameters() if 'lora' in name}
        torch.save(initial_lora_params, config.federate.save_to + '.init')
    return model
