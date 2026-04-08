import os
from federatedscope.llm.model.adapter_builder import AdapterModel

# Enable offline mode if environment variable is set
OFFLINE_MODE = os.environ.get('HF_HUB_OFFLINE', '0') == '1' or \
               os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'

# Also set environment variables to prevent any network calls
if OFFLINE_MODE:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


def get_model_from_huggingface(model_name, config):
    """
    Load a causal language model from HuggingFace transformers library.

    Args:
        model_name (str): The name of the pre-trained model to load.
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        AutoModelForCausalLM: A causal language model object.
    """
    from transformers import AutoModelForCausalLM

    kwargs = {'trust_remote_code': True}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    # Use SDPA (Scaled Dot-Product Attention) for faster attention if available
    try:
        kwargs['attn_implementation'] = 'sdpa'
    except Exception:
        pass

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_model_from_modelscope(model_name, config):
    """
    Load a causal language model from ModelScope models library.

    Args:
        model_name (str): The name of the pre-trained model to load.
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        Model: A causal language model object.
    """
    from modelscope import AutoModelForCausalLM

    kwargs = {'trust_remote_code': True}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    # Always try local_files_only first
    try:
        kwargs['local_files_only'] = True
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except OSError:
        # Fall back to downloading if local files not found and not in offline mode
        if OFFLINE_MODE:
            raise OSError(
                f"Cannot load model '{model_name}' in offline mode. "
                f"Please run make_offline.py first to download required models."
            )
        kwargs['local_files_only'] = False
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config):
    """
    Get a causal language model based on the configuration.

    Args:
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        AdapterModel: A causal language model object with optional adapter
            layers.
    """
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name, config=config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # Resize LLM model based on settings
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len,
                      model_hub)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)
    
    # for FFA-LoRA
    if config.federate.freeze_A:
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False
    return model
