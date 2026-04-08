import os
import os
from transformers import AutoTokenizer
from datasets import load_dataset

# Enable offline mode if environment variable is set or if we detect no network
OFFLINE_MODE = os.environ.get('HF_HUB_OFFLINE', '0') == '1' or \
               os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1' or \
               os.environ.get('HF_DATASETS_OFFLINE', '0') == '1'

# Also set environment variables to prevent any network calls
if OFFLINE_MODE:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_glue_dataset(config=None, **kwargs):
    model_name, _ = config.model.type.split('@')
    task_name, _ = config.data.type.split('@')
    glue_cfg = config.data.glue
    
    # Set cache directory for datasets
    cache_dir = config.data.root if config.data.root else None
    
    # Download and cache the dataset (not from cache-only mode)
    # This ensures the dataset is downloaded if not cached
    # Use the full repository path 'nyu-mll/glue' for proper resolution
    datasets = load_dataset(
        "nyu-mll/glue", 
        task_name, 
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
        config.data.label_list = label_list    # added by me, update the config object of the label list
    else:
        num_labels = 1
    config.data.num_labels = num_labels    # added by me, update the config object of the number of labels
    
    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Set model cache directory
    model_cache_dir = config.llm.cache.model if config.llm.cache.model else None
    
    # load tokenizer with proper caching
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_cache_dir,
        trust_remote_code=True
    )
    
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding='max_length', max_length=config.llm.tok_len, truncation=True)
        return result
    
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    def limit_split(split_dataset, size, seed_offset=0):
        if size is None or size <= 0 or len(split_dataset) <= size:
            return split_dataset
        shuffled = split_dataset.shuffle(seed=glue_cfg.sample_seed +
                                         seed_offset)
        return shuffled.select(range(size))

    train_dataset = limit_split(datasets["train"], glue_cfg.train_size, 0)
    if task_name == "mnli":
        eval_dataset = datasets["validation_matched" if config.data.matched else "validation_mismatched"]
        # test_dataset = datasets["test_matched" if config.data.matched else "test_mismatched"]
    else:
        eval_dataset = datasets["validation"]
        # test_dataset = datasets["test"]

    eval_dataset = limit_split(eval_dataset, glue_cfg.val_size, 1)
    
    dataset = (train_dataset, eval_dataset, [])

    return dataset, config
