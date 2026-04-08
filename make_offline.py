#!/usr/bin/env python
"""
make_offline.py

This script ensures all models and datasets required for offline execution are
downloaded and cached locally. It also sets up environment variables for
HuggingFace offline mode.

Usage:
    python make_offline.py [--cache-dir /path/to/cache] [--download-only] [--check-only]

Models:
    - FacebookAI/roberta-large (for GLUE tasks)
    - meta-llama/Meta-Llama-3-8B (for LLM tasks - requires authentication)

Datasets:
    - GLUE: cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli
    - LLM: gsm8k

After running this script, set the following environment variables before
running the main script offline:
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default storage root – sourced from STORAGE_ROOT env var set in .env files.
# Nothing should ever be written to ~/.cache.
DEFAULT_STORAGE_ROOT = os.environ.get("STORAGE_ROOT", "/drive1/mahi")
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_STORAGE_ROOT, "huggingface", "hub")

# Models used in the yaml configs
GLUE_MODELS = [
    "FacebookAI/roberta-large",
]

LLM_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
]

# GLUE task names used in configs
GLUE_TASKS = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

# GSM8K URLs for LLM tasks
GSM8K_URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/grade_school_math/data/test.jsonl",
}


def check_model_cached(model_name: str, cache_dir: str) -> Tuple[bool, str]:
    """
    Check if a model AND its tokenizer are already cached locally.
    
    Returns:
        Tuple of (is_cached, path_or_message)
    """
    # Convert model name to cache format
    cache_model_name = f"models--{model_name.replace('/', '--')}"
    
    # Check various possible locations
    possible_paths = [
        os.path.join(cache_dir, "hub", cache_model_name),
        os.path.join(cache_dir, cache_model_name),
        # Also check parent directories for hf_transformers structure
        os.path.join(os.path.dirname(cache_dir), "hf_transformers", cache_model_name),
        os.path.join(os.path.dirname(cache_dir), "data", cache_model_name),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            snapshots_dir = os.path.join(path, "snapshots")
            if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                # Check if tokenizer files exist in the snapshot
                snapshot_dirs = os.listdir(snapshots_dir)
                if snapshot_dirs:
                    snapshot_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                    # Check for essential tokenizer files
                    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json']
                    has_tokenizer = any(
                        os.path.exists(os.path.join(snapshot_path, f)) 
                        for f in tokenizer_files
                    )
                    if has_tokenizer:
                        return True, path
                    else:
                        return False, f"Model '{model_name}' found but TOKENIZER FILES ARE MISSING at {path}"
            # Also check for blobs (alternative cache structure)
            blobs_dir = os.path.join(path, "blobs")
            if os.path.exists(blobs_dir) and os.listdir(blobs_dir):
                # For blobs, we need to check refs to see what files are cached
                refs_dir = os.path.join(path, "refs")
                if os.path.exists(refs_dir):
                    return True, path
    
    return False, f"Model '{model_name}' not found in cache"


def check_glue_dataset_cached(task_name: str, cache_dir: str) -> Tuple[bool, str]:
    """
    Check if a GLUE dataset is already cached locally.
    """
    # HuggingFace datasets cache structure
    glue_cache_path = os.path.join(cache_dir, "glue", task_name)
    
    if os.path.exists(glue_cache_path) and os.listdir(glue_cache_path):
        return True, glue_cache_path
    
    # Also check for arrow files directly
    for root, dirs, files in os.walk(cache_dir):
        if task_name in root.lower() and "glue" in root.lower():
            arrow_files = [f for f in files if f.endswith('.arrow')]
            if arrow_files:
                return True, root
    
    return False, f"GLUE dataset '{task_name}' not found in cache"


def check_gsm8k_cached(data_dir: str) -> Tuple[bool, str]:
    """
    Check if GSM8K dataset files exist.
    """
    train_file = os.path.join(data_dir, "gsm8k_train.jsonl")
    test_file = os.path.join(data_dir, "gsm8k_test.jsonl")
    
    missing = []
    if not os.path.exists(train_file):
        missing.append("gsm8k_train.jsonl")
    if not os.path.exists(test_file):
        missing.append("gsm8k_test.jsonl")
    
    if missing:
        return False, f"Missing GSM8K files: {', '.join(missing)}"
    
    return True, f"GSM8K files found in {data_dir}"


def download_model(model_name: str, cache_dir: str, token: Optional[str] = None) -> bool:
    """
    Download a model and its tokenizer to the cache directory.
    """
    logger.info(f"Downloading model: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
        
        kwargs = {"cache_dir": cache_dir}
        if token:
            kwargs["token"] = token
        
        # Download tokenizer
        logger.info(f"  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        
        # Download model - try different model types
        logger.info(f"  Downloading model weights...")
        if "llama" in model_name.lower() or "gpt" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        elif "roberta" in model_name.lower() or "bert" in model_name.lower():
            # Download both base model and sequence classification model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=3, **kwargs  # Default to 3 labels for MNLI
            )
        else:
            model = AutoModel.from_pretrained(model_name, **kwargs)
        
        logger.info(f"  Successfully downloaded: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"  Failed to download {model_name}: {e}")
        return False


def download_glue_dataset(task_name: str, cache_dir: str) -> bool:
    """
    Download a GLUE dataset to the cache directory.
    """
    logger.info(f"Downloading GLUE dataset: {task_name}")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("glue", task_name, cache_dir=cache_dir)
        logger.info(f"  Successfully downloaded: glue/{task_name}")
        logger.info(f"    Train samples: {len(dataset['train'])}")
        if "validation" in dataset:
            logger.info(f"    Validation samples: {len(dataset['validation'])}")
        return True
        
    except Exception as e:
        logger.error(f"  Failed to download glue/{task_name}: {e}")
        return False


def download_gsm8k(data_dir: str) -> bool:
    """
    Download GSM8K dataset files.
    """
    logger.info("Downloading GSM8K dataset...")
    
    import urllib.request
    
    os.makedirs(data_dir, exist_ok=True)
    
    success = True
    for split, url in GSM8K_URLS.items():
        output_file = os.path.join(data_dir, f"gsm8k_{split}.jsonl")
        
        if os.path.exists(output_file):
            logger.info(f"  {split} already exists, skipping...")
            continue
            
        try:
            logger.info(f"  Downloading {split}...")
            temp_file = os.path.join(data_dir, f"{split}.jsonl")
            urllib.request.urlretrieve(url, temp_file)
            os.rename(temp_file, output_file)
            logger.info(f"    Saved to: {output_file}")
        except Exception as e:
            logger.error(f"  Failed to download {split}: {e}")
            success = False
    
    return success


def check_all(cache_dir: str, data_dir: str, verbose: bool = True) -> dict:
    """
    Check all required resources and return status.
    """
    results = {
        "models": {},
        "datasets": {},
        "all_ready": True
    }
    
    if verbose:
        logger.info("=" * 60)
        logger.info("Checking cached resources...")
        logger.info("=" * 60)
    
    # Check GLUE models
    if verbose:
        logger.info("\n--- GLUE Models ---")
    for model in GLUE_MODELS:
        is_cached, msg = check_model_cached(model, cache_dir)
        results["models"][model] = {"cached": is_cached, "message": msg}
        if verbose:
            status = "✓ CACHED" if is_cached else "✗ MISSING"
            logger.info(f"  {status}: {model}")
        if not is_cached:
            results["all_ready"] = False
    
    # Check LLM models
    if verbose:
        logger.info("\n--- LLM Models ---")
    for model in LLM_MODELS:
        is_cached, msg = check_model_cached(model, cache_dir)
        results["models"][model] = {"cached": is_cached, "message": msg}
        if verbose:
            status = "✓ CACHED" if is_cached else "✗ MISSING"
            logger.info(f"  {status}: {model}")
        if not is_cached:
            results["all_ready"] = False
    
    # Check GLUE datasets
    if verbose:
        logger.info("\n--- GLUE Datasets ---")
    for task in GLUE_TASKS:
        is_cached, msg = check_glue_dataset_cached(task, data_dir)
        results["datasets"][f"glue/{task}"] = {"cached": is_cached, "message": msg}
        if verbose:
            status = "✓ CACHED" if is_cached else "✗ MISSING"
            logger.info(f"  {status}: glue/{task}")
        if not is_cached:
            results["all_ready"] = False
    
    # Check GSM8K
    if verbose:
        logger.info("\n--- LLM Datasets ---")
    is_cached, msg = check_gsm8k_cached(data_dir)
    results["datasets"]["gsm8k"] = {"cached": is_cached, "message": msg}
    if verbose:
        status = "✓ CACHED" if is_cached else "✗ MISSING"
        logger.info(f"  {status}: gsm8k")
    if not is_cached:
        results["all_ready"] = False
    
    return results


def download_all(cache_dir: str, data_dir: str, token: Optional[str] = None) -> dict:
    """
    Download all required resources.
    """
    results = {
        "models": {},
        "datasets": {},
        "all_success": True
    }
    
    logger.info("=" * 60)
    logger.info("Downloading required resources...")
    logger.info("=" * 60)
    
    # Download GLUE models
    logger.info("\n--- Downloading GLUE Models ---")
    for model in GLUE_MODELS:
        is_cached, _ = check_model_cached(model, cache_dir)
        if is_cached:
            logger.info(f"  Skipping {model} (already cached)")
            results["models"][model] = True
        else:
            success = download_model(model, cache_dir, token)
            results["models"][model] = success
            if not success:
                results["all_success"] = False
    
    # Download LLM models
    logger.info("\n--- Downloading LLM Models ---")
    for model in LLM_MODELS:
        is_cached, _ = check_model_cached(model, cache_dir)
        if is_cached:
            logger.info(f"  Skipping {model} (already cached)")
            results["models"][model] = True
        else:
            success = download_model(model, cache_dir, token)
            results["models"][model] = success
            if not success:
                results["all_success"] = False
    
    # Download GLUE datasets
    logger.info("\n--- Downloading GLUE Datasets ---")
    for task in GLUE_TASKS:
        is_cached, _ = check_glue_dataset_cached(task, data_dir)
        if is_cached:
            logger.info(f"  Skipping glue/{task} (already cached)")
            results["datasets"][f"glue/{task}"] = True
        else:
            success = download_glue_dataset(task, data_dir)
            results["datasets"][f"glue/{task}"] = success
            if not success:
                results["all_success"] = False
    
    # Download GSM8K
    logger.info("\n--- Downloading LLM Datasets ---")
    is_cached, _ = check_gsm8k_cached(data_dir)
    if is_cached:
        logger.info("  Skipping gsm8k (already cached)")
        results["datasets"]["gsm8k"] = True
    else:
        success = download_gsm8k(data_dir)
        results["datasets"]["gsm8k"] = success
        if not success:
            results["all_success"] = False
    
    return results


def print_offline_instructions(cache_dir: str, data_dir: str):
    """
    Print instructions for running in offline mode.
    """
    logger.info("\n" + "=" * 60)
    logger.info("OFFLINE MODE INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("""
Before running the main script offline, set these environment variables:

    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1  
    export HF_DATASETS_OFFLINE=1

In your YAML config files, ensure you set:

    llm:
      cache:
        model: '{cache_dir}'
    data:
      root: '{data_dir}'

Or pass them as command line arguments:

    python federatedscope/main.py --cfg your_config.yaml \\
        llm.cache.model={cache_dir} \\
        data.root={data_dir}
""".format(cache_dir=cache_dir, data_dir=data_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare resources for offline execution of FederatedScope"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for HuggingFace model cache (default: $STORAGE_ROOT/huggingface/hub)"
    )
    parser.add_argument(
        "--data-dir", 
        type=str,
        default=None,
        help="Directory for dataset storage (default: $STORAGE_ROOT/data)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check what's cached without downloading"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download missing resources without detailed check report"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (e.g., Llama)"
    )
    parser.add_argument(
        "--glue-only",
        action="store_true",
        help="Only download GLUE-related resources"
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only download LLM-related resources"
    )
    
    args = parser.parse_args()
    
    # Determine cache directory (resolve from args → env → default)
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.environ.get(
            "HF_HOME",
            os.environ.get("HUGGINGFACE_HUB_CACHE", DEFAULT_CACHE_DIR),
        )
    cache_dir = os.path.expanduser(cache_dir)
    
    # Determine data directory (resolve from args → env → default)
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.environ.get(
            "HF_DATASETS_CACHE",
            os.path.join(DEFAULT_STORAGE_ROOT, "data"),
        )
    data_dir = os.path.expanduser(data_dir)
    
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    # Filter models/datasets based on flags
    global GLUE_MODELS, LLM_MODELS, GLUE_TASKS
    if args.glue_only:
        LLM_MODELS = []
    if args.llm_only:
        GLUE_MODELS = []
        GLUE_TASKS = []
    
    # Ensure directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    if args.check_only:
        # Only check status
        results = check_all(cache_dir, data_dir)
        
        logger.info("\n" + "=" * 60)
        if results["all_ready"]:
            logger.info("✓ All resources are cached and ready for offline use!")
            print_offline_instructions(cache_dir, data_dir)
            sys.exit(0)
        else:
            logger.info("✗ Some resources are missing. Run without --check-only to download.")
            sys.exit(1)
    
    elif args.download_only:
        # Download without detailed checking
        results = download_all(cache_dir, data_dir, args.token)
        
        if results["all_success"]:
            logger.info("\n✓ All resources downloaded successfully!")
            print_offline_instructions(cache_dir, data_dir)
            sys.exit(0)
        else:
            logger.info("\n✗ Some downloads failed. Check the logs above.")
            sys.exit(1)
    
    else:
        # Default: check first, then download missing
        check_results = check_all(cache_dir, data_dir)
        
        if check_results["all_ready"]:
            logger.info("\n✓ All resources are already cached!")
            print_offline_instructions(cache_dir, data_dir)
            sys.exit(0)
        
        logger.info("\nSome resources are missing. Starting download...")
        download_results = download_all(cache_dir, data_dir, args.token)
        
        # Final verification
        logger.info("\n" + "=" * 60)
        logger.info("Final verification...")
        logger.info("=" * 60)
        final_results = check_all(cache_dir, data_dir)
        
        if final_results["all_ready"]:
            logger.info("\n✓ All resources ready for offline use!")
            print_offline_instructions(cache_dir, data_dir)
            sys.exit(0)
        else:
            logger.info("\n✗ Some resources still missing. Manual intervention may be needed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
