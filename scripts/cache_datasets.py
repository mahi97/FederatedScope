#!/usr/bin/env python
"""
Script to pre-download and cache all datasets, models, and tokenizers
for GLUE tasks and LLM tasks (GSM8K).

Usage:
    python scripts/cache_datasets.py --cache_dir ./data --model_cache_dir ./models_cache
"""

import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


# All GLUE tasks
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

# Common models for GLUE
GLUE_MODELS = [
    "FacebookAI/roberta-large",
    "FacebookAI/roberta-base",
    "bert-base-uncased",
    "bert-large-uncased",
]

# LLM models for GSM8K
LLM_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
]


def cache_glue_datasets(cache_dir: str):
    """Download and cache all GLUE datasets."""
    print("=" * 60)
    print("Caching GLUE datasets...")
    print("=" * 60)
    
    for task in GLUE_TASKS:
        print(f"\nDownloading GLUE task: {task}")
        try:
            # Use the full repository path 'nyu-mll/glue' for proper resolution
            dataset = load_dataset(
                "nyu-mll/glue", 
                task, 
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            print(f"  ✓ {task} cached successfully")
            print(f"    Train size: {len(dataset['train'])}")
            if 'validation' in dataset:
                print(f"    Validation size: {len(dataset['validation'])}")
        except Exception as e:
            print(f"  ✗ Error caching {task}: {e}")


def cache_gsm8k_dataset(cache_dir: str):
    """Download and cache GSM8K dataset."""
    print("\n" + "=" * 60)
    print("Caching GSM8K dataset...")
    print("=" * 60)
    
    try:
        # GSM8K from HuggingFace datasets
        dataset = load_dataset(
            "gsm8k", 
            "main",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("  ✓ GSM8K cached successfully")
        print(f"    Train size: {len(dataset['train'])}")
        print(f"    Test size: {len(dataset['test'])}")
    except Exception as e:
        print(f"  Note: GSM8K HuggingFace version: {e}")
        
    # Also download the raw JSONL files used by the LLM dataloader
    from federatedscope.core.data.utils import download_url
    
    train_fp = os.path.join(cache_dir, 'gsm8k_train.jsonl')
    test_fp = os.path.join(cache_dir, 'gsm8k_test.jsonl')
    
    if not os.path.exists(train_fp):
        print("  Downloading GSM8K train JSONL...")
        download_url(
            'https://raw.githubusercontent.com/openai/grade-school-math'
            '/3101c7d5072418e28b9008a6636bde82a006892c/'
            'grade_school_math/data/train.jsonl', cache_dir)
        os.rename(os.path.join(cache_dir, 'train.jsonl'), train_fp)
        print("  ✓ GSM8K train JSONL cached")
    else:
        print("  ✓ GSM8K train JSONL already cached")
        
    if not os.path.exists(test_fp):
        print("  Downloading GSM8K test JSONL...")
        download_url(
            'https://raw.githubusercontent.com/openai/'
            'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'
            'grade_school_math/data/test.jsonl', cache_dir)
        os.rename(os.path.join(cache_dir, 'test.jsonl'), test_fp)
        print("  ✓ GSM8K test JSONL cached")
    else:
        print("  ✓ GSM8K test JSONL already cached")


def cache_tokenizers_and_models(model_cache_dir: str, models: list, model_type: str = "sequence_classification"):
    """Download and cache tokenizers and models."""
    print("\n" + "=" * 60)
    print(f"Caching tokenizers and models ({model_type})...")
    print("=" * 60)
    
    for model_name in models:
        print(f"\nCaching model: {model_name}")
        try:
            # Cache tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=model_cache_dir,
                trust_remote_code=True
            )
            print(f"  ✓ Tokenizer cached")
            
            # Cache model (optional - can be large)
            if model_type == "sequence_classification":
                # Don't actually load the full model to save memory, just cache config
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    model_name,
                    cache_dir=model_cache_dir,
                    trust_remote_code=True
                )
                print(f"  ✓ Model config cached")
            elif model_type == "causal_lm":
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    model_name,
                    cache_dir=model_cache_dir,
                    trust_remote_code=True
                )
                print(f"  ✓ Model config cached")
                
        except Exception as e:
            print(f"  ✗ Error caching {model_name}: {e}")


def cache_evaluation_metrics():
    """Download and cache evaluation metrics."""
    print("\n" + "=" * 60)
    print("Caching evaluation metrics...")
    print("=" * 60)
    
    try:
        import evaluate
        
        metrics = ["accuracy", "f1", "matthews_correlation", "pearsonr", "spearmanr"]
        for metric_name in metrics:
            try:
                metric = evaluate.load(metric_name)
                print(f"  ✓ {metric_name} cached")
            except Exception as e:
                print(f"  ✗ Error caching {metric_name}: {e}")
                
        # GLUE metric
        try:
            glue_metric = evaluate.load("glue", "mrpc")
            print(f"  ✓ glue metric cached")
        except Exception as e:
            print(f"  Note: glue metric: {e}")
            
    except ImportError:
        print("  Note: 'evaluate' package not installed, skipping metrics caching")


def main():
    parser = argparse.ArgumentParser(description="Cache datasets, models, and tokenizers")
    parser.add_argument("--cache_dir", type=str, default="./data", 
                        help="Directory to cache datasets")
    parser.add_argument("--model_cache_dir", type=str, default="./models_cache",
                        help="Directory to cache models and tokenizers")
    parser.add_argument("--skip_glue", action="store_true",
                        help="Skip GLUE dataset caching")
    parser.add_argument("--skip_gsm8k", action="store_true", 
                        help="Skip GSM8K dataset caching")
    parser.add_argument("--skip_models", action="store_true",
                        help="Skip model/tokenizer caching")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip evaluation metrics caching")
    parser.add_argument("--glue_models", type=str, nargs="+", default=GLUE_MODELS,
                        help="GLUE models to cache")
    parser.add_argument("--llm_models", type=str, nargs="+", default=LLM_MODELS,
                        help="LLM models to cache")
    
    args = parser.parse_args()
    
    # Create cache directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.model_cache_dir, exist_ok=True)
    
    print(f"Dataset cache directory: {args.cache_dir}")
    print(f"Model cache directory: {args.model_cache_dir}")
    
    # Cache datasets
    if not args.skip_glue:
        cache_glue_datasets(args.cache_dir)
    
    if not args.skip_gsm8k:
        cache_gsm8k_dataset(args.cache_dir)
    
    # Cache models and tokenizers
    if not args.skip_models:
        cache_tokenizers_and_models(args.model_cache_dir, args.glue_models, "sequence_classification")
        cache_tokenizers_and_models(args.model_cache_dir, args.llm_models, "causal_lm")
    
    # Cache evaluation metrics
    if not args.skip_metrics:
        cache_evaluation_metrics()
    
    print("\n" + "=" * 60)
    print("Caching complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
