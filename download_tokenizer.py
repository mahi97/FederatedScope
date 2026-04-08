#!/usr/bin/env python
"""
Quick script to download tokenizer files for FacebookAI/roberta-large.
Run this when you have internet access.
Prefer using `bash cache_offline.sh` which downloads everything.
"""
import os

# Resolve cache directory from env (set by .env.online or STORAGE_ROOT)
storage_root = os.environ.get('STORAGE_ROOT', '/drive1/mahi')
cache_dir = os.environ.get('HF_HOME', os.path.join(storage_root, 'huggingface'))
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(cache_dir, 'hub'))
os.environ.setdefault('HF_HOME', cache_dir)

from transformers import AutoTokenizer

print("Downloading tokenizer for FacebookAI/roberta-large...")
tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/roberta-large",
    cache_dir=cache_dir
)
print(f"Tokenizer downloaded to: {cache_dir}")
print("Done! You can now run offline.")
