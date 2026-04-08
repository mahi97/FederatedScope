#!/usr/bin/env python
"""
Quick script to download tokenizer files for FacebookAI/roberta-large.
Run this when you have internet access.
"""
import os

# Set the cache directory
cache_dir = '/home/dwseo/scratch/mahi/hf_transformers'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

from transformers import AutoTokenizer

print("Downloading tokenizer for FacebookAI/roberta-large...")
tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/roberta-large",
    cache_dir=cache_dir
)
print(f"Tokenizer downloaded to: {cache_dir}")
print("Done! You can now run offline.")
