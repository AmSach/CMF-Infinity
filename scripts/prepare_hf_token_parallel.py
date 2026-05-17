import os
import sys
import time
import json
import argparse
import multiprocessing
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

def tokenize_batch(batch, tokenizer, text_column, append_eos):
    texts = batch[text_column]
    tokenized = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    if append_eos:
        eos_id = tokenizer.eos_token_id
        for ids in tokenized:
            ids.append(eos_id)
    return {"tokens": tokenized}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceTB/smollm-corpus")
    parser.add_argument("--dataset-name", default="fineweb-edu-dedup")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, default=os.cpu_count())
    parser.add_argument("--append-eos", action="store_true")
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset}...")
    # Using streaming=True but with a large buffer to allow some parallelization 
    # or just non-streaming if we have enough RAM (Kaggle has 30GB).
    # For 10B tokens, we better use streaming to avoid OOM during download.
    dataset = load_dataset(args.dataset, args.dataset_name, split=args.split, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    tokens_seen = 0
    shard_idx = 0
    shard_tokens = []
    
    start_time = time.perf_counter()
    
    # We'll pull in batches and tokenize them
    batch_size = 1000
    current_batch = []
    
    print(f"Starting parallel tokenization with {args.num_proc} processes...")
    
    def save_shard(tokens_list, idx):
        flat_tokens = [t for sublist in tokens_list for t in sublist]
        tensor = torch.tensor(flat_tokens, dtype=torch.int32)
        filename = f"tokens_{idx:06d}.pt"
        shard_meta = {
            "format": "cmf.token_cache_shard.v1",
            "path": filename,
            "shard_index": idx,
            "tokens_count": tensor.numel(),
            "dtype": "int32"
        }
        torch.save({"tokens": tensor, **shard_meta}, args.output_dir / filename)
        with open(args.output_dir / f"{filename}.json", "w") as f:
            json.dump(shard_meta, f)
        return tensor.numel()

    shard_token_count = 0
    for row in dataset:
        current_batch.append(row[args.text_column])
        
        if len(current_batch) >= batch_size:
            # Tokenize current batch
            tokenized = tokenizer(current_batch, add_special_tokens=False)["input_ids"]
            for ids in tokenized:
                if args.append_eos:
                    ids.append(tokenizer.eos_token_id)
                shard_tokens.append(ids)
                count = len(ids)
                shard_token_count += count
                tokens_seen += count
                
                if shard_token_count >= args.shard_tokens:
                    save_shard(shard_tokens, shard_idx)
                    print(f"Wrote shard {shard_idx}, tokens={tokens_seen:,}, tok/s={tokens_seen/(time.perf_counter()-start_time):.0f}")
                    shard_idx += 1
                    shard_tokens = []
                    shard_token_count = 0
            
            current_batch = []
            if tokens_seen >= args.target_tokens:
                break
                
    if shard_tokens:
        save_shard(shard_tokens, shard_idx)
        
    # Write manifest
    manifest = {
        "format": "cmf.token_cache_dir.v1",
        "tokens_count": tokens_seen,
        "shards": [{"path": f"tokens_{i:06d}.pt", "tokens_count": args.shard_tokens} for i in range(shard_idx + 1)]
    }
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    print(f"Total tokens prepared: {tokens_seen:,}")

if __name__ == "__main__":
    main()
