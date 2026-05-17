import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

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

    print(f"Loading dataset split train[:2000000] from {args.dataset} (streaming=False for parallel download)...")
    
    # Load dataset slice non-streaming to download required parquet files in parallel at max network speed (~150MB/s)
    # 2,000,000 rows will easily yield 2.0+ Billion tokens of high-quality FineWeb-Edu text.
    dataset = load_dataset(args.dataset, args.dataset_name, split="train[:2000000]", streaming=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing shards to resume
    existing_shards = sorted(args.output_dir.glob("tokens_*.pt"))
    tokens_saved = 0
    if existing_shards:
        for shard_file in existing_shards:
            try:
                payload = torch.load(shard_file, map_location="cpu")
                tokens_saved += payload["tokens"].numel()
            except Exception as e:
                print(f"Warning: Failed to load existing shard {shard_file}: {e}")
        print(f"\n--- [RESUME] Found {len(existing_shards)} existing shards on disk ({tokens_saved:,} tokens). Resuming... ---\n")

    # True multi-processed tokenization using PyArrow/C++ and Rust FastTokenizer
    print(f"Tokenizing dataset in parallel using {args.num_proc} processes...")
    def tokenize_func(batch):
        return tokenizer(batch[args.text_column], add_special_tokens=False)
        
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=True,
        batch_size=5000,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names
    )
    
    print("\n--- Parallel tokenization complete! Writing shards to disk... ---\n")

    tokens_seen = 0
    shard_idx = 0
    shard_tokens = []
    shard_token_count = 0
    start_time = time.perf_counter()
    
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
        temp_filename = f"{filename}.tmp"
        torch.save({"tokens": tensor, **shard_meta}, args.output_dir / temp_filename)
        (args.output_dir / temp_filename).replace(args.output_dir / filename)
        with open(args.output_dir / f"{filename}.json", "w") as f:
            json.dump(shard_meta, f)
        return tensor.numel()

    for row in tokenized_dataset:
        ids = row["input_ids"]
        if args.append_eos:
            ids.append(tokenizer.eos_token_id)
            
        count = len(ids)
        if tokens_seen < tokens_saved:
            tokens_seen += count
            continue
            
        shard_tokens.append(ids)
        shard_token_count += count
        tokens_seen += count
        
        if shard_token_count >= args.shard_tokens:
            save_shard(shard_tokens, shard_idx)
            elapsed = time.perf_counter() - start_time
            tok_s = (tokens_seen - tokens_saved) / max(elapsed, 1e-6)
            percentage = (tokens_seen / args.target_tokens) * 100.0
            print(f"[SHARD WRITTEN] Saved shard {shard_idx} ({shard_token_count:,} tokens). Total: {tokens_seen:,}/{args.target_tokens:,} tokens ({percentage:.1f}%) | Speed: {tok_s:.0f} tok/s")
            shard_idx += 1
            shard_tokens = []
            shard_token_count = 0
            
        if tokens_seen >= args.target_tokens:
            break
            
    if shard_tokens and tokens_seen < args.target_tokens:
        save_shard(shard_tokens, shard_idx)
        shard_idx += 1
        
    # Write manifest
    manifest = {
        "format": "cmf.token_cache_dir.v1",
        "tokens_count": tokens_seen,
        "shards": [{"path": f"tokens_{i:06d}.pt", "tokens_count": args.shard_tokens if i < shard_idx - 1 else shard_token_count} for i in range(shard_idx)]
    }
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    print(f"\n--- [COMPLETE] Total tokens prepared: {tokens_seen:,} ---")

if __name__ == "__main__":
    main()
