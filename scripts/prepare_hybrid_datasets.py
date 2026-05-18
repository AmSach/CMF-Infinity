import os
import sys
import time
import json
import argparse
from pathlib import Path
import itertools

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

# The Master AGI Recipe Mix (Expanded with strict ratio balance)
DATASET_CONFIGS = [
    {
        "name": "fineweb-edu",
        "path": "HuggingFaceTB/smollm-corpus",
        "name_subset": "fineweb-edu-dedup",
        "text_column": "text",
        "ratio": 0.35, # 35% general high-density educational knowledge
    },
    {
        "name": "cosmopedia-v2",
        "path": "HuggingFaceTB/cosmopedia-v2",
        "name_subset": None,
        "text_column": "text",
        "ratio": 0.25, # 25% synthetic textbooks and courses
    },
    {
        "name": "stack-code",
        "path": "HuggingFaceTB/smollm-corpus",
        "name_subset": "stack-edu-dedup",
        "text_column": "text",
        "ratio": 0.15, # 15% clean algorithm logic
    },
    {
        "name": "open-web-math",
        "path": "open-web-math/open-web-math",
        "name_subset": None,
        "text_column": "text",
        "ratio": 0.10, # 10% rigorous LaTeX mathematical text
    },
    {
        "name": "proof-pile-2",
        "path": "EleutherAI/proof-pile-2",
        "name_subset": "algebraic-stack",
        "text_column": "text",
        "ratio": 0.10, # 10% scientific papers & formal proofs
    },
    {
        "name": "math-cot-reasoning",
        "path": "Qwen/Qwen2.5-Math-1.1M-CoT",
        "name_subset": None,
        "text_column": "cot_content", # Uses explicit step-by-step thinking traces
        "ratio": 0.05, # 5% high-intensity chain-of-thought planning
    }
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--shard-tokens", type=int, default=25_000_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--append-eos", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # 1. Check for existing shards to resume tokenization
    existing_shards = sorted(args.output_dir.glob("tokens_*.pt"))
    tokens_saved = 0
    if existing_shards:
        for shard_file in existing_shards:
            try:
                payload = torch.load(shard_file, map_location="cpu")
                tokens_saved += payload["tokens"].numel()
            except Exception as e:
                print(f"Warning: Failed to load existing shard {shard_file}: {e}")
        print(f"\n--- [RESUME] Found {len(existing_shards)} existing shards on disk ({tokens_saved:,} tokens). Fast-forwarding mix generator... ---\n")

    # 2. Setup streaming dataset iterators
    print("--- [STREAMING INITIALIZATION] Opening AGI-Recipe Datasets ---")
    iterators = {}
    for cfg in DATASET_CONFIGS:
        print(f"Streaming {cfg['name']} from {cfg['path']} (Ratio: {cfg['ratio']:.0%})...")
        try:
            ds = load_dataset(
                cfg["path"],
                cfg["name_subset"],
                split="train",
                streaming=True
            )
            iterators[cfg["name"]] = iter(ds)
        except Exception as e:
            # Fallback for Qwen dataset structure if needed
            print(f"Note: Standard split load failed for {cfg['name']}, trying default load: {e}")
            ds = load_dataset(cfg["path"], split="train", streaming=True)
            iterators[cfg["name"]] = iter(ds)

    # Calculate token allocation targets per dataset config
    tokens_per_source = {cfg["name"]: int(args.target_tokens * cfg["ratio"]) for cfg in DATASET_CONFIGS}
    
    tokens_seen = 0
    shard_idx = 0
    shard_tokens = []
    shard_token_count = 0
    
    skipped_tokens = 0
    skipping_mode = (tokens_saved > 0)
    
    start_time = time.perf_counter()
    
    # We round-robin through datasets based on target ratios to keep a uniform sequence mix
    # This prevents training from seeing all math, then all code, then all textbooks
    cycle_index = 0
    batch_size = 2000
    
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

    print("\n--- Starting Parallel Tokenization and Interleaved Mixing ---\n")
    
    while tokens_seen < args.target_tokens:
        batch_texts = []
        # Draw batch items proportionally from the round-robin mix
        for cfg in DATASET_CONFIGS:
            num_items = max(1, int(batch_size * cfg["ratio"]))
            iterator = iterators[cfg["name"]]
            col = cfg["text_column"]
            
            for _ in range(num_items):
                try:
                    row = next(iterator)
                    text = row.get(col, "")
                    if text:
                        batch_texts.append(text)
                except StopIteration:
                    # Reinstate iterator if exhausted to keep ratio stable
                    print(f"\n[INFO] Dataset {cfg['name']} exhausted. Restarting stream...")
                    ds = load_dataset(cfg["path"], cfg["name_subset"], split="train", streaming=True)
                    iterators[cfg["name"]] = iter(ds)
                    try:
                        row = next(iterators[cfg["name"]])
                        text = row.get(col, "")
                        if text:
                            batch_texts.append(text)
                    except Exception:
                        pass
        
        if not batch_texts:
            print("Warning: All dataset streams returned empty items. Retrying...")
            time.sleep(1)
            continue
            
        # Parallel Tokenize batch
        tokenized = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        
        # Fast skipping loop to resume instantly
        if skipping_mode:
            batch_counts = [len(ids) + (1 if args.append_eos else 0) for ids in tokenized]
            sum_counts = sum(batch_counts)
            if skipped_tokens + sum_counts < tokens_saved:
                skipped_tokens += sum_counts
                # Print status
                pct = (skipped_tokens / tokens_saved) * 100.0
                print(f"[Resume-Mix] Skipping already tokenized rows... {skipped_tokens:,}/{tokens_saved:,} ({pct:.1f}%)", end="\r", flush=True)
                continue
                
        # Processing active tokens
        for ids in tokenized:
            if args.append_eos:
                ids.append(tokenizer.eos_token_id)
            
            count = len(ids)
            if skipping_mode:
                skipped_tokens += count
                if skipped_tokens >= tokens_saved:
                    skipping_mode = False
                    tokens_seen = tokens_saved
                    shard_idx = len(existing_shards)
                    print(f"\n--- [RESUME COMPLETE] Mixed streams fast-forwarded successfully. Resuming active token cache generation at Shard {shard_idx}! ---\n")
                continue
            
            shard_tokens.append(ids)
            shard_token_count += count
            tokens_seen += count
            
            if shard_token_count >= args.shard_tokens:
                save_shard(shard_tokens, shard_idx)
                print(f"\n[SHARD SAVED] Shard {shard_idx} successfully written ({shard_token_count:,} tokens). Total processed: {tokens_seen:,}/{args.target_tokens:,} tokens.\n")
                shard_idx += 1
                shard_tokens = []
                shard_token_count = 0
                
        if not skipping_mode:
            elapsed = time.perf_counter() - start_time
            tok_s = (tokens_seen - tokens_saved) / max(elapsed, 1e-6)
            percentage = (tokens_seen / args.target_tokens) * 100.0
            
            remaining_tokens = max(0, args.target_tokens - tokens_seen)
            eta_seconds = remaining_tokens / max(tok_s, 1e-6)
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds / 3600:.2f} hrs"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds / 60:.2f} mins"
            else:
                eta_str = f"{eta_seconds:.0f} secs"
                
            print(f"[Mixer Progress] {tokens_seen:,} / {args.target_tokens:,} tokens ({percentage:.4f}%) | "
                  f"Shard {shard_idx} progress: {shard_token_count:,}/{args.shard_tokens:,} | "
                  f"Speed: {tok_s:.0f} tok/s | ETA: {eta_str}", end="\r", flush=True)

    if shard_tokens and tokens_seen < args.target_tokens:
        save_shard(shard_tokens, shard_idx)
        shard_idx += 1
        
    # Write CMF conformant manifest file
    manifest = {
        "format": "cmf.token_cache_dir.v1",
        "tokens_count": tokens_seen,
        "shards": [{"path": f"tokens_{i:06d}.pt", "tokens_count": args.shard_tokens if i < shard_idx - 1 else shard_token_count} for i in range(shard_idx)]
    }
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    print(f"\n--- [HYBRID COMPLETE] World-Class AGI-Recipe Token Cache Ready: {tokens_seen:,} tokens! ---")

if __name__ == "__main__":
    main()
