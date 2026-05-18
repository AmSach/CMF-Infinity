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
    parser.add_argument("--max-ahead", type=int, default=0, help="Limit tokenization to at most N shards ahead of currently training shard to save disk space (0 or negative to disable)")
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset} (streaming=True for instant startup)...")
    
    # Load dataset streaming to fetch only required files on-the-fly, bypassing the 350GB download choke
    dataset = load_dataset(args.dataset, args.dataset_name, split=args.split, streaming=True)
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
        print(f"\n--- [RESUME] Found {len(existing_shards)} existing shards on disk ({tokens_saved:,} tokens). Fast-forwarding generator... ---\n")

    tokens_seen = 0
    shard_idx = 0
    shard_tokens = []
    
    start_time = time.perf_counter()
    
    # High-speed batch processing leverages HF Tokenizer's internal C++/Rust parallel threads
    batch_size = 5000
    current_batch = []
    
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

    shard_token_count = 0
    skipped_tokens = 0
    skipping_mode = (tokens_saved > 0)
    
    for row in dataset:
        current_batch.append(row[args.text_column])
        
        if len(current_batch) >= batch_size:
            # Adaptive Disk Throttle:
            # Check if the tokenizer is too far ahead of the trainer.
            if args.max_ahead > 0:
                while True:
                    existing_shards = sorted(args.output_dir.glob("tokens_*.pt"))
                    if existing_shards:
                        indices = []
                        for p in existing_shards:
                            parts = p.stem.split("_")
                            if len(parts) >= 2 and parts[-1].isdigit():
                                indices.append(int(parts[-1]))
                        if indices:
                            min_active = min(indices)
                            if shard_idx - min_active >= args.max_ahead:
                                print(f"[Adaptive Throttle] Shard {shard_idx} is {shard_idx - min_active} shards ahead of training (min active: {min_active}). Pausing downloader to save disk space...", end="\r", flush=True)
                                time.sleep(2)
                                continue
                    break

            # Tokenize batch in parallel via HF Tokenizer Rust multi-threading
            tokenized = tokenizer(current_batch, add_special_tokens=False)["input_ids"]
            
            # Ultra-fast batch skipping to bypass Python row loop bottleneck
            if skipping_mode:
                batch_counts = [len(ids) + (1 if args.append_eos else 0) for ids in tokenized]
                sum_counts = sum(batch_counts)
                if skipped_tokens + sum_counts < tokens_saved:
                    skipped_tokens += sum_counts
                    current_batch = []
                    
                    # Print fast-forward logs occasionally
                    pct = (skipped_tokens / tokens_saved) * 100.0
                    print(f"[Fast-Forward] Skipping already processed data... {skipped_tokens:,} / {tokens_saved:,} tokens skipped ({pct:.1f}%)", end="\r", flush=True)
                    continue
            
            # Boundary transition or active tokenization
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
                        print(f"\n--- [RESUME COMPLETE] Fast-forwarded past {tokens_saved:,} tokens. Resuming tokenization at Shard {shard_idx}! ---\n")
                    continue
                
                shard_tokens.append(ids)
                shard_token_count += count
                tokens_seen += count
                
                if shard_token_count >= args.shard_tokens:
                    save_shard(shard_tokens, shard_idx)
                    print(f"\n[SHARD WRITTEN] Saved shard {shard_idx} ({shard_token_count:,} tokens). Total: {tokens_seen:,}/{args.target_tokens:,} tokens.\n")
                    shard_idx += 1
                    shard_tokens = []
                    shard_token_count = 0
            
            # Print real-time progress update
            if not skipping_mode:
                elapsed = time.perf_counter() - start_time
                tok_s = (tokens_seen - tokens_saved) / max(elapsed, 1e-6)
                percentage = (tokens_seen / args.target_tokens) * 100.0
                
                # Estimate ETA
                remaining_tokens = max(0, args.target_tokens - tokens_seen)
                eta_seconds = remaining_tokens / max(tok_s, 1e-6)
                
                if eta_seconds > 3600:
                    eta_str = f"{eta_seconds / 3600:.2f} hrs"
                elif eta_seconds > 60:
                    eta_str = f"{eta_seconds / 60:.2f} mins"
                else:
                    eta_str = f"{eta_seconds:.0f} secs"
                    
                print(f"[Progress] {tokens_seen:,} / {args.target_tokens:,} tokens ({percentage:.4f}%) | "
                      f"Shard {shard_idx} progress: {shard_token_count:,}/{args.shard_tokens:,} | "
                      f"Speed: {tok_s:.0f} tok/s | ETA: {eta_str}", end="\r", flush=True)
            
            current_batch = []
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
