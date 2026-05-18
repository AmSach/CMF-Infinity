import os
import sys
import time
import json
import argparse
import queue
import threading
from pathlib import Path

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

# Dedicated high-speed background pre-fetch worker
def dataset_producer(cfg: dict, q: queue.Queue):
    while True:
        try:
            # Streams chunks asynchronously from HF servers
            ds = load_dataset(
                cfg["path"],
                cfg["name_subset"],
                split="train",
                streaming=True
            )
            col = cfg["text_column"]
            for row in ds:
                text = row.get(col, "")
                if text and len(text.strip()) > 0:
                    # Blocks automatically when the queue hits capacity to prevent RAM exhaustion
                    q.put(text, block=True)
            print(f"\n[Producer INFO] Stream {cfg['name']} reached end. Loop restarting...")
        except Exception as e:
            # Automatic fault-tolerant reconnect on network glitches
            print(f"\n[Producer Reconnect] Stream {cfg['name']} encountered a connection hiccup: {e}. Retrying in 2 seconds...")
            time.sleep(2)

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
        print(f"\n--- [RESUME] Found {len(existing_shards)} shards on disk ({tokens_saved:,} tokens). Fast-forwarding active mix... ---\n")

    # 2. Spawn dedicated background worker threads for each dataset stream
    print("--- [ASYNCHRONOUS ENGINE] Launching Multithreaded Pre-Fetching Queues ---")
    queues = {}
    threads = []
    for cfg in DATASET_CONFIGS:
        # Buffer up to 30,000 texts in RAM per stream to completely absorb internet latency spikes
        q = queue.Queue(maxsize=30000)
        queues[cfg["name"]] = q
        
        t = threading.Thread(
            target=dataset_producer, 
            args=(cfg, q), 
            name=f"producer_{cfg['name']}",
            daemon=True
        )
        t.start()
        threads.append(t)
        print(f"   -> Spawned background pre-fetch thread: {t.name} (Max Queue: 30,000 docs)")

    tokens_seen = 0
    shard_idx = 0
    shard_tokens = []
    shard_token_count = 0
    
    skipped_tokens = 0
    skipping_mode = (tokens_saved > 0)
    
    start_time = time.perf_counter()
    
    # High-throughput batch size for Rust parallel tokenizer
    batch_size = 4000
    
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

    print("\n--- [BLAZING FAST] Active Tokenization Loop Engaged ---\n")
    
    # Wait briefly for queues to warm up and buffer initial content
    print("Pre-warming queues (waiting 5 seconds)...")
    time.sleep(5)
    
    while tokens_seen < args.target_tokens:
        batch_texts = []
        
        # Interleave records based on ratios from pre-fetched queues
        for cfg in DATASET_CONFIGS:
            num_items = max(1, int(batch_size * cfg["ratio"]))
            q = queues[cfg["name"]]
            
            for _ in range(num_items):
                try:
                    # Fast non-blocking pull
                    text = q.get_nowait()
                    batch_texts.append(text)
                except queue.Empty:
                    # If empty, move to next queue to prevent stalling the loop
                    break
        
        if not batch_texts:
            # Queues momentarily starving; sleep briefly to allow producers to buffer
            time.sleep(0.05)
            continue
            
        # Re-batch to maximize Rust parallel tokenization throughput
        tokenized = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        
        # High-speed parallel fast-skipping resume logic
        if skipping_mode:
            batch_counts = [len(ids) + (1 if args.append_eos else 0) for ids in tokenized]
            sum_counts = sum(batch_counts)
            if skipped_tokens + sum_counts < tokens_saved:
                skipped_tokens += sum_counts
                pct = (skipped_tokens / tokens_saved) * 100.0
                print(f"[Asynchronous Resume] Skipping cached tokens... {skipped_tokens:,}/{tokens_saved:,} ({pct:.1f}%)", end="\r", flush=True)
                continue
                
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
                    print(f"\n--- [RESUME COMPLETE] Asynchronous queues synchronized. Starting token cache at Shard {shard_idx}! ---\n")
                continue
            
            shard_tokens.append(ids)
            shard_token_count += count
            tokens_seen += count
            
            if shard_token_count >= args.shard_tokens:
                save_shard(shard_tokens, shard_idx)
                print(f"\n[SHARD WRITE SUCCESS] Shard {shard_idx} saved ({shard_token_count:,} tokens). Total processed: {tokens_seen:,}/{args.target_tokens:,}.\n")
                shard_idx += 1
                shard_tokens = []
                shard_token_count = 0
                
        if not skipping_mode:
            elapsed = time.perf_counter() - start_time
            tok_s = (tokens_seen - tokens_saved) / max(elapsed, 1e-6)
            percentage = (tokens_seen / args.target_tokens) * 100.0
            
            # Queue fills tracking to monitor thread efficiency
            queue_stats = ", ".join([f"{cfg['name']}: {queues[cfg['name']].qsize()}" for cfg in DATASET_CONFIGS])
            
            remaining_tokens = max(0, args.target_tokens - tokens_seen)
            eta_seconds = remaining_tokens / max(tok_s, 1e-6)
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds / 3600:.2f} hrs"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds / 60:.2f} mins"
            else:
                eta_str = f"{eta_seconds:.0f} secs"
                
            print(f"[Asynchronous Mixer] {tokens_seen:,} / {args.target_tokens:,} tokens ({percentage:.4f}%) | "
                  f"Shard {shard_idx}: {shard_token_count:,}/{args.shard_tokens:,} | "
                  f"Speed: {tok_s:.0f} tok/s | ETA: {eta_str} | Queues ({queue_stats})", end="\r", flush=True)

    if shard_tokens and tokens_seen < args.target_tokens:
        save_shard(shard_tokens, shard_idx)
        shard_idx += 1
        
    # Final CMF manifest
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
