"""
CMF-Infinity AGI Pre-Training Pipeline (Parallel Streaming & Concurrent Training)
=================================================================================
- Streams multiple HF datasets in parallel using multi-threaded downloaders (max NIC speed).
- Queues downloaded texts to a thread-safe buffer queue (memory-capped to prevent OOM).
- Tokenizes background data and writes .pt shards.
- Deletes consumed shards automatically to preserve disk limits.
- Launches tokenizer in background and trainer in foreground concurrently.
"""

import os
import json
import random
import subprocess
import sys
import time
import torch
import threading
import queue

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/AmSach/CMF-Infinity.git"
WORKSPACE_DIR = "/kaggle/working"
CMF_DIR = os.path.join(WORKSPACE_DIR, "CMF")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "agi_shards")

TARGET_TOKENS = 2_500_000_000
SHARD_TOKENS = 25_000_000  # 25M tokens per shard

# ─────────────────────────────────────────────────────────────────────────────
# DATASET MIX CONFIGURATION (Direct HF Streaming)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_MIX = [
    # (hf_id, subset, split, weight, formatter)
    # SlimPajama requires zstandard to decompress stream files
    ("cerebras/SlimPajama-627B",                   None,                    "train", 2,
     lambda r: r.get("text", "")),
    # Correct path is FinanceInc/auditor_sentiment
    ("FinanceInc/auditor_sentiment",               None,                    "train", 1,
     lambda r: f"User: Analyze this financial statement.\nAssistant: {r.get('sentence', '')}"),
    ("wikimedia/wikipedia",                        "20231101.en",           "train", 1,
     lambda r: r.get("text", "")),
    # Correct path is Qwen/Qwen2.5-Math-1.1M-CoT
    ("Qwen/Qwen2.5-Math-1.1M-CoT",                 None,                    "train", 1,
     lambda r: f"User: {r.get('problem', '')}\nAssistant: {r.get('cot_content', '')}"),
    ("HuggingFaceH4/CodeAlpaca_20K",               None,                    "train", 1,
     lambda r: f"User: {r.get('prompt', '')}\nAssistant: {r.get('completion', '')}"),
    ("teknium/OpenHermes-2.5",                     None,                    "train", 1,
     lambda r: (
         f"User: {r['conversations'][0]['value']}\nAssistant: {r['conversations'][1]['value']}"
         if len(r.get("conversations", [])) >= 2 else ""
     )),
]

def setup_environment():
    print("=" * 60)
    print("0. Environment Setup & Repo Sync")
    print("=" * 60)
    if os.path.exists(CMF_DIR):
        print("Pulling latest architectural updates...")
        subprocess.run(["git", "-C", CMF_DIR, "pull"], check=True)
    else:
        print(f"Cloning codebase from {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, CMF_DIR], check=True)
    
    # Must install zstandard for SlimPajama streaming decompressions
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "transformers", "tiktoken", "accelerate", "zstandard"], check=True)
    print("All packages installed.\n")

def tokenization_worker():
    """Background worker that streams directly from HF, tokenizes, and saves shards.
    Uses a single-threaded round-robin approach with active garbage collection to minimize RAM usage.
    """
    print("=" * 60)
    print("1. Background Tokenizer Initializing")
    print("=" * 60)
    import gc
    from transformers import AutoTokenizer
    from datasets import load_dataset

    os.makedirs(CACHE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id

    # Initialize streams in a single thread to save network socket and buffer memory
    iterators = []
    for hf_id, subset, split, weight, fmt in DATASET_MIX:
        try:
            kwargs = {"split": split, "streaming": True}
            if subset:
                it = iter(load_dataset(hf_id, subset, **kwargs))
            else:
                it = iter(load_dataset(hf_id, **kwargs))
            iterators.append((it, fmt, weight, hf_id))
        except Exception as e:
            print(f"Warning: Failed to load dataset {hf_id}: {e}")

    if not iterators:
        print("Error: No datasets successfully loaded. Aborting background tokenizer.")
        return

    print(f"[Tokenizer] Active round-robin stream loaded. Preparing shards...")

    tokens_seen = 0
    shard_idx = 0
    buffer = []
    manifest_shards = []

    def save_shard(toks, idx):
        tensor = torch.tensor(toks[:SHARD_TOKENS], dtype=torch.int32)
        fname = f"tokens_{idx:06d}.pt"
        meta = {
            "format": "cmf.token_cache_shard.v1",
            "path": fname,
            "shard_index": idx,
            "tokens_count": len(toks[:SHARD_TOKENS]),
        }
        torch.save({**meta, "tokens": tensor}, os.path.join(CACHE_DIR, fname))
        with open(os.path.join(CACHE_DIR, f"{fname}.json"), "w") as mf:
            json.dump(meta, mf)
        
        manifest_shards.append({"path": fname, "tokens_count": len(toks[:SHARD_TOKENS])})
        # Write/Update manifest.json
        manifest = {
            "format": "cmf.token_cache_dir.v1",
            "complete": tokens_seen >= TARGET_TOKENS,
            "tokens_count": tokens_seen,
            "shards": manifest_shards,
            "tokenizer": {"type": "hf_auto", "name": "gpt2", "vocab_size": tokenizer.vocab_size},
        }
        with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[Tokenizer] Saved shard {fname} ({len(toks[:SHARD_TOKENS]):,} tokens). Total: {tokens_seen:,}/{TARGET_TOKENS:,}")
        
        # Explicitly clean up memory
        del tensor
        gc.collect()
        return toks[SHARD_TOKENS:]

    active_iterators = list(iterators)
    
    while active_iterators and tokens_seen < TARGET_TOKENS:
        next_active = []
        for it, fmt, weight, hf_id in active_iterators:
            try:
                # Retrieve records based on the dataset ratio weight
                for _ in range(weight):
                    row = next(it)
                    text = fmt(row)
                    if text and text.strip():
                        encoded = tokenizer.encode(text, add_special_tokens=False) + [eos_id]
                        buffer.extend(encoded)
                        tokens_seen += len(encoded)

                        while len(buffer) >= SHARD_TOKENS:
                            buffer = save_shard(buffer, shard_idx)
                            shard_idx += 1
                next_active.append((it, fmt, weight, hf_id))
            except StopIteration:
                print(f"[Tokenizer] Dataset {hf_id} fully consumed.")
            except Exception as e:
                # If there's a temporary connection hiccup, retain the stream
                next_active.append((it, fmt, weight, hf_id))
                time.sleep(0.5)
        
        active_iterators = next_active

    if buffer and tokens_seen < TARGET_TOKENS:
        save_shard(buffer, shard_idx)
        shard_idx += 1

    # Mark manifest complete
    manifest = {
        "format": "cmf.token_cache_dir.v1",
        "complete": True,
        "tokens_count": tokens_seen,
        "shards": manifest_shards,
        "tokenizer": {"type": "hf_auto", "name": "gpt2", "vocab_size": tokenizer.vocab_size},
    }
    with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Tokenizer] Finished streaming. Total tokens prepared: {tokens_seen:,}")

def find_free_port():
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    except Exception:
        return random.randint(25000, 29999)

def launch_training():
    print("=" * 60)
    print("2. Launching Concurrently Distributed Trainer")
    print("=" * 60)
    os.chdir(CMF_DIR)

    port = find_free_port()
    print(f"Using dynamically allocated master port: {port}")

    # Reduce micro-batch-size from 16 to 2 and increase grad-accum to 32
    # This maintains the exact same global batch size (128) but uses 8x less VRAM!
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        f"--master_port={port}",
        "scripts/train_distributed.py",
        "--preset",                 "infinity-reasoning-0.12b",
        "--token-cache-dir",        CACHE_DIR,
        "--seq-len",                "1024",
        "--micro-batch-size",       "2",
        "--grad-accum",             "32",
        "--steps",                  "25000",
        "--save-every",             "10",
        "--checkpoint-dir",         WORKSPACE_DIR,
        "--package-out",            os.path.join(WORKSPACE_DIR, "cmf_agi_120m_final.package.pt"),
        "--gradient-checkpointing",
        "--fsdp",
        "--amp",
        "--tf32",
        "--apply-sft-mask",
        "--ponder-weight",          "0.05",
        "--warmup-steps",           "500",
        "--min-lr-ratio",           "0.05",
        "--clip-grad-norm",         "1.0",
        "--empty-cache-every",      "50",
        "--log-every",              "1",
        "--delete-consumed-shards",
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    # Configure env to prevent VRAM fragmentation as recommended by PyTorch
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    # Run torchrun with live streaming output to catch all errors immediately
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    # Read output line-by-line as it prints
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

if __name__ == "__main__":
    setup_environment()

    # Start the orchestrator background thread
    t_thread = threading.Thread(target=tokenization_worker, daemon=True)
    t_thread.start()

    # Give the download threads 15 seconds to connect, download, and write the first shard
    print("Waiting for tokenizer to begin writing...")
    time.sleep(15.0)

    # Launch the trainer (it blocks/waits if tokens_000000.pt isn't fully written yet)
    launch_training()
