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
    # FineWeb: high-quality deduplicated web text (open access, replaces gated SlimPajama)
    ("HuggingFaceFW/fineweb",                      "sample-10BT",           "train", 2,
     lambda r: r.get("text", "")),
    # Financial domain knowledge
    ("FinanceInc/auditor_sentiment",               None,                    "train", 1,
     lambda r: f"User: Analyze this financial statement.\nAssistant: {r.get('sentence', '')}"),
    # Encyclopedic knowledge
    ("wikimedia/wikipedia",                        "20231101.en",           "train", 1,
     lambda r: r.get("text", "")),
    # NVIDIA OpenMathReasoning: open-access math CoT (replaces gated Qwen dataset)
    # Note: no 'train' split - available: 'cot', 'tir', 'genselect', 'additional_problems'
    ("nvidia/OpenMathReasoning",                   None,                    "cot",   1,
     lambda r: f"User: {r.get('problem', '')}\nAssistant: {r.get('generated_solution', '')}"),
    # Code instruction following
    ("HuggingFaceH4/CodeAlpaca_20K",               None,                    "train", 1,
     lambda r: f"User: {r.get('prompt', '')}\nAssistant: {r.get('completion', '')}"),
    # Multi-turn assistant chat (coding, logic, writing)
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
    
    # Check GPU visibility first
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs detected:", torch.cuda.device_count())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("\n" + "!"*80)
            print("CRITICAL WARNING: CUDA GPU IS NOT VISIBLE TO PYTORCH!")
            print("Please ensure that you have selected a GPU accelerator (e.g., 'GPU T4 x2')")
            print("in the Kaggle notebook settings sidebar under Accelerator.")
            print("!"*80 + "\n")
    except Exception as e:
        print(f"GPU Detection Warning: {e}")
        
    # Install dependencies without upgrading torch/torchvision to protect CUDA driver bindings
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "zstandard", "tiktoken"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "datasets", "transformers", "accelerate"], check=True)
    print("All packages installed.\n")

def tokenization_worker():
    """Background worker that streams directly from HF, tokenizes, and saves shards.
    To minimize RAM usage, it processes one dataset at a time, batches tokenization,
    and runs aggressive garbage collection.
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

    # List of datasets to iterate sequentially
    datasets_list = []
    for hf_id, subset, split, weight, fmt in DATASET_MIX:
        for _ in range(weight):
            datasets_list.append((hf_id, subset, split, fmt))

    dataset_idx = 0
    while tokens_seen < TARGET_TOKENS and datasets_list:
        hf_id, subset, split, fmt = datasets_list[dataset_idx % len(datasets_list)]
        print(f"[Tokenizer] Opening dataset stream: {hf_id} ({split})")
        
        try:
            kwargs = {"split": split, "streaming": True}
            if subset:
                ds = load_dataset(hf_id, subset, **kwargs)
            else:
                ds = load_dataset(hf_id, **kwargs)
                
            it = iter(ds)
            
            # Read a batch of records (e.g. 5,000,000 tokens worth) to keep RAM footprint low
            target_batch_tokens = 5_000_000
            batch_tokens = 0
            batch_texts = []
            
            while batch_tokens < target_batch_tokens and tokens_seen < TARGET_TOKENS:
                try:
                    row = next(it)
                    text = fmt(row)
                    if text and text.strip():
                        batch_texts.append(text)
                        batch_tokens += len(text.split())
                except StopIteration:
                    break
                except Exception as e:
                    print(f"[Tokenizer] Stream read warning: {e}")
                    time.sleep(1.0)
                    
            if batch_texts:
                # Fast batched tokenization
                encoded_batch = tokenizer(batch_texts, add_special_tokens=False)
                for enc in encoded_batch["input_ids"]:
                    buffer.extend(enc)
                    buffer.append(eos_id)
                    tokens_seen += len(enc) + 1
                    
                while len(buffer) >= SHARD_TOKENS:
                    buffer = save_shard(buffer, shard_idx)
                    shard_idx += 1
                    
            # Delete references and collect garbage to free Arrow/HTTP cache memory
            del it
            del ds
            del batch_texts
            del encoded_batch
            gc.collect()
            
        except Exception as e:
            print(f"[Tokenizer] Failed to process dataset {hf_id}: {e}")
            time.sleep(2.0)
            
        dataset_idx += 1

    # Save remaining tokens
    if buffer:
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
