"""
CMF-Infinity AGI Pre-Training Pipeline (Direct Streaming & Concurrent Training)
================================───────────────────────────────────────────────
- Streams directly from HF to memory, tokenizes, and saves .pt shards (no raw JSONL file).
- Deletes consumed shards automatically to maintain < 1GB disk footprint.
- Launches tokenizer in the background and trainer in the foreground concurrently.
- Training starts immediately as soon as the first shard is written.
- Checkpoints saved directly to /kaggle/working root every 10 steps.
"""

import os
import json
import random
import subprocess
import sys
import time
import torch
import threading

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/YOUR_USERNAME/CMF.git"
WORKSPACE_DIR = "/kaggle/working"
CMF_DIR = os.path.join(WORKSPACE_DIR, "CMF")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "agi_shards")

TARGET_TOKENS = 2_500_000_000
SHARD_TOKENS = 25_000_000  # 25M tokens per shard

# ─────────────────────────────────────────────────────────────────────────────
# DATASET MIX CONFIGURATION (Direct HF Streaming)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_MIX = [
    ("cerebras/SlimPajama-627B",                   None,                    "train", 2,
     lambda r: r.get("text", "")),
    ("FinanceInc/auditor_sentiment_mined",          None,                    "train", 1,
     lambda r: f"User: Analyze this financial statement.\nAssistant: {r.get('sentence', '')}"),
    ("wikimedia/wikipedia",                        "20231101.en",           "train", 1,
     lambda r: r.get("text", "")),
    ("Qwen/Qwen2.5-Math-1.5M",                    None,                    "train", 1,
     lambda r: f"User: {r.get('problem', '')}\nAssistant: {r.get('solution', '')}"),
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
    
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "transformers", "tiktoken", "accelerate"], check=True)
    print("All packages installed.\n")

def tokenization_worker():
    """Background worker that streams directly from HF, tokenizes, and saves shards."""
    print("=" * 60)
    print("1. Background Tokenizer Initializing")
    print("=" * 60)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(CACHE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id

    iterators = []
    weights = []
    for hf_id, subset, split, weight, fmt in DATASET_MIX:
        try:
            kwargs = {"split": split, "streaming": True}
            if subset:
                it = iter(load_dataset(hf_id, subset, **kwargs))
            else:
                it = iter(load_dataset(hf_id, **kwargs))
            iterators.append((it, fmt))
            weights.append(weight)
        except Exception as e:
            print(f"Warning: Failed to load dataset {hf_id}: {e}")

    if not iterators:
        print("Error: No datasets successfully loaded. Aborting background tokenizer.")
        return

    pool = []
    for i, w in enumerate(weights):
        pool.extend([i] * w)

    tokens_seen = 0
    shard_idx = 0
    buffer = []
    exhausted = set()
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
        return toks[SHARD_TOKENS:]

    print("[Tokenizer] Direct HF stream active. Generating shards...")
    random.seed(42)
    
    while len(exhausted) < len(iterators) and tokens_seen < TARGET_TOKENS:
        random.shuffle(pool)
        for idx in pool:
            if idx in exhausted:
                continue
            it, fmt = iterators[idx]
            try:
                row = next(it)
                text = fmt(row)
                if text and text.strip():
                    encoded = tokenizer.encode(text, add_special_tokens=False) + [eos_id]
                    buffer.extend(encoded)
                    tokens_seen += len(encoded)

                    while len(buffer) >= SHARD_TOKENS:
                        buffer = save_shard(buffer, shard_idx)
                        shard_idx += 1
            except StopIteration:
                exhausted.add(idx)
            except Exception:
                pass

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

def launch_training():
    print("=" * 60)
    print("2. Launching Concurrently Distributed Trainer")
    print("=" * 60)
    os.chdir(CMF_DIR)

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "scripts/train_distributed.py",
        "--preset",                 "infinity-reasoning-0.12b",
        "--token-cache-dir",        CACHE_DIR,
        "--seq-len",                "1024",
        "--micro-batch-size",       "16",
        "--grad-accum",             "4",
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
        "--delete-consumed-shards", # Automatically delete consumed shards to preserve Kaggle disk space!
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    setup_environment()

    # Start the tokenizer in a background thread
    t_thread = threading.Thread(target=tokenization_worker, daemon=True)
    t_thread.start()

    # Give the tokenizer a small head start to write the first file metadata
    print("Waiting for tokenizer to begin writing...")
    time.sleep(10.0)

    # Launch the trainer (which blocks and waits for the first shard if it's not ready yet)
    launch_training()
