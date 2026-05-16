import os
import sys
import subprocess
from pathlib import Path

def run(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # 1. Install dependencies
    print("--- Installing dependencies ---")
    run([sys.executable, "-m", "pip", "install", "-e", ".[scale,vision,power]"])
    run([sys.executable, "-m", "pip", "install", "ninja"]) # Ensure ninja is available for potential extensions

    # 2. Prepare full FineWeb-Edu dataset shards
    # We'll target 10 billion tokens for an "extensive" start.
    # On Kaggle, this will take some time to download and tokenize.
    data_dir = Path("records/data/fineweb_edu_full")
    target_tokens = 10_000_000_000 # 10B tokens
    
    if not data_dir.exists() or not (data_dir / "manifest.json").exists():
        print(f"--- Preparing {target_tokens:,} tokens from FineWeb-Edu ---")
        run([
            sys.executable, "scripts/prepare_hf_token_shards.py",
            "--dataset", "HuggingFaceTB/smollm-corpus",
            "--dataset-name", "fineweb-edu-dedup",
            "--target-tokens", str(target_tokens),
            "--shard-tokens", "100000000", # 100M tokens per shard
            "--output-dir", str(data_dir),
            "--overwrite"
        ])

    # 3. Start Distributed Training (2x T4 GPUs)
    print("--- Starting Distributed Training (0.5B Model, 2x T4) ---")
    # Using torchrun to manage the 2 GPUs on Kaggle
    # micro-batch-size 2 * 16 grad_accum * 512 seq_len = 16,384 tokens/step per GPU
    # Total 32,768 tokens per optimizer step across 2 GPUs.
    # To reach 10B tokens: 10B / 32,768 approx 305,175 steps.
    
    run([
        "torchrun",
        "--nproc_per_node=2",
        "scripts/train_distributed.py",
        "--preset", "infinity-0.5b",
        "--token-cache-dir", str(data_dir),
        "--micro-batch-size", "2",
        "--grad-accum", "16",
        "--steps", "305175",
        "--lr", "2e-4",
        "--seq-len", "512",
        "--amp",
        "--tf32",
        "--compile", # Enable if torch version supports it
        "--log-every", "10",
        "--save-every", "1000",
        "--package-out", "records/checkpoints/cmf_0.5b_fineweb_edu.package.pt"
    ])

if __name__ == "__main__":
    main()
