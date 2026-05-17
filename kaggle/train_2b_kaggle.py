import os
import sys
import subprocess
from pathlib import Path

# Resolve root path of the project absolute
ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # 1. Install dependencies
    print("--- Installing dependencies ---")
    run([sys.executable, "-m", "pip", "install", "-e", str(ROOT / "[scale,vision,power]")])
    run([sys.executable, "-m", "pip", "install", "ninja"])

    # 2. Parallel Tokenization (Fast Path)
    data_dir = ROOT / "records" / "data" / "fineweb_edu_2b"
    target_tokens = 20_000_000_000 # 20B tokens for the 2B model
    
    if not data_dir.exists() or not (data_dir / "manifest.json").exists():
        print(f"--- Preparing {target_tokens:,} tokens using Parallel Tokenizer ---")
        run([
            sys.executable, str(ROOT / "scripts" / "prepare_hf_token_parallel.py"),
            "--dataset", "HuggingFaceTB/smollm-corpus",
            "--dataset-name", "fineweb-edu-dedup",
            "--target-tokens", str(target_tokens),
            "--shard-tokens", "200000000",
            "--output-dir", str(data_dir),
            "--append-eos"
        ])

    # 3. Start Deep 2B Reasoning Training
    print("--- Starting Deep 2B Reasoning Training (2x T4) ---")
    # Using gradient-checkpointing to squeeze a 2B model into 16GB VRAM
    # We use a micro-batch of 1 per GPU due to the massive depth/size
    # but compensate with high grad-accum.
    run([
        "torchrun",
        "--nproc_per_node=2",
        str(ROOT / "scripts" / "train_distributed.py"),
        "--preset", "infinity-reasoning-2b",
        "--token-cache-dir", str(data_dir),
        "--micro-batch-size", "1",
        "--grad-accum", "32",
        "--lr", "1e-4",
        "--seq-len", "512",
        "--amp",
        "--tf32",
        "--gradient-checkpointing",
        "--log-every", "1",
        "--save-every", "500",
        "--package-out", str(ROOT / "records" / "checkpoints" / "cmf_2b_reasoning.package.pt")
    ])

if __name__ == "__main__":
    main()
