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
    run([sys.executable, "-m", "pip", "install", "-e", f"{str(ROOT)}[scale,vision,power]"])
    run([sys.executable, "-m", "pip", "install", "ninja"])

    # 2. Parallel Tokenization (Fast Path)
    data_dir = ROOT / "records" / "data" / "fineweb_edu_2b"
    target_tokens = 1_500_000_000 # 1.5B tokens is the perfect scientific target for a 12-hour 2x T4 run
    
    if not data_dir.exists() or not (data_dir / "manifest.json").exists():
        # Count existing shards to reassure the user
        existing_shards = sorted(data_dir.glob("tokens_*.pt")) if data_dir.exists() else []
        if existing_shards:
            print(f"\n--- [INFO] Found {len(existing_shards)} existing token shards already on disk! ---")
            print(f"--- The tokenizer will resume downloading and parsing starting at Shard {len(existing_shards)} ---\n")
        else:
            print(f"\n--- Preparing {target_tokens:,} tokens using Parallel Tokenizer (BACKGROUND PROCESS) ---\n")
        
        # Launch tokenization as a background process redirecting stdout/stderr to a dedicated log file
        log_file = ROOT / "records" / "tokenizer_output.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "w", encoding="utf-8")
        
        tok_proc = subprocess.Popen([
            sys.executable, str(ROOT / "scripts" / "prepare_hf_token_parallel.py"),
            "--dataset", "HuggingFaceTB/smollm-corpus",
            "--dataset-name", "fineweb-edu-dedup",
            "--target-tokens", str(target_tokens),
            "--shard-tokens", "25000000",
            "--output-dir", str(data_dir),
            "--append-eos"
        ], stdout=log_handle, stderr=subprocess.STDOUT)
        
        print(f"--- Parallel Tokenizer launched in background. ---")
        print(f"--- All downloader/tokenizer output is logged to: {log_file} ---")
        print("--- You can view the live download/tokenizer progress anytime by running: !tail -n 20 /kaggle/working/records/tokenizer_output.log ---\n")
    else:
        print("\n--- [INFO] Full dataset tokenization manifest already exists. Skipping downloader. ---\n")
        tok_proc = None


    # 3. Start Deep 1.2B Reasoning Training
    print("--- Starting Deep 1.2B Reasoning Training (2x T4) ---")
    # Using gradient-checkpointing to squeeze a 1.2B model into 16GB VRAM
    # We use a micro-batch of 1 per GPU due to the massive depth/size
    # but compensate with high grad-accum.
    run([
        "torchrun",
        "--nproc_per_node=2",
        str(ROOT / "scripts" / "train_distributed.py"),
        "--preset", "infinity-reasoning-1.2b",
        "--token-cache-dir", str(data_dir),
        "--micro-batch-size", "1",
        "--grad-accum", "32",
        "--lr", "5e-5",

        "--seq-len", "512",
        "--amp",
        "--tf32",
        "--gradient-checkpointing",
        "--fsdp",
        "--log-every", "1",
        "--save-every", "500",
        "--package-out", str(ROOT / "records" / "checkpoints" / "cmf_1.2b_reasoning.package.pt"),
        "--checkpoint-dir", str(ROOT / "records" / "checkpoints" / "cmf_1.2b_steps")
    ])

    if tok_proc is not None:
        print("Waiting for background parallel tokenizer to finish...")
        tok_proc.wait()

if __name__ == "__main__":
    main()
